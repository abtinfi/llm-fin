"""
Pair-level scoring for the MCQ counterfactual pairs from `bigbench_uq.py
--dataset mcqpairs`.

THE MEASUREMENT THIS EXISTS FOR
-------------------------------
Causal Consistency on flip pairs, on its own, is not evidence of reasoning. A
model with the rule "if the prompt changed, change the answer" scores 100% on
flip pairs while understanding nothing. The control pairs -- two different
wrong options for the same question -- are the discriminator: there the answer
must NOT change.

So the headline is not one number but a gap:

    flip rate on FLIP pairs      should be high  (the label really did change)
    flip rate on CONTROL pairs   should be low   (the label did not)
    discrimination = the difference

A model answering at random gets the same rate on both and a discrimination of
zero, whatever its raw accuracy looks like.

Reported per source dataset and pooled, with paired bootstrap CIs on the
discrimination, and McNemar on the two pair types where they share a question.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bigbench_analysis import auroc, boot_auroc_ci, SIGNALS, ORDER  # noqa: E402


def load(path):
    return [json.loads(l) for l in Path(path).open()]


def group_pairs(recs):
    by = collections.defaultdict(list)
    for r in recs:
        by[r["meta"]["pair_id"]].append(r)
    return {k: v for k, v in by.items() if len(v) == 2}


def _disc_ci(pairs, n_boot=5000, seed=0):
    """
    Percentile bootstrap CI for the discrimination.

    The two pair types are resampled independently because they are separate
    item sets, not two measurements of the same items. Without a CI the sign of
    a small discrimination is not interpretable at all.
    """
    def rates(ptype):
        return np.array([int(x["pred"] != y["pred"])
                         for (x, y) in pairs.values()
                         if x["meta"]["pair_type"] == ptype
                         and x["pred"] is not None and y["pred"] is not None])
    a, b = rates("flip"), rates("control")
    if len(a) < 10 or len(b) < 10:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
         for _ in range(n_boot)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def summarise(recs, name):
    pairs = group_pairs(recs)
    rows = {}
    for ptype in ("flip", "control"):
        sub = {k: v for k, v in pairs.items()
               if v[0]["meta"]["pair_type"] == ptype}
        if not sub:
            continue
        both_right, changed, n = 0, 0, len(sub)
        for k, (x, y) in sub.items():
            ok = (x["pred"] == x["label"]) and (y["pred"] == y["label"])
            both_right += int(ok)
            # did the model's own answer differ between the two arms?
            if x["pred"] is not None and y["pred"] is not None:
                changed += int(x["pred"] != y["pred"])
        rows[ptype] = {"n": n, "consistency": both_right / n,
                       "flip_rate": changed / n}

    items = [r for r in recs if r["pred"] is not None]
    acc = sum(r["pred"] == r["label"] for r in items) / max(len(items), 1)
    dist = collections.Counter(r["pred"] for r in recs)

    print(f"\n{'='*70}\n{name}   items={len(recs)}   pairs={len(pairs)}\n{'='*70}")
    print(f"item accuracy: {acc:.3f}    predictions: {dict(dist)}")
    gold = collections.Counter(r["label"] for r in recs)
    print(f"gold balance : {dict(gold)}")

    print(f"\n{'pair type':>10} {'n':>6} {'consistency':>12} {'answer-flip rate':>18}")
    for ptype, d in rows.items():
        print(f"{ptype:>10} {d['n']:6d} {d['consistency']:12.3f} "
              f"{d['flip_rate']:18.3f}")

    if "flip" in rows and "control" in rows:
        disc = rows["flip"]["flip_rate"] - rows["control"]["flip_rate"]
        lo, hi = _disc_ci(pairs)
        print(f"\ndiscrimination (flip-rate on FLIP minus flip-rate on "
              f"CONTROL): {disc:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
        if lo <= 0 <= hi:
            print("  CI covers zero: no measurable difference between the two "
                  "pair types.")
        if disc <= 0.02:
            print("  -> the model changes its answer just as readily when the "
                  "truth did NOT change.\n     Its consistency score on flip "
                  "pairs carries no evidence of reasoning.")
        elif disc < 0.15:
            print("  -> weak discrimination: it flips only slightly more often "
                  "when the truth flips.")
        else:
            print("  -> the model flips substantially more when the truth "
                  "flips. Real, if partial, sensitivity.")

    # The item set is NOT balanced: each question yields one CORRECT and three
    # INCORRECT items, so "always answer INCORRECT" scores 0.75. Report the
    # best constant answer, not the model's own most frequent one -- otherwise
    # a badly biased model looks like it is clearing a bar it is far below.
    print("\nconstant-answer baselines (the design is 1 CORRECT : 3 INCORRECT)")
    best_lab, best_acc = None, -1.0
    for lab in sorted(gold):
        a = gold[lab] / len(recs)
        print(f"  always answering {lab:<10s} -> item accuracy {a:.3f}")
        if a > best_acc:
            best_lab, best_acc = lab, a
    verdict = ("BELOW" if acc < best_acc else "above")
    print(f"  the model achieves {acc:.3f} -- {verdict} the best constant "
          f"answer ({best_lab}, {best_acc:.3f})")
    if dist:
        top, cnt = dist.most_common(1)[0]
        print(f"  response bias: says {top} on {cnt/len(recs):.1%} of items "
              f"while {gold[top]/len(recs):.1%} of items are {top}")
    return rows


def uq_table(recs, n_boot):
    items = [r for r in recs if r["pred"] is not None]
    wrong = np.array([r["pred"] != r["label"] for r in items], bool)
    if wrong.all() or not wrong.any():
        print("\n(all items same correctness; AUROC undefined)")
        return
    print(f"\nuncertainty signals as wrongness detectors (n={len(items)})")
    print(f"{'signal':>16} {'AUROC':>8} {'95% CI':>18}")
    for sig in ORDER:
        field, to_unc, _ = SIGNALS[sig]
        usable = [r for r in items if r.get(field) is not None]
        if len(usable) < 20:
            continue
        unc = np.array([to_unc(r[field]) for r in usable], float)
        w = np.array([r["pred"] != r["label"] for r in usable], bool)
        if w.all() or not w.any():
            continue
        a = auroc(unc, w)
        lo, hi = boot_auroc_ci(unc, w, n_boot=n_boot)
        print(f"{sig:>16} {a:8.3f}   [{lo:.3f}, {hi:.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    recs = load(args.preds)
    for r in recs:
        r["meta"] = r.get("meta") or {}

    by_src = collections.defaultdict(list)
    for r in recs:
        by_src[r["meta"].get("source", "?")].append(r)
    for src, sub in sorted(by_src.items()):
        summarise(sub, src)
    if len(by_src) > 1:
        summarise(recs, "POOLED")
    uq_table(recs, args.n_boot)


if __name__ == "__main__":
    main()
