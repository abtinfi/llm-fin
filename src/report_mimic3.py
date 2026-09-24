"""
The MIMIC v3.1 results, read the way the audit says they must be read.

src/audit_mimic_dupes.py found three properties of the v3 arm that make the
headline numbers in summary_*.json misleading on their own. None is a bug in
the pipeline (every number reproduces exactly, see audit_mimic_preds.py); all
three are properties of the benchmark that a table must carry:

1. DUPLICATE PROMPTS. The note is minimal (age, sex, one value, drug), so the
   166,264 test items are only 13,870 distinct prompts. Greedy decoding gives
   an identical prompt an identical answer (up to batch effects, item 4), so
   an item-level bootstrap treats ~12 copies of one observation as 12
   observations and its interval is several times too narrow. Intervals here
   resample DISTINCT PROMPTS (accuracy) and distinct PAIR PROMPT-TUPLES (CC).

2. CONTRADICTORY GROUND TRUTH. metformin_egfr30 and metformin_egfr45 print
   the same sentence ("... serum creatinine of X ... Metformin is being
   considered") but apply thresholds 30 and 45. For eGFR in [30, 45) the same
   prompt is SAFE in one family and UNSAFE in the other; 21% of test items sit
   on such a prompt and any model must be wrong on one copy. The two families
   are reported separately, and a "conflict-free" view drops those items.

3. TRAIN/TEST PROMPT OVERLAP. The split is patient-disjoint, not
   prompt-disjoint. Nothing is trained on train except the constraint layer,
   whose adapter saw 1,000 pairs (data/mimic_v3_cl); 38% of test items share a
   prompt with them. cl / nsai_uq_cl are therefore also reported on the pairs
   neither of whose prompts the adapter saw (heldout has no overlap at all).

4. BATCH EFFECTS. Under stock batching identical prompts did not always get
   identical answers (bfloat16 with left padding made near-ties depend on the
   batch). The v3b arm runs through run_eval_v3, which removes this by
   construction; the per-row count stays in the table as a check and should
   read 0.

Items 2 and the rounding defect are fixed in data/mimic_v3b itself
(--question_version v2, printed-precision guard), so its conflict-free view
should equal "all". The views stay so a regression would show.

Also reported: the always-SAFE baseline (the label is SAFE on ~68% of test
items, so accuracy must be read against it), and a row count of items whose
label disagrees with their own printed value (one warfarin pair: the builder's
rounding guard used 2 decimals, INR is printed with 1; fixed in build_mimic).

PRINTS AND WRITES AGGREGATES ONLY.

  python src/report_mimic3.py            # writes results/mimic_v3/AUDIT_MIMIC3.md
"""

import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from audit_mimic_labels import label_from_prompt  # noqa: E402

CSAI = Path("/home/asosoft/abtin/paper/csai")
MODELS = ["biomistral-7b", "llama3-openbiollm-8b", "mistral-7b-instruct-v0-2"]
VARIANTS = ["base", "sym", "rag", "uq", "nsai", "nsai_uq", "cl", "nsai_uq_cl"]
# The corrected arm (question_version v2, printed-precision guard, run
# through run_eval_v3). Both arms write into results/mimic_v3b/<model>/,
# told apart by tag.
RES = Path(os.environ.get("V3B_RESULTS", HERE / "results/mimic_v3b"))  # env: tests only
ARMS = {
    "v3b": (HERE / "data/mimic_v3b", RES, "_mimic3b"),
    "note": (HERE / "data/mimic_v3b_note", RES, "_mimic3bnote"),
}
SWAP_TAG = "_mimic3bswap"
CL_TRAIN = HERE / "data/mimic_v3b_cl/counterfactual_train.jsonl"
N_BOOT = 2000


def cluster_ci(keys, flags, seed=0):
    """95% bootstrap CI of mean(flags), resampling clusters (keys)."""
    if not flags:
        return (None, None)
    s, c = collections.defaultdict(float), collections.defaultdict(int)
    for k, f in zip(keys, flags):
        s[k] += f; c[k] += 1
    ks = list(s)
    S = np.array([s[k] for k in ks]); C = np.array([c[k] for k in ks])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ks), size=(N_BOOT, len(ks)))
    m = S[idx].sum(1) / C[idx].sum(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def fmt_ci(ci):
    return "—" if ci[0] is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def load_items(data, split):
    items = {}
    for line in open(data / f"counterfactual_{split}.jsonl"):
        r = json.loads(line)
        items[r["id"]] = r
    labels = collections.defaultdict(set)
    for r in items.values():
        labels[r["prompt"]].add(r["label"])
    conflict = {p for p, l in labels.items() if len(l) > 1}
    bad_label = {i for i, r in items.items()
                 if label_from_prompt(r["family"], r["vignette"])[0] != r["label"]}
    return items, conflict, bad_label


def metrics(preds, items, keep=lambda r: True):
    """Accuracy and CC with prompt-cluster CIs, over items passing `keep`."""
    rs = [p for p in preds if keep(items[p["id"]])]
    if not rs:
        return None
    ok = [float((not p["abstained"]) and p["pred"] == p["label"]) for p in rs]
    acc = float(np.mean(ok))
    acc_ci = cluster_ci([items[p["id"]]["prompt"] for p in rs], ok)
    base_rate = float(np.mean([p["label"] == "SAFE" for p in rs]))
    pairs = collections.defaultdict(list)
    for p in rs:
        pairs[p["pair_id"]].append(p)
    ccf, cck, sf = [], [], []
    for pid, arms in pairs.items():
        if len(arms) != 2:
            continue
        both = all((not a["abstained"]) and a["pred"] is not None for a in arms)
        if items[arms[0]["id"]].get("is_control"):
            if both:
                sf.append(float(arms[0]["pred"] != arms[1]["pred"]))
            continue
        ccf.append(float(all((not a["abstained"]) and a["pred"] == a["label"]
                             for a in arms)))
        cck.append(tuple(sorted(items[a["id"]]["prompt"] for a in arms)))
    return {"n": len(rs), "acc": acc, "acc_ci": acc_ci, "safe_base": base_rate,
            "cc": float(np.mean(ccf)) if ccf else None,
            "cc_ci": cluster_ci(cck, ccf), "n_pairs": len(ccf),
            "sf": float(np.mean(sf)) if sf else None,
            "cov": float(np.mean([not p["abstained"] for p in rs]))}


def main():
    cl_prompts = ({json.loads(l)["prompt"] for l in open(CL_TRAIN)}
                  if CL_TRAIN.exists() else set())
    out = ["# MIMIC-IV v3.1 — results as the audit requires them to be read",
           "", "Generated by `src/report_mimic3.py`; see its docstring for why "
           "each view exists. Intervals resample distinct prompts, not items.",
           ""]
    for arm, (data, root, tag) in ARMS.items():
        for split in ["test", "heldout"]:
            if not (data / f"counterfactual_{split}.jsonl").exists():
                continue
            items, conflict, bad_label = load_items(data, split)
            fams = sorted({r["family"] for r in items.values()})
            out += [f"## {arm} / {split}", "",
                    f"{len(items):,} items, "
                    f"{len({r['prompt'] for r in items.values()}):,} distinct "
                    f"prompts; {sum(r['prompt'] in conflict for r in items.values()):,} "
                    f"items on a prompt carrying both labels; "
                    f"{len(bad_label):,} items whose label disagrees with their "
                    f"printed value (excluded from every row below).", ""]
            # One representative pair per distinct (prompt, prompt) tuple: the
            # effective sample, with every duplicate counted once.
            first_pair, rep = {}, set()
            by_pair = collections.defaultdict(list)
            for r in items.values():
                by_pair[r["pair_id"]].append(r)
            for pid in sorted(by_pair):
                key = tuple(sorted(a["prompt"] for a in by_pair[pid]))
                if key not in first_pair:
                    first_pair[key] = pid
                    rep.add(pid)
            views = [("all", lambda r: r["id"] not in bad_label),
                     ("distinct pairs", lambda r: r["pair_id"] in rep
                      and r["id"] not in bad_label),
                     ("conflict-free", lambda r: r["id"] not in bad_label
                      and r["prompt"] not in conflict)]
            views += [(f, (lambda f: lambda r: r["family"] == f
                           and r["id"] not in bad_label)(f)) for f in fams]
            out += ["| model | variant | view | n | always-SAFE acc | acc [95% CI] "
                    "| CC [95% CI] | pairs | spurious flip | coverage "
                    "| same prompt, different pred |",
                    "|---|---|---|---|---|---|---|---|---|---|---|"]
            for model in MODELS:
                for v in VARIANTS:
                    pf = root / model / f"preds_{split}_{v}_seed0{tag}.jsonl"
                    if not pf.exists():
                        continue
                    preds = [json.loads(l) for l in open(pf)]
                    byp = collections.defaultdict(set)
                    for p in preds:
                        byp[items[p["id"]]["prompt"]].add(p["pred"])
                    incons = sum(len(s) > 1 for s in byp.values())
                    vv = list(views)
                    if v in ("cl", "nsai_uq_cl") and cl_prompts and arm == "v3b":
                        pair_seen = collections.defaultdict(bool)
                        for r in items.values():
                            pair_seen[r["pair_id"]] |= r["prompt"] in cl_prompts
                        vv.append(("unseen by adapter",
                                   lambda r, ps=pair_seen: not ps[r["pair_id"]]
                                   and r["id"] not in bad_label))
                    for name, keep in vv:
                        m = metrics(preds, items, keep)
                        if m is None:
                            continue
                        out.append(
                            f"| {model} | {v} | {name} | {m['n']:,} | "
                            f"{fmt(m['safe_base'])} | {fmt(m['acc'])} "
                            f"{fmt_ci(m['acc_ci'])} | {fmt(m['cc'])} "
                            f"{fmt_ci(m['cc_ci'])} | {m['n_pairs']:,} | "
                            f"{fmt(m['sf'])} | {fmt(m['cov'])} | "
                            f"{incons if name == 'all' else ''} |")
            out.append("")
    # Option order: base with "SAFE or UNSAFE" vs "UNSAFE or SAFE", same items.
    data, root, tag = ARMS["v3b"]
    out += ["## Option-order diagnostic (v3b test, base)", "",
            "A model reading the case gives the same answer whichever option "
            "is listed first. `changed` is the share of items whose answer "
            "flips when only the order of the two options in the instruction "
            "line is swapped.", "",
            "| model | n | SAFE share, SAFE-first | SAFE share, UNSAFE-first "
            "| answer changed | acc SAFE-first | acc UNSAFE-first |",
            "|---|---|---|---|---|---|---|"]
    for model in MODELS:
        a_p = root / model / f"preds_test_base_seed0{tag}.jsonl"
        b_p = root / model / f"preds_test_base_seed0{SWAP_TAG}.jsonl"
        if not (a_p.exists() and b_p.exists()):
            continue
        a = {r["id"]: r for r in map(json.loads, open(a_p))}
        b = {r["id"]: r for r in map(json.loads, open(b_p))}
        ids = sorted(set(a) & set(b))
        n = len(ids)
        out.append(
            f"| {model} | {n:,} | "
            f"{sum(a[i]['pred'] == 'SAFE' for i in ids) / n:.3f} | "
            f"{sum(b[i]['pred'] == 'SAFE' for i in ids) / n:.3f} | "
            f"{sum(a[i]['pred'] != b[i]['pred'] for i in ids) / n:.3f} | "
            f"{sum(a[i]['pred'] == a[i]['label'] for i in ids) / n:.3f} | "
            f"{sum(b[i]['pred'] == b[i]['label'] for i in ids) / n:.3f} |")
    out.append("")
    dst = RES / "AUDIT_MIMIC3B.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out) + "\n")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
