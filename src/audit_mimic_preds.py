"""
Independent audit of MIMIC prediction files against their dataset and summary.

Per preds_<split>_<variant>_seed0<tag>.jsonl:
  1. every data item scored exactly once, no extras; label copied correctly
  2. pred == model.parse_answer(raw), recomputed
  3. accuracy / causal consistency / violation / spurious-flip recomputed
     here from scratch (not via metrics.score) and compared with the summary
  4. identical prompts -> identical pred (greedy decoding is deterministic; a
     disagreement means batch composition or padding changed an answer)
  5. gate firing per family (renal families are rendered WITHOUT eGFR, so
     the rules gate should stay silent there; potassium and INR are rendered
     for the extractor, so it should fire)
  6. a random sample of 200 items re-checked field by field

PRINTS COUNTS AND RATES ONLY.

  python src/audit_mimic_preds.py <data_dir> <results_dir> <tag> [split]
"""

import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/home/asosoft/abtin/paper/csai/src")
from model import parse_answer                    # noqa: E402


def load(p):
    return [json.loads(l) for l in open(p)]


def my_scores(recs):
    n = len(recs)
    acc = sum((not r["abstained"]) and r["pred"] == r["label"] for r in recs) / n
    uns = [r for r in recs if r["label"] == "UNSAFE"]
    viol = sum((not r["abstained"]) and r["pred"] == "SAFE" for r in uns) / len(uns)
    pairs = collections.defaultdict(list)
    for r in recs:
        pairs[r["pair_id"]].append(r)
    cc, sf = [], []
    for arms in pairs.values():
        if len(arms) != 2:
            continue
        both = all((not a["abstained"]) and a["pred"] is not None for a in arms)
        if any(a.get("is_control") for a in arms):
            if both:
                sf.append(arms[0]["pred"] != arms[1]["pred"])
            continue
        cc.append(all((not a["abstained"]) and a["pred"] == a["label"] for a in arms))
    return {"accuracy": acc, "violation_rate": viol,
            "causal_consistency": sum(cc) / len(cc),
            "spurious_flip_rate": (sum(sf) / len(sf)) if sf else None}


def main(data, res, tag, split="test"):
    data, res = Path(data), Path(res)
    items = {r["id"]: r for r in load(data / f"counterfactual_{split}.jsonl")}
    summ = json.load(open(res / f"summary_{split}{tag}.json"))
    rng = random.Random(0)
    for pf in sorted(res.glob(f"preds_{split}_*_seed0{tag}.jsonl")):
        variant = pf.name[len(f"preds_{split}_"):-len(f"_seed0{tag}.jsonl")]
        preds = load(pf)
        print(f"\n--- {res.name} / {split} / {variant}  ({len(preds):,} preds)")
        ids = [p["id"] for p in preds]
        c = collections.Counter()
        c["missing"] = len(set(items) - set(ids))
        c["extra"] = len(set(ids) - set(items))
        c["duplicate_ids"] = len(ids) - len(set(ids))
        for p in preds:
            it = items.get(p["id"])
            if it is None:
                continue
            p["is_control"] = it.get("is_control", False)
            c["label_mismatch"] += p["label"] != it["label"]
            c["pair_mismatch"] += p["pair_id"] != it["pair_id"]
            c["pred_not_parse_of_raw"] += p["pred"] != parse_answer(p["raw"])
            c["unparsable"] += p["pred"] is None
        print("  integrity:", {k: v for k, v in c.items()})
        # 3. metrics recomputed independently
        row = next(r for r in summ if r.get("variant") == variant
                   and r.get("seed", 0) == 0)
        mine = my_scores(preds)
        for k, v in mine.items():
            s = row.get(k)
            ok = (v is None and s is None) or (v is not None and s is not None
                                               and abs(v - s) < 1e-12)
            print(f"  {k:20s} mine={v if v is None else round(v, 6)!s:>10} "
                  f"summary={s if s is None else round(s, 6)!s:>10}  "
                  f"{'OK' if ok else 'MISMATCH'}")
        # 4. same prompt -> same answer
        by_prompt = collections.defaultdict(set)
        for p in preds:
            by_prompt[items[p["id"]]["prompt"]].add(p["pred"])
        incons = sum(1 for s in by_prompt.values() if len(s) > 1)
        print(f"  identical prompts with DIFFERENT preds: {incons:,} of "
              f"{len(by_prompt):,}")
        # 5. gate firing per family
        g = collections.defaultdict(lambda: [0, 0])
        for p in preds:
            g[p["family"]][0] += bool(p.get("gate_fired"))
            g[p["family"]][1] += 1
        print("  gate_fired rate:", {f: round(a / b, 3) for f, (a, b) in sorted(g.items())})
        print("  pred SAFE share:",
              round(sum(p["pred"] == "SAFE" for p in preds) / len(preds), 3),
              "| label SAFE share:",
              round(sum(p["label"] == "SAFE" for p in preds) / len(preds), 3))
        # 6. random sample, field by field
        bad = 0
        for p in rng.sample(preds, min(200, len(preds))):
            it = items[p["id"]]
            bad += not (p["family"] == it["family"] and p["label"] == it["label"]
                        and p["pair_id"] == it["pair_id"]
                        and p["pred"] == parse_answer(p["raw"]))
        print(f"  random sample of 200: {bad} with any field wrong")


if __name__ == "__main__":
    main(*sys.argv[1:])
