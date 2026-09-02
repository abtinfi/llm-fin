"""
Did retrieval actually work? (P3 in HANDOFF.md)

`base->rag` is not significant on the test split and Causal Consistency drops on
held-out. Before reporting "RAG does not help", this separates the two possible
causes:

  - retrieval hit rate is LOW  -> TF-IDF is the problem, not the model. Try a
    dense retriever before drawing any conclusion about RAG as a method.
  - retrieval hit rate is HIGH -> the correct guideline was in the context and
    the model ignored it. That is a genuine, reportable finding.

The gold document for a case of family F is `guideline::F`. Every prediction
record stores the retrieved doc ids in the `retrieved` field.

Reports hit@k for k=1..K, precision@K, mean rank of the gold doc, the fraction
of retrieved slots taken by distractors, and a per-family breakdown so a single
badly-worded guideline does not hide behind the average. Also splits accuracy by
whether the gold doc was retrieved -- if retrieval works but accuracy is flat
across that split, the model is ignoring the context.

  python src/retrieval_diagnostic.py --preds results/preds_test_rag_seed0.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--compare",
                    help="matching base-variant preds, to check whether having "
                         "the gold doc changed the answer vs. no-RAG")
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.preds).open()]
    if not recs:
        print("no records")
        return
    if not any(r.get("retrieved") for r in recs):
        print("no `retrieved` field populated -- is this a RAG variant?")
        return

    k_max = max(len(r.get("retrieved") or []) for r in recs)
    print(f"{len(recs)} cases, top-k = {k_max}\n")

    hits_at = defaultdict(int)      # k -> n cases with gold in top-k
    ranks = []                      # 1-indexed rank of gold, None if absent
    n_distractor_slots = 0
    n_slots = 0
    per_family = defaultdict(lambda: {"n": 0, "hit": 0, "correct": 0,
                                      "correct_hit": 0, "correct_miss": 0,
                                      "n_hit": 0, "n_miss": 0})

    for r in recs:
        gold = f"guideline::{r['family']}"
        got = r.get("retrieved") or []
        n_slots += len(got)
        n_distractor_slots += sum(1 for d in got if d.startswith("distractor::"))

        rank = got.index(gold) + 1 if gold in got else None
        ranks.append(rank)
        for k in range(1, k_max + 1):
            if rank is not None and rank <= k:
                hits_at[k] += 1

        f = per_family[r["family"]]
        f["n"] += 1
        correct = r["pred"] == r["label"]
        f["correct"] += int(correct)
        if rank is not None:
            f["hit"] += 1
            f["n_hit"] += 1
            f["correct_hit"] += int(correct)
        else:
            f["n_miss"] += 1
            f["correct_miss"] += int(correct)

    n = len(recs)
    print("=== retrieval hit rate (gold = guideline::{family}) ===")
    for k in range(1, k_max + 1):
        print(f"  hit@{k}: {hits_at[k]}/{n} = {hits_at[k]/n:.3f}")
    found = [r for r in ranks if r is not None]
    if found:
        print(f"  mean rank of gold doc when retrieved: {sum(found)/len(found):.2f}")
    # exactly one gold doc exists per case, so precision@K is hit@K / K
    print(f"  precision@{k_max}: {hits_at[k_max]/(n*k_max):.3f} "
          f"(1 gold doc exists per case, so the ceiling is {1/k_max:.3f})")
    print(f"  distractor share of retrieved slots: "
          f"{n_distractor_slots}/{n_slots} = {n_distractor_slots/n_slots:.3f}")

    print("\n=== per family ===")
    print(f"{'family':>28} {'n':>4} {'hit@k':>7} {'acc':>7} "
          f"{'acc|hit':>8} {'acc|miss':>9}")
    for fam in sorted(per_family):
        f = per_family[fam]
        ah = f["correct_hit"] / f["n_hit"] if f["n_hit"] else float("nan")
        am = f["correct_miss"] / f["n_miss"] if f["n_miss"] else float("nan")
        print(f"{fam:>28} {f['n']:4d} {f['hit']/f['n']:7.3f} "
              f"{f['correct']/f['n']:7.3f} {ah:8.3f} {am:9.3f}")

    hit_recs = [r for r, rk in zip(recs, ranks) if rk is not None]
    miss_recs = [r for r, rk in zip(recs, ranks) if rk is None]
    print("\n=== accuracy split by whether the gold doc was retrieved ===")
    for name, group in [("gold retrieved", hit_recs), ("gold missed", miss_recs)]:
        if group:
            acc = sum(r["pred"] == r["label"] for r in group) / len(group)
            print(f"  {name:>15}: n={len(group):4d}  acc={acc:.3f}")
        else:
            print(f"  {name:>15}: n=0")

    overall_hit = hits_at[k_max] / n
    print("\n=== reading ===")
    if overall_hit < 0.5:
        print(f"Hit rate {overall_hit:.3f} is LOW. TF-IDF retrieval is failing, "
              f"so the flat RAG row does not license any claim about RAG as a\n"
              f"method. Try a dense retriever before concluding anything.")
    else:
        print(f"Hit rate {overall_hit:.3f} is HIGH -- the correct guideline was "
              f"in the context for most cases. If accuracy is flat across the\n"
              f"hit/miss split above, the model is ignoring retrieved context. "
              f"That is a genuine, reportable finding.")

    if args.compare:
        base = {r["id"]: r for r in
                (json.loads(l) for l in Path(args.compare).open())}
        changed = flipped_right = flipped_wrong = 0
        for r in hit_recs:
            b = base.get(r["id"])
            if b is None or b["pred"] == r["pred"]:
                continue
            changed += 1
            if r["pred"] == r["label"]:
                flipped_right += 1
            elif b["pred"] == b["label"]:
                flipped_wrong += 1
        print(f"\n=== among cases where the gold doc WAS retrieved (n={len(hit_recs)}) ===")
        print(f"  answer changed vs. no-RAG: {changed}")
        print(f"    wrong -> right: {flipped_right}")
        print(f"    right -> wrong: {flipped_wrong}")
        if changed == 0:
            print("  The context had zero effect on the decision even when the "
                  "correct guideline was present.")


if __name__ == "__main__":
    main()
