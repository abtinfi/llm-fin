"""
The constraint-layer (Aim 3) training set for the v3.1 arm: data/mimic_v3_cl.

WHY A SUBSAMPLE
  constraint_layer.py optimises one PAIR per step with no batching and caches a
  full-vocabulary base distribution per record on the CPU. On all 49,803 v3
  train pairs at the arm's 8 epochs that is ~400k steps (many hours, with no
  checkpoint inside training, so the 03:01 reboot would restart it) and, for
  Llama-3's 128k vocabulary, ~50 GB of cached distributions. The only MIMIC
  precedent, adapter_mimic.npz, was trained on 19 pairs. So the protocol is
  kept -- same epochs, layer, graph, loss weights -- and the data is a seeded
  sample, stated as such.

WHY CAUSAL PAIRS ONLY
  data/mimic, the arm adapter_mimic.npz was trained on, has no control pairs.
  constraint_layer.py would accept them (L_ontology is skipped for a
  same-label pair), but training on them would change what the adapter is
  taught relative to that precedent. Controls stay in the EVALUATION.

WHAT test/heldout HERE ARE FOR
  constraint_layer.py reports before/after accuracy and CC on the --data
  directory's own test and heldout splits. On the full v3 splits that
  diagnostic alone would take hours, so it gets a sample too. The HEADLINE
  cl / nsai_uq_cl rows are scored by run_eval.py on the FULL v3 test and
  heldout splits with the saved adapter (run_mimic_v3_cl.sh).

  python src/build_mimic_cl_subset.py --src <csai>/data/mimic_v3 --out data/mimic_v3_cl
"""

import argparse
import collections
import json
import random
from pathlib import Path

SIZES = {"train": 1000, "test": 1000, "heldout": 500}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/mimic_v3_cl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    meta = json.loads((src / "build_meta.json").read_text())
    assert meta["credentialed"], f"{src} is not the v3.1 arm"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    counts = {}
    for split, n in SIZES.items():
        by = collections.defaultdict(list)
        for line in open(src / f"counterfactual_{split}.jsonl"):
            r = json.loads(line)
            if not r["is_control"]:
                by[r["pair_id"]].append(r)
        pids = sorted(by)
        rng.shuffle(pids)
        keep = sorted(pids[:n])
        with (out / f"counterfactual_{split}.jsonl").open("w") as fh:
            for p in keep:
                for r in by[p]:
                    fh.write(json.dumps(r) + "\n")
        counts[split] = 2 * len(keep)
        print(f"  {split}: {len(keep):,} of {len(pids):,} causal pairs")
    (out / "rag_corpus.jsonl").write_text((src / "rag_corpus.jsonl").read_text())
    meta.update({"derived_from": str(src), "subset_seed": args.seed,
                 "subset_pairs": SIZES, "controls_included": False,
                 "purpose": "constraint-layer training + its internal "
                            "before/after diagnostic only; headline rows are "
                            "scored on the full v3 splits",
                 "counts": counts})
    (out / "build_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
