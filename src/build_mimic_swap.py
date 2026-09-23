"""
Option-order diagnostic for the v3.1 arm: the same items with the answer
options listed as "UNSAFE or SAFE" instead of "SAFE or UNSAFE".

WHY. On 16 Demo items OpenBioLLM answered SAFE 16/16 with the original order
and 0/16 with the order swapped: it copies the first option and reads none of
the clinical content. A model that is doing the task should give the SAME
answer under both orders. Scoring base under the swapped order, for every
model, measures that directly -- and tells a reader which models' rows carry
information about the case at all. Nothing else changes: same items, same ids,
same labels; only the option order in the instruction line.

  python src/build_mimic_swap.py --src data/mimic_v3b --out data/mimic_v3b_swap
"""

import argparse
import json
import shutil
from pathlib import Path

ORIG = "Answer with exactly one word on the first line: SAFE or UNSAFE."
SWAP = "Answer with exactly one word on the first line: UNSAFE or SAFE."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in ["calib", "test", "heldout"]:
        n = 0
        with (out / f"counterfactual_{split}.jsonl").open("w") as fh:
            for line in open(src / f"counterfactual_{split}.jsonl"):
                r = json.loads(line)
                assert r["prompt"].count(ORIG) == 1, r["id"]
                r["prompt"] = r["prompt"].replace(ORIG, SWAP)
                r["option_order"] = "UNSAFE_first"
                fh.write(json.dumps(r) + "\n"); n += 1
        print(f"  {split}: {n:,} items")
    shutil.copy(src / "rag_corpus.jsonl", out / "rag_corpus.jsonl")
    meta = json.loads((src / "build_meta.json").read_text())
    meta.update({"derived_from": str(src), "option_order": "UNSAFE_first",
                 "purpose": "option-order (position bias) diagnostic"})
    (out / "build_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
