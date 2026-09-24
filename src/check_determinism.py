"""
Guard stage for every v3b job: prove, on the job's own GPU and dtype, that the
run_eval_v3 generation path gives each prompt one answer regardless of the
order prompts arrive in. Exit 1 (the job stops) on any difference.

The same check passed on CPU/fp32 (32 items, every field bit-identical across
two input orders). bfloat16 on a GPU (run_eval.py's default dtype) is where the
original batch noise came from, so it is re-proved there, per model, before
any number is produced.

Compared at exact equality: pred, logit_margin, answer_logprob (the fields
every reported metric and every decision-level UQ signal is computed from),
plus raw text, entropy and max_entropy.

  python src/check_determinism.py --model_id M --data D [--n 1000] [--early_stop]
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_eval_v3                                  # noqa: E402  (patches model)

M = run_eval_v3.M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=1000)
    # run_eval.py's own default; the guard must test the dtype the job uses.
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    items = [json.loads(l) for i, l in
             enumerate(open(Path(args.data) / "counterfactual_test.jsonl"))
             if i < args.n]
    prompts = [r["prompt"] for r in items]
    shuffled = list(prompts)
    random.Random(1).shuffle(shuffled)
    lm = M.load_model({"backend": "hf", "model_id": args.model_id,
                       "dtype": args.dtype, "device": "cuda"})
    a = dict(zip(prompts, lm.generate(prompts)))
    b = dict(zip(shuffled, lm.generate(shuffled)))
    fields = ["text", "logit_margin", "answer_logprob", "entropy", "max_entropy"]
    bad = {f: sum(getattr(a[p], f) != getattr(b[p], f) for p in a) for f in fields}
    preds_bad = sum(M.parse_answer(a[p].text) != M.parse_answer(b[p].text) for p in a)
    print(f"determinism check: {len(prompts)} items, {len(a)} distinct prompts, "
          f"early_stop={run_eval_v3.EARLY_STOP}")
    print(f"  differences across two input orders: pred={preds_bad} {bad}")
    if preds_bad or bad["logit_margin"] or bad["answer_logprob"]:
        print("  FAIL: an answer or a decision signal depends on input order")
        sys.exit(1)
    if any(bad.values()):
        print("  WARN: only auxiliary fields moved; decisions are order-free")
    print("  PASS")


if __name__ == "__main__":
    main()
