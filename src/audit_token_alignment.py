"""
Audit 2, live part: what token ids does each model ACTUALLY generate for its
answer, and are they the ids the decision margin is read from?

For each model: 5 random test prompts (original option order), plus, for any
model with parsed-but-margin-less rows, 5 of those rows' prompts. Each is run
through plain `model.generate` (greedy, 10 new tokens, fp32 on CPU, no logit
processors), and the first 10 generated ids and pieces are printed next to the
pipeline's SAFE/UNSAFE id sets (lm_common.answer_token_ids), the step the
pipeline would read its margin at, and the answer parsed from the text.

Only generated tokens are printed, with digits masked: the prompts are
credentialed rows and never leave this machine.

  HF_HUB_OFFLINE=1 python src/audit_token_alignment.py
"""

import json
import random
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/home/asosoft/abtin/paper/csai/src")
from lm_common import answer_token_ids, wrap_prompt   # noqa: E402
from model import parse_answer                        # noqa: E402

RES = HERE / "results/mimic_v3b"
MODELS = [("BioMistral/BioMistral-7B", "biomistral-7b"),
          ("aaditya/Llama3-OpenBioLLM-8B", "llama3-openbiollm-8b"),
          ("mistralai/Mistral-7B-Instruct-v0.2", "mistral-7b-instruct-v0-2")]
mask = lambda s: re.sub(r"\d", "#", s)  # noqa: E731


def read(p):
    return [json.loads(l) for l in open(p)]


def main():
    torch.set_num_threads(32)
    test = {r["id"]: r for r in read(HERE / "data/mimic_v3b/counterfactual_test.jsonl")}
    swap = {r["id"]: r for r in read(HERE / "data/mimic_v3b_swap/counterfactual_test.jsonl")}
    for mid, tag in MODELS:
        tok = AutoTokenizer.from_pretrained(mid)
        ids = answer_token_ids(tok, verbose=False)
        allans = ids["SAFE"] | ids["UNSAFE"]
        rng = random.Random(0)
        sample = [("random, original order", test[i]["prompt"], None)
                  for i in rng.sample(sorted(test), 5)]
        # rows whose text names an answer but that carry no margin
        nomargin = []
        for pf in sorted(RES.glob(f"{tag}/preds_test_*_seed0_mimic3b*.jsonl")):
            if "note" in pf.name:
                continue
            src = swap if "swap" in pf.name else test
            for r in read(pf):
                if r.get("logit_margin") is None and parse_answer(r.get("raw") or ""):
                    if "rag" in pf.name or "nsai" in pf.name:
                        continue  # a RAG prompt is not the record's prompt
                    nomargin.append((pf.name, src[r["id"]]["prompt"]))
        for name, p in random.Random(1).sample(nomargin, min(5, len(nomargin))):
            sample.append((f"no margin in {name}", p, None))
        print(f"\n=== {mid}")
        print(f"pipeline SAFE ids   {[(i, tok.convert_ids_to_tokens(i)) for i in sorted(ids['SAFE'])]}")
        print(f"pipeline UNSAFE ids {[(i, tok.convert_ids_to_tokens(i)) for i in sorted(ids['UNSAFE'])]}")
        print(f"rows with a parsed answer and no margin (non-RAG files): {len(nomargin):,}")
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32)
        model.eval()
        for name, prompt, _ in sample:
            enc = tok(wrap_prompt(tok, prompt), return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=10, do_sample=False,
                                     temperature=None, top_p=None,
                                     output_scores=True, return_dict_in_generate=True,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            g = out.sequences[0, enc["input_ids"].shape[1]:].tolist()
            text = tok.decode(g, skip_special_tokens=True)
            step = next((s for s, t in enumerate(g) if t in allans), None)
            margin = None
            if step is not None:
                lg = out.scores[step][0]
                margin = (max(lg[i].item() for i in ids["SAFE"])
                          - max(lg[i].item() for i in ids["UNSAFE"]))
            print(f"  [{name}]")
            print(f"    first 10 ids   {g}")
            print(f"    pieces         {[mask(tok.convert_ids_to_tokens(t)) for t in g]}")
            print(f"    text           {mask(text)!r}")
            print(f"    parsed answer  {parse_answer(text)}   margin step {step}   "
                  f"margin {None if margin is None else round(margin, 3)}   "
                  f"sign agrees {None if margin is None else (margin > 0) == (parse_answer(text) == 'SAFE')}")
        del model


if __name__ == "__main__":
    main()
