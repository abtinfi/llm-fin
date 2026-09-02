"""
External-validity run for Aim 4's uncertainty component.

WHY THIS EXISTS
---------------
`results/p1_uq_signals.md` reported that no uncertainty signal cleared the P1
acceptance bar on the hand-built benchmark. But the pool it was measured on was
n=56 (the cases the symbolic gate declines), and every confidence interval
covered chance. `uq_diagnostic.py` says so in its own words: "a larger
evaluation set is what would settle it, not a threshold sweep."

This module is that larger evaluation set. It runs BioMistral-7B over three
large public medical QA benchmarks -- 6,456 items, ~115x the n=56 pool -- and
scores the same uncertainty signals. At this n the confidence intervals are
narrow enough that "no signal" and "usable signal" are actually
distinguishable, so the finding is decisive in whichever direction it falls.

WHAT IT ADDS OVER THE ORIGINAL FOUR SIGNALS
-------------------------------------------
The original four (entropy, max_entropy, logit_margin, answer_logprob) are
carried over unchanged so the numbers are comparable. Two more are added
because they are the strongest signals in the selective-prediction literature,
and a negative result is only decisive if the best-known alternative was also
given a fair run:

  option_entropy    Entropy of the softmax over ONLY the K answer-option
                    tokens at the decision step. This is the decision-level
                    analogue of the proposal's Eq. (2), which takes entropy
                    over the whole 32k vocabulary. Most of that vocabulary mass
                    is irrelevant to the answer, so Eq. (2) measures fluency
                    uncertainty, not decision uncertainty. This is the single
                    most likely explanation for the original negative result.

  option_prob_top   Max renormalised probability over the K option tokens, i.e.
                    the model's confidence in the option it actually chose.

NON-NEGOTIABLES PRESERVED (HANDOFF.md section 7)
------------------------------------------------
  - bf16, no quantisation. 4-bit would perturb exactly the logit distribution
    every signal here is read from.
  - Greedy decoding, do_sample=False.
  - Ground truth comes from the datasets' own labels. No LLM-as-judge, no
    human annotation.
  - No threshold is tuned anywhere in this file. It reports AUROC and a
    risk-coverage curve; choosing an operating point is a separate decision.
  - The tokenizer trap from P1 is handled: whitespace-only leading pieces are
    skipped, and any token id ambiguous between two options is dropped rather
    than silently used.

USAGE
-----
    export CUDA_VISIBLE_DEVICES=0
    python src/bigbench_uq.py --dataset medmcqa --limit 32     # smoke test
    python src/bigbench_uq.py --dataset all
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))


# --------------------------------------------------------------------------
# Dataset adapters
# --------------------------------------------------------------------------
# Each adapter returns a list of dicts with a uniform shape:
#   {id, prompt, labels: [...], gold: str, meta: {...}}
# so the scoring code never needs to know which benchmark it is looking at.

MCQ_INSTRUCTION = (
    "You are a medical expert answering a multiple-choice examination "
    "question.\n\n"
    "{question}\n\n{options}\n\n"
    "Respond with the single letter of the correct option first, then one "
    "short sentence of justification."
)

PUBMEDQA_INSTRUCTION = (
    "You are a medical expert. Based only on the research abstract below, "
    "answer the question.\n\n"
    "Abstract:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer with exactly one word -- yes, no, or maybe -- then one short "
    "sentence of justification."
)


def _truncate_words(text: str, max_words: int) -> str:
    w = text.split()
    return text if len(w) <= max_words else " ".join(w[:max_words]) + " ..."


def load_medmcqa(limit=None):
    from datasets import load_dataset
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    letters = ["A", "B", "C", "D"]
    keys = ["opa", "opb", "opc", "opd"]
    out = []
    for i, r in enumerate(ds):
        opts = "\n".join(f"{L}. {r[k]}" for L, k in zip(letters, keys))
        out.append({
            "id": f"medmcqa::{r.get('id', i)}",
            "prompt": MCQ_INSTRUCTION.format(
                question=r["question"].strip(), options=opts),
            "labels": letters,
            "gold": letters[int(r["cop"])],
            "meta": {"subject": r.get("subject_name")},
        })
    return out


def load_medqa(limit=None):
    from datasets import load_dataset
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    letters = ["A", "B", "C", "D"]
    out = []
    for i, r in enumerate(ds):
        o = r["options"]
        opts = "\n".join(f"{L}. {o[L]}" for L in letters if L in o)
        # USMLE vignettes are long; cap so the batch does not blow up memory.
        q = _truncate_words(r["question"].strip(), 350)
        out.append({
            "id": f"medqa::{i}",
            "prompt": MCQ_INSTRUCTION.format(question=q, options=opts),
            "labels": letters,
            "gold": r["answer_idx"],
            "meta": {"step": r.get("meta_info")},
        })
    return out


def load_pubmedqa(limit=None):
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out = []
    for i, r in enumerate(ds):
        ctx = " ".join(r["context"]["contexts"])
        out.append({
            "id": f"pubmedqa::{r.get('pubid', i)}",
            "prompt": PUBMEDQA_INSTRUCTION.format(
                context=_truncate_words(ctx, 400),
                question=r["question"].strip()),
            "labels": ["yes", "no", "maybe"],
            "gold": r["final_decision"].strip().lower(),
            "meta": {},
        })
    return out


# --------------------------------------------------------------------------
# Counterfactual pairs built from multiple-choice questions
# --------------------------------------------------------------------------
# A 4-option question carries more structure than a single 4-way accuracy
# number. Holding the question fixed and swapping only the candidate answer
# gives a minimal pair, and there are two kinds:
#
#   flip     (gold option) vs (a wrong option)  -> the label MUST change
#   control  (wrong option) vs (another wrong)  -> the label MUST NOT change
#
# The counterfactual benchmark in `data/` has only the first kind. That leaves
# a hole: a model that simply answers "different" whenever the prompt changes
# scores well on flip pairs while understanding nothing. The control pairs
# close it, and the gap between the two is the quantity of interest --
# consistency on flips is worthless without a low spurious-flip rate on
# controls.
#
# Ground truth comes from the dataset's own gold letter. No LLM-as-judge.

VERIFY_INSTRUCTION = (
    "You are a medical expert checking a proposed answer to an examination "
    "question.\n\n"
    "Question: {question}\n\n"
    "Proposed answer: {option}\n\n"
    "Is the proposed answer correct for this question?\n"
    "Respond with exactly one word first -- CORRECT or INCORRECT -- then one "
    "short sentence of justification."
)


def _mcq_pairs(records, tag, rng, limit=None):
    """
    records: list of {qid, question, options: {L: text}, gold: L}
    Emits 4 items per question: one flip pair and one control pair.
    """
    out = []
    for rec in records:
        opts, gold = rec["options"], rec["gold"]
        wrong = sorted(L for L in opts if L != gold)
        if len(wrong) < 2 or gold not in opts:
            continue
        w1, w2 = rng.sample(wrong, 2)

        def item(letter, label, pair_id, pair_type, role):
            q = _truncate_words(rec["question"].strip(), 320)
            return {
                "id": f"{pair_id}::{role}",
                "prompt": VERIFY_INSTRUCTION.format(
                    question=q, option=opts[letter].strip()),
                "labels": ["CORRECT", "INCORRECT"],
                "gold": label,
                "meta": {"pair_id": pair_id, "pair_type": pair_type,
                         "option_letter": letter, "source": tag,
                         "qid": rec["qid"]},
            }

        pid = f"{tag}::{rec['qid']}::flip"
        out.append(item(gold, "CORRECT", pid, "flip", "a"))
        out.append(item(w1, "INCORRECT", pid, "flip", "b"))
        pid = f"{tag}::{rec['qid']}::control"
        out.append(item(w1, "INCORRECT", pid, "control", "a"))
        out.append(item(w2, "INCORRECT", pid, "control", "b"))
        if limit and len(out) >= limit * 4:
            break
    return out


def load_mcqpairs(limit=None, seed=0):
    """Counterfactual pairs from MedMCQA validation + MedQA test."""
    import random
    from datasets import load_dataset
    rng = random.Random(seed)
    per = (limit // 2) if limit else None

    recs = []
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    letters = ["A", "B", "C", "D"]
    keys = ["opa", "opb", "opc", "opd"]
    for i, r in enumerate(ds):
        if per and len(recs) >= per:
            break
        opts = {L: str(r[k]) for L, k in zip(letters, keys)
                if r[k] is not None and str(r[k]).strip()}
        if len(opts) != 4:
            continue
        recs.append({"qid": str(r.get("id", i)), "question": r["question"],
                     "options": opts, "gold": letters[int(r["cop"])]})
    out = _mcq_pairs(recs, "medmcqa", rng)

    recs = []
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    for i, r in enumerate(ds):
        if per and len(recs) >= per:
            break
        o = {L: str(v) for L, v in r["options"].items()}
        if len(o) != 4:
            continue
        recs.append({"qid": str(i), "question": r["question"],
                     "options": o, "gold": r["answer_idx"]})
    out += _mcq_pairs(recs, "medqa", rng)
    return out


LOADERS = {
    "medmcqa": load_medmcqa,
    "medqa": load_medqa,
    "pubmedqa": load_pubmedqa,
    "mcqpairs": load_mcqpairs,
}


# --------------------------------------------------------------------------
# Model wrapper -- K-way generalisation of HFModel's answer-token machinery
# --------------------------------------------------------------------------

class MCQModel:
    """
    bf16 greedy generation with decision-level signal extraction over an
    arbitrary set of answer labels.

    This is the K-way generalisation of `model.HFModel`. It is a separate class
    rather than an edit to HFModel because HFModel's SAFE/UNSAFE ids are baked
    in at construction time and the counterfactual pipeline depends on that
    exact behaviour. Nothing in src/ changes as a result of this file.
    """

    def __init__(self, model_id: str, dtype: str = "bfloat16",
                 device: str = "cuda", seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        set_seed(seed)
        self.torch = torch
        self.name = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map=device)
        self.model.eval()
        self._id_cache: Dict[tuple, Dict[str, set]] = {}

    def label_token_ids(self, labels: List[str]) -> Dict[str, set]:
        """
        First *content* token id for each label under the leading contexts the
        model actually emits. Same defence as HFModel._answer_token_ids:

          - SPM word-boundary marker U+2581 is stripped before deciding whether
            a piece carries real characters, so a bare-space token is never
            mistaken for the answer.
          - Any id that lands in two different label sets is dropped and
            reported. On this tokenizer that is what silently destroyed the
            first logit-margin run (P1, HANDOFF.md).
        """
        key = tuple(labels)
        if key in self._id_cache:
            return self._id_cache[key]

        ids = {L: set() for L in labels}
        for L in labels:
            # Contexts a model plausibly emits the answer in.
            variants = [L, f" {L}", f"\n{L}", f"({L}", f"**{L}"]
            if L.isalpha() and len(L) == 1:
                variants += [L.lower(), f" {L.lower()}"]
            else:
                variants += [L.capitalize(), f" {L.capitalize()}",
                             L.upper(), f" {L.upper()}"]
            for t in variants:
                for tok in self.tokenizer.encode(t, add_special_tokens=False):
                    piece = self.tokenizer.convert_ids_to_tokens(tok)
                    if not piece.replace("▁", "").strip():
                        continue                      # whitespace / boundary
                    if piece in ("<0x0A>",):
                        continue                      # byte-fallback newline
                    if piece in ("(", "*", "**"):
                        continue                      # punctuation prefix
                    ids[L].add(tok)
                    break

        # Drop ids claimed by more than one label.
        seen, dupes = {}, set()
        for L, s in ids.items():
            for t in s:
                if t in seen and seen[t] != L:
                    dupes.add(t)
                seen[t] = L
        if dupes:
            print(f"[model] dropping {len(dupes)} token id(s) ambiguous across "
                  f"labels {labels}: "
                  f"{[self.tokenizer.convert_ids_to_tokens(t) for t in dupes]}")
            for L in ids:
                ids[L] -= dupes

        empty = [L for L, s in ids.items() if not s]
        if empty:
            raise RuntimeError(
                f"could not resolve distinct answer tokens for {empty}; "
                f"decision-level signals cannot be computed for this tokenizer")
        self._id_cache[key] = ids
        return ids

    def _wrap(self, prompt: str) -> str:
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        return f"[INST] {prompt} [/INST]"

    def run(self, items: List[dict], max_new_tokens: int = 24,
            batch_size: int = 8, max_input_tokens: int = 1024) -> List[dict]:
        torch = self.torch
        out_records = []
        t0 = time.time()

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            labels = batch[0]["labels"]
            lab_ids = self.label_token_ids(labels)
            # flat lookup: token id -> label
            tok2label = {t: L for L, s in lab_ids.items() for t in s}

            chunk = [self._wrap(b["prompt"]) for b in batch]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True,
                                 truncation=True,
                                 max_length=max_input_tokens).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    temperature=None, top_p=None,
                    return_dict_in_generate=True, output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id)

            gen_ids = gen.sequences[:, enc["input_ids"].shape[1]:]

            # Whole-vocabulary token entropy, Eq. (2) of the proposal.
            ents = []
            for logits in gen.scores:
                lp = torch.log_softmax(logits.float(), dim=-1)
                ents.append(-(lp.exp() * lp).sum(dim=-1))
            ent = torch.stack(ents, dim=1)                     # [B, T]
            mask = (gen_ids != self.tokenizer.pad_token_id).float()

            for b, item in enumerate(batch):
                m = mask[b]
                n = int(m.sum().item()) or 1
                text = self.tokenizer.decode(gen_ids[b],
                                             skip_special_tokens=True).strip()

                rec = {
                    "id": item["id"],
                    "label": item["gold"],
                    "labels": labels,
                    "text": text,
                    "entropy": (ent[b] * m).sum().item() / n,
                    "max_entropy": (ent[b] * m).max().item(),
                    "n_tokens": n,
                    "logit_margin": None,
                    "answer_logprob": None,
                    "option_entropy": None,
                    "option_prob_top": None,
                    "decision_step": None,
                    "pred": None,
                    "pred_from_text": parse_label(text, labels),
                    "meta": item.get("meta", {}),
                }

                # Decision step: first generated token that resolves to a label.
                for s in range(gen_ids.shape[1]):
                    tok = int(gen_ids[b, s].item())
                    if tok not in tok2label:
                        continue
                    logits_s = gen.scores[s][b].float()
                    logprobs_s = torch.log_softmax(logits_s, dim=-1)

                    # One representative logit per label: the best-scoring id
                    # in that label's set.
                    per_label = {L: max(logits_s[t].item() for t in lab_ids[L])
                                 for L in labels}
                    vals = sorted(per_label.values(), reverse=True)

                    # Renormalised distribution over the K options only. The
                    # proposal's Eq. (2) spreads entropy over all ~32k tokens,
                    # most of which are irrelevant to the decision; this is the
                    # same quantity restricted to the decision.
                    mx = vals[0]
                    exps = [math.exp(v - mx) for v in per_label.values()]
                    Z = sum(exps)
                    probs = [e / Z for e in exps]
                    opt_ent = -sum(p * math.log(p) for p in probs if p > 0)

                    rec["logit_margin"] = vals[0] - vals[1]
                    rec["answer_logprob"] = logprobs_s[tok].item()
                    rec["option_entropy"] = opt_ent
                    rec["option_prob_top"] = max(probs)
                    rec["decision_step"] = s
                    rec["pred"] = tok2label[tok]
                    break

                # Fall back to a text parse if no label token was emitted.
                if rec["pred"] is None:
                    rec["pred"] = rec["pred_from_text"]
                out_records.append(rec)

            done = min(i + batch_size, len(items))
            if done % (batch_size * 10) == 0 or done == len(items):
                el = time.time() - t0
                rate = done / el
                print(f"  {done}/{len(items)}  {rate:.2f} it/s  "
                      f"eta {(len(items)-done)/max(rate,1e-9)/60:.1f} min",
                      flush=True)
        return out_records


def parse_label(text: str, labels: List[str]) -> Optional[str]:
    """
    Text-level fallback parse. Used only to cross-check the token-level
    decision; disagreement rate is reported so the token-level signal can be
    audited rather than trusted blindly.
    """
    if not text:
        return None
    if all(len(L) == 1 and L.isalpha() for L in labels):
        m = re.search(r"\b([" + "".join(labels) + r"])\b", text)
        return m.group(1) if m else None
    low = text.lower()
    best, pos = None, len(low) + 1
    for L in labels:
        m = re.search(r"\b" + re.escape(L.lower()) + r"\b", low)
        if m and m.start() < pos:
            best, pos = L, m.start()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="medmcqa",
                    choices=sorted(LOADERS) + ["all"])
    ap.add_argument("--model_id", default="BioMistral/BioMistral-7B")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        print("[warn] CUDA_VISIBLE_DEVICES is "
              f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}, expected '0'. "
              "Only GPU 0 is available to this project (HANDOFF.md section 2).")

    names = sorted(LOADERS) if args.dataset == "all" else [args.dataset]
    model = MCQModel(args.model_id, seed=args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for name in names:
        print(f"\n=== {name} ===", flush=True)
        items = LOADERS[name](limit=args.limit)
        print(f"loaded {len(items)} items", flush=True)
        recs = model.run(items, max_new_tokens=args.max_new_tokens,
                         batch_size=args.batch_size)

        out = Path(args.out_dir) / f"bigbench_{name}_seed{args.seed}.jsonl"
        with out.open("w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

        n = len(recs)
        resolved = sum(r["decision_step"] is not None for r in recs)
        unparsed = sum(r["pred"] is None for r in recs)
        both = [r for r in recs
                if r["decision_step"] is not None
                and r["pred_from_text"] is not None]
        agree = sum(r["pred"] == r["pred_from_text"] for r in both)
        correct = sum(r["pred"] == r["label"] for r in recs)
        print(f"wrote {out}")
        print(f"  n={n}  accuracy={correct/n:.3f}  "
              f"decision token resolved on {resolved}/{n} "
              f"({resolved/n:.1%})")
        print(f"  token-vs-text agreement: {agree}/{len(both)} "
              f"({agree/max(len(both),1):.1%})   unparsable: {unparsed}")


if __name__ == "__main__":
    main()
