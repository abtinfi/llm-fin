"""
run_eval.py from the MAIN checkout, with generation made deterministic per
prompt and deduplicated. Used for every model on every v3.1 arm.

WHY
  The v3 arm's prompts repeat (the rendered note is age, sex, one value, the
  drug): 166,264 test items are ~17.5k distinct prompts. The stock generate()
  batches items in file order with left padding, so a prompt's padded length,
  and with it the bfloat16 reduction order, depends on its batch neighbours. The
  audit found BioMistral giving DIFFERENT answers to identical prompts in 311
  of 13,870 cases (src/audit_mimic_preds.py) -- noise that lands in the
  spurious-flip rate and cannot be told apart from it.

WHAT IT DOES (monkeypatches model.HFModel.generate; nothing on disk changes)
  1. Each DISTINCT prompt is generated once; every copy gets that result.
     Identical prompts therefore get identical answers by construction, and
     the arm costs ~1/10 of the GPU time.
  2. Distinct prompts are grouped by tokenized length and batched only with
     prompts of the SAME length, so no batch is ever padded; a short final
     batch is filled up to batch_size with copies of its last prompt, so every
     forward pass has the same shape. A prompt's result then depends on the
     prompt alone -- not on which other prompts share its batch.
  3. EARLY_STOP=1 adds the answer-fixed stopping criterion (Mistral-Instruct
     only): see AnswerFixed below for why pred and every decision-level signal
     are unchanged, verified on 16 items at exact float equality.

WHAT IT CHANGES relative to stock batching: only near-ties that the old
batching resolved by accident of padding. That is the point.

  python src/run_eval_v3.py <run_eval.py args...>
"""

import collections
import os
import runpy
import sys
from pathlib import Path

CSAI_SRC = Path("/home/asosoft/abtin/paper/csai/src")
sys.path.insert(0, str(CSAI_SRC))

import torch                                      # noqa: E402
from transformers import StoppingCriteria, StoppingCriteriaList  # noqa: E402

import model as M                                 # noqa: E402  (main checkout)

assert Path(M.__file__).resolve().parent == CSAI_SRC, M.__file__
EARLY_STOP = os.environ.get("EARLY_STOP", "0") == "1"


class AnswerFixed(StoppingCriteria):
    """
    Stop a row once (a) the first model.ANSWER_RE match in its decoded text
    has a character after it -- parse_answer takes the first match, which can
    then no longer move or grow -- AND (b) a SAFE/UNSAFE answer token has been
    generated, the step every decision-level signal is read from. Otherwise
    the row runs to max_new_tokens as before.
    """

    def __init__(self, tokenizer, prompt_len, answer_ids):
        self.tok, self.L = tokenizer, prompt_len
        self.ans = torch.tensor(sorted(answer_ids))

    def __call__(self, input_ids, scores, **kw):
        gen = input_ids[:, self.L:]
        has_ans = torch.isin(gen, self.ans.to(gen.device)).any(dim=1)
        done = torch.zeros(gen.shape[0], dtype=torch.bool, device=gen.device)
        for b in torch.nonzero(has_ans).flatten().tolist():
            text = self.tok.decode(gen[b], skip_special_tokens=True)
            m = M.ANSWER_RE.search(text)
            if m and m.end() < len(text):
                done[b] = True
        return done


_orig_generate = M.HFModel.generate
STATS = collections.Counter()


def _run_fixed_batch(self, chunk, max_new_tokens, batch_size):
    hf = self.model
    inner = hf.generate
    if EARLY_STOP:
        ans = (set(self.answer_token_ids["SAFE"])
               | set(self.answer_token_ids["UNSAFE"]))

        def patched(**kw):
            kw["stopping_criteria"] = StoppingCriteriaList(
                [AnswerFixed(self.tokenizer, kw["input_ids"].shape[1], ans)])
            return inner(**kw)
        hf.generate = patched
    try:
        return _orig_generate(self, chunk, max_new_tokens=max_new_tokens,
                              batch_size=batch_size)
    finally:
        hf.generate = inner


def generate(self, prompts, max_new_tokens=64, batch_size=8):
    uniq = list(dict.fromkeys(prompts))
    by_len = collections.defaultdict(list)
    for p in uniq:
        by_len[len(self.tokenizer(self._wrap(p)).input_ids)].append(p)
    out = {}
    for L in sorted(by_len):
        # Canonical order inside a length group, so batch composition does
        # not depend on the order items arrive in. Without it, pred, margin
        # and every decision-level signal were already bit-identical across
        # two input orders, but the full-vocabulary entropy/max_entropy moved
        # by ~1e-7 with batch neighbours (fp32 CPU check, 32 items).
        group = sorted(by_len[L])
        for i in range(0, len(group), batch_size):
            chunk = group[i:i + batch_size]
            fill = batch_size - len(chunk)
            gens = _run_fixed_batch(self, chunk + [chunk[-1]] * fill,
                                    max_new_tokens, batch_size)
            for p, g in zip(chunk, gens):
                out[p] = g
            STATS["batches"] += 1
            STATS["filler_rows"] += fill
    STATS["items"] += len(prompts)
    STATS["distinct"] += len(uniq)
    print(f"  [run_eval_v3] {len(prompts):,} items -> {len(uniq):,} distinct "
          f"prompts in {len(by_len)} length groups"
          f"{'  (early stop)' if EARLY_STOP else ''}", flush=True)
    return [out[p] for p in prompts]


M.HFModel.generate = generate

if __name__ == "__main__":
    sys.argv[0] = str(CSAI_SRC / "run_eval.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
