"""
run_eval.py from the MAIN checkout, with generation stopped as soon as the
answer is fixed. Used for Mistral-7B-Instruct on the v3.1 arms only.

WHY
  Mistral-Instruct explains itself: on the Demo arm it generated 37.8 tokens
  on average and 40% of its answers ran into the 64-token cap. A batch of 8
  decodes until its longest member stops, so ~98% of batches ran all 64 steps
  (1 - 0.6^8). BioMistral answers in ~3 tokens. Measured on v3, BioMistral does
  52 items/s at ~150 ms per batch; at 64 steps Mistral would do ~4, putting
  its v3 job near 100 hours.

WHAT IT GUARANTEES
  A row stops only when BOTH hold:
    (a) the first match of model.ANSWER_RE in the decoded text has at least
        one character after it. parse_answer takes the FIRST match, and a
        first match followed by a character can no longer move or grow: any
        earlier match would have to span it, and every pattern is a fixed
        word bounded by \\b. So `pred` is what the full 64 tokens would give.
    (b) a token in answer_token_ids SAFE|UNSAFE has been generated. The
        decision-level signals (logit_margin, answer_logprob, and the
        decision_entropy every reported UQ row uses) are read at the FIRST
        such step. Without (b), "Contraindicated. ... UNSAFE" would stop
        before that step exists and change those signals.
  If either never happens the row runs to 64 tokens exactly as before.
  Rows of a batch are computed independently, so a finished row being fed
  pad tokens does not change another row's logits.

WHAT IT CHANGES
  Only fields that summarise ALL generated tokens: `raw` (shorter), the mean
  `entropy`, `max_entropy`, and `n_tokens`. None is used by any reported
  metric (every UQ row in the repo is decision_entropy). This is verified,
  not assumed: see verify in run_mimic_v3b.sh's header.

  python src/run_eval_earlystop.py <run_eval.py args...>
"""

import runpy
import sys
from pathlib import Path

CSAI_SRC = Path("/home/asosoft/abtin/paper/csai/src")
sys.path.insert(0, str(CSAI_SRC))

import torch                                      # noqa: E402
from transformers import StoppingCriteria, StoppingCriteriaList  # noqa: E402

import model as M                                 # noqa: E402  (main checkout)

assert Path(M.__file__).resolve().parent == CSAI_SRC, M.__file__


class AnswerFixed(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len, answer_ids):
        self.tok = tokenizer
        self.L = prompt_len
        self.ans = torch.tensor(sorted(answer_ids))
        self.stops = 0

    def __call__(self, input_ids, scores, **kw):
        gen = input_ids[:, self.L:]
        has_ans = torch.isin(gen, self.ans.to(gen.device)).any(dim=1)
        done = torch.zeros(gen.shape[0], dtype=torch.bool, device=gen.device)
        for b in torch.nonzero(has_ans).flatten().tolist():
            text = self.tok.decode(gen[b], skip_special_tokens=True)
            m = M.ANSWER_RE.search(text)
            if m and m.end() < len(text):
                done[b] = True
        self.stops += int(done.sum())
        return done


_orig_generate = M.HFModel.generate


def generate(self, prompts, max_new_tokens=64, batch_size=8):
    hf = self.model
    inner = hf.generate
    ans = set(self.answer_token_ids["SAFE"]) | set(self.answer_token_ids["UNSAFE"])

    def patched(**kw):
        kw["stopping_criteria"] = StoppingCriteriaList(
            [AnswerFixed(self.tokenizer, kw["input_ids"].shape[1], ans)])
        return inner(**kw)

    hf.generate = patched
    try:
        return _orig_generate(self, prompts, max_new_tokens=max_new_tokens,
                              batch_size=batch_size)
    finally:
        hf.generate = inner


M.HFModel.generate = generate

if __name__ == "__main__":
    sys.argv[0] = str(CSAI_SRC / "run_eval.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
