"""
Model wrapper.

Greedy decoding, fp16, no quantisation. Quantisation is deliberately avoided:
it perturbs the logit distribution and would invalidate the predictive-entropy
term used by the UQ variant (Eq. 2 of the proposal).

Returns, for every generation:
    text            -- decoded continuation
    entropy         -- mean token-level predictive entropy over generated tokens
    max_entropy     -- max token entropy (used as a secondary UQ signal)
    n_tokens
    logit_margin    -- signed logit(SAFE) - logit(UNSAFE) at the first generated
                       position that resolves to either answer word (None if the
                       generation never produces one). See P1 in HANDOFF.md:
                       decision-level alternative to whole-vocabulary entropy.
    answer_logprob  -- log-probability of the actual answer token at that same
                       position (None if unresolved).
"""

import math
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Generation:
    text: str
    entropy: float
    max_entropy: float
    n_tokens: int
    logit_margin: Optional[float] = None
    answer_logprob: Optional[float] = None


class MockLM:
    """
    Deterministic stand-in so the whole pipeline can be validated end-to-end
    without a GPU. It is NOT a baseline and must never appear in reported
    results -- it exists only to prove the plumbing works.
    """

    name = "mock"

    def __init__(self, seed: int = 0, skill: float = 0.62):
        import random
        self.rng = random.Random(seed)
        self.skill = skill

    def attach_adapter(self, path: str, layer: int, alpha: float = 1.0):
        # The mock has no hidden states; it records the request so the plumbing
        # check still covers the code path, and shifts its skill so the variant
        # is distinguishable in a mock run. It is not a result either way.
        self.adapter = (path, layer, alpha)

    def detach_adapter(self):
        self.adapter = None

    def generate(self, prompts: List[str], max_new_tokens: int = 64,
                 batch_size: int = 8) -> List[Generation]:
        outs = []
        for p in prompts:
            # crude heuristic + noise, only to exercise downstream code paths
            risky = any(k in p.lower() for k in
                        ["egfr of 1", "egfr of 2", "pregnant", "severe", "clarithromycin"])
            correct = self.rng.random() < self.skill
            says_unsafe = risky if correct else (not risky)
            ans = "UNSAFE" if says_unsafe else "SAFE"
            ent = self.rng.uniform(0.05, 1.6)
            magnitude = self.rng.uniform(0.5, 6.0)
            margin = -magnitude if says_unsafe else magnitude  # + favors SAFE
            logprob = -abs(self.rng.gauss(0.3, 0.4))
            outs.append(Generation(
                text=f"{ans}\nThis is a mock justification.",
                entropy=ent, max_entropy=ent * 1.8, n_tokens=12,
                logit_margin=margin, answer_logprob=logprob))
        return outs


class HFModel:
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
            model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device,
        )
        self.model.eval()
        self.answer_token_ids = self._answer_token_ids()
        self._adapter_handle = None

    def _answer_token_ids(self):
        """
        First *content* token ids for SAFE / UNSAFE under the leading contexts
        the model actually sees (bare, space-prefixed, newline-prefixed), since
        SPM tokenizers split a word differently depending on what precedes it.
        Used to find the decision step in generate() and to compute the logit
        margin between the two answer classes.

        Leading whitespace-only tokens are skipped: on this tokenizer " SAFE"
        and " UNSAFE" both begin with the bare-space token, so taking enc[0]
        naively would put one id in BOTH classes and make the margin
        meaningless. Any id that still lands in both classes is dropped and
        reported rather than silently used.
        """
        variants = {
            "SAFE": ["SAFE", " SAFE", "\nSAFE"],
            "UNSAFE": ["UNSAFE", " UNSAFE", "\nUNSAFE"],
        }
        ids = {"SAFE": set(), "UNSAFE": set()}
        for label, texts in variants.items():
            for t in texts:
                for tok in self.tokenizer.encode(t, add_special_tokens=False):
                    piece = self.tokenizer.convert_ids_to_tokens(tok)
                    # SPM marks a word boundary with U+2581; strip it before
                    # deciding whether the piece carries any actual characters.
                    if not piece.replace("▁", "").strip():
                        continue          # pure whitespace / boundary marker
                    if piece in ("<0x0A>",):
                        continue          # byte-fallback newline
                    ids[label].add(tok)
                    break

        overlap = ids["SAFE"] & ids["UNSAFE"]
        if overlap:
            print(f"[model] WARNING: dropping {len(overlap)} token id(s) "
                  f"ambiguous between SAFE and UNSAFE: "
                  f"{[self.tokenizer.convert_ids_to_tokens(i) for i in overlap]}")
            ids["SAFE"] -= overlap
            ids["UNSAFE"] -= overlap
        if not ids["SAFE"] or not ids["UNSAFE"]:
            raise RuntimeError(
                "could not resolve distinct SAFE/UNSAFE answer tokens; "
                "the logit-margin signal cannot be computed for this tokenizer")
        return ids

    def attach_adapter(self, path: str, layer: int, alpha: float = 1.0):
        """
        Insert the trained Constraint-Aware Layer of Aim 3 into the forward
        pass: h'_l = h_l + alpha * P_causal(h_l).

        This exists so the constraint layer can be scored as a ROW OF THE
        ABLATION TABLE rather than in a separate report with its own readout.
        `constraint_layer.py` evaluates by argmax over the two answer logits;
        every row of the table is scored by generating text and parsing the
        first decision word. Those two readouts do not have to agree, so a
        number produced by one cannot be placed in a table built from the
        other. Attaching the adapter here puts it through the identical path.
        """
        import numpy as np
        torch = self.torch
        z = np.load(path)
        W_down = torch.tensor(z["W_down"], device=self.device).float()
        W_up = torch.tensor(z["W_up"], device=self.device).float()
        b = torch.tensor(z["b"], device=self.device).float()

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            delta = torch.nn.functional.gelu(h.float() @ W_down + b) @ W_up
            h = h + alpha * delta.to(h.dtype)
            return (h,) + output[1:] if isinstance(output, tuple) else h

        self.detach_adapter()
        self._adapter_handle = self.model.model.layers[layer]\
            .register_forward_hook(hook)
        print(f"[model] constraint layer attached at layer {layer}, "
              f"alpha={alpha}, rank={W_down.shape[1]}")

    def detach_adapter(self):
        if self._adapter_handle is not None:
            self._adapter_handle.remove()
            self._adapter_handle = None

    def _wrap(self, prompt: str) -> str:
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        return f"[INST] {prompt} [/INST]"

    @staticmethod
    def _entropy_from_scores(scores, seq_ids, torch):
        """Token-level predictive entropy, Eq. (2) of the proposal."""
        ents = []
        for step, logits in enumerate(scores):
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            probs = logprobs.exp()
            h = -(probs * logprobs).sum(dim=-1)   # nats
            ents.append(h)
        return torch.stack(ents, dim=1)           # [batch, steps]

    def generate(self, prompts: List[str], max_new_tokens: int = 64,
                 batch_size: int = 8) -> List[Generation]:
        torch = self.torch
        results: List[Generation] = []
        for i in range(0, len(prompts), batch_size):
            chunk = [self._wrap(p) for p in prompts[i:i + batch_size]]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            gen_ids = out.sequences[:, enc["input_ids"].shape[1]:]
            ent = self._entropy_from_scores(out.scores, gen_ids, torch)  # [B, T]
            mask = (gen_ids != self.tokenizer.pad_token_id).float()
            safe_ids = self.answer_token_ids["SAFE"]
            unsafe_ids = self.answer_token_ids["UNSAFE"]
            for b in range(gen_ids.shape[0]):
                m = mask[b]
                n = int(m.sum().item()) or 1
                e = (ent[b] * m).sum().item() / n
                emax = (ent[b] * m).max().item()
                text = self.tokenizer.decode(gen_ids[b], skip_special_tokens=True)

                # Decision-level signals: find the first generated step whose
                # token is a SAFE/UNSAFE first-token, and read the two answer
                # logits off that step. P1 in HANDOFF.md.
                margin, answer_lp = None, None
                for s in range(gen_ids.shape[1]):
                    tok = int(gen_ids[b, s].item())
                    if tok in safe_ids or tok in unsafe_ids:
                        logits_s = out.scores[s][b].float()
                        logprobs_s = torch.log_softmax(logits_s, dim=-1)
                        safe_logit = max(logits_s[i].item() for i in safe_ids)
                        unsafe_logit = max(logits_s[i].item() for i in unsafe_ids)
                        margin = safe_logit - unsafe_logit   # + favours SAFE
                        answer_lp = logprobs_s[tok].item()
                        break

                results.append(Generation(
                    text=text.strip(), entropy=e, max_entropy=emax, n_tokens=n,
                    logit_margin=margin, answer_logprob=answer_lp))
        return results


ANSWER_RE = re.compile(r"\b(UNSAFE|NOT SAFE|CONTRAINDICATED|SAFE)\b", re.I)


def parse_answer(text: str) -> Optional[str]:
    """
    Extract the SAFE/UNSAFE decision. Returns None if unparsable -- those are
    reported separately rather than silently counted as wrong.
    """
    m = ANSWER_RE.search(text)
    if not m:
        return None
    tok = m.group(1).upper()
    return "SAFE" if tok == "SAFE" else "UNSAFE"


def load_model(cfg):
    if cfg["backend"] == "mock":
        return MockLM(seed=cfg.get("seed", 0))
    return HFModel(cfg["model_id"], cfg.get("dtype", "float16"),
                   cfg.get("device", "cuda"), cfg.get("seed", 0))
