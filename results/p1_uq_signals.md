# P1 — Alternative uncertainty signals

BioMistral-7B, test split, seed 0, bf16, greedy decoding, GPU 0.
All numbers regenerated after adding the new signals; the accuracy/consistency
rows are unchanged from the previous run, confirming decoding was not perturbed.

AUROC = probability that a wrong answer looks more uncertain than a correct one.
0.5 is no signal. All signals are mapped to a "higher = more uncertain"
convention so they are directly comparable:

| signal | uncertainty defined as |
|---|---|
| `entropy` | mean token-level predictive entropy (proposal Eq. 2) |
| `max_entropy` | max token entropy over the generation |
| `logit_margin` | `-abs(logit(SAFE) - logit(UNSAFE))` at the decision token |
| `answer_logprob` | `-logprob` of the emitted answer token |

## Result 1 — entropy has no signal, and this replicates

| population | n | entropy AUROC | 95% CI |
|---|---|---|---|
| base (all cases) | 128 | 0.469 | [0.369, 0.573] |
| rag (all cases) | 128 | 0.434 | [0.331, 0.541] |
| nsai (gate-declined only) | 56 | 0.374 | [0.228, 0.535] |
| heldout base | 32 | 0.402 | [0.200, 0.609] |

Confirmed on every population. `max_entropy` is no better (0.293–0.532).
This falsifies the assumption in §4.7 of the proposal.

## Result 2 — the decision-level signal is reliably better than entropy

Paired bootstrap of `AUROC(logit_margin) - AUROC(entropy)` on identical items
(5000 resamples). Comparing two independent CIs is not a test of the difference,
so the difference itself is resampled paired.

| population | n | logit_margin | entropy | delta | 95% CI | p |
|---|---|---|---|---|---|---|
| base | 128 | 0.668 | 0.469 | **+0.199** | [+0.058, +0.336] | 0.004 |
| rag | 128 | 0.553 | 0.434 | **+0.119** | [+0.033, +0.202] | 0.004 |
| nsai (gate-declined) | 56 | 0.544 | 0.374 | **+0.170** | [+0.037, +0.306] | 0.016 |
| heldout base | 32 | 0.605 | 0.402 | +0.203 | [-0.118, +0.522] | 0.214 |

Same sign and similar magnitude on all four populations; significant on the
three that have enough items. Held-out is underpowered (16 pairs), as expected.

**This is a positive finding**: uncertainty tied to the binary decision carries
information that whole-vocabulary entropy does not. The proposal's §4.7
assumption fails for the *specific estimator* it names, not for the idea of
neural uncertainty as such.

## Result 3 — but it does not rescue the UQ variant row

The `nsai_uq` row only governs cases where the symbolic gate declines. On that
population:

| signal | n | AUROC | 95% CI | clears P1 bar? |
|---|---|---|---|---|
| entropy | 56 | 0.374 | [0.228, 0.535] | no |
| max_entropy | 56 | 0.514 | [0.354, 0.671] | no |
| logit_margin | 56 | 0.544 | [0.389, 0.701] | no |
| answer_logprob | 56 | 0.560 | [0.406, 0.714] | **point estimate only** |

`answer_logprob` clears 0.55 on the point estimate but its CI lower bound is
0.406. Per the P1 acceptance criterion this is **not** evidence of a usable
signal, and no threshold was tuned to chase it.

The likely mechanism: the gate decides 72/128 cases and makes 0 errors on them.
It removes precisely the cases where the answer is determinable, leaving a
residual population that is genuinely hard. A signal with AUROC 0.668 on the
full population drops to 0.544 on that residual. **The gate and the UQ engine
compete for the same easy cases**, which is a structural point about the
architecture, not a tuning failure.

## Verdict against the P1 acceptance criterion

> "If AUROC stays below 0.55, stop and report the negative result — do not tune
> thresholds until a number looks good."

On the population the UQ engine actually governs, no signal clears 0.55 with a
CI excluding chance. **Report the negative result for the `nsai_uq` row.**

Report alongside it the positive Result 2: the failure is specific to
whole-vocabulary entropy, and decision-level uncertainty is measurably better
even though it is still not good enough to act on here.

## Not done

Self-consistency (k=5 samples at temperature > 0, answer disagreement rate) is
the third alternative listed in P1. It breaks the deterministic-decoding
guarantee that makes the paired McNemar tests valid, so per HANDOFF it needs its
own variant row rather than replacing the existing one. Not attempted yet.

## Implementation notes

- `src/model.py` computes both new signals at the first generated token that
  resolves to a SAFE/UNSAFE first-token. Populated on 128/128 test cases.
- **Tokenizer trap, fixed:** on this SPM tokenizer `" SAFE"` and `" UNSAFE"`
  both begin with the bare-space token `28705`, so taking `encode(...)[0]`
  naively puts one id in *both* answer classes and makes the margin
  meaningless. `_answer_token_ids` now skips leading whitespace-only pieces and
  drops (with a warning) any id still ambiguous between the classes. Resolved
  ids: SAFE `{▁SA, SA}`, UNSAFE `{▁UN, UN}`. An eval run made before this fix
  was discarded.
- Seeds 1–4 predictions are byte-identical to seed 0 (this is P2). Their pred
  files were left without the new fields rather than spending GPU time
  regenerating identical content; fixing P2 will regenerate all of them.

## Reproduce

```bash
cd ~/abtin/paper/csai && export CUDA_VISIBLE_DEVICES=0 && export HF_HUB_OFFLINE=1
python src/run_eval.py --backend hf --model_id BioMistral/BioMistral-7B \
    --seeds 0 --batch_size 4
python src/uq_diagnostic.py --preds results/preds_test_base_seed0.jsonl --all
python src/uq_diagnostic.py --preds results/preds_test_base_seed0.jsonl \
    --paired logit_margin entropy
```
