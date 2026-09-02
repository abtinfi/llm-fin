# P1 settled: the UQ signal at n=6,456

**Verdict: the code was not broken. Equation (2) of the proposal measures the
wrong quantity.**

Token-level predictive entropy over the whole vocabulary is a near-useless
wrongness detector (pooled AUROC **0.525**, 95% CI [0.511, 0.539]). The same
entropy restricted to the answer options is a real and usable signal (**0.687**,
95% CI [0.675, 0.700]). The difference, measured paired on identical items, is
**+0.163** with 95% CI [+0.144, +0.181], p < 0.0001 Holm-corrected. It
replicates independently on all three benchmarks.

---

## 1. Why this run exists

`results/p1_uq_signals.md` reported a negative result on the hand-built
counterfactual benchmark. But the population the UQ engine actually governs
there is n=56, and every confidence interval in that report covered chance —
the best signal was `answer_logprob` at 0.560 with CI [0.406, 0.714]. A CI that
wide is not evidence of absence. The honest reading of P1 was **undetermined**,
not **no signal**, and `src/uq_diagnostic.py` says so in its own closing text:

> "a larger evaluation set is what would settle it, not a threshold sweep."

This is that larger evaluation set.

| | P1 (`p1_uq_signals.md`) | This run |
|---|---|---|
| Items scored | 56 | **6,456** |
| Widest CI on the best signal | 0.308 | **0.026** |
| Can it distinguish 0.50 from 0.60? | No | **Yes** |

The confidence interval is ~12x narrower. "Undetermined" is no longer an
available answer in either direction.

## 2. Setup

| | |
|---|---|
| Model | `BioMistral/BioMistral-7B`, bf16, **no quantisation** |
| Decoding | greedy, `do_sample=False`, `max_new_tokens=24` |
| GPU | GPU 0 only (`CUDA_VISIBLE_DEVICES=0`) |
| Seed | 0 (decoding is deterministic, so seeds do not vary the output — see P2) |
| Ground truth | each dataset's own gold labels. No LLM-as-judge, no human annotation |
| Thresholds tuned | **none** |

| Dataset | Split | n | Accuracy | Chance |
|---|---|---|---|---|
| MedMCQA (`openlifescienceai/medmcqa`) | validation | 4,183 | 0.388 | 0.250 |
| MedQA-USMLE (`GBaker/MedQA-USMLE-4-options`) | test | 1,273 | 0.397 | 0.250 |
| PubMedQA (`qiaojin/PubMedQA`, `pqa_labeled`) | train | 1,000 | 0.721 | 0.333 |
| **Pooled** | | **6,456** | **0.442** | |

Accuracy is above chance on all three, so wrongness is a meaningful target. The
zero-shot numbers sit slightly below the few-shot figures BioMistral reports for
itself, which is the expected direction and serves as a sanity check that the
model is wired up correctly.

Two of the six signals are new in this run. They exist because a negative result
is only decisive if the best-known alternative was also given a fair run:

- **`option_entropy`** — entropy of the softmax over *only* the K answer-option
  tokens at the decision step. This is the decision-level analogue of Eq. (2).
- **`option_prob_top`** — renormalised probability of the chosen option.

### Data quality

| Dataset | Decision token resolved | Token-vs-text agreement | Unparsable |
|---|---|---|---|
| MedMCQA | 4183/4183 (100%) | 4181/4181 (100%) | 0 |
| MedQA | 1273/1273 (100%) | 1273/1273 (100%) | 0 |
| PubMedQA | 1000/1000 (100%) | 1000/1000 (100%) | 0 |

Every item yielded a decision token, and the token-level decision agreed with an
independent regex parse of the generated text on every single item. The
uncertainty values are being read off the step where the answer was actually
committed, not off an arbitrary position.

The **tokenizer trap from P1 fired again** and was caught: `▁**` and `▁(` are
shared first-tokens across answer labels on this SPM tokenizer. They were
detected and dropped. Left in, they would have put one token id in two answer
classes and made every margin meaningless — the bug that destroyed one earlier
eval run.

## 3. Result

AUROC as a wrongness detector. 0.5 = no signal. Bootstrap CIs, 5,000 resamples.

| Signal | MedMCQA | MedQA | PubMedQA | **Pooled** | Pooled 95% CI |
|---|---|---|---|---|---|
| `entropy` — **Eq. (2)** | 0.544 | 0.490 | 0.604 | **0.525** | [0.511, 0.539] |
| `max_entropy` | 0.549 | 0.505 | 0.572 | **0.502** | [0.487, 0.516] |
| `logit_margin` | 0.625 | 0.626 | 0.668 | **0.657** | [0.643, 0.670] |
| `answer_logprob` | 0.629 | 0.621 | 0.670 | **0.674** | [0.660, 0.687] |
| `option_prob_top` | 0.641 | 0.629 | 0.668 | **0.678** | [0.665, 0.691] |
| **`option_entropy`** | **0.648** | 0.624 | 0.661 | **0.687** | **[0.675, 0.700]** |

On MedQA, Eq. (2) entropy scores **0.490** — the point estimate is *below*
chance, matching the inversion first seen on the counterfactual benchmark.

### The decisive contrast

Paired bootstrap of AUROC(signal) − AUROC(`entropy`) on identical items,
Holm-corrected within each dataset.

| Signal | MedMCQA | MedQA | PubMedQA | **Pooled** | Pooled 95% CI | p (Holm) |
|---|---|---|---|---|---|---|
| `option_entropy` | +0.104 | +0.134 | +0.056 | **+0.163** | [+0.144, +0.181] | <0.0001 |
| `option_prob_top` | +0.097 | +0.139 | +0.064 | **+0.154** | [+0.135, +0.173] | <0.0001 |
| `answer_logprob` | +0.084 | +0.131 | +0.066 | **+0.149** | [+0.130, +0.167] | <0.0001 |
| `logit_margin` | +0.081 | +0.136 | +0.064 | **+0.132** | [+0.113, +0.151] | <0.0001 |
| `max_entropy` | +0.004 | +0.014 | −0.032 | **−0.023** | [−0.034, −0.012] | <0.0001 *(worse)* |

Every decision-level signal beats Eq. (2) on every dataset, with the CI
excluding zero in all four pooled comparisons and in all three per-dataset
comparisons after correction. `max_entropy` is the control: it is another
whole-vocabulary quantity, and it does not help. **The thing that matters is
restricting attention to the decision, not the choice of functional form.**

### Why Eq. (2) fails

Eq. (2) averages entropy over a ~32,000-token vocabulary at every generated
position. Almost all of that probability mass concerns *how to phrase the
justification*, not *which answer is right*. A model can be fluent and wrong
(low entropy, wrong answer) or hesitant in wording and right (high entropy,
right answer). Eq. (2) measures wording uncertainty. Restricting the same
entropy to the K answer tokens measures decision uncertainty, and that is the
quantity the UQ engine actually needs.

This is a specification error in the proposal, not a defect in the
implementation. Nothing in `src/model.py` computes Eq. (2) incorrectly — it
computes it correctly, and the quantity itself is not fit for the purpose.

## 4. Does deferral actually buy anything?

Pooled risk-coverage using `option_entropy`. AURC 0.3973.

| Coverage | Error | vs. full coverage |
|---|---|---|
| 1.00 | 0.558 | — |
| 0.80 | 0.516 | −0.042 |
| 0.60 | 0.457 | −0.101 |
| 0.40 | 0.382 | −0.176 |
| 0.20 | 0.282 | −0.276 |
| 0.10 | 0.200 | −0.359 |

Monotone in the right direction at every step. Contrast this with the same curve
under Eq. (2) on the counterfactual benchmark, which *rose* from 0.411 to 0.545
as coverage fell — the old signal was not merely useless, it was actively
selecting for wrong answers.

### Calibration

`option_prob_top` is a genuine probability, so it can be asked whether it is
calibrated and not merely whether it ranks.

| Dataset | ECE | Mean confidence | Accuracy | |
|---|---|---|---|---|
| MedMCQA | 0.132 | 0.520 | 0.388 | overconfident |
| MedQA | 0.165 | 0.556 | 0.397 | overconfident |
| PubMedQA | **0.034** | 0.717 | 0.721 | **well calibrated** |
| Pooled | 0.116 | 0.558 | 0.442 | overconfident |

Overconfidence tracks difficulty: where the model is competent (PubMedQA, 0.721
accuracy) its confidence is almost exactly right; where it is near chance it is
substantially overconfident. Ranking works even where calibration does not, which
is why AUROC and ECE are reported separately.

## 5. Carrying this back to the counterfactual benchmark

### The two experiments are the same experiment

For a binary decision the entropy of the renormalised two-way distribution is a
strictly decreasing function of `|logit_margin|`. Verified on the existing test
predictions (n=128): Spearman ρ = **0.9992**, AUROC 0.6698 vs 0.6680. So on the
counterfactual benchmark `logit_margin` **is** the decision-restricted entropy,
and the benchmark had already measured it:

| Signal, `base` variant, n=128 | AUROC |
|---|---|
| `entropy` — Eq. (2) | 0.469 |
| decision-restricted entropy | 0.670 |
| `-abs(logit_margin)` | 0.668 |

Same finding, same direction, found independently on two unrelated datasets.

### Rebuilding row 4

`src/run_eval.py` gained a `--uq_signal` flag. **The default is still `entropy`,
so every previously reported number is reproduced exactly by the default
invocation** — confirmed: rows 1–3 of the re-run are prediction-for-prediction
identical to the reported run on both splits.

The signal was chosen on the **external** datasets above, which share no items
with the benchmark. That is out-of-sample model selection, not threshold tuning
on the test set. (Disclosure: P1 had already reported a paired +0.199 for
`logit_margin` on the benchmark's `base` variant, so the choice is not purely
blind to the benchmark.)

**Test split, row 4 only** (rows 1–3 unchanged):

| UQ signal | Accuracy (strict) | Sel. Acc. | CC | Violation | Coverage |
|---|---|---|---|---|---|
| `entropy` (Eq. 2) | 0.562 | 0.973 | 0.562 | 0.000 | 0.578 |
| **`logit_margin`** | **0.617** | 0.940 | 0.562 | 0.016 | **0.656** |

Paired McNemar on identical items: **B01 = 7, B10 = 0, p = 0.0156.** Seven items
improved, none regressed — the same zero-regression pattern the symbolic gate
showed. Held-out reproduces the direction (B01 = 2, B10 = 0) but at n=32 is
underpowered (p = 0.5).

Deferral quality on the gate-declined pool (n=56), error at matched coverage:

| Coverage | `entropy` error | `logit_margin` error |
|---|---|---|
| 1.00 | 0.411 | 0.411 |
| 0.80 | 0.422 | **0.378** |
| 0.57 | 0.500 | **0.406** |
| 0.39 | 0.545 | **0.318** |

Eq. (2) makes the retained set *worse* at every operating point. The margin
makes it better at every operating point.

**The cost, stated plainly:** the margin-based engine answers more (coverage
0.656 vs 0.578) and is right more often overall, but it lets through **one**
unsafe case that the entropy engine had abstained on (violation rate 0.016 vs
0.000). On a clinical safety task that trade is not automatically worth taking.
The operating point is a deliberate choice, and the curve above is what it should
be chosen from — not a single number.

## 5b. Qualification found later: this holds for SELECTION, not verification

`results/mcqpairs.md` re-ran the same six signals on a **verification** task —
"is this proposed answer correct?" — over 8,000 items, and **the ordering
reverses**:

| Signal | AUROC there | 95% CI |
|---|---|---|
| `entropy` — Eq. (2) | **0.540** | [0.528, 0.553] |
| `logit_margin` / `option_entropy` / `option_prob_top` | **0.475** | [0.462, 0.487] |

The decision-level signals sit *below* chance. The reason is visible in the
same run: on that task the model answers CORRECT to 73.4% of items while only
25.0% are correct. When a model has collapsed onto a near-constant answer, its
confidence at the decision token measures how strongly it holds that prior, not
whether the prior fits this item. Confidence tracks the bias, the bias is
usually wrong, so confidence anti-correlates with accuracy.

**Revised claim.** Restricting uncertainty to the decision helps when the model
is genuinely choosing between options — the setting measured in this report,
where it scores well above chance on all three benchmarks. It does not help,
and can invert, when the model is answering degenerately. **Uncertainty
quantification ranks cases within a functioning policy; it cannot rescue a
broken one.** Quote the +0.163 with this condition attached.

## 6. Limitations

- **Single seed, deterministic decoding.** P2 is unaffected by this run: seeds
  still do not vary the output. The benchmark-instantiation fix P2 describes is
  still outstanding.
- **Answer-position bias.** On the MCQ sets the model over-predicts A and D
  (MedMCQA: 1,861 A-predictions against 1,348 A-golds). A known LLM artifact.
  It inflates the error rate but there is no reason it would favour one
  uncertainty signal over another.
- **PubMedQA `maybe` is never predicted** — 0/110 correct on that class. Those
  110 items are deterministically wrong, which makes the UQ task *harder*, so
  the positive result here is conservative.
- **MedMCQA carries known label noise.** MedQA and PubMedQA are cleaner and show
  the same effect, so the conclusion does not rest on it.
- **`option_entropy` is usable, not excellent.** 0.687 supports triage and a
  reported risk-coverage curve. It does not support a claim of reliable
  per-case error detection.
- **These are MCQ benchmarks, not counterfactual pairs.** They settle the
  *signal* question; they do not measure Causal Consistency, which remains the
  counterfactual benchmark's job.

## 7. What is settled and what is not

**Settled** (for the selection setting -- see 5b for the verification caveat)**.**
1. Eq. (2) whole-vocabulary entropy is not a usable wrongness detector for this
   model on medical QA selection. 0.525 [0.511, 0.539] at n=6,456. This is now
   a measurement, not a guess.
2. Restricting uncertainty to the decision fixes it: +0.163 AUROC, p < 0.0001,
   replicated on three independent benchmarks.
3. Uncertainty-aware deferral does work once the signal is right — pooled error
   0.558 → 0.282 at 20% coverage.
4. The implementation was never the problem. §4.7 of the proposal needs its
   uncertainty term respecified.

**Not settled.**
1. Whether the UQ engine helps *on the 56 gate-declined counterfactual items*.
   Direction is right and McNemar is significant (7–0, p = 0.0156), but AUROC on
   that pool is 0.544 with a CI still covering chance. That is now a
   sample-size statement, not a signal statement — the benchmark needs to grow.
2. Whether the violation-rate cost (0.000 → 0.016) is acceptable. That is a
   clinical judgement, not a statistical one.
3. Self-consistency (k=5 sampling) is still unattempted; it breaks the
   deterministic-decoding guarantee and needs its own variant row.

## 8. Reproducing

```bash
export CUDA_VISIBLE_DEVICES=0

# generation, ~20 min for all three datasets on one 4090
python src/bigbench_uq.py --dataset all --batch_size 16

# analysis
python src/bigbench_analysis.py \
    --preds results/bigbench_medmcqa_seed0.jsonl \
            results/bigbench_medqa_seed0.jsonl \
            results/bigbench_pubmedqa_seed0.jsonl \
    --json_out results/bigbench_uq.json

# rebuilt row 4 (default --uq_signal entropy reproduces the original)
python src/run_eval.py --backend hf --model_id BioMistral/BioMistral-7B \
    --seeds 0 --batch_size 4 --uq_signal logit_margin --tag _margin
python src/make_table.py --split test --tag _margin \
    --out results/table_test_margin.md
```

Artefacts: `results/bigbench_{medmcqa,medqa,pubmedqa}_seed0.jsonl`,
`results/bigbench_uq.json`, `results/table_{test,heldout}_margin.md`.
