# Counterfactual pairs from multiple-choice questions — 8,000 items

**Verdict: on 2,000 paired medical questions, BioMistral-7B changes its answer
at the same rate whether or not the correct answer changed.** Discrimination
−0.013, 95% CI [−0.037, +0.013]. Its consistency score on genuine
counterfactual pairs therefore carries no evidence of reasoning.

---

## 1. The design, and the hole it fills

The counterfactual benchmarks in `data/` and `data/medcalc` contain only one
kind of pair: two arms whose correct label **differs**. That leaves a hole. A
model following the rule *"if the prompt changed, change the answer"* scores
perfectly on such pairs while understanding nothing, and no existing measurement
would catch it.

Holding a multiple-choice question fixed and swapping only the candidate answer
gives both kinds of pair:

| Pair type | Arms | Correct label |
|---|---|---|
| **flip** | gold option vs. a wrong option | **must change** |
| **control** | one wrong option vs. another wrong option | **must not change** |

The quantity of interest is not either rate alone but the **discrimination**:

```
discrimination = flip-rate on FLIP pairs  −  flip-rate on CONTROL pairs
```

A model that reasons flips much more on FLIP pairs. A model that merely reacts
to the prompt changing flips equally on both and scores zero, whatever its raw
consistency looks like.

Task given to the model: *"Is the proposed answer correct for this question?
Respond CORRECT or INCORRECT."* Ground truth is the dataset's own gold letter —
no LLM-as-judge, no annotation.

| Source | Questions | Items |
|---|---|---|
| MedMCQA (validation) | 1,000 | 4,000 |
| MedQA-USMLE (test) | 1,000 | 4,000 |
| **Total** | **2,000** | **8,000** |

Each question yields 4 items (2 flip arms + 2 control arms), so the label mix
is 1 CORRECT : 3 INCORRECT by construction.

**Data quality.** Decision token resolved on 7,991/8,000 (99.9%); token-level
decision agreed with an independent text parse on 7,991/7,991 (100%); 0
unparsable.

## 2. Result

| | MedMCQA | MedQA | **Pooled** | 95% CI |
|---|---|---|---|---|
| flip-rate on FLIP pairs | 0.158 | 0.256 | **0.207** | |
| flip-rate on CONTROL pairs | 0.167 | 0.272 | **0.220** | |
| **discrimination** | −0.009 | −0.016 | **−0.013** | **[−0.037, +0.013]** |
| consistency, FLIP pairs | 0.096 | 0.169 | 0.133 | |
| consistency, CONTROL pairs | 0.083 | 0.273 | 0.178 | |

The CI is tight and straddles zero. At n=2,000 pairs of each type this is not
an underpowered null: there is no measurable difference between the two pair
types. Swapping the gold option for a wrong one moves the model's answer no
more than swapping one wrong option for another.

### The model is worse than a constant answer

| | Item accuracy |
|---|---|
| Always answer CORRECT | 0.250 |
| **Always answer INCORRECT** | **0.750** |
| BioMistral-7B | **0.408** |

It says CORRECT on **73.4%** of items while only 25.0% are correct. Asked to
verify a proposed answer, it agrees with almost anything put in front of it.

This is the same failure shape seen elsewhere in the project — the base model
answering UNSAFE to 63.5% of real notes, and answering SAFE to all 170
held-out items — a strong prior over the answer token that the input barely
moves.

## 3. What this does and does not show

**Does show.** Counterfactual consistency alone is not evidence of causal
reasoning, and this benchmark family needs control pairs to be interpretable.
Any future consistency number in this project — including the 0.986 on
`data/medcalc` — should be reported alongside a spurious-flip rate.

**Does not show.** That the model cannot answer medical questions. In the
4-way selection format the same model scores 0.388 on MedMCQA and 0.397 on
MedQA against a 0.250 chance level (`results/bigbench_uq.md`) — well above
chance. Verification is a different and evidently harder format for it than
selection, and the response bias is specific to the verification framing.

**Not tested here.** RAG and the symbolic gate. Neither applies: these
questions carry no extractable numeric fact and no threshold, and there is no
guideline corpus that answers them. This is precisely why the four-stage
ablation needed `data/medcalc` rather than an MCQ set — a point worth keeping
in mind before proposing to scale the benchmark by converting more MCQ data.

## 4. An important qualification to `results/bigbench_uq.md`

That report concluded that decision-restricted uncertainty (`option_entropy`,
`option_prob_top`) is decisively better than the proposal's whole-vocabulary
Eq. (2) entropy: +0.163 AUROC pooled over 6,456 items, p < 0.0001.

**On this task the ordering reverses.**

| Signal | AUROC | 95% CI |
|---|---|---|
| `entropy` — Eq. (2) | **0.540** | [0.528, 0.553] |
| `answer_logprob` | 0.508 | [0.495, 0.521] |
| `max_entropy` | 0.507 | [0.494, 0.520] |
| `logit_margin` | **0.475** | [0.462, 0.487] |
| `option_entropy` | **0.475** | [0.462, 0.487] |
| `option_prob_top` | **0.475** | [0.462, 0.487] |

The three decision-level signals sit *below* chance: higher confidence is
slightly predictive of being **wrong**.

The explanation is consistent with the response bias above. When a model
answers CORRECT to 73.4% of items regardless of content, its confidence at the
decision token measures how strongly it holds that prior, not whether the prior
is right on this item. Confidence tracks the bias; the bias is usually wrong;
so confidence anti-correlates with accuracy.

**The revised claim.** Restricting uncertainty to the decision helps when the
model is genuinely choosing between options — the 4-way selection setting where
it scores well above chance. It does not help, and can invert, when the model
has collapsed onto a near-constant answer. Uncertainty quantification cannot
rescue a degenerate policy; it can only rank cases within a functioning one.
That is a condition on §4.7 of the proposal, not a refutation of the earlier
result, and it should be stated wherever the +0.163 figure is.

## 5. Limitations

- **Verification framing.** Results here describe "is this answer correct?",
  not "which answer is correct?". They do not transfer to the selection format
  without the caveat above.
- **One wrong option is sampled at random** per question for the flip pair, and
  two for the control pair. Some wrong options are obviously wrong and some are
  near-misses; that variance is not stratified.
- **Label mix is 1:3 by construction**, so item accuracy must always be read
  against the 0.750 constant-answer baseline, never against 0.5.
- **Single seed, greedy decoding.** As everywhere else in this project, P2 is
  untouched.
- **MedMCQA carries known label noise**; MedQA is cleaner and shows the same
  discrimination.

## 6. Reproducing

```bash
export CUDA_VISIBLE_DEVICES=0
python src/bigbench_uq.py --dataset mcqpairs --limit 2000 --batch_size 16
python src/mcqpair_analysis.py --preds results/bigbench_mcqpairs_seed0.jsonl
```
