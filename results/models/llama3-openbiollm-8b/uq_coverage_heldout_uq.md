# UQ coverage report -- `heldout`, variant `uq`

Signal: **`decision_entropy`**. Target error rate on retained items: alpha = 0.1. n = 32.

## 1. The frozen split-conformal threshold, as shipped

| | coverage | error on retained items | target |
|---|---|---|---|
| frozen tau (calibration split) | 0.125 | 0.500 | 0.10 |

Target **violated** on this split.

A frozen threshold is only valid while the test stream is exchangeable with the calibration split. This split is a different rule family, so it is not, and the guarantee does not transfer. That is not a bug in the calibration -- it is the documented limit of split conformal, and the reason the proposal names the adaptive variant.

## 2. Adaptive Conformal Inference on the same items

alpha_t updated after every item with gamma = 0.05; the threshold is the running (1 - alpha_t) quantile of uncertainties seen so far.

| | coverage | error on retained items |
|---|---|---|
| frozen tau | 0.125 | 0.500 |
| **ACI** | **0.938** | **0.500** |

alpha_t ends at 0.0001 (started at 0.1). 
ACI does not reach the target within this many items: the guarantee is a long-run average and this split is only 32 items long. The direction of travel is the thing to read, not the endpoint.

## 3. Conditional coverage by subgroup

Marginal coverage can hide a subgroup the system has quietly stopped answering. `violation rate` is the share of truly UNSAFE cases in that subgroup that were answered SAFE.


**by `family`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| nitrofurantoin_renal | 16 | 0.125 | 0.500 | 0.125 |
| ondansetron_qt | 16 | 0.125 | 0.500 | 0.125 |

**by `label`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| SAFE | 16 | 0.125 | 1.000 | 0.000 |
| UNSAFE | 16 | 0.125 | 0.000 | 0.125 |

**by `gate_fired`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| False | 32 | 0.125 | 0.500 | 0.125 |

Widest coverage gap between subgroups of n>=5: **0.000** (min 0.125, max 0.125). Marginal coverage is 0.125.

