# UQ coverage report -- `test`, variant `uq`

Signal: **`decision_entropy`**. Target error rate on retained items: alpha = 0.1. n = 128.

## 1. The frozen split-conformal threshold, as shipped

| | coverage | error on retained items | target |
|---|---|---|---|
| frozen tau (calibration split) | 0.078 | 0.300 | 0.10 |

Target **violated** on this split.

A frozen threshold is only valid while the test stream is exchangeable with the calibration split. This split is a different rule family, so it is not, and the guarantee does not transfer. That is not a bug in the calibration -- it is the documented limit of split conformal, and the reason the proposal names the adaptive variant.

## 2. Adaptive Conformal Inference on the same items

alpha_t updated after every item with gamma = 0.05; the threshold is the running (1 - alpha_t) quantile of uncertainties seen so far.

| | coverage | error on retained items |
|---|---|---|
| frozen tau | 0.078 | 0.300 |
| **ACI** | **0.977** | **0.496** |

alpha_t ends at 0.0001 (started at 0.1). 
ACI does not reach the target within this many items: the guarantee is a long-run average and this split is only 128 items long. The direction of travel is the thing to read, not the endpoint.

## 3. Conditional coverage by subgroup

Marginal coverage can hide a subgroup the system has quietly stopped answering. `violation rate` is the share of truly UNSAFE cases in that subgroup that were answered SAFE.


**by `family`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| acei_pregnancy | 16 | 0.062 | 1.000 | 0.000 |
| aspirin_reye | 16 | 0.000 | — | 0.000 |
| betablocker_asthma | 16 | 0.062 | 1.000 | 0.000 |
| metformin_renal | 16 | 0.125 | 0.500 | 0.125 |
| nsaid_renal | 16 | 0.000 | — | 0.000 |
| spironolactone_hyperkalaemia | 16 | 0.000 | — | 0.000 |
| statin_macrolide | 16 | 0.125 | 1.000 | 0.000 |
| warfarin_inr | 16 | 0.250 | 0.500 | 0.250 |

**by `label`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| SAFE | 64 | 0.109 | 1.000 | 0.000 |
| UNSAFE | 64 | 0.047 | 0.000 | 0.047 |

**by `gate_fired`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| False | 128 | 0.078 | 0.700 | 0.047 |

Widest coverage gap between subgroups of n>=5: **0.250** (min 0.000, max 0.250). Marginal coverage is 0.078.

