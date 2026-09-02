# UQ coverage report -- `test`, variant `nsai_uq`

Signal: **`entropy`**. Target error rate on retained items: alpha = 0.1. n = 128.

## 1. The frozen split-conformal threshold, as shipped

| | coverage | error on retained items | target |
|---|---|---|---|
| frozen tau (calibration split) | 0.570 | 0.014 | 0.10 |

Target holds on this split.

## 2. Adaptive Conformal Inference on the same items

alpha_t updated after every item with gamma = 0.05; the threshold is the running (1 - alpha_t) quantile of uncertainties seen so far.

| | coverage | error on retained items |
|---|---|---|
| frozen tau | 0.570 | 0.014 |
| **ACI** | **0.875** | **0.161** |

alpha_t ends at 0.0151 (started at 0.1). 
ACI does not reach the target within this many items: the guarantee is a long-run average and this split is only 128 items long. The direction of travel is the thing to read, not the endpoint.

## 3. Conditional coverage by subgroup

Marginal coverage can hide a subgroup the system has quietly stopped answering. `violation rate` is the share of truly UNSAFE cases in that subgroup that were answered SAFE.


**by `family`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| acei_pregnancy | 16 | 0.500 | 1.000 | 0.000 |
| aspirin_reye | 16 | 1.000 | 1.000 | 0.000 |
| betablocker_asthma | 16 | 0.500 | 1.000 | 0.000 |
| metformin_renal | 16 | 0.500 | 1.000 | 0.000 |
| nsaid_renal | 16 | 0.500 | 1.000 | 0.000 |
| spironolactone_hyperkalaemia | 16 | 0.500 | 1.000 | 0.000 |
| statin_macrolide | 16 | 0.562 | 0.889 | 0.000 |
| warfarin_inr | 16 | 0.500 | 1.000 | 0.000 |

**by `label`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| SAFE | 64 | 0.578 | 0.973 | 0.000 |
| UNSAFE | 64 | 0.562 | 1.000 | 0.000 |

**by `gate_fired`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| False | 56 | 0.018 | 0.000 | 0.000 |
| True | 72 | 1.000 | 1.000 | 0.000 |

Widest coverage gap between subgroups of n>=5: **0.982** (min 0.018, max 1.000). Marginal coverage is 0.570.

