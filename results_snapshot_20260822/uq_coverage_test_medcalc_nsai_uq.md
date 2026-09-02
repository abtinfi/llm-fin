# UQ coverage report -- `test_medcalc`, variant `nsai_uq`

Signal: **`entropy`**. Target error rate on retained items: alpha = 0.1. n = 180.

## 1. The frozen split-conformal threshold, as shipped

| | coverage | error on retained items | target |
|---|---|---|---|
| frozen tau (calibration split) | 1.000 | 0.006 | 0.10 |

Target holds on this split.

## 2. Adaptive Conformal Inference on the same items

alpha_t updated after every item with gamma = 0.05; the threshold is the running (1 - alpha_t) quantile of uncertainties seen so far.

| | coverage | error on retained items |
|---|---|---|
| frozen tau | 1.000 | 0.006 |
| **ACI** | **0.489** | **0.000** |

alpha_t ends at 0.9950 (started at 0.1). 
ACI holds the target on this split.

## 3. Conditional coverage by subgroup

Marginal coverage can hide a subgroup the system has quietly stopped answering. `violation rate` is the share of truly UNSAFE cases in that subgroup that were answered SAFE.


**by `family`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| metformin_renal | 180 | 1.000 | 0.994 | 0.000 |

**by `label`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| SAFE | 90 | 1.000 | 0.989 | 0.000 |
| UNSAFE | 90 | 1.000 | 1.000 | 0.000 |

**by `gate_fired`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| False | 2  ⚠️ n<5 | 1.000 | 0.500 | 0.000 |
| True | 178 | 1.000 | 1.000 | 0.000 |

Widest coverage gap between subgroups of n>=5: **0.000** (min 1.000, max 1.000). Marginal coverage is 1.000.

