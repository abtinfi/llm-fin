# UQ coverage report -- `test_mimic3b`, variant `nsai_uq_cl`

Signal: **`decision_entropy`**. Target error rate on retained items: alpha = 0.1. n = 166264.

## 1. The frozen split-conformal threshold, as shipped

| | coverage | error on retained items | target |
|---|---|---|---|
| frozen tau (calibration split) | 0.474 | 0.015 | 0.10 |

Target holds on this split.

## 2. Adaptive Conformal Inference on the same items

alpha_t updated after every item with gamma = 0.05; the threshold is the running (1 - alpha_t) quantile of uncertainties seen so far.

| | coverage | error on retained items |
|---|---|---|
| frozen tau | 0.474 | 0.015 |
| **ACI** | **0.149** | **0.225** |

alpha_t ends at 0.9990 (started at 0.1). 
ACI does not reach the target within this many items: the guarantee is a long-run average and this split is only 166264 items long. The direction of travel is the thing to read, not the endpoint.

## 3. Conditional coverage by subgroup

Marginal coverage can hide a subgroup the system has quietly stopped answering. `violation rate` is the share of truly UNSAFE cases in that subgroup that were answered SAFE.


**by `family`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| metformin_egfr30 | 36724 | 0.137 | 0.758 | 0.000 |
| metformin_egfr45 | 64714 | 0.138 | 1.000 | 0.000 |
| spironolactone_k5_5 | 64826 | 1.000 | 1.000 | 0.000 |

**by `label`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| SAFE | 112844 | 0.489 | 0.978 | 0.000 |
| UNSAFE | 53420 | 0.442 | 1.000 | 0.000 |

**by `gate_fired`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| False | 101438 | 0.138 | 0.913 | 0.000 |
| True | 64826 | 1.000 | 1.000 | 0.000 |

Widest coverage gap between subgroups of n>=5: **0.863** (min 0.137, max 1.000). Marginal coverage is 0.474.

