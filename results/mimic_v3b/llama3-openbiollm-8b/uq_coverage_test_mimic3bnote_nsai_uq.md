# UQ coverage report -- `test_mimic3bnote`, variant `nsai_uq`

Signal: **`decision_entropy`**. Target error rate on retained items: alpha = 0.1. n = 3916.

## 1. The frozen split-conformal threshold, as shipped

| | coverage | error on retained items | target |
|---|---|---|---|
| frozen tau (calibration split) | 0.423 | 0.066 | 0.10 |

Target holds on this split.

## 2. Adaptive Conformal Inference on the same items

alpha_t updated after every item with gamma = 0.05; the threshold is the running (1 - alpha_t) quantile of uncertainties seen so far.

| | coverage | error on retained items |
|---|---|---|
| frozen tau | 0.423 | 0.066 |
| **ACI** | **0.919** | **0.230** |

alpha_t ends at 0.0301 (started at 0.1). 
ACI does not reach the target within this many items: the guarantee is a long-run average and this split is only 3916 items long. The direction of travel is the thing to read, not the endpoint.

## 3. Conditional coverage by subgroup

Marginal coverage can hide a subgroup the system has quietly stopped answering. `violation rate` is the share of truly UNSAFE cases in that subgroup that were answered SAFE.


**by `family`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| metformin_egfr30 | 968 | 0.054 | 0.519 | 0.079 |
| metformin_egfr45 | 1484 | 0.096 | 0.585 | 0.111 |
| spironolactone_k5_5 | 1464 | 1.000 | 0.982 | 0.065 |

**by `label`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| SAFE | 2664 | 0.440 | 1.000 | 0.000 |
| UNSAFE | 1252 | 0.388 | 0.774 | 0.088 |

**by `gate_fired`**

| value | n | coverage | selective accuracy | violation rate |
|---|---|---|---|---|
| False | 2258 | 0.000 | — | 0.000 |
| True | 1658 | 1.000 | 0.934 | 0.226 |

Widest coverage gap between subgroups of n>=5: **1.000** (min 0.000, max 1.000). Marginal coverage is 0.423.

