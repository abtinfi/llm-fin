# Automated audit of the v3b outputs (2026-09-24T18:59:06+03:00)

```

=== /home/asosoft/abtin/paper/csai/.claude/worktrees/mimic-v3/data/mimic_v3b ===
  [train]
    metformin_egfr30     causal  items= 11,080  label!=prompt:      0 ( 0.00%)
    metformin_egfr30     control items= 10,826  label!=prompt:      0 ( 0.00%)
    metformin_egfr45     causal  items= 19,930  label!=prompt:      0 ( 0.00%)
    metformin_egfr45     control items= 18,898  label!=prompt:      0 ( 0.00%)
    spironolactone_k5_5  causal  items= 20,206  label!=prompt:      0 ( 0.00%)
    spironolactone_k5_5  control items= 18,666  label!=prompt:      0 ( 0.00%)
  [calib]
    metformin_egfr30     causal  items=  7,552  label!=prompt:      0 ( 0.00%)
    metformin_egfr30     control items=  7,368  label!=prompt:      0 ( 0.00%)
    metformin_egfr45     causal  items= 13,438  label!=prompt:      0 ( 0.00%)
    metformin_egfr45     control items= 12,782  label!=prompt:      0 ( 0.00%)
    spironolactone_k5_5  causal  items= 13,388  label!=prompt:      0 ( 0.00%)
    spironolactone_k5_5  control items= 12,410  label!=prompt:      0 ( 0.00%)
  [test]
    metformin_egfr30     causal  items= 18,582  label!=prompt:      0 ( 0.00%)
    metformin_egfr30     control items= 18,142  label!=prompt:      0 ( 0.00%)
    metformin_egfr45     causal  items= 33,182  label!=prompt:      0 ( 0.00%)
    metformin_egfr45     control items= 31,532  label!=prompt:      0 ( 0.00%)
    spironolactone_k5_5  causal  items= 33,724  label!=prompt:      0 ( 0.00%)
    spironolactone_k5_5  control items= 31,102  label!=prompt:      0 ( 0.00%)
  [heldout]
    warfarin_inr4        causal  items= 18,510  label!=prompt:      0 ( 0.00%)
    warfarin_inr4        control items= 18,032  label!=prompt:      0 ( 0.00%)
  identical prompt text carrying BOTH labels: 0 of 28,854 distinct prompts

=== /home/asosoft/abtin/paper/csai/.claude/worktrees/mimic-v3/data/mimic_v3b_note ===
  [train]
    metformin_egfr30     causal  items=    124  label!=prompt:      0 ( 0.00%)
    metformin_egfr30     control items=    120  label!=prompt:      0 ( 0.00%)
    metformin_egfr45     causal  items=    252  label!=prompt:      0 ( 0.00%)
    metformin_egfr45     control items=    240  label!=prompt:      0 ( 0.00%)
    spironolactone_k5_5  causal  items=    224  label!=prompt:      0 ( 0.00%)
    spironolactone_k5_5  control items=    204  label!=prompt:      0 ( 0.00%)
  [calib]
    metformin_egfr30     causal  items=    122  label!=prompt:      0 ( 0.00%)
    metformin_egfr30     control items=    118  label!=prompt:      0 ( 0.00%)
    metformin_egfr45     causal  items=    274  label!=prompt:      0 ( 0.00%)
    metformin_egfr45     control items=    260  label!=prompt:      0 ( 0.00%)
    spironolactone_k5_5  causal  items=    204  label!=prompt:      0 ( 0.00%)
    spironolactone_k5_5  control items=    196  label!=prompt:      0 ( 0.00%)
  [test]
    metformin_egfr30     causal  items=    490  label!=prompt:      0 ( 0.00%)
    metformin_egfr30     control items=    478  label!=prompt:      0 ( 0.00%)
    metformin_egfr45     causal  items=    762  label!=prompt:      0 ( 0.00%)
    metformin_egfr45     control items=    722  label!=prompt:      0 ( 0.00%)
    spironolactone_k5_5  causal  items=    748  label!=prompt:      0 ( 0.00%)
    spironolactone_k5_5  control items=    716  label!=prompt:      0 ( 0.00%)
  [heldout]
    warfarin_inr4        causal  items=  1,000  label!=prompt:      0 ( 0.00%)
    warfarin_inr4        control items=    978  label!=prompt:      0 ( 0.00%)
  identical prompt text carrying BOTH labels: 0 of 7,452 distinct prompts
=== /home/asosoft/abtin/paper/csai/.claude/worktrees/mimic-v3/data/mimic_v3b ===
items vs distinct prompts:
  train    items=  99,606  distinct= 14,844  ratio=  6.7x
      metformin_egfr30     items= 21,906  distinct= 3,793
      metformin_egfr45     items= 38,828  distinct= 3,894
      spironolactone_k5_5  items= 38,872  distinct= 7,157
  calib    items=  66,938  distinct= 12,918  ratio=  5.2x
      metformin_egfr30     items= 14,920  distinct= 3,261
      metformin_egfr45     items= 26,220  distinct= 3,364
      spironolactone_k5_5  items= 25,798  distinct= 6,293
  test     items= 166,264  distinct= 17,550  ratio=  9.5x
      metformin_egfr30     items= 36,724  distinct= 4,669
      metformin_egfr45     items= 64,714  distinct= 4,783
      spironolactone_k5_5  items= 64,826  distinct= 8,098
  heldout  items=  36,542  distinct=  7,425  ratio=  4.9x
      warfarin_inr4        items= 36,542  distinct= 7,425
prompt overlap between splits (distinct prompts in both):
  calib    & train   : 10,279 prompts (92,465 train items)
  calib    & test    : 11,101 prompts (154,048 test items)
  test     & train   : 12,249 prompts (96,381 train items)
prompts carrying BOTH labels: 0; by families: {}
  train   :       0 items (0.00%) sit on such a prompt
  calib   :       0 items (0.00%) sit on such a prompt
  test    :       0 items (0.00%) sit on such a prompt
  heldout :       0 items (0.00%) sit on such a prompt

--- biomistral-7b / test / base  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.688062 summary=  0.688062  OK
  violation_rate       mine=   0.87149 summary=   0.87149  OK
  causal_consistency   mine=  0.109746 summary=  0.109746  OK
  spurious_flip_rate   mine=  0.057319 summary=  0.057319  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.927 | label SAFE share: 0.679
  random sample of 200: 0 with any field wrong

--- biomistral-7b / test / cl  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.951018 summary=  0.951018  OK
  violation_rate       mine=   0.09283 summary=   0.09283  OK
  causal_consistency   mine=  0.855138 summary=  0.855138  OK
  spurious_flip_rate   mine=  0.044469 summary=  0.044469  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.689 | label SAFE share: 0.679
  random sample of 200: 0 with any field wrong

--- biomistral-7b / test / nsai  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 43886, 'unparsable': 0}
  accuracy             mine=  0.605543 summary=  0.605543  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=  0.394512 summary=  0.394512  OK
  spurious_flip_rate   mine=  0.012702 summary=  0.012702  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.284 | label SAFE share: 0.679
  random sample of 200: 48 with any field wrong

--- biomistral-7b / test / nsai_uq_cl  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 3245, 'unparsable': 0}
  accuracy             mine=  0.657075 summary=  0.657075  OK
  violation_rate       mine=   0.08231 summary=   0.08231  OK
  causal_consistency   mine=  0.399027 summary=  0.399027  OK
  spurious_flip_rate   mine=  0.007009 summary=  0.007009  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.805 | label SAFE share: 0.679
  random sample of 200: 4 with any field wrong

--- biomistral-7b / test / nsai_uq  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 43886, 'unparsable': 0}
  accuracy             mine=  0.414185 summary=  0.414185  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=  0.394488 summary=  0.394488  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.284 | label SAFE share: 0.679
  random sample of 200: 50 with any field wrong

--- biomistral-7b / test / rag  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.341589 summary=  0.341589  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=  0.000211 summary=  0.000211  OK
  spurious_flip_rate   mine=  0.083218 summary=  0.083218  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.02 | label SAFE share: 0.679
  random sample of 200: 0 with any field wrong

--- biomistral-7b / test / sym  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 16524, 'unparsable': 0}
  accuracy             mine=  0.787446 summary=  0.787446  OK
  violation_rate       mine=   0.66155 summary=   0.66155  OK
  causal_consistency   mine=  0.394488 summary=  0.394488  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.891 | label SAFE share: 0.679
  random sample of 200: 29 with any field wrong

--- biomistral-7b / test / uq  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=       0.0 summary=       0.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=      None summary=      None  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.927 | label SAFE share: 0.679
  random sample of 200: 0 with any field wrong

--- biomistral-7b / test / base  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.518641 summary=  0.518641  OK
  violation_rate       mine=  0.574281 summary=  0.574281  OK
  causal_consistency   mine=      0.05 summary=      0.05  OK
  spurious_flip_rate   mine=   0.09499 summary=   0.09499  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.566 | label SAFE share: 0.68
  random sample of 200: 0 with any field wrong

--- biomistral-7b / test / nsai  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 1249, 'unparsable': 0}
  accuracy             mine=  0.593718 summary=  0.593718  OK
  violation_rate       mine=  0.091054 summary=  0.091054  OK
  causal_consistency   mine=      0.35 summary=      0.35  OK
  spurious_flip_rate   mine=  0.005219 summary=  0.005219  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.054, 'metformin_egfr45': 0.096, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.332 | label SAFE share: 0.68
  random sample of 200: 65 with any field wrong

--- biomistral-7b / test / nsai_uq  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 1249, 'unparsable': 0}
  accuracy             mine=  0.395301 summary=  0.395301  OK
  violation_rate       mine=  0.087859 summary=  0.087859  OK
  causal_consistency   mine=     0.348 summary=     0.348  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.054, 'metformin_egfr45': 0.096, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.332 | label SAFE share: 0.68
  random sample of 200: 62 with any field wrong

--- biomistral-7b / test / rag  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=   0.33095 summary=   0.33095  OK
  violation_rate       mine=  0.003195 summary=  0.003195  OK
  causal_consistency   mine=     0.011 summary=     0.011  OK
  spurious_flip_rate   mine=  0.021921 summary=  0.021921  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.013 | label SAFE share: 0.68
  random sample of 200: 0 with any field wrong

--- biomistral-7b / test / sym  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 970, 'unparsable': 0}
  accuracy             mine=  0.752554 summary=  0.752554  OK
  violation_rate       mine=  0.543131 summary=  0.543131  OK
  causal_consistency   mine=     0.375 summary=     0.375  OK
  spurious_flip_rate   mine=  0.051148 summary=  0.051148  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.054, 'metformin_egfr45': 0.096, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.78 | label SAFE share: 0.68
  random sample of 200: 46 with any field wrong

--- biomistral-7b / test / uq  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=       0.0 summary=       0.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=      None summary=      None  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.566 | label SAFE share: 0.68
  random sample of 200: 0 with any field wrong

--- biomistral-7b / heldout / base  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.333069 summary=  0.333069  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.0 | label SAFE share: 0.667
  random sample of 200: 0 with any field wrong

--- biomistral-7b / heldout / cl  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.463795 summary=  0.463795  OK
  violation_rate       mine=  0.077644 summary=  0.077644  OK
  causal_consistency   mine=  0.087628 summary=  0.087628  OK
  spurious_flip_rate   mine=  0.327307 summary=  0.327307  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.182 | label SAFE share: 0.667
  random sample of 200: 0 with any field wrong

--- biomistral-7b / heldout / nsai  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 24371, 'unparsable': 0}
  accuracy             mine=       1.0 summary=       1.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       1.0 summary=       1.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.667 | label SAFE share: 0.667
  random sample of 200: 129 with any field wrong

--- biomistral-7b / heldout / nsai_uq_cl  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 14275, 'unparsable': 0}
  accuracy             mine=       1.0 summary=       1.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       1.0 summary=       1.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.667 | label SAFE share: 0.667
  random sample of 200: 68 with any field wrong

--- biomistral-7b / heldout / nsai_uq  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 24371, 'unparsable': 0}
  accuracy             mine=       1.0 summary=       1.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       1.0 summary=       1.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.667 | label SAFE share: 0.667
  random sample of 200: 138 with any field wrong

--- biomistral-7b / heldout / rag  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.333069 summary=  0.333069  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.0 | label SAFE share: 0.667
  random sample of 200: 0 with any field wrong

--- biomistral-7b / heldout / sym  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 24371, 'unparsable': 0}
  accuracy             mine=       1.0 summary=       1.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       1.0 summary=       1.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.667 | label SAFE share: 0.667
  random sample of 200: 127 with any field wrong

--- biomistral-7b / heldout / uq  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=       0.0 summary=       0.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=      None summary=      None  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.0 | label SAFE share: 0.667
  random sample of 200: 0 with any field wrong

--- biomistral-7b / heldout / base  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.338726 summary=  0.338726  OK
  violation_rate       mine=  0.001497 summary=  0.001497  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=   0.00818 summary=   0.00818  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.002 | label SAFE share: 0.662
  random sample of 200: 0 with any field wrong

--- biomistral-7b / heldout / nsai  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 1300, 'unparsable': 0}
  accuracy             mine=  0.974722 summary=  0.974722  OK
  violation_rate       mine=   0.07485 summary=   0.07485  OK
  causal_consistency   mine=     0.928 summary=     0.928  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.688 | label SAFE share: 0.662
  random sample of 200: 135 with any field wrong

--- biomistral-7b / heldout / nsai_uq  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 1300, 'unparsable': 0}
  accuracy             mine=  0.974722 summary=  0.974722  OK
  violation_rate       mine=   0.07485 summary=   0.07485  OK
  causal_consistency   mine=     0.928 summary=     0.928  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.688 | label SAFE share: 0.662
  random sample of 200: 130 with any field wrong

--- biomistral-7b / heldout / rag  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.368049 summary=  0.368049  OK
  violation_rate       mine=  0.001497 summary=  0.001497  OK
  causal_consistency   mine=     0.026 summary=     0.026  OK
  spurious_flip_rate   mine=  0.034765 summary=  0.034765  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.031 | label SAFE share: 0.662
  random sample of 200: 0 with any field wrong

--- biomistral-7b / heldout / sym  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 1358, 'unparsable': 0}
  accuracy             mine=  0.974722 summary=  0.974722  OK
  violation_rate       mine=   0.07485 summary=   0.07485  OK
  causal_consistency   mine=     0.928 summary=     0.928  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.688 | label SAFE share: 0.662
  random sample of 200: 133 with any field wrong

--- biomistral-7b / heldout / uq  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=       0.0 summary=       0.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=      None summary=      None  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.002 | label SAFE share: 0.662
  random sample of 200: 0 with any field wrong

--- llama3-openbiollm-8b / test / base  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.678704 summary=  0.678704  OK
  violation_rate       mine=       1.0 summary=       1.0  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 1.0 | label SAFE share: 0.679
  random sample of 200: 0 with any field wrong

--- llama3-openbiollm-8b / test / cl  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 1753}
  accuracy             mine=   0.89966 summary=   0.89966  OK
  violation_rate       mine=  0.059753 summary=  0.059753  OK
  causal_consistency   mine=  0.768622 summary=  0.768622  OK
  spurious_flip_rate   mine=  0.121893 summary=  0.121893  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.621 | label SAFE share: 0.679
  random sample of 200: 0 with any field wrong

--- llama3-openbiollm-8b / test / nsai  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 34943, 'unparsable': 91814}
  accuracy             mine=  0.431374 summary=  0.431374  OK
  violation_rate       mine=  0.051067 summary=  0.051067  OK
  causal_consistency   mine=  0.394488 summary=  0.394488  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.339 | label SAFE share: 0.679
  random sample of 200: 42 with any field wrong

--- llama3-openbiollm-8b / test / nsai_uq_cl  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 34166, 'unparsable': 0}
  accuracy             mine=  0.440865 summary=  0.440865  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=  0.394488 summary=  0.394488  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.281 | label SAFE share: 0.679
  random sample of 200: 30 with any field wrong

--- llama3-openbiollm-8b / test / nsai_uq  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 34943, 'unparsable': 91814}
  accuracy             mine=  0.395925 summary=  0.395925  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=  0.394488 summary=  0.394488  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.339 | label SAFE share: 0.679
  random sample of 200: 47 with any field wrong

--- llama3-openbiollm-8b / test / rag  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 126402}
  accuracy             mine=  0.221208 summary=  0.221208  OK
  violation_rate       mine=  0.057712 summary=  0.057712  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.24 | label SAFE share: 0.679
  random sample of 200: 0 with any field wrong

--- llama3-openbiollm-8b / test / sym  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 18080, 'unparsable': 0}
  accuracy             mine=  0.787446 summary=  0.787446  OK
  violation_rate       mine=   0.66155 summary=   0.66155  OK
  causal_consistency   mine=  0.394488 summary=  0.394488  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.891 | label SAFE share: 0.679
  random sample of 200: 21 with any field wrong

--- llama3-openbiollm-8b / test / uq  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.076451 summary=  0.076451  OK
  violation_rate       mine=  0.015631 summary=  0.015631  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 1.0 | label SAFE share: 0.679
  random sample of 200: 0 with any field wrong

--- llama3-openbiollm-8b / test / base  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 75}
  accuracy             mine=  0.667518 summary=  0.667518  OK
  violation_rate       mine=  0.963259 summary=  0.963259  OK
  causal_consistency   mine=     0.004 summary=     0.004  OK
  spurious_flip_rate   mine=  0.004283 summary=  0.004283  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.971 | label SAFE share: 0.68
  random sample of 200: 0 with any field wrong

--- llama3-openbiollm-8b / test / nsai  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 479, 'unparsable': 117}
  accuracy             mine=  0.752043 summary=  0.752043  OK
  violation_rate       mine=  0.609425 summary=  0.609425  OK
  causal_consistency   mine=     0.363 summary=     0.363  OK
  spurious_flip_rate   mine=  0.035599 summary=  0.035599  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.054, 'metformin_egfr45': 0.096, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.834 | label SAFE share: 0.68
  random sample of 200: 27 with any field wrong

--- llama3-openbiollm-8b / test / nsai_uq  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 479, 'unparsable': 117}
  accuracy             mine=  0.395301 summary=  0.395301  OK
  violation_rate       mine=  0.087859 summary=  0.087859  OK
  causal_consistency   mine=     0.348 summary=     0.348  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.054, 'metformin_egfr45': 0.096, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.834 | label SAFE share: 0.68
  random sample of 200: 17 with any field wrong

--- llama3-openbiollm-8b / test / rag  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 204}
  accuracy             mine=  0.635087 summary=  0.635087  OK
  violation_rate       mine=  0.816294 summary=  0.816294  OK
  causal_consistency   mine=     0.041 summary=     0.041  OK
  spurious_flip_rate   mine=  0.081678 summary=  0.081678  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.857 | label SAFE share: 0.68
  random sample of 200: 0 with any field wrong

--- llama3-openbiollm-8b / test / sym  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 397, 'unparsable': 55}
  accuracy             mine=  0.768386 summary=  0.768386  OK
  violation_rate       mine=  0.672524 summary=  0.672524  OK
  causal_consistency   mine=     0.351 summary=     0.351  OK
  spurious_flip_rate   mine=  0.004251 summary=  0.004251  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.054, 'metformin_egfr45': 0.096, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.885 | label SAFE share: 0.68
  random sample of 200: 22 with any field wrong

--- llama3-openbiollm-8b / test / uq  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 75}
  accuracy             mine=       0.0 summary=       0.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=      None summary=      None  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.971 | label SAFE share: 0.68
  random sample of 200: 0 with any field wrong

--- llama3-openbiollm-8b / heldout / base  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.666931 summary=  0.666931  OK
  violation_rate       mine=       1.0 summary=       1.0  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 1.0 | label SAFE share: 0.667
  random sample of 200: 0 with any field wrong

--- llama3-openbiollm-8b / heldout / cl  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.333069 summary=  0.333069  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.0 | label SAFE share: 0.667
  random sample of 200: 0 with any field wrong

--- llama3-openbiollm-8b / heldout / nsai  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 12171, 'unparsable': 0}
  accuracy             mine=       1.0 summary=       1.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       1.0 summary=       1.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.667 | label SAFE share: 0.667
  random sample of 200: 71 with any field wrong

--- llama3-openbiollm-8b / heldout / nsai_uq_cl  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 24371, 'unparsable': 0}
  accuracy             mine=       1.0 summary=       1.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       1.0 summary=       1.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.667 | label SAFE share: 0.667
  random sample of 200: 125 with any field wrong

--- llama3-openbiollm-8b / heldout / nsai_uq  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 12171, 'unparsable': 0}
  accuracy             mine=       1.0 summary=       1.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       1.0 summary=       1.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.667 | label SAFE share: 0.667
  random sample of 200: 62 with any field wrong

--- llama3-openbiollm-8b / heldout / rag  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.666931 summary=  0.666931  OK
  violation_rate       mine=       1.0 summary=       1.0  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 1.0 | label SAFE share: 0.667
  random sample of 200: 0 with any field wrong

--- llama3-openbiollm-8b / heldout / sym  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 12171, 'unparsable': 0}
  accuracy             mine=       1.0 summary=       1.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       1.0 summary=       1.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.667 | label SAFE share: 0.667
  random sample of 200: 73 with any field wrong

--- llama3-openbiollm-8b / heldout / uq  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=       0.0 summary=       0.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=      None summary=      None  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 1.0 | label SAFE share: 0.667
  random sample of 200: 0 with any field wrong

--- llama3-openbiollm-8b / heldout / base  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 4}
  accuracy             mine=  0.660263 summary=  0.660263  OK
  violation_rate       mine=  0.998503 summary=  0.998503  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.997 | label SAFE share: 0.662
  random sample of 200: 0 with any field wrong

--- llama3-openbiollm-8b / heldout / nsai  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 687, 'unparsable': 0}
  accuracy             mine=  0.974722 summary=  0.974722  OK
  violation_rate       mine=   0.07485 summary=   0.07485  OK
  causal_consistency   mine=     0.928 summary=     0.928  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.688 | label SAFE share: 0.662
  random sample of 200: 70 with any field wrong

--- llama3-openbiollm-8b / heldout / nsai_uq  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 687, 'unparsable': 0}
  accuracy             mine=  0.974722 summary=  0.974722  OK
  violation_rate       mine=   0.07485 summary=   0.07485  OK
  causal_consistency   mine=     0.928 summary=     0.928  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.688 | label SAFE share: 0.662
  random sample of 200: 81 with any field wrong

--- llama3-openbiollm-8b / heldout / rag  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 97}
  accuracy             mine=  0.634985 summary=  0.634985  OK
  violation_rate       mine=  0.790419 summary=  0.790419  OK
  causal_consistency   mine=     0.066 summary=     0.066  OK
  spurious_flip_rate   mine=  0.047312 summary=  0.047312  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.849 | label SAFE share: 0.662
  random sample of 200: 0 with any field wrong

--- llama3-openbiollm-8b / heldout / sym  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 622, 'unparsable': 0}
  accuracy             mine=  0.974722 summary=  0.974722  OK
  violation_rate       mine=   0.07485 summary=   0.07485  OK
  causal_consistency   mine=     0.928 summary=     0.928  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.688 | label SAFE share: 0.662
  random sample of 200: 68 with any field wrong

--- llama3-openbiollm-8b / heldout / uq  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 4}
  accuracy             mine=       0.0 summary=       0.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=      None summary=      None  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.997 | label SAFE share: 0.662
  random sample of 200: 0 with any field wrong

--- mistral-7b-instruct-v0-2 / test / base  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.646863 summary=  0.646863  OK
  violation_rate       mine=  0.959023 summary=  0.959023  OK
  causal_consistency   mine=   0.01881 summary=   0.01881  OK
  spurious_flip_rate   mine=  0.178989 summary=  0.178989  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.942 | label SAFE share: 0.679
  random sample of 200: 0 with any field wrong

--- mistral-7b-instruct-v0-2 / test / cl  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.905512 summary=  0.905512  OK
  violation_rate       mine=  0.090453 summary=  0.090453  OK
  causal_consistency   mine=   0.76888 summary=   0.76888  OK
  spurious_flip_rate   mine=  0.117015 summary=  0.117015  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.642 | label SAFE share: 0.679
  random sample of 200: 0 with any field wrong

--- mistral-7b-instruct-v0-2 / test / nsai  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 8893, 'unparsable': 0}
  accuracy             mine=  0.794622 summary=  0.794622  OK
  violation_rate       mine=  0.228323 summary=  0.228323  OK
  causal_consistency   mine=  0.522436 summary=  0.522436  OK
  spurious_flip_rate   mine=  0.245816 summary=  0.245816  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.62 | label SAFE share: 0.679
  random sample of 200: 8 with any field wrong

--- mistral-7b-instruct-v0-2 / test / nsai_uq_cl  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 12241, 'unparsable': 0}
  accuracy             mine=  0.466457 summary=  0.466457  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=  0.394979 summary=  0.394979  OK
  spurious_flip_rate   mine=  0.000924 summary=  0.000924  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.539 | label SAFE share: 0.679
  random sample of 200: 12 with any field wrong

--- mistral-7b-instruct-v0-2 / test / nsai_uq  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 8893, 'unparsable': 0}
  accuracy             mine=  0.555484 summary=  0.555484  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=  0.398535 summary=  0.398535  OK
  spurious_flip_rate   mine=  0.022148 summary=  0.022148  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.62 | label SAFE share: 0.679
  random sample of 200: 10 with any field wrong

--- mistral-7b-instruct-v0-2 / test / rag  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.741135 summary=  0.741135  OK
  violation_rate       mine=  0.232741 summary=  0.232741  OK
  causal_consistency   mine=  0.382931 summary=  0.382931  OK
  spurious_flip_rate   mine=  0.316678 summary=  0.316678  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.569 | label SAFE share: 0.679
  random sample of 200: 0 with any field wrong

--- mistral-7b-instruct-v0-2 / test / sym  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 24544, 'unparsable': 0}
  accuracy             mine=  0.794483 summary=  0.794483  OK
  violation_rate       mine=  0.639648 summary=  0.639648  OK
  causal_consistency   mine=  0.395658 summary=  0.395658  OK
  spurious_flip_rate   mine=  0.026394 summary=  0.026394  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.884 | label SAFE share: 0.679
  random sample of 200: 31 with any field wrong

--- mistral-7b-instruct-v0-2 / test / uq  (166,264 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.089454 summary=  0.089454  OK
  violation_rate       mine=  0.028547 summary=  0.028547  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 17,550
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.942 | label SAFE share: 0.679
  random sample of 200: 0 with any field wrong

--- mistral-7b-instruct-v0-2 / test / base  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.660878 summary=  0.660878  OK
  violation_rate       mine=  0.867412 summary=  0.867412  OK
  causal_consistency   mine=     0.046 summary=     0.046  OK
  spurious_flip_rate   mine=   0.09499 summary=   0.09499  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.896 | label SAFE share: 0.68
  random sample of 200: 0 with any field wrong

--- mistral-7b-instruct-v0-2 / test / nsai  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 295, 'unparsable': 0}
  accuracy             mine=  0.746936 summary=  0.746936  OK
  violation_rate       mine=  0.340256 summary=  0.340256  OK
  causal_consistency   mine=     0.426 summary=     0.426  OK
  spurious_flip_rate   mine=  0.158664 summary=  0.158664  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.054, 'metformin_egfr45': 0.096, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.645 | label SAFE share: 0.68
  random sample of 200: 16 with any field wrong

--- mistral-7b-instruct-v0-2 / test / nsai_uq  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 295, 'unparsable': 0}
  accuracy             mine=  0.395301 summary=  0.395301  OK
  violation_rate       mine=  0.087859 summary=  0.087859  OK
  causal_consistency   mine=     0.348 summary=     0.348  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.054, 'metformin_egfr45': 0.096, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.645 | label SAFE share: 0.68
  random sample of 200: 17 with any field wrong

--- mistral-7b-instruct-v0-2 / test / rag  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.701226 summary=  0.701226  OK
  violation_rate       mine=  0.339457 summary=  0.339457  OK
  causal_consistency   mine=      0.31 summary=      0.31  OK
  spurious_flip_rate   mine=  0.213987 summary=  0.213987  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.599 | label SAFE share: 0.68
  random sample of 200: 0 with any field wrong

--- mistral-7b-instruct-v0-2 / test / sym  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 498, 'unparsable': 0}
  accuracy             mine=  0.782431 summary=  0.782431  OK
  violation_rate       mine=  0.607827 summary=  0.607827  OK
  causal_consistency   mine=     0.381 summary=     0.381  OK
  spurious_flip_rate   mine=   0.04071 summary=   0.04071  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.054, 'metformin_egfr45': 0.096, 'spironolactone_k5_5': 1.0}
  pred SAFE share: 0.851 | label SAFE share: 0.68
  random sample of 200: 28 with any field wrong

--- mistral-7b-instruct-v0-2 / test / uq  (3,916 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=       0.0 summary=       0.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=      None summary=      None  OK
  identical prompts with DIFFERENT preds: 0 of 3,513
  gate_fired rate: {'metformin_egfr30': 0.0, 'metformin_egfr45': 0.0, 'spironolactone_k5_5': 0.0}
  pred SAFE share: 0.896 | label SAFE share: 0.68
  random sample of 200: 0 with any field wrong

--- mistral-7b-instruct-v0-2 / heldout / base  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.704012 summary=  0.704012  OK
  violation_rate       mine=   0.88867 summary=   0.88867  OK
  causal_consistency   mine=  0.032091 summary=  0.032091  OK
  spurious_flip_rate   mine=  0.112467 summary=  0.112467  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.963 | label SAFE share: 0.667
  random sample of 200: 0 with any field wrong

--- mistral-7b-instruct-v0-2 / heldout / cl  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.412457 summary=  0.412457  OK
  violation_rate       mine=  0.000164 summary=  0.000164  OK
  causal_consistency   mine=  0.007455 summary=  0.007455  OK
  spurious_flip_rate   mine=  0.306233 summary=  0.306233  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.079 | label SAFE share: 0.667
  random sample of 200: 0 with any field wrong

--- mistral-7b-instruct-v0-2 / heldout / nsai  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 5697, 'unparsable': 0}
  accuracy             mine=       1.0 summary=       1.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       1.0 summary=       1.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.667 | label SAFE share: 0.667
  random sample of 200: 31 with any field wrong

--- mistral-7b-instruct-v0-2 / heldout / nsai_uq_cl  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 5707, 'unparsable': 0}
  accuracy             mine=       1.0 summary=       1.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       1.0 summary=       1.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.667 | label SAFE share: 0.667
  random sample of 200: 36 with any field wrong

--- mistral-7b-instruct-v0-2 / heldout / nsai_uq  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 5697, 'unparsable': 0}
  accuracy             mine=       1.0 summary=       1.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       1.0 summary=       1.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.667 | label SAFE share: 0.667
  random sample of 200: 24 with any field wrong

--- mistral-7b-instruct-v0-2 / heldout / rag  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.844097 summary=  0.844097  OK
  violation_rate       mine=  0.274012 summary=  0.274012  OK
  causal_consistency   mine=  0.550297 summary=  0.550297  OK
  spurious_flip_rate   mine=  0.089618 summary=  0.089618  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.694 | label SAFE share: 0.667
  random sample of 200: 0 with any field wrong

--- mistral-7b-instruct-v0-2 / heldout / sym  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 10816, 'unparsable': 0}
  accuracy             mine=       1.0 summary=       1.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       1.0 summary=       1.0  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.667 | label SAFE share: 0.667
  random sample of 200: 60 with any field wrong

--- mistral-7b-instruct-v0-2 / heldout / uq  (36,542 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.003284 summary=  0.003284  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=      None summary=      None  OK
  identical prompts with DIFFERENT preds: 0 of 7,425
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.963 | label SAFE share: 0.667
  random sample of 200: 0 with any field wrong

--- mistral-7b-instruct-v0-2 / heldout / base  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.557634 summary=  0.557634  OK
  violation_rate       mine=  0.107784 summary=  0.107784  OK
  causal_consistency   mine=     0.148 summary=     0.148  OK
  spurious_flip_rate   mine=  0.274029 summary=  0.274029  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.293 | label SAFE share: 0.662
  random sample of 200: 0 with any field wrong

--- mistral-7b-instruct-v0-2 / heldout / nsai  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 363, 'unparsable': 0}
  accuracy             mine=  0.974722 summary=  0.974722  OK
  violation_rate       mine=   0.07485 summary=   0.07485  OK
  causal_consistency   mine=     0.928 summary=     0.928  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.688 | label SAFE share: 0.662
  random sample of 200: 41 with any field wrong

--- mistral-7b-instruct-v0-2 / heldout / nsai_uq  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 363, 'unparsable': 0}
  accuracy             mine=  0.974722 summary=  0.974722  OK
  violation_rate       mine=   0.07485 summary=   0.07485  OK
  causal_consistency   mine=     0.928 summary=     0.928  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.688 | label SAFE share: 0.662
  random sample of 200: 31 with any field wrong

--- mistral-7b-instruct-v0-2 / heldout / rag  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=  0.831648 summary=  0.831648  OK
  violation_rate       mine=  0.302395 summary=  0.302395  OK
  causal_consistency   mine=      0.48 summary=      0.48  OK
  spurious_flip_rate   mine=  0.126789 summary=  0.126789  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.698 | label SAFE share: 0.662
  random sample of 200: 0 with any field wrong

--- mistral-7b-instruct-v0-2 / heldout / sym  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 919, 'unparsable': 0}
  accuracy             mine=  0.974722 summary=  0.974722  OK
  violation_rate       mine=   0.07485 summary=   0.07485  OK
  causal_consistency   mine=     0.928 summary=     0.928  OK
  spurious_flip_rate   mine=       0.0 summary=       0.0  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 1.0}
  pred SAFE share: 0.688 | label SAFE share: 0.662
  random sample of 200: 83 with any field wrong

--- mistral-7b-instruct-v0-2 / heldout / uq  (1,978 preds)
  integrity: {'missing': 0, 'extra': 0, 'duplicate_ids': 0, 'label_mismatch': 0, 'pair_mismatch': 0, 'pred_not_parse_of_raw': 0, 'unparsable': 0}
  accuracy             mine=       0.0 summary=       0.0  OK
  violation_rate       mine=       0.0 summary=       0.0  OK
  causal_consistency   mine=       0.0 summary=       0.0  OK
  spurious_flip_rate   mine=      None summary=      None  OK
  identical prompts with DIFFERENT preds: 0 of 1,891
  gate_fired rate: {'warfarin_inr4': 0.0}
  pred SAFE share: 0.293 | label SAFE share: 0.662
  random sample of 200: 0 with any field wrong
```
