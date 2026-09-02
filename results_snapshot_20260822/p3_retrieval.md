# P3 — Does retrieval actually work?

BioMistral-7B, seed 0, TF-IDF retriever, k=3, 22-passage corpus
(10 guidelines + 12 distractors). Gold document for a case of family F is
`guideline::F`, read from the `retrieved` field of the prediction records.

Run with `python src/retrieval_diagnostic.py --preds <rag preds> --compare <base preds>`.

## Headline: retrieval is NOT the bottleneck

| split | hit@1 | hit@3 | precision@3 | mean rank of gold |
|---|---|---|---|---|
| test (128 cases, 8 families) | 0.500 | **0.648** | 0.216 | 1.29 |
| heldout (32 cases, 2 families) | — | **1.000** | 0.333 (ceiling) | 1.53 |

On held-out the correct guideline was retrieved for **every single case**, at the
theoretical maximum precision@3. Accuracy there is 0.500 and Causal Consistency
is **0.000**. The model had the right guidance in context and still failed.

## Accuracy does not improve when the gold doc is present

Test split:

| | n | accuracy |
|---|---|---|
| gold doc retrieved | 83 | 0.614 |
| gold doc missed | 45 | 0.667 |

Accuracy is if anything *lower* when the correct guideline is in the context.

Among the 83 test cases where the gold doc was retrieved, the answer changed
versus no-RAG in only 10 cases (7 wrong→right, 3 right→wrong). On held-out,
4 of 32 changed (2 wrong→right, 2 right→wrong) — a wash.

## Per-family breakdown (test)

| family | n | hit@3 | acc | acc given hit | acc given miss |
|---|---|---|---|---|---|
| acei_pregnancy | 16 | 0.062 | 0.750 | 1.000 | 0.733 |
| aspirin_reye | 16 | 0.125 | 0.688 | 0.500 | 0.714 |
| betablocker_asthma | 16 | 1.000 | 0.500 | 0.500 | — |
| metformin_renal | 16 | 0.500 | 0.688 | 0.875 | 0.500 |
| nsaid_renal | 16 | 0.500 | 0.562 | 0.500 | 0.625 |
| spironolactone_hyperkalaemia | 16 | 1.000 | 0.500 | 0.500 | — |
| statin_macrolide | 16 | 1.000 | 0.500 | 0.500 | — |
| warfarin_inr | 16 | 1.000 | 0.875 | 0.875 | — |

Two families retrieve badly (`acei_pregnancy` 0.062, `aspirin_reye` 0.125) —
their vignettes present the causal factor in prose ("is 12 weeks pregnant",
a paediatric age) that shares few terms with the guideline text, which is
exactly where TF-IDF's lexical matching breaks down. Distractors occupy 57.6% of
retrieved slots on the test split.

Note those two families have *above-average* accuracy despite the worst
retrieval, and the four families with perfect retrieval sit at 0.500. There is
no relationship between retrieval quality and accuracy in either direction.

## Verdict against the P3 acceptance criterion

> "A stated retrieval hit rate. The claim 'RAG is insufficient' is only
> defensible once retrieval is shown to have worked."

Stated: **hit@3 = 0.648 on test, 1.000 on held-out.**

The held-out result settles it: retrieval was perfect and RAG still produced
0.000 Causal Consistency. The failure is the model ignoring retrieved context,
not a plumbing failure. **"RAG is insufficient on this benchmark" is now a
defensible claim.**

A dense retriever would raise the test-split hit rate (TF-IDF clearly struggles
on the two prose-presented families), but held-out shows that would not change
the conclusion: at hit rate 1.000 the accuracy is still chance.

## Honest caveats

- Held-out is 16 pairs. It is decisive about *retrieval working*, which is a
  deterministic property of the retriever, but its accuracy numbers carry wide
  CIs.
- The corpus is 22 passages. Retrieval into a realistic corpus would be harder;
  this setup is generous to RAG, which strengthens the negative result rather
  than weakening it.
