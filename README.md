# Ablation of the proposal's four contributions, against a baseline

Implements the ablation matrix of §4.6 of the proposal on the smallest possible
dataset first, per the supervisor's two instructions:

1. **Every contribution is measured against the baseline**, in one table, not
   only against the row above it.
2. **Start small and simple, then scale.** Templated synthetic vignettes
   (128 items) → real PMC clinical notes (600 items) → MedMCQA / MedQA /
   PubMedQA (6,456 items) for the UQ question.

## The table

Eight rows, produced by `src/make_table.py`:

| # | Row | RAG | Gate | UQ | Constraint layer |
|---|---|:-:|:-:|:-:|:-:|
| 1 | Base LLM | | | | |
| 2 | + RAG | ✓ | | | |
| 3 | + Symbolic Gate (NS-AI) | ✓ | ✓ | | |
| 4 | + UQ Engine (NS-AI+UQ) | ✓ | ✓ | ✓ | |
| 5 | Base + Symbolic Gate only | | ✓ | | |
| 6 | Base + UQ only | | | ✓ | |
| 7 | Base + Constraint Layer only | | | | ✓ |
| 8 | All four | ✓ | ✓ | ✓ | ✓ |

Rows 1–4 are the proposal's cumulative ladder. Rows 5–7 isolate one
contribution each, because a cumulative ladder cannot attribute an effect: on
real notes RAG *lowers* Causal Consistency, so anything measured on top of it
starts from a damaged state. Row 7 is the trained Constraint-Aware Layer of
Aim 3, scored through the **same** readout as every other row.

## Run order

```bash
pip install -r requirements.txt

# 1. build both benchmarks (integrity checks run at build time and abort on
#    failure -- see "Data integrity" below)
python src/build_dataset.py --out data                 # synthetic, 128 items
python src/build_medcalc.py --out data/medcalc         # real notes, 600 items

# 2. plumbing check, no GPU needed -- results are NOT scientific
python src/run_eval.py --backend mock --seeds 0
python src/make_table.py --split test

# 3. everything, on GPU 0
bash run_full_pipeline.sh
```

`run_full_pipeline.sh` runs the nine stages in order: both ablations, the
constraint-layer training and its controls, the constraint-layer table rows,
the RQ3 perplexity check, the tables, the conformal/coverage reports, and the
Aim 1 SAE. It pins `CUDA_VISIBLE_DEVICES=0`.

## What each file does

| File | Aim | What it produces |
|---|---|---|
| `src/build_dataset.py` | — | synthetic counterfactual pairs + integrity checks |
| `src/build_medcalc.py` | — | counterfactual pairs from real PMC notes |
| `src/run_eval.py` | 3, 4 | the eight ablation rows |
| `src/make_table.py` | 4 | the deliverable table + paired tests |
| `src/sae.py` | **1** | sparse autoencoder, FIS, feature knock-out |
| `src/probe.py` | 1 | is the decisive fact linearly decodable? |
| `src/patching.py` | 2 | activation patching, ACE by layer |
| `src/steering.py` | 3 | fixed-direction steering (the null result) |
| `src/constraint_layer.py` | 3 | the trained Constraint-Aware Layer |
| `src/coverage_report.py` | 4 | adaptive conformal + conditional coverage |
| `src/perplexity.py` | 3 | RQ3: does the layer cost language ability? |

## Design decisions that are deliberately not shortcuts

- **fp16/bf16, no 4-bit quantisation.** Quantisation perturbs the logit
  distribution and would invalidate the predictive-entropy term (Eq. 2).
- **Ground truth is generated, not judged.** No LLM-as-judge, no human
  annotation. Each counterfactual pair flips exactly one causal variable across
  a threshold, so the label flip is guaranteed.
- **Pair-level Causal Consistency.** A model that always answers UNSAFE gets
  50% per-arm but 0% at pair level. Only the pair-level number is reported.
- **Paired statistics.** Identical prompts, decoding, item order and seeds
  across variants, so McNemar is valid. Holm correction over the comparisons.
- **The symbolic gate never guesses.** If it cannot extract the relevant fact
  it defers to the neural answer.
- **No fake error bars.** Decoding is greedy, so a second seed reproduces the
  run exactly; the table reports item-level bootstrap CIs instead of a
  seed-to-seed standard deviation that would always be 0.000.

## Data integrity

`build_dataset.py` refuses to write a split that fails any of:

- every pair has exactly two arms, one SAFE and one UNSAFE;
- the two arms are **minimal** — identical once the causal factor is blanked
  out (compared on the stored `skeleton`, so categorical factors work too);
- wherever the symbolic gate can read a vignette, its decision agrees with the
  label.

These exist because three real defects got through without them, all of which
are fixed in the current builder:

1. the two arms drew **different distractor labs**, so 64/64 test pairs were
   not minimal pairs;
2. the implicit β-blocker renderer branched on the substring `"asthma"`, which
   the *safe* values also contain (`"no history of asthma"`), so those arms
   described severe asthma while labelled SAFE — items that contradicted their
   own ground truth;
3. clinically impossible patients: pregnant women aged 67–84 (the pregnancy
   family drew from the default older-adult age pool), a 15-year-old described
   as being in primary school, and implicit renal arms whose hard-coded
   creatinine implied an eGFR far from the near-threshold value the explicit
   arm stated.

The pre-fix data and results are kept under `data/archive_pre_fix/` and
`results/archive_pre_fix/`.

### Data provenance — no credentialed source is needed to run

`python src/check_data.py` reports where every dataset came from, verifies the
files the pipeline reads, re-checks pair structure and cross-split leakage, and
exits non-zero under `--strict`. It runs as stage 0 of `run_full_pipeline.sh`.

| Source | Needs credentials? | What it is |
|---|---|---|
| `data/` | no | Synthetic vignettes, generated in-repo by `build_dataset.py` |
| `data/medcalc/` | no | **Real clinical prose** — 600 items from MedCalc-Bench PMC case reports |
| `data/external/` | no | MedCalc-Bench source CSVs; open download, git-ignored (54 MB) |
| `data/umls/` | **to rebuild only** | Causal graph + cached UMLS responses. Running makes zero HTTP calls; rebuilding needs a free UTS key in `../.env` |

**MIMIC-IV / PhysioNet is not a dependency.** The proposal names MIMIC-IV in
§4.6, but no loader, data path or token check exists anywhere in `src/` — it is
an *unimplemented proposal element*, not a blocked one, and every Aim 1–4 stage
runs without it. The real-clinical-text arm of every benchmark is
`data/medcalc`, which is open access.

This distinction is worth keeping straight in the write-up. A synthetic note
corpus would not close the §4.6 gap: evaluating on generated notes measures the
generator, not the model. What MIMIC-IV Demo v2.2 (open access, no
credentialing) would add is *real structured `labevents`* — which is the only
thing that addresses the which-value-is-current problem below, and it carries
no notes at all.

## Known limitation to state in the write-up

**The gate's accuracy is partly circular and must never be quoted alone.** It
evaluates the same constraint and threshold the labels were generated from, and
on the MedCalc benchmark the same extractor that filters the dataset runs
inside the gate. The uncircular numbers are its **coverage** — 42.7% of real
renal notes, against 56.2% of the templated vignettes — and its accuracy on the
subset where it *declines*, both of which the table now reports.

## GPU

BioMistral-7B in bf16 needs ≈16 GB plus activations → a 24 GB card. The full
pipeline is a few hours on one 4090. **GPU 0 only** on this machine.
