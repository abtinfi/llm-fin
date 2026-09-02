"""
Builds the synthetic counterfactual clinical dataset.

Output: data/synthetic_control/counterfactual_{split}.jsonl

Each record is one arm of a counterfactual pair:
    pair_id      -- identical for the two arms
    arm          -- "unsafe" | "safe"
    label        -- "UNSAFE" | "SAFE"  (deterministic ground truth)
    facts        -- structured values, used ONLY for auditing and for scoring,
                    never given to the model or to the symbolic extractor
Splits:
    calib  -- 2 templates per family, used to tune the UQ abstention threshold
    test   -- 8 templates per family, used for the reported table
    heldout-- the two unseen rule families
"""

import argparse
import json
import random
from pathlib import Path

from rules import get_families, RULE_FAMILIES, AGES, SEXES

INSTRUCTION = (
    "You are reviewing a proposed prescription for safety.\n"
    "Answer with exactly one word on the first line: SAFE or UNSAFE.\n"
    "Then give one short sentence of justification.\n\n"
    "Case:\n{vignette}\n\n"
    "Is the proposed prescription safe for this patient?"
)


def build_records(family, n_templates, template_offset, rng, hard=True):
    from hardening import render
    records = []
    for i in range(n_templates):
        idx = template_offset + i
        implicit = hard and (idx % 2 == 1)   # alternate explicit / implicit
        # Family-specific pools where the default would produce an impossible
        # patient (see RuleFamily.ages / .sexes). The default older-adult pool
        # gave the pregnancy family eight pregnant patients aged 67-84.
        age_pool = family.ages or AGES
        sex_pool = family.sexes or SEXES
        age = age_pool[idx % len(age_pool)]
        sex = sex_pool[idx % len(sex_pool)]
        pair_id = f"{family.name}__{idx:02d}"

        # BUG FIX (see HANDOFF.md section 5, bug 5): both arms must draw the
        # SAME distractor labs. `render` consumes `rng`, so passing the shared
        # generator to each arm in turn advanced its state and gave the two
        # arms different distractors -- 64/64 test pairs differed in CRP,
        # sodium, platelets and haemoglobin as well as in the causal value.
        # That makes them not minimal pairs: a model could flip its answer
        # because the irrelevant labs changed, and Causal Consistency could
        # not tell that apart from responding to the causal factor. Seeding a
        # fresh generator per arm from one per-pair seed keeps distractors
        # identical within a pair and still varied across pairs.
        pair_seed = rng.randrange(2 ** 32)

        for arm, values, label in (
            ("unsafe", family.unsafe_values, "UNSAFE"),
            ("safe", family.safe_values, "SAFE"),
        ):
            value = values[idx % len(values)]
            ctx = {"age": age, "sex": sex, "value": value}
            arm_rng = random.Random(pair_seed)
            if hard:
                vignette, skeleton = render(family, ctx, implicit, arm_rng,
                                            unsafe=(arm == "unsafe"))
            else:
                vignette = family.vignette(ctx)
                skeleton = family.vignette({**ctx, "value": "<FACTOR>"})
            records.append({
                "id": f"{pair_id}__{arm}",
                "pair_id": pair_id,
                "family": family.name,
                "held_out": family.held_out,
                "arm": arm,
                "drug": family.drug,
                "factor": family.factor,
                "factor_value": value,
                "label": label,
                "vignette": vignette,
                "presentation": "implicit" if implicit else "explicit",
                # the vignette with the causal factor blanked out; both arms of
                # a pair must agree on this exactly (see validate())
                "skeleton": skeleton,
                "prompt": INSTRUCTION.format(vignette=vignette),
                "facts": {"age": age, "sex": sex, family.factor: value},
            })
    return records


def validate(records, split_name):
    """
    Integrity checks that run at build time and abort the build on failure.

    These exist because the previous version of this file shipped a dataset in
    which all 64 test pairs differed in their distractor labs as well as in the
    causal value. The fix was made in `build_records`, but nothing checked the
    output, so the broken data stayed on disk and every reported number was
    computed from it. A fix without a check is not a fix.

    Checked:
      1. Every pair has exactly two arms, one SAFE and one UNSAFE.
      2. The two arms are MINIMAL: they differ in exactly one contiguous run of
         tokens. Anything else means the pair is not a counterfactual.
      3. Where the symbolic gate can read the vignette at all, its decision
         agrees with the label. A disagreement means the text contradicts its
         own ground truth.
    """
    from components import SymbolicGate
    from rules import RULE_FAMILIES as _FAMS

    pairs = {}
    for r in records:
        pairs.setdefault(r["pair_id"], {})[r["arm"]] = r

    n_bad_arms = n_nonminimal = n_gate_conflict = 0
    gate = SymbolicGate(_FAMS)
    for pid, arms in pairs.items():
        if set(arms) != {"safe", "unsafe"} or \
                {arms["safe"]["label"], arms["unsafe"]["label"]} != {"SAFE", "UNSAFE"}:
            n_bad_arms += 1
            continue
        # Minimal-pair test. Comparing the two vignettes word by word cannot
        # do this on its own: where the causal factor is categorical the two
        # renderings share words ("a urine hCG ... positive" vs "a urine hCG
        # ... negative"), which looks like several separate edits while being
        # one substitution of the factor. So the comparison is made on the
        # SKELETON -- the vignette with the factor text blanked out -- which is
        # exact for every family, categorical or numeric.
        same_skeleton = arms["unsafe"]["skeleton"] == arms["safe"]["skeleton"]
        distinct_text = arms["unsafe"]["vignette"] != arms["safe"]["vignette"]
        if not (same_skeleton and distinct_text):
            n_nonminimal += 1
            if n_nonminimal <= 3:
                why = ("arms are textually IDENTICAL" if not distinct_text
                       else "skeletons differ outside the causal factor")
                print(f"    NOT MINIMAL {pid}: {why}")
                print(f"      unsafe: {arms['unsafe']['vignette']}")
                print(f"      safe  : {arms['safe']['vignette']}")
        for arm in arms.values():
            g = gate(arm["vignette"])
            if g.fired and ("UNSAFE" if g.violated else "SAFE") != arm["label"]:
                n_gate_conflict += 1
                if n_gate_conflict <= 2:
                    print(f"    GATE CONFLICT {arm['id']}: gate says "
                          f"{'UNSAFE' if g.violated else 'SAFE'}, label "
                          f"{arm['label']}\n      {arm['vignette']}")

    print(f"  [{split_name}] integrity: {len(pairs)} pairs, "
          f"{n_bad_arms} malformed, {n_nonminimal} non-minimal, "
          f"{n_gate_conflict} gate/label conflicts")
    if n_bad_arms or n_nonminimal or n_gate_conflict:
        raise SystemExit(f"integrity check FAILED on split {split_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_control")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_hardening", action="store_true",
                    help="reproduce the original easy, fully explicit vignettes")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from hardening import apply_hardening
    apply_hardening(RULE_FAMILIES, near_threshold=not args.no_hardening)

    main_families = get_families(include_held_out=False)
    held = [f for f in RULE_FAMILIES if f.held_out]

    calib, test, heldout = [], [], []
    for fam in main_families:
        calib += build_records(fam, 2, 0, rng, hard=not args.no_hardening)
        test += build_records(fam, 8, 2, rng, hard=not args.no_hardening)
    for fam in held:
        heldout += build_records(fam, 8, 0, rng, hard=not args.no_hardening)

    for name, recs in (("calib", calib), ("test", test), ("heldout", heldout)):
        validate(recs, name)
        p = out / f"counterfactual_{name}.jsonl"
        with p.open("w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        n_pairs = len({r["pair_id"] for r in recs})
        print(f"{p}: {len(recs)} records / {n_pairs} pairs / "
              f"{len({r['family'] for r in recs})} families")

    # guideline corpus for RAG (all families, incl. held out -- retrieval is
    # allowed to see guidelines, that is the point of RAG)
    corpus = []
    for fam in RULE_FAMILIES:
        corpus.append({"id": f"guideline::{fam.name}", "text": fam.guideline})
    # distractors: real but irrelevant guidance, so retrieval is non-trivial
    distractors = [
        "Vitamin D supplementation is recommended for adults with limited sun exposure.",
        "Blood pressure should be measured in both arms at the first assessment.",
        "Annual influenza vaccination is recommended for adults over 65 years.",
        "Smoking cessation support should be offered at every clinical contact.",
        "Statins are first line for secondary prevention of cardiovascular disease.",
        "Metformin is the preferred first line oral agent in type 2 diabetes.",
        "Paracetamol is first line for mild to moderate musculoskeletal pain.",
        "Atrial fibrillation stroke risk should be assessed with the CHA2DS2-VASc score.",
        "Serum creatinine should be rechecked one to two weeks after starting an ACE inhibitor.",
        "Inhaler technique should be reviewed before escalating asthma therapy.",
        "Bone protection should be considered for patients on long term oral corticosteroids.",
        "Proton pump inhibitors should be reviewed for deprescribing after eight weeks.",
    ]
    for i, d in enumerate(distractors):
        corpus.append({"id": f"distractor::{i}", "text": d})

    with (out / "rag_corpus.jsonl").open("w") as f:
        for c in corpus:
            f.write(json.dumps(c) + "\n")
    print(f"{out/'rag_corpus.jsonl'}: {len(corpus)} passages "
          f"({len(RULE_FAMILIES)} guidelines + {len(distractors)} distractors)")


if __name__ == "__main__":
    main()
