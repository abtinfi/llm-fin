"""
Independent audit: is every MIMIC item's label what its OWN PROMPT implies?

The label must follow from what the model is shown, not from the stored
measurement. This re-derives it from the printed text alone:

  * the lab value is parsed out of the vignette, at the precision it was
    PRINTED (potassium and INR at 1 decimal, creatinine at 2);
  * for the renal families eGFR is recomputed from that printed creatinine
    and the printed age and sex with the project's own renal.ckd_epi;
  * the family's threshold and operator are applied;
  * the result is compared with the stored label.

It also checks, on the same pass, that no prompt text appears with two
different labels anywhere in a dataset (the ground truth must be a function
of the prompt), that the printed age/sex match `facts`, and that each causal
pair's two arms straddle the threshold as printed.

PRINTS COUNTS ONLY. Row-level v3.1 content is credentialed and must not reach
an online service, including an assistant reading this output.

  python src/audit_mimic_labels.py <data_dir> [<data_dir> ...]
"""

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/asosoft/abtin/paper/csai/src")
from renal import ckd_epi                          # noqa: E402

RULES = {
    "metformin_egfr30": ("creat", 30.0, "<"),
    "metformin_egfr45": ("creat", 45.0, "<"),
    "spironolactone_k5_5": ("k", 5.5, ">"),
    "warfarin_inr4": ("inr", 4.0, ">"),
}
PAT = {
    "creat": re.compile(r"serum creatinine of (\d+(?:\.\d+)?) mg/dL"),
    "k": re.compile(r"Serum potassium is (\d+(?:\.\d+)?) mmol/L"),
    "inr": re.compile(r"Today's INR is (\d+(?:\.\d+)?)"),
}
WHO = re.compile(r"A (\d+)-year-old (woman|man)")


def label_from_prompt(fam, vig):
    kind, thr, op = RULES[fam]
    # The rendered sentence is the LAST line of the vignette; the note arm
    # prepends an excerpt that may itself contain numbers.
    tail = vig.strip().split("\n")[-1]
    m = PAT[kind].search(tail)
    w = WHO.search(tail)
    if not m or not w:
        return None, None, None, None
    shown = float(m.group(1))
    age, sex = int(w.group(1)), ("female" if w.group(2) == "woman" else "male")
    q = ckd_epi(shown, age, sex) if kind == "creat" else shown
    unsafe = q < thr if op == "<" else q > thr
    return ("UNSAFE" if unsafe else "SAFE"), shown, age, sex


def audit(d):
    d = Path(d)
    print(f"\n=== {d} ===")
    by_prompt = collections.defaultdict(set)
    for split in ["train", "calib", "test", "heldout"]:
        p = d / f"counterfactual_{split}.jsonl"
        if not p.exists():
            continue
        c = collections.Counter()
        pairs = collections.defaultdict(list)
        for line in p.open():
            r = json.loads(line)
            fam = r["family"]
            kind = "control" if r.get("is_control") else "causal"
            lab, shown, age, sex = label_from_prompt(fam, r["vignette"])
            c[(fam, kind, "n")] += 1
            if lab is None:
                c[(fam, kind, "unparsed")] += 1
                continue
            if lab != r["label"]:
                c[(fam, kind, "LABEL_DISAGREES_WITH_PROMPT")] += 1
            f = r.get("facts", {})
            if f.get("age") != age or str(f.get("sex", "")).lower() != sex:
                c[(fam, kind, "age_sex_mismatch")] += 1
            by_prompt[r["prompt"]].add(r["label"])
            pairs[r["pair_id"]].append(lab)
        bad_pairs = collections.Counter()
        for pid, labs in pairs.items():
            fam = pid.split("__")[0]
            if pid.endswith("__ctrl") or pid.endswith("__ctrl__note"):
                if len(set(labs)) != 1:
                    bad_pairs[(fam, "control_pair_straddles_as_printed")] += 1
            elif set(labs) != {"SAFE", "UNSAFE"}:
                bad_pairs[(fam, "causal_pair_same_side_as_printed")] += 1
        print(f"  [{split}]")
        for fam in RULES:
            for kind in ["causal", "control"]:
                n = c[(fam, kind, "n")]
                if not n:
                    continue
                bad = c[(fam, kind, "LABEL_DISAGREES_WITH_PROMPT")]
                extra = {k[2]: v for k, v in c.items()
                         if k[0] == fam and k[1] == kind
                         and k[2] not in ("n", "LABEL_DISAGREES_WITH_PROMPT")}
                flag = "  <-- WRONG LABELS" if bad else ""
                print(f"    {fam:20s} {kind:7s} items={n:>7,}  "
                      f"label!=prompt: {bad:>6,} ({bad / n:6.2%}){flag}"
                      + (f"  {extra}" if extra else ""))
        for (fam, what), v in sorted(bad_pairs.items()):
            print(f"    {fam:20s} {what}: {v:,} pairs")
    conflicting = sum(1 for labs in by_prompt.values() if len(labs) > 1)
    print(f"  identical prompt text carrying BOTH labels: {conflicting:,} "
          f"of {len(by_prompt):,} distinct prompts")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        audit(d)
