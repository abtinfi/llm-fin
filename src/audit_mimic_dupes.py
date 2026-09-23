"""
How many INDEPENDENT prompts does a MIMIC arm really have?

The rendered note is minimal by design (age, sex, one lab value, the drug),
so two different real patients with the same age, sex and value receive the
byte-identical prompt. The splits are patient-disjoint, not prompt-disjoint.
The model decodes greedily, so identical prompts get identical answers: they
are one observation counted several times. This reports, per split and
family, items vs distinct prompts; prompts shared between splits (which is
train-to-test leakage for anything TRAINED, i.e. the constraint layer); and
prompts that carry both labels (unanswerable: the model must be wrong on one).

PRINTS COUNTS ONLY.

  python src/audit_mimic_dupes.py <data_dir>
"""

import collections
import json
import sys
from pathlib import Path


def main(d):
    d = Path(d)
    splits = ["train", "calib", "test", "heldout"]
    prompts = {}                     # split -> Counter(prompt)
    fam_of = collections.defaultdict(set)
    labels = collections.defaultdict(set)
    per = collections.defaultdict(lambda: [0, set()])
    for s in splits:
        prompts[s] = collections.Counter()
        for line in (d / f"counterfactual_{s}.jsonl").open():
            r = json.loads(line)
            p = r["prompt"]
            prompts[s][p] += 1
            fam_of[p].add(r["family"])
            labels[p].add(r["label"])
            key = (s, r["family"])
            per[key][0] += 1
            per[key][1].add(p)
    print(f"=== {d} ===")
    print("items vs distinct prompts:")
    for s in splits:
        n, u = sum(prompts[s].values()), len(prompts[s])
        print(f"  {s:8s} items={n:>8,}  distinct={u:>7,}  ratio={n / u:5.1f}x")
        for (ss, fam), (n2, ps) in sorted(per.items()):
            if ss == s:
                print(f"      {fam:20s} items={n2:>7,}  distinct={len(ps):>6,}")
    print("prompt overlap between splits (distinct prompts in both):")
    for a in splits:
        for b in splits:
            if a < b:
                ov = set(prompts[a]) & set(prompts[b])
                if ov:
                    items_b = sum(prompts[b][p] for p in ov)
                    print(f"  {a:8s} & {b:8s}: {len(ov):>6,} prompts "
                          f"({items_b:,} {b} items)")
    conf = {p for p, l in labels.items() if len(l) > 1}
    fams = collections.Counter(tuple(sorted(fam_of[p])) for p in conf)
    print(f"prompts carrying BOTH labels: {len(conf):,}; by families: "
          f"{dict(fams)}")
    for s in splits:
        k = sum(c for p, c in prompts[s].items() if p in conf)
        n = sum(prompts[s].values())
        print(f"  {s:8s}: {k:>7,} items ({k / n:.2%}) sit on such a prompt")


if __name__ == "__main__":
    main(sys.argv[1])
