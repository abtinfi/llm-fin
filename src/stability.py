"""
Feature stability across seeds and layers -- proposal section 4.4's
"evaluate stability across layers, prompts, and model seeds", which the
implementation status file lists as NOT IMPLEMENTED.

Two dictionaries trained on the same activations with different seeds are not
comparable feature-by-feature: index 413 in one has nothing to do with index
413 in the other. The comparison has to be made between the SUBSPACES they
span, and between the concepts they select.

Three measures, each answering a different question:

  dictionary agreement   for every decoder row in A, the largest |cosine| to
                         any row in B, averaged. "Does B contain, somewhere,
                         a direction that means what this one means?" A high
                         value with a low Jaccard below would mean the same
                         directions are found but ranked differently.

  concept Jaccard        overlap of the sets of CONCEPTS the top-N features
                         select. Index-free by construction, and it is the
                         quantity every claim in the report actually rests
                         on -- "the dictionary has a creatinine feature" is a
                         statement about this set.

  per-concept counts     how many top-N features each concept gets in each
                         run. The attribution layer weights by these, so a
                         concept that appears in one seed and not another is
                         a direct threat to section 4.2's conclusions.

Across LAYERS the same three are computed, with a caveat that must be read
with them: a low cross-layer agreement is the expected result, not a failure.
Layer 16 and layer 24 are different representations and there is no reason a
feature should survive between them. The number is reported so that "we chose
layer 20" stops being an unexamined choice.
"""

import argparse
import glob
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np


def load_dict(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    W = np.asarray(z["W_dec"], dtype=np.float32)      # [F, d]
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
    return W


def load_fis(p):
    d = json.loads(Path(p).read_text())
    return d


def agreement(A, B, chunk=2048):
    """Mean over rows of A of the largest |cosine| to any row of B."""
    best = np.empty(A.shape[0], dtype=np.float32)
    for i in range(0, A.shape[0], chunk):
        sims = np.abs(A[i:i + chunk] @ B.T)
        best[i:i + chunk] = sims.max(axis=1)
    return float(best.mean()), float(np.median(best))


def alive(W, fis):
    """Rows worth comparing: dead features are noise, not disagreement."""
    live = {f["feature"] for f in fis["features"]}
    idx = sorted(live)
    return W[idx] if idx else W


def concept_sets(fis, top):
    feats = fis["features"][:top]
    return {f["concept"] for f in feats if f.get("concept")}


def concept_counts(fis, top):
    out = {}
    for f in fis["features"][:top]:
        c = f.get("concept")
        if c:
            out[c] = out.get(c, 0) + 1
    return out


def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", nargs="+", required=True,
                    help="npz dictionaries to compare; each needs a sibling "
                         "*_fis.json")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="display name per --sae entry; defaults to the "
                         "filename")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default="results/stability.md")
    ap.add_argument("--json_out", default="results/stability.json")
    args = ap.parse_args()

    labels = args.labels or [Path(p).stem for p in args.sae]
    if len(labels) != len(args.sae):
        raise SystemExit("--labels must have one entry per --sae")

    runs = {}
    for lab, p in zip(labels, args.sae):
        fis_p = str(p).replace(".npz", "_fis.json")
        if not Path(fis_p).is_file():
            raise SystemExit(f"missing {fis_p}; run `sae.py score` first")
        fis = load_fis(fis_p)
        runs[lab] = {
            "npz": str(p), "fis": fis_p,
            "W": alive(load_dict(p), fis),
            "fis_obj": fis,
            "layer": _layer_of(p),
            "seed": fis.get("seed"),
            "concepts": concept_sets(fis, args.top),
            "counts": concept_counts(fis, args.top),
            "fvu": fis.get("fvu"), "l0": fis.get("l0"),
            "dead": fis.get("dead"), "d_hidden": fis.get("d_hidden"),
        }

    pairs = []
    for a, b in combinations(labels, 2):
        m_ab, md_ab = agreement(runs[a]["W"], runs[b]["W"])
        m_ba, md_ba = agreement(runs[b]["W"], runs[a]["W"])
        pairs.append({
            "a": a, "b": b,
            "same_layer": runs[a]["layer"] == runs[b]["layer"],
            "mean_max_cosine_a_to_b": m_ab,
            "mean_max_cosine_b_to_a": m_ba,
            "median_max_cosine_a_to_b": md_ab,
            "concept_jaccard": jaccard(runs[a]["concepts"],
                                       runs[b]["concepts"]),
            "concepts_only_in_a": sorted(runs[a]["concepts"]
                                         - runs[b]["concepts"]),
            "concepts_only_in_b": sorted(runs[b]["concepts"]
                                         - runs[a]["concepts"]),
        })

    out = {"top": args.top,
           "runs": {k: {kk: vv for kk, vv in v.items()
                        if kk not in ("W", "fis_obj", "concepts")}
                    for k, v in runs.items()},
           "pairs": pairs}
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(out, indent=2, default=list))

    L = ["# Feature stability across seeds and layers", "",
         "*Generated by `src/stability.py`. Do not edit the numbers by hand.*",
         "",
         "Proposal section 4.4 asks for stability \"across layers, prompts, and "
         "model seeds\". Two dictionaries trained with different seeds cannot "
         "be compared feature-by-feature -- index 413 in one has nothing to do "
         "with index 413 in the other -- so the comparison is between the "
         "subspaces they span and the concepts they select.",
         "",
         "## Runs compared", "",
         "| run | layer | seed | FVU | L0 | dead | live features |",
         "|---|---|---|---|---|---|---|"]
    for lab, r in runs.items():
        L.append(f"| `{lab}` | {r['layer']} | {r['seed']} | "
                 f"{_f(r['fvu'])} | {_f(r['l0'], 1)} | {r['dead']} | "
                 f"{r['W'].shape[0]} |")
    L += ["", "## Pairwise agreement", "",
          "`mean max |cos|` is, for every live decoder row in A, the largest "
          "absolute cosine to any row in B, averaged. It asks whether B "
          "contains *somewhere* a direction meaning what A's row means, "
          "independent of index or rank.",
          "",
          "`concept Jaccard` is the overlap of the sets of concepts the top-"
          f"{args.top} features select. That is the quantity every claim in "
          "the reports actually rests on.",
          "",
          "| A | B | same layer | mean max &#124;cos&#124; A→B | B→A | "
          "concept Jaccard | only in A | only in B |",
          "|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        L.append(f"| `{p['a']}` | `{p['b']}` | "
                 f"{'yes' if p['same_layer'] else 'no'} | "
                 f"{_f(p['mean_max_cosine_a_to_b'])} | "
                 f"{_f(p['mean_max_cosine_b_to_a'])} | "
                 f"{_f(p['concept_jaccard'])} | "
                 f"{', '.join(p['concepts_only_in_a']) or '—'} | "
                 f"{', '.join(p['concepts_only_in_b']) or '—'} |")

    L += ["", "## Features per concept", "",
          "The attribution layer of section 4.2 weights by these counts, so a "
          "concept present in one run and absent in another is a direct threat "
          "to its conclusions, not a curiosity.",
          "",
          "| concept | " + " | ".join(f"`{l}`" for l in labels) + " |",
          "|---" * (len(labels) + 1) + "|"]
    all_c = sorted({c for r in runs.values() for c in r["counts"]})
    for c in all_c:
        L.append(f"| {c} | " +
                 " | ".join(str(runs[l]["counts"].get(c, 0)) for l in labels)
                 + " |")

    cross = [p for p in pairs if not p["same_layer"]]
    if cross:
        L += ["", "## Reading the cross-layer rows", "",
              "A low agreement between different layers is the expected "
              "result, not a failure: layer 16 and layer 24 are different "
              "representations and no feature is obliged to survive between "
              "them. The number is here so that the choice of layer 20 stops "
              "being an unexamined default.", ""]
    L += ["", "## Reproduce", "",
          "```bash",
          "for S in 0 1 2; do",
          "  python src/sae.py train --acts results/sae/acts_medcalc_train_L20.npz \\",
          "      --kind topk --expansion 4 --k 32 --epochs 30 --seed $S \\",
          "      --out results/sae/sae_topk_L20_s$S.npz",
          "  python src/sae.py score --sae results/sae/sae_topk_L20_s$S.npz \\",
          "      --top 25 --causal_items -1 --causal_features 0",
          "done",
          "python src/stability.py --sae results/sae/sae_topk_L20_s*.npz",
          "```", ""]
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out} and {args.json_out}")


def _layer_of(p):
    m = re.search(r"_L(\d+)", str(p))
    return int(m.group(1)) if m else None


def _f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


if __name__ == "__main__":
    main()
