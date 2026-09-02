"""
Baseline (regex) vs UMLS-grounded Aim 1, feature by feature.

The two runs have DIFFERENT dictionaries -- the vocabulary reaches the SAE
through collect -- so feature ids are not comparable across runs. What is
comparable is the distribution of S_semantic, which concepts win, and whether
the causal null survives. This script reports exactly those and refuses to
pair features by id.
"""
import json, sys
from pathlib import Path


def load(p):
    d = json.loads(Path(p).read_text())
    return d


def summarise(tag, d):
    feats = d["features"]
    sem = [f["s_semantic"] for f in feats]
    fis = [f.get("fis", 0.0) for f in feats]
    causal = [f["s_causal"] for f in feats if f.get("s_causal") is not None]
    by_concept = {}
    for f in feats:
        by_concept.setdefault(f["concept"], []).append(f["s_semantic"])
    print(f"\n### {tag}  (source={d.get('concept_source','regex')})")
    print(f"  kind={d['kind']}  FVU={d['fvu']:.3f}  L0={d['l0']:.1f}  "
          f"dead={d['dead']}/{d['d_hidden']}")
    print(f"  S_semantic: max={max(sem):.3f}  mean={sum(sem)/len(sem):.3f}  "
          f"n={len(sem)}")
    print(f"  FIS:        max={max(fis):.3f}  mean={sum(fis)/len(fis):.3f}")
    if causal:
        print(f"  S_causal:   max={max(causal):.4f}  min={min(causal):.4f}  "
              f"(a decision flip needs ~1-5 logits)")
    print(f"  top feature: #{feats[0]['feature']} "
          f"'{feats[0]['concept']}' S_sem={feats[0]['s_semantic']:.3f}")
    print("  concepts represented in the top-N: " +
          ", ".join(f"{c}({len(v)})" for c, v in
                    sorted(by_concept.items(), key=lambda kv: -len(kv[1]))))
    return d


def umls_only_hits(slugs):
    """
    What UMLS atoms ALONE match on the real prompts, per concept.

    Term counts are NOT hit counts, and using them to judge grounding is the
    mistake this function exists to prevent: `qt_interval` carries 13 UMLS
    atoms and matches ZERO of them on real notes, because notes write "QTc"
    and UMLS stores the normalised form. Only the empirical column decides
    whether a concept may be called UMLS-matched.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataclasses import asdict
    from umls_grounding import (CausalKnowledgeGraph, ConceptGrounding,
                                NON_UMLS_PATTERNS)
    g = CausalKnowledgeGraph.load(Path("data/umls/causal_graph.json"))
    recs = []
    for f in ("data/synthetic_control/counterfactual_test.jsonl",
              "data/medcalc/counterfactual_test.jsonl"):
        if Path(f).is_file():
            recs += [json.loads(l) for l in Path(f).open()]
    out = {}
    for slug in slugs:
        gr = g.groundings.get(slug)
        full = uo = 0
        if gr:
            full = sum(len(gr.matcher().findall(r["prompt"])) for r in recs)
            if gr.cui and slug not in NON_UMLS_PATTERNS:
                bare = ConceptGrounding(**{**asdict(gr), "lexical_variants": []})
                uo = sum(len(bare.matcher().findall(r["prompt"])) for r in recs)
        out[slug] = (full, uo)
    return out


def provenance_table(d):
    pr = d.get("concept_provenance") or {}
    if not pr:
        print("\n  (no provenance recorded -- regex run)")
        return
    hits = umls_only_hits(list(pr))
    print(f"\n  {'concept':16s} {'CUI':11s} {'atoms':>6s} {'cur':>4s} "
          f"{'hits':>6s} {'umls-only':>10s}  grounding")
    for slug, v in pr.items():
        atoms, cur = v["n_umls_terms"], v["n_curated_terms"]
        full, uo = hits.get(slug, (0, 0))
        if atoms == 0:
            note = "LEXICAL FALLBACK -- no CUI, not UMLS-grounded at all"
        elif full == 0:
            note = "UNMEASURABLE -- concept never appears in these prompts"
        elif uo == 0:
            note = "CUI-anchored ONLY -- zero UMLS atoms match real text"
        elif uo < full / 2:
            note = f"mostly curated ({uo}/{full} from UMLS atoms)"
        else:
            note = f"UMLS-matched ({uo}/{full} from UMLS atoms)"
        print(f"  {slug:16s} {v['cui'] or '-':11s} {atoms:6d} {cur:4d} "
              f"{full:6d} {uo:10d}  {note}")


if __name__ == "__main__":
    base, new = sys.argv[1], sys.argv[2]
    b = summarise("BASELINE (hand-written regexes)", load(base))
    n = summarise("UMLS-GROUNDED", load(new))
    provenance_table(n)
    print("\n### What moved")
    bs = max(f["s_semantic"] for f in b["features"])
    ns = max(f["s_semantic"] for f in n["features"])
    print(f"  best S_semantic {bs:.3f} -> {ns:.3f}  ({ns-bs:+.3f})")
    bf = max(f.get("fis", 0) for f in b["features"])
    nf = max(f.get("fis", 0) for f in n["features"])
    print(f"  best FIS        {bf:.3f} -> {nf:.3f}  ({nf-bf:+.3f})")
    print("\n  Feature ids are NOT comparable across these two runs: the "
          "vocabulary\n  changes which tokens are collected, so the "
          "dictionaries are different.")
