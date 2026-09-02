"""
Baseline vs. post-fix comparison for the B1/B2/B3 re-run.

Reads the artifacts the original pipeline wrote into `results/` and the ones
`rerun_affected_stages.sh` wrote into `results/rerun_fixes/`, and prints the
quantities each defect could have moved, side by side.

WHY A SCRIPT AND NOT A DIFF
---------------------------
Two of the three fixes change the *shape* of the output, not just the numbers:
the steering sweep gained a negative arm (7 alphas -> 13), and the SAE
knock-out now reports which module it hooked. A textual diff of the JSON is
therefore unreadable, and the interesting comparison -- "did the null survive"
-- is a handful of scalars buried in it.

WHAT TO LOOK FOR, PER DEFECT
----------------------------
  B1  `s_causal` / `excess` per SAE feature. The baseline measured these with
      a dictionary fitted on h_20 applied to h_21. If the null was an artefact
      of that mismatch, the excess figures rise here. If it was real, they stay
      in the 0.00x range and the Aim 2 conclusion is now independently
      supported rather than merely repeated.
  B2  the sign of the best alpha, and whether the NEGATIVE arm of the sweep
      (which never existed before) shows anything. The baseline could only
      steer toward SAFE on a model that already answered SAFE everywhere.
  B3  the last layer's ACE row. Only `layer == n_layers - 1` was affected;
      every other row must reproduce to the digit, and the script checks that
      explicitly -- an unchanged row elsewhere is evidence the fix was surgical.

  python src/compare_rerun.py --out results/rerun_fixes/COMPARISON.md
"""

import argparse
import json
from pathlib import Path


def load(p):
    p = Path(p)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def pair_status(a, b, tol=1e-9):
    """How to read a baseline/rerun pair when one side is missing."""
    if a is None and b is None:
        return "both missing"
    if a is None:
        return "no baseline"
    if b is None:
        return "not re-run"
    return "identical" if abs(a - b) <= tol else f"{b - a:+.4f}"


# --------------------------------------------------------------- B1: SAE
def section_sae(base_p, new_p, L):
    L.append("## B1 — SAE knock-out (`sae.py::causal_knockout`)\n")
    L.append("The dictionary was fitted on `hidden_states[20]` but the hook "
             "fired on `layers[20]`, whose output is `hidden_states[21]`. The "
             "hook now fires on `layers[19]`. `S_semantic` is untouched by "
             "this defect and must reproduce exactly; `S_causal` is the "
             "quantity at risk.\n")
    b, n = load(base_p), load(new_p)
    if b is None or n is None:
        L.append(f"*Missing: baseline={'ok' if b else 'ABSENT'}, "
                 f"rerun={'ok' if n else 'ABSENT'} "
                 f"(`{base_p}` / `{new_p}`)*\n")
        return
    bf = {f["feature"]: f for f in b.get("features", [])}
    nf = {f["feature"]: f for f in n.get("features", [])}
    L.append("| feature | concept | S_sem base | S_sem rerun | S_causal base "
             "| S_causal rerun | excess base | excess rerun |")
    L.append("|---|---|---|---|---|---|---|---|")
    sem_moved = 0
    for k in sorted(bf, key=lambda k: -bf[k]["s_semantic"]):
        fb, fn = bf[k], nf.get(k)
        if fn is None:
            continue
        if abs(fb["s_semantic"] - fn["s_semantic"]) > 1e-9:
            sem_moved += 1
        eb = (fb.get("causal_detail") or {}).get("excess")
        en = (fn.get("causal_detail") or {}).get("excess")
        L.append(f"| #{k} | {fb.get('concept')} | {fmt(fb['s_semantic'],3)} | "
                 f"{fmt(fn['s_semantic'],3)} | {fmt(fb.get('s_causal'))} | "
                 f"{fmt(fn.get('s_causal'))} | {fmt(eb)} | {fmt(en)} |")
    L.append("")
    if sem_moved:
        L.append(f"⚠️ **`S_semantic` moved on {sem_moved} feature(s).** It must "
                 f"not: this defect never touched the semantic term. "
                 f"Investigate before reading the causal column.\n")
    else:
        L.append("`S_semantic` reproduced exactly on every feature, as it must "
                 "— the fix was confined to the causal stage.\n")
    eb = [(f.get("causal_detail") or {}).get("excess") for f in bf.values()]
    en = [(f.get("causal_detail") or {}).get("excess") for f in nf.values()]
    eb = [x for x in eb if x is not None]
    en = [x for x in en if x is not None]
    if eb and en:
        L.append(f"Largest |excess| over a matched random control: baseline "
                 f"**{max(abs(x) for x in eb):.4f}**, rerun "
                 f"**{max(abs(x) for x in en):.4f}** logits. A decision flip "
                 f"needs ~1–5 logits.\n")


# ---------------------------------------------------------- B2: steering
def section_steering(base_p, new_p, L):
    L.append("## B2 — Steering sign and layer alignment (`steering.py`)\n")
    L.append("The direction points toward UNSAFE and was applied with a "
             "negated coefficient over a non-negative sweep, so every steer "
             "pushed toward SAFE on a model already answering SAFE on all "
             "items. The sweep is now two-sided and the hook fires on the "
             "module that produced the representation the direction was "
             "fitted on.\n")
    b, n = load(base_p), load(new_p)
    if n is None:
        L.append(f"*Rerun artifact absent (`{new_p}`).*\n")
        return

    def cells(d):
        return [(int(l), r) for l, v in d["layers"].items()
                for r in v["rows"]]

    L.append("> **The two `alpha` columns do not mean the same thing.** The "
             "baseline recorded the alpha it was *asked* for while applying "
             "its negation, so a baseline row logged at `+4.00×` was "
             "physically a steer of `-4.00×` — toward SAFE. Every baseline "
             "alpha below should be read with its sign flipped. That is the "
             "defect, not a reporting choice, and it is why the baseline's "
             "'positive' arm is not comparable to the rerun's.\n")
    L.append("| run | cells swept | alphas (as recorded) | best cell | CC | "
             "random CC | gain |")
    L.append("|---|---|---|---|---|---|---|")
    for name, d in (("baseline", b), ("rerun", n)):
        if d is None:
            L.append(f"| {name} | — | — | *absent* | — | — | — |")
            continue
        cs = cells(d)
        bl, br = max(cs, key=lambda t: t[1]["cc"] - t[1]["cc_rand"])
        alphas = sorted({r["alpha_rel"] for _, r in cs})
        rng = f"{min(alphas):+.2f} … {max(alphas):+.2f}"
        L.append(f"| {name} | {len(cs)} | {rng} | L{bl} @ "
                 f"{br['alpha_rel']:+.2f}× | {br['cc']:.3f} | "
                 f"{br['cc_rand']:.3f} | "
                 f"{br['cc']-br['cc_rand']:+.3f} |")
    L.append("")

    cs = cells(n)
    pos = [(l, r) for l, r in cs if r["alpha_rel"] > 0]
    neg = [(l, r) for l, r in cs if r["alpha_rel"] < 0]
    L.append("**The arm that did not exist before.** Positive alpha steers "
             "toward UNSAFE; that arm is the one capable of moving a model "
             "stuck on SAFE.\n")
    L.append("| arm | best gain over matched random control | at |")
    L.append("|---|---|---|")
    for name, arm in (("toward UNSAFE (alpha > 0) — **never tested before**",
                       pos),
                      ("toward SAFE (alpha < 0) — the only arm the baseline "
                       "actually applied", neg)):
        if not arm:
            L.append(f"| {name} | — | — |")
            continue
        l, r = max(arm, key=lambda t: t[1]["cc"] - t[1]["cc_rand"])
        L.append(f"| {name} | {r['cc']-r['cc_rand']:+.3f} | "
                 f"L{l} @ {r['alpha_rel']:+.2f}× |")
    L.append("")
    accs = sorted({round(r["acc"], 3) for _, r in cs})
    L.append(f"Item accuracies observed across the rerun sweep: "
             f"{accs[:8]}{' …' if len(accs) > 8 else ''}. In the baseline "
             f"every cell sat at 0.500 (the model answering SAFE on "
             f"everything); accuracy moving off 0.500 is the direct signature "
             f"that steering now reaches the decision.\n")


# ---------------------------------------------------------- B3: patching
def section_patching(pairs, L):
    L.append("## B3 — Post-final-norm donor at the last layer "
             "(`patching.py`)\n")
    L.append("`hidden_states[n_layers]` is emitted after `model.norm`, so the "
             "deepest ACE row wrote a normed tensor into a pre-norm residual "
             "stream. `Patcher.run` now captures the last decoder layer's "
             "true output and substitutes it. **Only the final row was "
             "affected** — every other row is a regression check.\n")
    for label, base_p, new_p in pairs:
        L.append(f"### {label}\n")
        b, n = load(base_p), load(new_p)
        if b is None or n is None:
            L.append(f"*Missing: baseline={'ok' if b else 'ABSENT'}, "
                     f"rerun={'ok' if n else 'ABSENT'}.*\n")
            continue
        br = {r["layer"]: r for r in b["rows"]}
        nr = {r["layer"]: r for r in n["rows"]}
        last = max(br)
        moved = [l for l in sorted(set(br) & set(nr))
                 if l != last and abs(br[l]["ace"] - nr[l]["ace"]) > 1e-9]
        L.append(f"- pairs used: {b.get('pairs_used')} → "
                 f"{n.get('pairs_used')}")
        L.append(f"- identity control max |err|: "
                 f"{fmt(b.get('identity_max_err'))} → "
                 f"{fmt(n.get('identity_max_err'))} "
                 f"(must be ~0 in both; it is the hook-placement check)")
        if last in br and last in nr:
            L.append(f"- **last layer ({last}) ACE: "
                     f"{fmt(br[last]['ace'])} → {fmt(nr[last]['ace'])}**, "
                     f"excess {fmt(br[last]['excess'])} → "
                     f"{fmt(nr[last]['excess'])}")
        if moved:
            L.append(f"- ⚠️ **{len(moved)} non-final row(s) also moved** "
                     f"(layers {moved[:10]}). Expected zero — the fix should "
                     f"be confined to the last layer.")
        else:
            L.append("- ✅ every non-final row reproduced to the digit, so the "
                     "fix is confined to the row it was meant to correct.")
        bmax = max(abs(r["excess"]) for r in b["rows"])
        nmax = max(abs(r["excess"]) for r in n["rows"])
        L.append(f"- largest |excess| over control, any layer: "
                 f"{bmax:.4f} → {nmax:.4f} logits\n")


# ------------------------------------------------------ B3: sufficiency
def section_sufficiency(pairs, L):
    L.append("## B3 (cont.) — Sufficiency / injection sweeps\n")
    L.append("`sufficiency` shares `Patcher.run`, so it inherits the fix. Its "
             "default layer list skips the last layer, so these are expected "
             "to reproduce exactly; a change here means `--layers` reached the "
             "affected row.\n")
    for label, base_p, new_p in pairs:
        b, n = load(base_p), load(new_p)
        if b is None or n is None:
            L.append(f"- **{label}**: missing "
                     f"(baseline={'ok' if b else 'ABSENT'}, "
                     f"rerun={'ok' if n else 'ABSENT'})")
            continue
        bk = {(r["layer"], r["alpha"]): r for r in b["rows"]}
        nk = {(r["layer"], r["alpha"]): r for r in n["rows"]}
        moved = [k for k in sorted(set(bk) & set(nk))
                 if abs(bk[k]["ace"] - nk[k]["ace"]) > 1e-9]
        bmax = max(abs(r["excess"]) for r in b["rows"])
        nmax = max(abs(r["excess"]) for r in n["rows"])
        bmono = sum(1 for v in b.get("monotone_layers", {}).values() if v)
        nmono = sum(1 for v in n.get("monotone_layers", {}).values() if v)
        L.append(f"- **{label}**: largest |excess| {bmax:.4f} → {nmax:.4f}; "
                 f"monotone layers {bmono} → {nmono}; "
                 f"cells changed: {len(moved)}/{len(set(bk) & set(nk))}; "
                 f"alpha=0 control {fmt(b.get('alpha0_max_err'))} → "
                 f"{fmt(n.get('alpha0_max_err'))}")
    L.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results")
    ap.add_argument("--rerun", default="results/rerun_fixes")
    ap.add_argument("--out", default="results/rerun_fixes/COMPARISON.md")
    args = ap.parse_args()
    B, R = Path(args.baseline), Path(args.rerun)

    L = ["# B1 / B2 / B3 fixes — baseline vs. re-run\n",
         f"Baseline: `{B}` (2026-09-01 full run + 2026-09-02 join pass). "
         f"Re-run: `{R}`.\n",
         "Every number below comes from the JSON artifacts, not from a log. "
         "A defect that changed no conclusion is still worth recording as "
         "such, so unchanged rows are reported rather than omitted.\n"]

    section_sae(B / "sae" / "sae_topk_L20_fis.json",
                R / "sae_topk_L20_fis.json", L)
    section_steering(B / "steering_qt.json", R / "steering_qt.json", L)
    section_patching([
        ("MedCalc test (renal / creatinine)",
         B / "patching_medcalc_test.json", R / "patching_medcalc_test.json"),
        ("MedCalc held-out (QT / ondansetron)",
         B / "patching_medcalc_heldout.json",
         R / "patching_medcalc_heldout.json"),
    ], L)
    section_sufficiency([
        ("MedCalc test", B / "sufficiency_medcalc_test.json",
         R / "sufficiency_medcalc_test.json"),
        ("MedCalc held-out", B / "sufficiency_medcalc_heldout.json",
         R / "sufficiency_medcalc_heldout.json"),
    ], L)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
