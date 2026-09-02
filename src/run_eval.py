"""
Runs the ablation matrix from section 4.6 of the proposal:

  (1) Base LLM
  (2) + RAG
  (3) + Symbolic Gate            (NS-AI)
  (4) + UQ Engine                (NS-AI + UQ)

Each variant adds exactly one component. Prompts, decoding, seeds and item
order are identical across variants so that paired tests (McNemar) are valid.

Usage
  python src/run_eval.py --backend mock --seeds 0 1 2 3 4
  python src/run_eval.py --backend hf --model_id BioMistral/BioMistral-7B --seeds 0 1 2
"""

import argparse
import json
from pathlib import Path

from components import SymbolicGate, TfidfRetriever, build_rag_prompt
from metrics import calibrate_threshold, score
from model import load_model, parse_answer
from rules import RULE_FAMILIES

# The cumulative ladder of section 4.6 of the proposal, PLUS the isolated
# variants and the constraint layer.
#
# Why the isolated ones exist: a purely cumulative ladder cannot attribute an
# effect to a component. On real notes RAG makes Causal Consistency WORSE
# (0.023 -> 0.000), so the symbolic gate's measured contribution is the
# distance from a damaged intermediate state, not from the baseline. Rows
# `sym` and `uq` add each component to the BASE model directly, which is what
# "the effect of adding this contribution" actually means.
#
#   name        RAG   gate   UQ    constraint layer
#   base         -     -      -     -
#   rag          Y     -      -     -
#   sym          -     Y      -     -      <- isolated
#   uq           -     -      Y     -      <- isolated
#   cl           -     -      -     Y      <- isolated (Aim 3)
#   nsai         Y     Y      -     -
#   nsai_uq      Y     Y      Y     -
#   nsai_uq_cl   Y     Y      Y     Y      <- everything
VARIANT_SPEC = {
    "base":       dict(rag=False, gate=False, uq=False, cl=False),
    "rag":        dict(rag=True,  gate=False, uq=False, cl=False),
    "sym":        dict(rag=False, gate=True,  uq=False, cl=False),
    "uq":         dict(rag=False, gate=False, uq=True,  cl=False),
    "cl":         dict(rag=False, gate=False, uq=False, cl=True),
    "nsai":       dict(rag=True,  gate=True,  uq=False, cl=False),
    "nsai_uq":    dict(rag=True,  gate=True,  uq=True,  cl=False),
    "nsai_uq_cl": dict(rag=True,  gate=True,  uq=True,  cl=True),
}
VARIANTS = list(VARIANT_SPEC)

# Which quantity the UQ engine defers on, mapped to a "higher = more uncertain"
# convention so calibrate_threshold() works unchanged for all of them.
#
# `entropy` is the proposal's Eq. (2) and remains the default, so every number
# already reported is reproduced exactly by the default invocation. The
# alternatives exist because Eq. (2) spreads entropy over the whole ~32k
# vocabulary, most of which is irrelevant to a binary SAFE/UNSAFE decision --
# see results/p1_uq_signals.md and results/bigbench_uq.md.
#
# NOTE for K=2: the entropy of the renormalised distribution over the two
# answer tokens is a strictly decreasing function of |logit_margin|, so on this
# benchmark `logit_margin` IS the decision-restricted entropy. They are
# rank-equivalent and therefore have identical AUROC.
def _decision_entropy(margin):
    """
    Entropy in nats of the decision distribution, restricted to the two answer
    tokens -- the respecified uncertainty term of proposal 4.7.

    Eq. (2) as written averages entropy over the whole ~32k vocabulary, and at
    n=6,456 that measures AUROC 0.525 [0.511, 0.539] -- chance
    (`results/bigbench_uq.md`). Restricting the SAME entropy to the tokens the
    decision is actually made over gives 0.687 [0.675, 0.700], a paired
    +0.163, p < 0.0001 Holm-corrected, replicated independently on MedMCQA,
    MedQA and PubMedQA. What matters is the restriction, not the functional
    form: `max_entropy`, a whole-vocabulary control, does not help.

    With K=2 answer tokens, softmax over the pair gives p = sigmoid(margin),
    so H = -(p ln p + (1-p) ln(1-p)), maximal at ln 2 = 0.6931 nats when the
    two logits are tied. This is a strictly decreasing function of |margin|,
    so `decision_entropy` and `logit_margin` are RANK-EQUIVALENT here and have
    identical AUROC and identical deferral behaviour. Both are provided: the
    entropy is the quantity 4.7 should name, the margin is what the n=6,456
    external run was reported under, and keeping both makes that equivalence
    checkable rather than asserted.
    """
    import math
    if margin is None:
        return float("nan")
    p = 1.0 / (1.0 + math.exp(-abs(float(margin))))
    q = 1.0 - p
    if p <= 0.0 or q <= 0.0:
        return 0.0
    return -(p * math.log(p) + q * math.log(q))


# Which quantity the UQ engine defers on, mapped to a "higher = more uncertain"
# convention so calibrate_threshold() works unchanged for all of them.
UQ_SIGNALS = {
    # Default since 2026-09-01: the decision-restricted form of Eq. (2).
    "decision_entropy": ("logit_margin", _decision_entropy),
    # The proposal's Eq. (2) exactly as written. Retained, not deleted: it is
    # what every pre-2026-09-01 number was produced under, and passing
    # `--uq_signal entropy` reproduces them.
    "entropy":        ("entropy",        lambda v: v),
    "max_entropy":    ("max_entropy",    lambda v: v),
    "logit_margin":   ("logit_margin",   lambda v: -abs(v)),
    "answer_logprob": ("answer_logprob", lambda v: -v),
}


def uncertainty(gen, signal):
    """
    Uncertainty of one generation under `signal`.

    A missing value (the generation never emitted an answer token, so there is
    no decision step to read a margin off) is treated as maximally uncertain
    rather than silently dropped or imputed. That keeps the UQ engine aligned
    with the gate's rule in HANDOFF.md section 7: when a component cannot
    establish the fact it needs, it must not guess.
    """
    field, to_unc = UQ_SIGNALS[signal]
    v = getattr(gen, field, None)
    return float("inf") if v is None else to_unc(v)


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open()]


def generate_for_split(lm, records, use_rag, retriever, batch_size):
    prompts, meta = [], []
    for r in records:
        if use_rag:
            p, docs = build_rag_prompt(r["vignette"], retriever)
            meta.append([d["id"] for d in docs])
        else:
            p, = (r["prompt"],)
            meta.append([])
        prompts.append(p)
    gens = lm.generate(prompts, batch_size=batch_size)
    return gens, meta


def apply_gate(gate, records, preds):
    """
    Symbolic gate overrides the neural decision when a hard constraint fires.
    Gate abstains (returns the neural answer unchanged) when it cannot extract
    the relevant fact -- it never guesses.
    """
    out, fired = [], []
    for r, p in zip(records, preds):
        g = gate(r["vignette"])
        if g.fired:
            out.append("UNSAFE" if g.violated else "SAFE")
            fired.append(True)
        else:
            out.append(p)
            fired.append(False)
    return out, fired


def run_variant(variant, lm, calib, test, retriever, gate, alpha,
                batch_size, uq_signal="entropy", adapter=None,
                adapter_layer=None, adapter_alpha=1.0):
    spec = VARIANT_SPEC[variant]
    use_rag, use_gate, use_uq, use_cl = (spec["rag"], spec["gate"],
                                         spec["uq"], spec["cl"])

    # The constraint layer is part of the model, so it must be attached before
    # ANY generation for this variant -- including the calibration pass, or the
    # UQ threshold would be calibrated on a different model from the one it
    # governs.
    if use_cl:
        if not adapter:
            raise SystemExit(f"variant {variant} needs --adapter")
        lm.attach_adapter(adapter, adapter_layer, adapter_alpha)
    else:
        lm.detach_adapter()

    tau, calib_diag = None, {}
    if use_uq:
        cg, _ = generate_for_split(lm, calib, use_rag, retriever, batch_size)
        cpred = [parse_answer(g.text) for g in cg]
        cfired = [False] * len(cpred)
        if use_gate:
            cpred, cfired = apply_gate(gate, calib, cpred)
        # Only calibrate on the population the UQ engine actually governs.
        # Gate-decided items are deterministic, so their entropies carry no
        # information about neural uncertainty; including them corrupts tau and
        # collapses the abstention set onto exactly the non-gated cases.
        pool = [(uncertainty(g, uq_signal), p == r["label"])
                for g, p, r, f in zip(cg, cpred, calib, cfired) if not f]
        if not pool:
            pool = [(uncertainty(g, uq_signal), p == r["label"])
                    for g, p, r in zip(cg, cpred, calib)]
        # An infinite uncertainty cannot be a threshold; drop those from
        # calibration only (they still abstain at test time by construction).
        finite = [(u, c) for u, c in pool if u != float("inf")]
        if finite:
            pool = finite
        tau = calibrate_threshold([e for e, _ in pool],
                                  [c for _, c in pool], alpha=alpha)
        # Record what the threshold achieved ON THE CALIBRATION SPLIT. When no
        # threshold reaches the target error rate, calibrate_threshold falls
        # back to the most conservative one and the engine abstains on almost
        # everything -- which is the correct behaviour for a model that is at
        # chance, but from the table alone it is indistinguishable from a bug.
        # These two numbers make the difference visible.
        kept = [(u, c) for u, c in pool if u <= tau]
        calib_diag = {
            "calib_n": len(pool),
            "calib_coverage_at_tau": len(kept) / len(pool) if pool else 0.0,
            "calib_error_at_tau": (1 - sum(c for _, c in kept) / len(kept))
            if kept else float("nan"),
            "calib_target_alpha": alpha,
        }

    gens, rag_meta = generate_for_split(lm, test, use_rag, retriever, batch_size)
    preds = [parse_answer(g.text) for g in gens]
    gate_fired = [False] * len(preds)
    if use_gate:
        preds, gate_fired = apply_gate(gate, test, preds)

    records = []
    for r, g, p, gf, rm in zip(test, gens, preds, gate_fired, rag_meta):
        # gate-decided items are deterministic -> never abstained on entropy
        unc = uncertainty(g, uq_signal)
        abstained = bool(use_uq and not gf and (p is None or unc > tau))
        if p is None and not use_uq:
            abstained = False
        records.append({
            "id": r["id"], "pair_id": r["pair_id"], "family": r["family"],
            "label": r["label"], "pred": p, "abstained": abstained,
            "entropy": g.entropy, "max_entropy": g.max_entropy,
            "uq_signal": uq_signal, "uq_uncertainty": (
                None if unc == float("inf") else unc),
            "logit_margin": g.logit_margin, "answer_logprob": g.answer_logprob,
            "gate_fired": gf, "retrieved": rm, "raw": g.text,
        })
    lm.detach_adapter()
    return records, tau, calib_diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "hf"], default="mock")
    ap.add_argument("--model_id", default="BioMistral/BioMistral-7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--data", default="data/synthetic_control",
                    help="benchmark root. Default is the SYNTHETIC CONTROL "
                         "benchmark (templated vignettes, hand-written "
                         "thresholds); pass data/medcalc for the real-notes arm.")
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", default="test",
                    help="name of the counterfactual_<split>.jsonl file to "
                         "score. `heldout` is the unseen rule family; on the "
                         "MedCalc data `heldout_all` is the pre-split 85-pair "
                         "version kept for continuity.")
    ap.add_argument("--uq_signal", default="decision_entropy",
                    choices=sorted(UQ_SIGNALS),
                    help="quantity the UQ engine defers on. Default since "
                         "2026-09-01 is `decision_entropy`, the "
                         "decision-restricted form of Eq. (2): AUROC 0.687 "
                         "against 0.525 for Eq. (2) as written, measured over "
                         "6,456 items. Pass `--uq_signal entropy` to "
                         "reproduce every number reported before that date.")
    ap.add_argument("--gate", default="rules", choices=["rules", "medcalc"],
                    help="'rules' is the regex gate for the templated "
                         "vignettes; 'medcalc' extracts labs from real "
                         "clinical prose and runs the clinical equation "
                         "itself. See src/medcalc_gate.py.")
    ap.add_argument("--variants", nargs="+", default=VARIANTS,
                    choices=VARIANTS,
                    help="which rows of the ablation to run")
    ap.add_argument("--adapter", default=None,
                    help="npz written by constraint_layer.py --save_adapter; "
                         "required for the 'cl' and 'nsai_uq_cl' variants")
    ap.add_argument("--adapter_layer", type=int, default=30)
    ap.add_argument("--adapter_alpha", type=float, default=1.0)
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames, so an alternative UQ "
                         "signal never overwrites the reported run.")
    args = ap.parse_args()

    data = Path(args.data)
    calib = read_jsonl(data / "counterfactual_calib.jsonl")
    test = read_jsonl(data / f"counterfactual_{args.split}.jsonl")
    retriever = TfidfRetriever(str(data / "rag_corpus.jsonl"), k=args.topk)
    if args.gate == "medcalc":
        from medcalc_gate import MedCalcGate
        gate = MedCalcGate()
    else:
        gate = SymbolicGate(RULE_FAMILIES)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # A mock run must never be able to overwrite a real one. The mock backend is
    # the DEFAULT, so a bare `python src/run_eval.py --split test` -- the exact
    # command a smoke test uses -- silently replaced the real `base` row in
    # results/summary_test.json AND results/preds_test_base_seed0.jsonl on
    # 2026-09-02. Mock output is forced onto its own `_mock` tag before ANY file
    # is written, and a mock row can never be merged into a file holding real
    # ones.
    if args.backend == "mock" and not args.tag.endswith("_mock"):
        args.tag = f"{args.tag}_mock"
        print("  NOTE: mock backend -- writing to the _mock tag so real "
              "artifacts cannot be clobbered.")

    summary = []
    for seed in args.seeds:
        lm = load_model({"backend": args.backend, "model_id": args.model_id,
                         "device": args.device, "dtype": args.dtype, "seed": seed})
        for variant in args.variants:
            recs, tau, calib_diag = run_variant(
                                    variant, lm, calib, test, retriever, gate,
                                    args.alpha, args.batch_size,
                                    uq_signal=args.uq_signal,
                                    adapter=args.adapter,
                                    adapter_layer=args.adapter_layer,
                                    adapter_alpha=args.adapter_alpha)
            s = score(recs)
            s.update({**calib_diag,
                      "variant": variant, "seed": seed, "tau": tau,
                      "gate_fired_rate": float(
                          sum(r["gate_fired"] for r in recs) / len(recs)),
                      "adapter": args.adapter if VARIANT_SPEC[variant]["cl"]
                      else None,
                      "uq_signal": args.uq_signal, "gate": args.gate,
                      "data": str(data),
                      "split": args.split, "backend": args.backend,
                      "model": args.model_id if args.backend == "hf" else "mock"})
            summary.append(s)

            with (outdir / f"preds_{args.split}_{variant}_seed{seed}{args.tag}.jsonl").open("w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(f"[seed {seed}] {variant:9s} "
                  f"acc={s['accuracy']:.3f} cc={s['causal_consistency']:.3f} "
                  f"viol={s['violation_rate']:.3f} cov={s['coverage']:.3f}")
        del lm

    # Merge into any existing summary rather than overwriting it: variants are
    # run in separate invocations (the constraint-layer rows need an adapter
    # that does not exist until it is trained), and clobbering the file would
    # silently drop the rows produced by the earlier call.
    spath = outdir / f"summary_{args.split}{args.tag}.json"
    if spath.exists():
        prev = json.load(spath.open())
        real = {r.get("model") for r in prev} - {"mock", None}
        if args.backend == "mock" and real:
            raise SystemExit(
                f"refusing to write mock rows into {spath}, which holds real "
                f"results from {sorted(real)}. Delete the file or use a "
                f"different --tag.")
        keep = [r for r in prev
                if (r["variant"], r["seed"]) not in
                {(x["variant"], x["seed"]) for x in summary}]
        summary = keep + summary
    order = {v: i for i, v in enumerate(VARIANTS)}
    summary.sort(key=lambda r: (r["seed"], order.get(r["variant"], 99)))
    with spath.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {outdir}/summary_{args.split}{args.tag}.json")


if __name__ == "__main__":
    main()
