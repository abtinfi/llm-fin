"""
Aim 3, smallest honest prototype: can the represented fact be MADE causal?

THE GAP THIS SITS IN
--------------------
Two results from the same benchmark family (`data/medcalc` held-out, QTc >
500 ms -> ondansetron unsafe):

  probe    a linear readout of the residual stream at layer 5 recovers the
           correct SAFE/UNSAFE label at 0.859 pair-consistency, against a
           null of 0.235. The fact is there, and it is there at the readout
           position.
  patching moving that same representation between the two arms of a pair
           changes the decision logit by ~0.02. It has no causal effect.
  model    answers SAFE on all 170 items. Pair-consistency 0.000.

So the information is present, available where the decision is made, and
ignored. The proposal's Aim 3 says to fix exactly this, with

    h'_l = h_l + alpha * P_causal(h_l)

This is that, in its simplest possible form: `P_causal` is the probe direction
already fitted, and the only free parameter is alpha. If adding the direction
moves the answer toward the truth, the representation was usable and the model
was failing to read it. If it does not, the probe was reading something the
computation downstream cannot consume, and Aim 3 needs a trained layer rather
than a direction.

WHAT IS AND IS NOT CLAIMED
--------------------------
This is a prototype, not Aim 3. It steers with a single fixed direction at one
layer and one position, applies no ontology or uncertainty term from the
proposal's objective, and trains nothing. A positive result licenses building
the real layer; it is not a substitute for it.

HONESTY GUARDS
--------------
  - The steering direction is fitted on the TRAINING folds only and applied to
    held-out items, fold by fold. A direction fitted on all items and then
    evaluated on those items would be circular and would always "work".
  - alpha is swept and the whole curve reported. No alpha is selected and then
    presented as the result.
  - A random-direction control of matched norm is run at every alpha. Steering
    with a random vector also perturbs the output; only the margin over that
    control is evidence.
  - The sign convention is checked against the probe's own classes, not
    assumed, AND the sweep runs both ways (positive alpha steers toward
    UNSAFE, negative toward SAFE). Until 2026-09-02 the caller negated alpha
    while sweeping only non-negative values, so every steer pushed toward
    SAFE on a model that already answered SAFE on all 170 items -- the one
    direction capable of changing an answer was never tested. Any result
    below that pre-dates that fix is not evidence either way.
  - The direction is injected into the SAME representation it was fitted on.
    The probe cache index l is hidden_states[l], the input to layer l, so the
    hook fires on layers[l-1]; see `Steerer._producer`.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import grouped_folds, read_jsonl        # noqa: E402


def fit_direction(X_tr, y_tr, C=0.05, n_pca=128):
    """
    Probe direction in activation space, as a unit vector pointing toward
    UNSAFE.

    Fitted through the same scaler+PCA+logistic pipeline the probe uses, then
    pulled back into the original 4096-dim space so it can be added to the
    residual stream: the composed map is linear, so the pullback is exact.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X_tr)
    Z = sc.transform(X_tr)
    k = int(min(n_pca, len(X_tr) - 1, X_tr.shape[1]))
    pca = PCA(n_components=k, random_state=0).fit(Z)
    clf = LogisticRegression(max_iter=5000, C=C).fit(pca.transform(Z), y_tr)

    # w in PCA space -> scaled space -> original space
    w = clf.coef_[0] @ pca.components_          # [D] in scaled space
    w = w / np.where(sc.scale_ == 0, 1.0, sc.scale_)
    w = w / (np.linalg.norm(w) + 1e-12)
    # clf.classes_ is sorted; make the vector point toward UNSAFE
    if clf.classes_[1] != "UNSAFE":
        w = -w
    return w


class Steerer:
    def __init__(self, model_id, dtype="bfloat16", max_input_tokens=1400):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map="cuda")
        self.model.eval()
        self.layers = self.model.model.layers
        self.max_input_tokens = max_input_tokens
        self.safe_ids, self.unsafe_ids = self._answer_ids()

    def _answer_ids(self):
        got = {}
        for label in ("SAFE", "UNSAFE"):
            ids = set()
            for t in (label, f" {label}", f"\n{label}"):
                for tid in self.tok.encode(t, add_special_tokens=False):
                    piece = self.tok.convert_ids_to_tokens(tid)
                    if not piece.replace("▁", "").strip() or piece == "<0x0A>":
                        continue
                    ids.add(tid)
                    break
            got[label] = ids
        overlap = got["SAFE"] & got["UNSAFE"]
        got["SAFE"] -= overlap
        got["UNSAFE"] -= overlap
        assert got["SAFE"] and got["UNSAFE"]
        return got["SAFE"], got["UNSAFE"]

    def _wrap(self, p):
        if self.tok.chat_template:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True)
        return f"[INST] {p} [/INST]"

    def _producer(self, layer):
        """
        The module whose output IS `hidden_states[layer]` (defect B2, fixed
        2026-09-02).

        The steering direction is fitted on the probe cache, whose index `l`
        is `hidden_states[l]` -- the INPUT to decoder layer l, i.e. the OUTPUT
        of layer l-1. Registering the hook on `layers[layer]` therefore fitted
        the direction in one representation and injected it into the next one
        down, so the vector being added was not the vector that was learned.
        """
        if layer == 0:
            return self.model.model.embed_tokens
        return self.layers[layer - 1]

    def margin(self, prompt, layer=None, vec=None, alpha=0.0):
        """logit(SAFE) - logit(UNSAFE), optionally steering at `layer`."""
        torch = self.torch
        enc = self.tok(self._wrap(prompt), return_tensors="pt",
                       truncation=True,
                       max_length=self.max_input_tokens).to("cuda")
        handle = None
        if vec is not None and alpha != 0.0:
            v = torch.tensor(vec, device="cuda")

            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                h = h.clone()
                # steer only the final position: that is where the decision is
                # read out, and it is the position the probe was fitted on
                h[0, -1, :] = h[0, -1, :] + (alpha * v).to(h.dtype)
                return (h,) + output[1:] if isinstance(output, tuple) else h

            handle = self._producer(layer).register_forward_hook(hook)
        try:
            with torch.no_grad():
                o = self.model(**enc)
            v = o.logits[0, -1].float()
            return (max(v[i].item() for i in self.safe_ids)
                    - max(v[i].item() for i in self.unsafe_ids))
        finally:
            if handle:
                handle.remove()


def score(records, margins):
    """Item accuracy and pair-level consistency from signed margins."""
    ok, by_pair = [], defaultdict(list)
    for r, m in zip(records, margins):
        pred = "SAFE" if m > 0 else "UNSAFE"
        good = pred == r["label"]
        ok.append(good)
        by_pair[r["pair_id"]].append(good)
    cc = float(np.mean([all(v) for v in by_pair.values()]))
    return float(np.mean(ok)), cc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True,
                    help="npz of activations from probe.py")
    ap.add_argument("--layer", type=int, nargs="+", required=True,
                    help="one or more layers to steer at")
    ap.add_argument("--pool", default="final", choices=["final", "mean"])
    ap.add_argument("--model_id", default="BioMistral/BioMistral-7B")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[-8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0,
                             0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
                    help="in MULTIPLES of that layer's mean residual norm. "
                         "The residual stream grows ~240x from layer 0 to 32 "
                         "in this model (0.13 -> 347), so an absolute alpha "
                         "means something different at every depth and a "
                         "fixed sweep is uninterpretable. The sweep is now "
                         "TWO-SIDED: positive steers toward UNSAFE, negative "
                         "toward SAFE. A one-sided sweep cannot distinguish "
                         "'this direction does nothing' from 'this direction "
                         "was applied backwards', which is exactly the "
                         "ambiguity the pre-2026-09-02 runs were in.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    recs = read_jsonl(args.data)
    y = np.array([r["label"] for r in recs])
    pair_ids = [r["pair_id"] for r in recs]
    z = np.load(args.cache)[args.pool]                     # [L+1, N, D]
    folds = grouped_folds(pair_ids, args.folds, args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"{len(recs)} items, {len(set(pair_ids))} pairs, "
          f"pooling {args.pool}, layers {args.layer}")

    S = Steerer(args.model_id)
    report = {"data": args.data, "pool": args.pool, "layers": {}}

    for layer in args.layer:
        acts = z[layer]
        norm = float(np.linalg.norm(acts, axis=1).mean())
        dirs = {f: fit_direction(acts[folds != f], y[folds != f])
                for f in range(args.folds)}
        rand_dirs = {}
        for f in range(args.folds):
            v = rng.normal(size=acts.shape[1])
            rand_dirs[f] = v / np.linalg.norm(v)

        print(f"\n=== layer {layer}: mean residual norm {norm:.2f} ===",
              flush=True)
        rows = []
        for rel in args.alphas:
            alpha = rel * norm
            m_probe, m_rand = [], []
            for i, r in enumerate(recs):
                f = folds[i]
                # SIGN (defect B2, fixed 2026-09-02).
                #
                # `fit_direction` returns a unit vector pointing toward
                # UNSAFE: sklearn's `coef_[0]` points at `classes_[1]`, and
                # `classes_` sorts to ["SAFE", "UNSAFE"], so the guard in
                # fit_direction never fires and the vector is already correct.
                # This call then passed `-alpha`, so with an all-positive
                # sweep every steer pushed toward SAFE -- and the base model
                # already answers SAFE on all 170 items, so the only direction
                # that could change an answer was never tested. Item accuracy
                # sat at exactly 0.500 for every layer and every alpha, which
                # is what that looks like.
                #
                # `alpha` is now passed through unnegated, so a POSITIVE alpha
                # steers toward UNSAFE, and the sweep carries both signs.
                m_probe.append(S.margin(r["prompt"], layer, dirs[f], alpha))
                m_rand.append(S.margin(r["prompt"], layer, rand_dirs[f],
                                       alpha))
            a_p, cc_p = score(recs, m_probe)
            a_r, cc_r = score(recs, m_rand)
            rows.append({"alpha_rel": rel, "alpha_abs": alpha,
                         "acc": a_p, "cc": cc_p,
                         "acc_rand": a_r, "cc_rand": cc_r})
            print(f"  alpha={rel:5.2f}x norm ({alpha:7.2f})  "
                  f"probe: acc={a_p:.3f} CC={cc_p:.3f}   "
                  f"random: acc={a_r:.3f} CC={cc_r:.3f}   "
                  f"margin={cc_p-cc_r:+.3f}", flush=True)
        report["layers"][str(layer)] = {"residual_norm": norm, "rows": rows}

    allrows = [(int(l), r) for l, v in report["layers"].items()
               for r in v["rows"]]
    bl, br = max(allrows, key=lambda t: t[1]["cc"] - t[1]["cc_rand"])
    gain = br["cc"] - br["cc_rand"]
    # The unsteered cell is the one at alpha=0, which is no longer the first
    # row now that the sweep starts at the most negative alpha.
    zero_cells = [r["cc"] for _, r in allrows if r["alpha_rel"] == 0]
    base = zero_cells[0] if zero_cells else float("nan")
    n_cells = len(allrows)

    # A fixed threshold on the BEST cell is not a test. With `n_cells` cells
    # swept, the maximum of that many noisy estimates is elevated by
    # construction, and a lone peak with no dose-response either side of it is
    # what noise looks like. Two things are checked instead:
    #   - how far the best margin stands above the control arm's own largest
    #     excursion, which is a same-data estimate of the noise scale;
    #   - whether the neighbouring alphas at the same layer are also elevated,
    #     i.e. whether there is any dose-response at all.
    ctrl_swing = max(abs(r["cc_rand"]) for _, r in allrows)

    # Dose-response must be read along ONE arm of the sweep. With the sweep now
    # two-sided, the numerically adjacent alphas either side of a best cell at
    # +0.25 are 0 (where the gain is 0 by construction, since no hook is
    # attached) and +0.5. Including alpha=0 would make `dose` false for every
    # small-alpha peak regardless of the evidence. Neighbours are therefore
    # taken among the alphas of the SAME SIGN, ordered by magnitude.
    sign = 1 if br["alpha_rel"] > 0 else -1
    same_arm = sorted((r for r in report["layers"][str(bl)]["rows"]
                       if r["alpha_rel"] != 0
                       and (r["alpha_rel"] > 0) == (sign > 0)),
                      key=lambda r: abs(r["alpha_rel"]))
    if br["alpha_rel"] == 0:
        nbrs = []                      # the unsteered cell cannot show a dose
    else:
        i = [j for j, r in enumerate(same_arm)
             if r["alpha_rel"] == br["alpha_rel"]][0]
        nbrs = [same_arm[j]["cc"] - same_arm[j]["cc_rand"]
                for j in (i - 1, i + 1) if 0 <= j < len(same_arm)]
    dose = all(x >= 0.5 * gain for x in nbrs) if nbrs else False

    print(f"\nunsteered CC on this split: {base:.3f}")
    print(f"best of {n_cells} (layer, alpha) cells: layer {bl}, "
          f"alpha {br['alpha_rel']:+.2f}x norm "
          f"({'toward UNSAFE' if br['alpha_rel'] > 0 else 'toward SAFE' if br['alpha_rel'] < 0 else 'unsteered'})"
          f" -> CC {br['cc']:.3f} vs random "
          f"{br['cc_rand']:.3f} ({gain:+.3f})")
    print(f"  largest excursion of the RANDOM control anywhere: "
          f"{ctrl_swing:.3f}")
    print(f"  neighbouring alphas at that layer: "
          f"{', '.join(f'{x:+.3f}' for x in nbrs) or 'n/a'}"
          f"   dose-response: {'yes' if dose else 'NO'}")

    if gain <= ctrl_swing * 1.5 or not dose:
        print("  -> NOT a result. The best cell is within the range the random "
              "control reaches\n     on its own, and/or it is an isolated peak "
              "with no dose-response.\n     Steering along a single probe "
              "direction does not make the fact causal;\n     Aim 3 needs a "
              "trained layer, not a vector.")
    else:
        print("  -> the represented fact CAN be made to drive the answer, with "
              "a dose-response.\n     Aim 3 has a concrete target.")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
