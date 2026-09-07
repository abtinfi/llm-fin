"""
Aim 2: average causal effect of the edited clinical value, by layer.

WHAT IS PATCHED, AND WHY THAT AND NOT SOMETHING ELSE
----------------------------------------------------
The two arms of a pair are the same note with one number changed. Tokenised,
they differ at a handful of positions and agree everywhere else. This patches
**only those differing positions**, at one layer, taking the residual stream
from the counterfactual run and writing it into the clean run.

The obvious alternative -- replacing the whole layer at every position -- is
close to useless: writing every position of layer l from run B makes the
forward pass *become* run B from layer l onward, so the curve just measures how
early you overwrote, not what the number does. Restricting the patch to the
positions the edit touched isolates the causal path from that number to the
decision, which is the quantity Aim 2 asks for:

    ACE(l) = E[margin | do(number := counterfactual at layer l)] - E[margin]

Reported in logits (raw), and normalised by the clean gap between the two arms
where that gap is large enough for the ratio to mean anything. On a family
where the model's two arms already produce the same margin, the normalised
figure divides by noise and is suppressed rather than printed -- see
`src/probe.py` for why that family needs a probe instead.

CONTROLS
--------
  position control  Patch the same number of positions, same layer, but chosen
                    at random from positions the edit did NOT touch. A real
                    effect at the edited positions must exceed this.
  identity control  Patch the clean run from itself. Must give exactly zero;
                    anything else means the hook is misplaced.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lm_common import answer_token_ids, decoder_layers, wrap_prompt


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open()]


class Patcher:
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
        self.layers = decoder_layers(self.model)
        self.max_input_tokens = max_input_tokens
        self.safe_ids, self.unsafe_ids = self._answer_ids()

    def _answer_ids(self):
        """
        Delegated to `lm_common.answer_token_ids`, which is tokenizer-agnostic.
        Getting this wrong makes every margin in this file meaningless
        (P1 in HANDOFF.md).
        """
        got = answer_token_ids(self.tok)
        return got["SAFE"], got["UNSAFE"]

    def _wrap(self, p):
        return wrap_prompt(self.tok, p)

    def encode(self, prompt):
        return self.tok(self._wrap(prompt), return_tensors="pt",
                        truncation=True,
                        max_length=self.max_input_tokens).to("cuda")

    def margin(self, logits):
        """logit(SAFE) - logit(UNSAFE) at the next-token position."""
        v = logits[0, -1].float()
        return (max(v[i].item() for i in self.safe_ids)
                - max(v[i].item() for i in self.unsafe_ids))

    def run(self, enc, capture=False):
        torch = self.torch

        # POST-NORM BOUNDARY (defect B3, fixed 2026-09-02).
        #
        # hidden_states[l] is the INPUT to layer l (l=0 is the embedding
        # output), so hidden_states[l+1] is what layer l produced -- for every
        # l EXCEPT the last. transformers appends the tensor a final time
        # AFTER `model.norm`, so hidden_states[n_layers] is post-final-norm and
        # is NOT the output of layer n_layers-1. Verified empirically against
        # transformers 5.15.0: hidden_states[n_layers] differs from the last
        # layer's captured output by up to 3.0 in absolute value on a random
        # 3-layer model.
        #
        # Writing that normed tensor into the pre-norm residual stream made the
        # final row of every ACE curve invalid. Rather than dropping the row --
        # the deepest layer is the one where a causal effect is most expected,
        # so silently omitting it would be the worse fix -- the true output of
        # the last decoder layer is captured directly and substituted for the
        # post-norm entry. Every donor `hs[l+1]` is then the output of layer l
        # for all l, and `sufficiency` is corrected by the same change.
        grabbed = {}

        def grab(module, args, output):
            grabbed["h"] = (output[0] if isinstance(output, tuple)
                            else output).detach()

        handle = (self.layers[-1].register_forward_hook(grab)
                  if capture else None)
        try:
            with torch.no_grad():
                o = self.model(**enc, output_hidden_states=capture)
        finally:
            if handle:
                handle.remove()
        m = self.margin(o.logits)
        if not capture:
            return m, None
        hs = [h.detach() for h in o.hidden_states]
        assert len(hs) == len(self.layers) + 1, (
            f"expected {len(self.layers)+1} hidden states, got {len(hs)} -- "
            f"the layer/donor alignment in this file assumes one per layer "
            f"plus the embedding output")
        hs[-1] = grabbed["h"]          # replace post-norm with the true output
        return m, hs

    def run_patched(self, enc, layer, positions, donor, mode="replace",
                    alpha=1.0):
        """
        Forward `enc`, intervening on the output of `layer` at `positions`.

        mode="replace"  h[pos] := donor
                        Causal NECESSITY, the do(F := F_counterfactual) that
                        the original analysis runs.

        mode="add"      h[pos] := h[pos] + alpha * donor
                        Causal SUFFICIENCY. `donor` here is a DIRECTION (the
                        counterfactual-minus-clean difference, or a control
                        direction), not a replacement state, and alpha is the
                        dose. This is the intervention the proposal's ACE
                        formula literally names:

                            ACE(df) = E[Y | do(F + df)] - E[Y | do(F)]

                        Replacement is the special case that jumps straight to
                        the counterfactual value and reports no dose. Additive
                        injection sweeps df, so a real causal channel shows a
                        monotone dose-response and an artefact does not -- the
                        property `steering.py` looked for and did not find with
                        a fixed probe direction.
        """
        torch = self.torch
        pos = torch.tensor(positions, device="cuda", dtype=torch.long)

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            if mode == "add":
                h[0, pos, :] = h[0, pos, :] + alpha * donor.to(h.dtype)
            else:
                h[0, pos, :] = donor.to(h.dtype)
            return (h,) + output[1:] if isinstance(output, tuple) else h

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                o = self.model(**enc)
            return self.margin(o.logits)
        finally:
            handle.remove()


def analyse(args):
    recs = read_jsonl(args.data)
    by_pair = defaultdict(dict)
    for r in recs:
        by_pair[r["pair_id"]][r["arm"]] = r
    # `set(v) == {"safe", "unsafe"}` rather than `len(v) == 2`: control pairs
    # carry arms named ctrl_a/ctrl_b and would raise a KeyError here. They have
    # no causal edit to patch, so skipping them is correct as well as safe.
    pairs = [(k, v["safe"], v["unsafe"])
             for k, v in by_pair.items() if set(v) == {"safe", "unsafe"}]
    pairs.sort()
    print(f"{len(pairs)} complete pairs in {args.data}")

    P = Patcher(args.model_id)
    n_layers = len(P.layers)
    rng = np.random.default_rng(args.seed)

    raw = defaultdict(list)      # layer -> [delta margin]
    ctrl = defaultdict(list)
    gaps, identity_err = [], []
    used, skipped = 0, defaultdict(int)

    for pid, safe_r, unsafe_r in pairs:
        if args.limit and used >= args.limit:
            break
        ea = P.encode(safe_r["prompt"])
        eb = P.encode(unsafe_r["prompt"])
        a, b = ea["input_ids"][0], eb["input_ids"][0]
        if a.shape[0] != b.shape[0]:
            skipped["different token length"] += 1
            continue
        diff = (a != b).nonzero().flatten().tolist()
        if not diff:
            skipped["identical tokenisation"] += 1
            continue
        if len(diff) > args.max_diff:
            skipped[f"more than {args.max_diff} differing positions"] += 1
            continue

        m_clean, _ = P.run(ea)
        m_cf, hs_b = P.run(eb, capture=True)
        gaps.append(m_cf - m_clean)

        # identity control: patch the clean run from its own activations
        _, hs_a = P.run(ea, capture=True)
        mid = n_layers // 2
        e = P.run_patched(ea, mid, diff, hs_a[mid + 1][0, diff, :]) - m_clean
        identity_err.append(abs(e))

        # position control: same count of positions, but not the edited ones
        pool = [i for i in range(a.shape[0]) if i not in set(diff)]
        cpos = sorted(rng.choice(pool, size=len(diff), replace=False).tolist())

        for l in range(n_layers):
            donor = hs_b[l + 1][0, diff, :]
            raw[l].append(P.run_patched(ea, l, diff, donor) - m_clean)
            donor_c = hs_b[l + 1][0, cpos, :]
            ctrl[l].append(P.run_patched(ea, l, cpos, donor_c) - m_clean)

        used += 1
        if used % 10 == 0:
            cap = len(pairs) if not args.limit else min(args.limit, len(pairs))
            print(f"  {used}/{cap} pairs", flush=True)

    if not used:
        print("no usable pairs"); return
    gap = float(np.mean(gaps))
    print(f"\nusable pairs: {used}")
    for k, v in skipped.items():
        print(f"  skipped {v}: {k}")
    print(f"identity control, max |error|: {max(identity_err):.2e} "
          + ("(hook is correctly placed)" if max(identity_err) < 1e-3
             else "<- NON-ZERO, the hook is wrong and nothing below is valid"))
    print(f"\nclean margin gap between arms (unsafe - safe): {gap:+.4f} logits")
    normalise = abs(gap) >= args.min_gap
    if not normalise:
        print(f"  |gap| < {args.min_gap}: the model's two arms are "
              f"output-equivalent, so a normalised effect would divide by "
              f"noise. Raw logits only.")

    rows = []
    print(f"\n{'layer':>6} {'ACE (logits)':>14} {'control':>10} "
          f"{'excess':>10}" + (f" {'normalised':>11}" if normalise else ""))
    for l in range(n_layers):
        a = float(np.mean(raw[l]))
        c = float(np.mean(ctrl[l]))
        row = {"layer": l, "ace": a, "control": c, "excess": a - c}
        line = f"{l:6d} {a:14.4f} {c:10.4f} {a-c:10.4f}"
        if normalise:
            row["normalised"] = a / gap
            line += f" {a/gap:11.3f}"
        rows.append(row)
        print(line)

    best = max(rows, key=lambda r: abs(r["excess"]))
    print(f"\nlargest effect over control at layer {best['layer']}: "
          f"{best['excess']:+.4f} logits")
    if abs(best["excess"]) < 0.05:
        print("  -> the edited value has essentially no causal effect on the "
              "decision at ANY layer.\n     The quantity is not on a causal "
              "path to the answer.")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"data": args.data, "pairs_used": used, "clean_gap": gap,
             "identity_max_err": max(identity_err), "rows": rows}, indent=2))
        print(f"wrote {args.out}")


def sufficiency(args):
    """
    Aim 2, causal SUFFICIENCY: does injecting the counterfactual direction
    PRODUCE the unsafe decision, and does it do so dose-dependently?

    Necessity (the `analyse` path) asks whether removing/overwriting the
    edited value destroys the behaviour. Sufficiency asks the converse, and
    the two are not redundant: a quantity can be necessary because it gates
    something else without being sufficient to drive the decision itself. The
    proposal names both (4.5) and only necessity was implemented.

    The intervention is
        h[edited positions] += alpha * (h_unsafe - h_safe)
    swept over alpha, giving the continuous ACE the proposal's formula asks
    for rather than a single replacement.

    THREE CONTROLS, all mandatory, all reported:
      random direction   a Gaussian direction rescaled to the SAME per-position
                         norm as the real one. This is the proposal's "negative
                         controls (random features)". If a matched-norm random
                         vector moves the margin as much, the effect is norm,
                         not content.
      shuffled direction the real difference vectors from a DIFFERENT pair,
                         same norm distribution and same "is a real activation
                         difference" character, but the wrong content.
      alpha = 0          must reproduce the clean margin exactly.

    WHAT WOULD FALSIFY THE EXISTING NULL: a monotone dose-response in the real
    direction that the matched controls do not show. The necessity run found
    effects ~100-500x too small to flip a decision; if injection at alpha=1
    also lands there, necessity and sufficiency agree and Aim 2's answer is a
    consistent no on this model.
    """
    recs = read_jsonl(args.data)
    by_pair = defaultdict(dict)
    for r in recs:
        by_pair[r["pair_id"]][r["arm"]] = r
    # `set(v) == {"safe", "unsafe"}` rather than `len(v) == 2`: control pairs
    # carry arms named ctrl_a/ctrl_b and would raise a KeyError here. They have
    # no causal edit to patch, so skipping them is correct as well as safe.
    pairs = [(k, v["safe"], v["unsafe"])
             for k, v in by_pair.items() if set(v) == {"safe", "unsafe"}]
    pairs.sort()
    print(f"{len(pairs)} complete pairs in {args.data}")
    print(f"SUFFICIENCY (injection): h += alpha * (h_unsafe - h_safe) "
          f"at the edited positions\n")

    P = Patcher(args.model_id)
    n_layers = len(P.layers)
    # The matched-norm random control below was drawn from torch's GLOBAL
    # generator, which nothing seeds. `--seed` therefore did not reach it and
    # `random_control` -- and so `excess`, the headline quantity -- changed
    # between otherwise identical runs. Two runs of the held-out split on
    # 2026-09-02 agreed on `ace` and `shuffled_control` to the last digit and
    # disagreed on `random_control` on 30/32 rows for exactly this reason.
    # A dedicated seeded generator on the activation device fixes it.
    gen = P.torch.Generator(device=P.model.device).manual_seed(args.seed)
    alphas = [float(a) for a in args.alphas.split(",")]
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(0, n_layers, max(1, n_layers // 8))))

    # real[layer][alpha] -> [delta margin]
    real = defaultdict(lambda: defaultdict(list))
    randc = defaultdict(lambda: defaultdict(list))
    shufc = defaultdict(lambda: defaultdict(list))
    zero_err = []
    prev_delta = {}                 # layer -> a previous pair's direction
    used, skipped = 0, defaultdict(int)

    for pid, safe_r, unsafe_r in pairs:
        if args.limit and used >= args.limit:
            break
        ea, eb = P.encode(safe_r["prompt"]), P.encode(unsafe_r["prompt"])
        a, b = ea["input_ids"][0], eb["input_ids"][0]
        if a.shape[0] != b.shape[0]:
            skipped["different token length"] += 1
            continue
        diff = (a != b).nonzero().flatten().tolist()
        if not diff or len(diff) > args.max_diff:
            skipped["no usable differing positions"] += 1
            continue

        m_clean, hs_a = P.run(ea, capture=True)
        _, hs_b = P.run(eb, capture=True)

        for l in layers:
            d = (hs_b[l + 1][0, diff, :] - hs_a[l + 1][0, diff, :])
            # matched-norm random control, per position
            g = P.torch.randn(d.shape, generator=gen,
                              device=d.device, dtype=d.dtype)
            g = g / g.norm(dim=-1, keepdim=True) * d.norm(dim=-1, keepdim=True)
            sh = prev_delta.get(l)
            for al in alphas:
                real[l][al].append(
                    P.run_patched(ea, l, diff, d, mode="add", alpha=al) - m_clean)
                randc[l][al].append(
                    P.run_patched(ea, l, diff, g, mode="add", alpha=al) - m_clean)
                if sh is not None and sh.shape == d.shape:
                    shufc[l][al].append(
                        P.run_patched(ea, l, diff, sh, mode="add",
                                      alpha=al) - m_clean)
            zero_err.append(abs(
                P.run_patched(ea, l, diff, d, mode="add", alpha=0.0) - m_clean))
            prev_delta[l] = d

        used += 1
        if used % 10 == 0:
            cap = len(pairs) if not args.limit else min(args.limit, len(pairs))
            print(f"  {used}/{cap} pairs", flush=True)

    if not used:
        print("no usable pairs"); return
    print(f"\nusable pairs: {used}")
    for k, v in skipped.items():
        print(f"  skipped {v}: {k}")
    z = max(zero_err) if zero_err else 0.0
    print(f"alpha=0 control, max |error|: {z:.2e} "
          + ("(injection hook is correctly placed)" if z < 1e-3
             else "<- NON-ZERO, the hook is wrong and nothing below is valid"))

    rows = []
    print(f"\n{'layer':>6} {'alpha':>7} {'ACE(inject)':>12} {'rand ctrl':>10} "
          f"{'shuf ctrl':>10} {'excess':>10}")
    for l in layers:
        for al in alphas:
            r = float(np.mean(real[l][al]))
            c = float(np.mean(randc[l][al])) if randc[l][al] else float("nan")
            sh = float(np.mean(shufc[l][al])) if shufc[l][al] else float("nan")
            ex = r - c
            rows.append({"layer": l, "alpha": al, "ace": r,
                         "random_control": c, "shuffled_control": sh,
                         "excess": ex, "n": len(real[l][al])})
            print(f"{l:6d} {al:7.2f} {r:12.4f} {c:10.4f} {sh:10.4f} {ex:10.4f}")

    best = max(rows, key=lambda r: abs(r["excess"]))
    print(f"\nlargest excess over the matched random control: "
          f"{best['excess']:+.4f} logits at layer {best['layer']}, "
          f"alpha={best['alpha']}")

    # Dose-response is the discriminating test, not the single best number.
    mono = {}
    for l in layers:
        seq = [float(np.mean(real[l][al])) for al in sorted(alphas)]
        mono[l] = bool(all(x <= y for x, y in zip(seq, seq[1:]))
                       or all(x >= y for x, y in zip(seq, seq[1:])))
    n_mono = sum(mono.values())
    print(f"monotone dose-response in {n_mono}/{len(layers)} layers tested")
    if abs(best["excess"]) < 0.05:
        print("  -> injection does NOT produce the decision at any dose "
              "tested.\n     Read together with the necessity run: the edited "
              "quantity is neither\n     necessary nor sufficient for the "
              "answer on this model.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"mode": "sufficiency", "data": args.data, "pairs_used": used,
             "alphas": alphas, "layers": layers,
             "alpha0_max_err": float(z),
             "monotone_layers": mono, "rows": rows}, indent=2))
        print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model_id", default="BioMistral/BioMistral-7B")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap on pairs evaluated; 0 (default) means "
                         "ALL complete pairs in --data, no cap")
    ap.add_argument("--max_diff", type=int, default=6)
    ap.add_argument("--min_gap", type=float, default=0.5)
    ap.add_argument("--mode", choices=["necessity", "sufficiency"],
                    default="necessity",
                    help="necessity = replace the edited positions with the "
                         "counterfactual state (the original analysis); "
                         "sufficiency = inject the counterfactual direction "
                         "at a swept dose (proposal 4.5, second test)")
    ap.add_argument("--alphas", default="0.5,1,2,4",
                    help="injection doses for --mode sufficiency")
    ap.add_argument("--layers", default=None,
                    help="comma-separated layers for --mode sufficiency; "
                         "default is 8 evenly spaced")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    (sufficiency if args.mode == "sufficiency" else analyse)(args)


if __name__ == "__main__":
    main()
