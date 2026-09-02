"""
Aim 3: the Constraint-Aware Layer, trained.

    h'_l = h_l + alpha * P_causal(h_l)

`src/steering.py` already tested the degenerate case where `P_causal` is a
fixed direction taken from a linear probe. That was a null: best margin +0.106
over a matched-norm random control that itself swings +-0.071, with no
dose-response across alpha. The conclusion there was that the fact is
*decodable* but not *consumable* as a rank-one nudge, and that Aim 3 needs a
trained map. This is that map.

THE OBJECTIVE
-------------
The proposal specifies
    L = L_LM + l1*L_ontology + l2*L_causal + l3*L_uncertainty
and all four terms are now implemented:

  L_causal      cross-entropy on the correct SAFE/UNSAFE token at the decision
                position. The label comes from the clinical rule, not a judge.

  L_LM          KL(base || adapted) over the full vocabulary at the same
                position, against the frozen base model's own distribution.
                This is the term that stops the layer from destroying the model
                to win the classification, and it is what makes the RQ3
                perplexity check meaningful rather than decorative.
                (The direction is the one torch's `kl_div(input, target)`
                computes, with the base distribution as the target. An earlier
                docstring called it KL(adapted || base); the code was always
                this way round. Either direction is a valid drift penalty, but
                the file must say which one it is.)

  L_ontology    a PAIRWISE term, and the only one that uses the ontology
                rather than the label. The constraint graph does not merely
                say which arm is unsafe -- it says the decision is MONOTONE in
                the decisive quantity: crossing the threshold in the unsafe
                direction can only make the case less safe, never more. So for
                the two arms of a pair the SAFE-minus-UNSAFE logit margin must
                be ordered, safe arm above unsafe arm, by at least a margin.
                A hinge on that ordering is a statement about the mechanism,
                not about either individual answer, and a model can satisfy
                every per-item label while violating it.

  L_uncertainty the constraint layer must not become confident where the note
                does not establish the fact. For each training item an ABLATED
                copy is built with the decisive number redacted; on those the
                decision distribution is pushed toward maximum entropy. Without
                this term nothing stops the adapter from learning a prior over
                the corpus ("these notes are usually unsafe") and stating it
                with full confidence on a note that says nothing.
                The redaction is done by `redact()` below, and the effect is
                measured after training, not assumed: the report prints mean
                decision entropy on redacted items before and after.

THE PREDICTION THIS TESTS
-------------------------
`results/aim123_internals.md` established, before any of this was trained:

  QT family     the decisive fact IS linearly decodable from the residual
                stream (pair-CC 0.976 mean-pooled, 0.859 at the readout
                position, null 0.20-0.24).
  renal family  it is NOT (0.042 / 0.093, null 0.03-0.04), even though the
                decision is trivially decodable from the raw facts (1.000).

So the prediction, made in advance: a constraint layer should be able to help
on QT and should NOT be able to help on renal, because no map can read a
quantity the representation does not carry. Training on renal and testing on
both is therefore a real test and not a fishing expedition -- a layer that
"works" on renal would falsify the probe result, and one that transfers to the
unseen QT family would confirm it.

HONESTY GUARDS
--------------
  - Train / calib / test / heldout are disjoint BY PAIR. No note the layer was
    trained on is ever scored.
  - The held-out family (QT) is a different drug, different quantity and
    different threshold, never seen in training.
  - The base model is scored under the IDENTICAL decision rule (argmax over the
    two answer logits at the decision position), so the comparison is not
    confounded by a change of readout.
  - alpha is fixed at 1.0 and the layer index is chosen from the probe curve
    BEFORE training, not swept for the best test number.
  - A control run trains the same adapter on SHUFFLED labels. If that also
    improves the test score, the gain is coming from the optimisation touching
    the model at all, not from the constraint.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open()]


def ontology_margins(graph_path, families, base_margin):
    """
    Required SAFE-minus-UNSAFE margin per rule family, read from the causal
    knowledge graph instead of being a constant.

    Before this, `L_ontology` applied `ont_margin` to every pair identically,
    which meant the term consulted no ontology at all -- the one objective term
    whose name outran its content. It now consults
    `CausalKnowledgeGraph.ontology_margin`, which returns

        0                  no contraindication edge for this drug/finding
                           -> the term does not fire at all
        base               the edge is expert-curated only
        base * 1.25        MED-RT independently attests the edge

    WHAT THIS DOES AND DOES NOT CHANGE, predicted before running:

      ondansetron_qt  is ABSENT from MED-RT, so it keeps the curated margin and
                      the headline Aim 3 number (held-out QT CC 0.000 -> 0.767)
                      should be UNCHANGED by this wiring. If it moves, the
                      cause is adapter stochasticity, not the ontology.
      metformin_renal is MED-RT-attested, so the renal run's required margin
                      rises by 25% and `constraint_renal.json` is expected to
                      move. The probe predicts renal cannot work regardless,
                      so this is not a rescue attempt.

    Returning `{}` (no graph) restores the old constant behaviour exactly, so
    the previous results remain reproducible.
    """
    if not graph_path:
        return {}, "constant (no graph): every pair required `ont_margin`"
    from umls_grounding import CausalKnowledgeGraph
    g = CausalKnowledgeGraph.load(graph_path)
    out, notes = {}, []
    for e in g.edges:
        if e.rel != "CONTRAINDICATED_WITH" or e.provenance != "curated":
            continue
        fam = e.source_vocab
        if fam not in families:
            continue
        m = g.ontology_margin(e.src, e.dst, base=base_margin)
        out[fam] = m
        notes.append(f"    {fam:30s} required margin {m:.3f}  "
                     f"[{g.attestation(e.src, e.dst)}]")
    missing = sorted(set(families) - set(out))
    for fam in missing:
        notes.append(f"    {fam:30s} NOT IN GRAPH -- term silent (0.0)")
        out[fam] = 0.0
    return out, "\n".join(notes)


# The decisive quantity, by family. Redacting it turns a decidable note into
# one where the correct behaviour is to be uncertain -- the supervision signal
# for L_uncertainty. Everything else in the note is untouched, so the ablated
# item is as close as possible to the real one.
REDACT_PATTERNS = {
    "metformin_renal": [
        re.compile(r"(creatin(?:ine|e)\b[^.\n;]{0,30}?)(\d+(?:\.\d+)?)"
                   r"(\s*(?:mg\s*/\s*d[lL]|[\u00b5u]mol\s*/\s*L))", re.I),
        re.compile(r"(\bCr\b[^.\n;]{0,12}?)(\d+(?:\.\d+)?)"
                   r"(\s*(?:mg\s*/\s*d[lL]|[\u00b5u]mol\s*/\s*L))"),
    ],
    "ondansetron_qt": [
        re.compile(r"(QT\s*(?:interval|duration)?[^.\n;]{0,25}?)"
                   r"(\d+(?:\.\d+)?)(\s*(?:ms|msec|milliseconds)\b)", re.I),
    ],
}


def redact(record):
    """
    The same note with the decisive number replaced by `[not recorded]`.

    Returns None when no pattern matches -- the item is then simply not used
    for L_uncertainty rather than being redacted approximately. A half-redacted
    note would teach the layer to be uncertain about a note that still contains
    the answer, which is worse than not training the term at all.
    """
    prompt = record["prompt"]
    for rx in REDACT_PATTERNS.get(record["family"], []):
        new, n = rx.subn(lambda m: m.group(1) + "[not recorded]", prompt,
                         count=1)
        if n:
            return new
    return None


class ConstraintAdapter:
    """Low-rank residual map, the trainable P_causal."""

    def __init__(self, dim, rank, torch, device="cuda"):
        self.torch = torch
        g = torch.Generator(device="cpu").manual_seed(0)
        # down-projection small, up-projection zero: the layer starts as an
        # exact identity, so training begins from the unmodified model and any
        # change is attributable to the objective rather than to initialisation.
        self.W_down = (torch.randn(dim, rank, generator=g) * (dim ** -0.5)
                       ).to(device).float().requires_grad_(True)
        self.W_up = torch.zeros(rank, dim, device=device).float(
            ).requires_grad_(True)
        self.b = torch.zeros(rank, device=device).float().requires_grad_(True)

    def params(self):
        return [self.W_down, self.W_up, self.b]

    def __call__(self, h):
        torch = self.torch
        x = h.float()
        z = torch.nn.functional.gelu(x @ self.W_down + self.b)
        return (z @ self.W_up).to(h.dtype)

    def state(self):
        return {k: v.detach().cpu().numpy()
                for k, v in (("W_down", self.W_down), ("W_up", self.W_up),
                             ("b", self.b))}


class Runner:
    def __init__(self, model_id, layer, rank, alpha, dtype="bfloat16",
                 max_input_tokens=1200):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map="cuda")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = self.model.model.layers
        self.layer = layer
        self.alpha = alpha
        self.max_input_tokens = max_input_tokens
        dim = self.model.config.hidden_size
        self.adapter = ConstraintAdapter(dim, rank, torch)
        self.safe_ids, self.unsafe_ids = self._answer_ids()
        self._handle = None

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
        ov = got["SAFE"] & got["UNSAFE"]
        got["SAFE"] -= ov
        got["UNSAFE"] -= ov
        assert got["SAFE"] and got["UNSAFE"], "answer tokens not separable"
        return sorted(got["SAFE"]), sorted(got["UNSAFE"])

    def _wrap(self, p):
        if self.tok.chat_template:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True)
        return f"[INST] {p} [/INST]"

    def encode(self, prompt):
        return self.tok(self._wrap(prompt), return_tensors="pt",
                        truncation=True,
                        max_length=self.max_input_tokens).to("cuda")

    def attach(self):
        """Insert h' = h + alpha * P_causal(h) at the chosen layer."""
        if self._handle is not None:
            return

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h + self.alpha * self.adapter(h)
            return (h,) + output[1:] if isinstance(output, tuple) else h

        self._handle = self.layers[self.layer].register_forward_hook(hook)

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def logits(self, enc):
        return self.model(**enc).logits[0, -1]

    def decision(self, logits):
        """argmax over the two answer classes -- the readout used everywhere."""
        s = max(logits[i].item() for i in self.safe_ids)
        u = max(logits[i].item() for i in self.unsafe_ids)
        return ("SAFE" if s > u else "UNSAFE"), s - u


def score(records, preds):
    ok, by_pair = [], defaultdict(list)
    for r, p in zip(records, preds):
        good = p == r["label"]
        ok.append(good)
        by_pair[r["pair_id"]].append(good)
    cc = float(np.mean([all(v) for v in by_pair.values()]))
    return float(np.mean(ok)), cc


def evaluate(R, records, with_adapter):
    torch = R.torch
    R.attach() if with_adapter else R.detach()
    preds = []
    with torch.no_grad():
        for r in records:
            p, _ = R.decision(R.logits(R.encode(r["prompt"])))
            preds.append(p)
    R.detach()
    return score(records, preds)


def decision_logits(R, prompt):
    """(logsumexp over SAFE ids, same over UNSAFE ids) at the decision step."""
    torch = R.torch
    lg = R.logits(R.encode(prompt)).float()
    a = torch.logsumexp(lg[R.safe_ids], dim=0)
    b = torch.logsumexp(lg[R.unsafe_ids], dim=0)
    return lg, a, b


def redacted_entropy(R, records, with_adapter):
    """
    Mean binary entropy of the SAFE/UNSAFE decision on ablated notes, in nats.

    ln 2 = 0.693 is maximal (no information). This is the measurement that
    makes L_uncertainty a claim rather than a decoration: it is reported for
    the base model and for the adapted model, on notes whose decisive number
    has been removed.
    """
    torch = R.torch
    R.attach() if with_adapter else R.detach()
    ents = []
    with torch.no_grad():
        for r in records:
            pr = redact(r)
            if pr is None:
                continue
            _, a, b = decision_logits(R, pr)
            p = torch.softmax(torch.stack([a, b]), dim=0)
            ents.append(float(-(p * torch.log(p + 1e-12)).sum()))
    R.detach()
    return (float(np.mean(ents)) if ents else float("nan")), len(ents)


def train(R, records, epochs, lr, lam_kl, lam_ont=0.0, lam_unc=0.0,
          ont_margin=1.0, shuffle_labels=False, seed=0, fam_margin=None):
    """
    L = L_causal + lam_kl*L_LM + lam_ont*L_ontology + lam_unc*L_uncertainty

    Optimisation is per PAIR, not per item, because L_ontology is defined on
    the two arms jointly: it constrains their ORDER, which no single-item loss
    can express. The other three terms are averaged over the arms of the pair,
    so lam_* keep their meaning regardless of pairing.
    """
    torch = R.torch
    rng = np.random.default_rng(seed)
    labels = {r["id"]: r["label"] for r in records}
    by_pair = defaultdict(list)
    for r in records:
        by_pair[r["pair_id"]].append(r)
    pairs = [v for v in by_pair.values() if len(v) == 2]
    if len(pairs) * 2 != len(records):
        print(f"  note: {len(records) - 2*len(pairs)} unpaired items are used "
              f"for the per-item terms only")

    if shuffle_labels:
        # Control: permute WITHIN each pair, preserving the one-of-each
        # structure so the task stays balanced but the association is gone.
        for arms in pairs:
            if rng.random() < 0.5:
                a, b = arms
                labels[a["id"]], labels[b["id"]] = labels[b["id"]], labels[a["id"]]

    # Base distributions, cached once, for the KL term.
    R.detach()
    base_logp = {}
    with torch.no_grad():
        for r in records:
            lg = R.logits(R.encode(r["prompt"])).float()
            base_logp[r["id"]] = torch.log_softmax(lg, dim=-1).cpu()

    opt = torch.optim.AdamW(R.adapter.params(), lr=lr, weight_decay=0.0)
    order = np.arange(len(pairs))
    R.attach()
    for ep in range(epochs):
        rng.shuffle(order)
        tot = defaultdict(float)
        n_ok = n_items = n_ont = 0
        for i in order:
            arms = pairs[i]
            losses, margins = [], {}
            for r in arms:
                lg, a, b = decision_logits(R, r["prompt"])
                lab = labels[r["id"]]
                tgt_a, tgt_b = (a, b) if lab == "SAFE" else (b, a)
                l_causal = -(tgt_a - torch.logsumexp(
                    torch.stack([tgt_a, tgt_b]), dim=0))

                bl = base_logp[r["id"]].to(lg.device)
                l_kl = torch.nn.functional.kl_div(
                    torch.log_softmax(lg, dim=-1), bl,
                    log_target=True, reduction="sum")

                losses.append(l_causal + lam_kl * l_kl)
                margins[r["label"]] = a - b     # SAFE-minus-UNSAFE logit
                tot["causal"] += float(l_causal)
                tot["kl"] += float(l_kl)
                n_ok += int((tgt_a > tgt_b).item())
                n_items += 1

                if lam_unc:
                    pr = redact(r)
                    if pr is not None:
                        _, ua, ub = decision_logits(R, pr)
                        pu = torch.softmax(torch.stack([ua, ub]), dim=0)
                        ent = -(pu * torch.log(pu + 1e-12)).sum()
                        # maximise entropy -> minimise its negative, floored at
                        # ln 2 so the term stops pushing once it is maximal
                        l_unc = torch.clamp(np.log(2.0) - ent, min=0.0)
                        losses.append(lam_unc * l_unc)
                        tot["unc"] += float(l_unc)

            loss = torch.stack(losses).sum() / len(arms)

            # L_ontology: the SAFE arm's margin must sit above the UNSAFE arm's
            # by at least `ont_margin`. Uses the TRUE labels even in the
            # shuffled control -- the ontology is not a label, and permuting
            # labels must not permute the mechanism the term encodes. In the
            # control run lam_ont is set to 0 by main() for exactly this
            # reason, so the control cannot be helped by it.
            # The required margin is now per-family and graph-derived. A
            # family the graph licenses no contraindication for gets 0.0, and
            # the term is silent for it -- which is the substantive difference
            # from the constant version, not the magnitude.
            req = (fam_margin.get(arms[0].get("family"), 0.0)
                   if fam_margin else ont_margin)
            if lam_ont and req > 0 and {"SAFE", "UNSAFE"} <= set(margins):
                hinge = torch.clamp(
                    req - (margins["SAFE"] - margins["UNSAFE"]), min=0.0)
                loss = loss + lam_ont * hinge
                tot["ont"] += float(hinge)
                n_ont += 1

            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        msg = (f"  epoch {ep+1}/{epochs}  L_causal={tot['causal']/n_items:.4f}"
               f"  L_KL={tot['kl']/n_items:.4f}")
        if lam_ont:
            msg += (f"  L_ont={tot['ont']/max(n_ont,1):.4f}"
                    f"(on {n_ont}/{len(pairs)} pairs)")
        if lam_unc:
            msg += f"  L_unc={tot['unc']/n_items:.4f}"
        print(msg + f"  train-decision-acc={n_ok/n_items:.3f}", flush=True)
    R.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/medcalc")
    ap.add_argument("--train_on", default="train",
                    help="split to train on. Use 'heldout_first' to train on "
                         "the first half of the QT family and test on the "
                         "second -- the configuration the probe result "
                         "predicts should WORK, as against renal which it "
                         "predicts should not.")
    ap.add_argument("--model_id", default="BioMistral/BioMistral-7B")
    ap.add_argument("--layer", type=int, default=30,
                    help="Chosen from the probe curve, not from test scores. "
                         "The mean-pooled probe peaks at layer 30 (0.976) and "
                         "the final-position probe still reads 0.753 there, so "
                         "the fact is present AND the path to the decision is "
                         "short. Layer 5 (the final-position peak, 0.859) was "
                         "tried first and the optimisation would not converge "
                         "-- 27 layers of processing sit between it and the "
                         "logits. That change was made on TRAINING behaviour, "
                         "before any test number was looked at.")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam_kl", type=float, default=0.01)
    ap.add_argument("--lam_ont", type=float, default=0.1,
                    help="weight of the pairwise ontology-monotonicity hinge")
    ap.add_argument("--lam_unc", type=float, default=0.1,
                    help="weight of the uncertainty term, applied to notes "
                         "whose decisive number has been redacted")
    ap.add_argument("--ont_margin", type=float, default=1.0,
                    help="BASE required margin. The graph scales it per family "
                         "(x1.25 where MED-RT attests the contraindication, "
                         "x1.0 where only curation does, x0 where the graph "
                         "licenses nothing).")
    ap.add_argument("--graph", default="data/umls/causal_graph.json",
                    help="causal knowledge graph consulted by L_ontology. "
                         "Pass '' to restore the pre-2026-09-01 constant-margin "
                         "behaviour, which reproduces the earlier results.")
    ap.add_argument("--shuffled_control", action="store_true")
    ap.add_argument("--out", default="results/constraint_layer.json")
    ap.add_argument("--save_adapter", default=None)
    args = ap.parse_args()

    d = Path(args.data)
    # The QT train/eval halves are now separate files written by
    # build_medcalc.py, so the split the adapter trains on is the same one
    # every other experiment sees, instead of being re-derived here. That is
    # what lets the constraint layer appear as a row of the ablation table:
    # `counterfactual_heldout.jsonl` no longer contains any note it was
    # trained on.
    if args.train_on == "heldout_first":
        tr = read_jsonl(d / "counterfactual_heldout_train.jsonl")
        ho = read_jsonl(d / "counterfactual_heldout.jsonl")
        te = read_jsonl(d / "counterfactual_test.jsonl")
        assert not ({r["pair_id"] for r in tr} & {r["pair_id"] for r in ho})
    else:
        # any split name may be trained on -- the synthetic benchmark has no
        # `train` file and uses `calib`, which is disjoint from test by
        # template and from heldout by family.
        tr = read_jsonl(d / f"counterfactual_{args.train_on}.jsonl")
        te = read_jsonl(d / "counterfactual_test.jsonl")
        ho = read_jsonl(d / "counterfactual_heldout.jsonl")
        assert not ({r["pair_id"] for r in tr} & {r["pair_id"] for r in te})
        assert not ({r["pair_id"] for r in tr} & {r["pair_id"] for r in ho})
    print(f"train {len(tr)} items / {len({r['pair_id'] for r in tr})} pairs")
    print(f"test  {len(te)} items / {len({r['pair_id'] for r in te})} pairs "
          f"(same family as train)")
    print(f"held  {len(ho)} items / {len({r['pair_id'] for r in ho})} pairs "
          f"({sorted({r['family'] for r in ho})})")
    print(f"train families: {sorted({r['family'] for r in tr})}")

    R = Runner(args.model_id, args.layer, args.rank, args.alpha)
    print(f"\nadapter: layer {args.layer}, rank {args.rank}, alpha "
          f"{args.alpha}, {sum(p.numel() for p in R.adapter.params()):,} "
          f"trainable parameters")

    print("\n=== base model, same decision rule ===")
    b_te = evaluate(R, te, False)
    b_ho = evaluate(R, ho, False)
    print(f"  test    acc={b_te[0]:.3f}  CC={b_te[1]:.3f}")
    print(f"  heldout acc={b_ho[0]:.3f}  CC={b_ho[1]:.3f}")

    # L_uncertainty is measured before training so the after-number means
    # something. Both are on notes whose decisive value has been redacted.
    u_te_base, n_red = redacted_entropy(R, te, False)
    u_ho_base, n_red_ho = redacted_entropy(R, ho, False)
    print(f"\n=== decision entropy on REDACTED notes, base model ===")
    print(f"  test    {u_te_base:.4f} nats over {n_red} items "
          f"(ln 2 = {np.log(2):.4f} is maximal)")
    print(f"  heldout {u_ho_base:.4f} nats over {n_red_ho} items")

    # The shuffled-label control must not be handed a term that encodes the
    # true mechanism, or it would no longer be a control.
    lam_ont = 0.0 if args.shuffled_control else args.lam_ont
    if args.shuffled_control and args.lam_ont:
        print("  [control] lam_ont forced to 0: the ontology term is defined "
              "on the true arm ordering and would leak the real labels back "
              "into a run whose whole purpose is to have none.")

    fams = {r.get("family") for r in tr if r.get("family")}
    graph_path = args.graph if (args.graph and Path(args.graph).is_file()) else None
    if args.graph and not graph_path:
        print(f"  [warn] --graph {args.graph} not found; L_ontology falls back "
              f"to the constant margin. Run "
              f"`python src/umls_grounding.py build` to create it.")
    fam_margin, ont_note = ontology_margins(graph_path, fams, args.ont_margin)
    print(f"\n=== L_ontology, required margin per family ===")
    print(ont_note)

    print(f"\n=== training on {len(tr)} items "
          f"({'SHUFFLED-LABEL CONTROL' if args.shuffled_control else 'real labels'}) ===")
    train(R, tr, args.epochs, args.lr, args.lam_kl,
          lam_ont=lam_ont, lam_unc=args.lam_unc,
          ont_margin=args.ont_margin,
          shuffle_labels=args.shuffled_control,
          fam_margin=(fam_margin or None))

    print("\n=== with constraint layer ===")
    a_te = evaluate(R, te, True)
    a_ho = evaluate(R, ho, True)
    print(f"  test    acc={a_te[0]:.3f}  CC={a_te[1]:.3f}   "
          f"(base {b_te[1]:.3f}, delta {a_te[1]-b_te[1]:+.3f})")
    print(f"  heldout acc={a_ho[0]:.3f}  CC={a_ho[1]:.3f}   "
          f"(base {b_ho[1]:.3f}, delta {a_ho[1]-b_ho[1]:+.3f})")

    u_te_ad, _ = redacted_entropy(R, te, True)
    u_ho_ad, _ = redacted_entropy(R, ho, True)
    print(f"\n=== decision entropy on REDACTED notes, with constraint layer ===")
    print(f"  test    {u_te_base:.4f} -> {u_te_ad:.4f} nats")
    print(f"  heldout {u_ho_base:.4f} -> {u_ho_ad:.4f} nats")

    rep = {"layer": args.layer, "rank": args.rank, "alpha": args.alpha,
           "epochs": args.epochs, "lr": args.lr, "lam_kl": args.lam_kl,
           "lam_ont": lam_ont, "lam_unc": args.lam_unc,
           "ont_margin": args.ont_margin,
           "ontology_graph": (str(graph_path) if graph_path else None),
           "ontology_required_margin": fam_margin,
           "train_on": args.train_on, "data": str(d),
           "n_train_pairs": len({r["pair_id"] for r in tr}),
           "redacted_entropy": {
               "n_test": n_red, "n_heldout": n_red_ho,
               "ln2": float(np.log(2)),
               "test_base": u_te_base, "test_adapted": u_te_ad,
               "heldout_base": u_ho_base, "heldout_adapted": u_ho_ad},
           "shuffled_control": args.shuffled_control,
           "base": {"test_acc": b_te[0], "test_cc": b_te[1],
                    "heldout_acc": b_ho[0], "heldout_cc": b_ho[1]},
           "adapted": {"test_acc": a_te[0], "test_cc": a_te[1],
                       "heldout_acc": a_ho[0], "heldout_cc": a_ho[1]}}
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}")
    if args.save_adapter:
        np.savez(args.save_adapter, **R.adapter.state())
        print(f"wrote {args.save_adapter}")


if __name__ == "__main__":
    main()
