"""
The Token-to-Concept Attribution Layer of proposal section 4.2, and its
faithfulness evaluation.

Section 4.2 promises a bridge:

    Token Generation -> Hidden Activation -> SAE Feature -> Probing Classifier
    -> Biomedical Concept -> Causal Graph Node -> Clinical Explanation

and says the layer "will be quantitatively evaluated for faithfulness using
standard sufficiency and comprehensiveness metrics". Before this file the
repository implemented the bridge as far as "Probe" and stopped: `sae.py`
produces features and scores them against concepts, `probe.py` decodes the
decisive fact from the residual stream, and nothing turned either into a
per-token statement about why THIS answer was produced for THIS note.

WHAT THE LAYER COMPUTES

For token t and concept c, at the layer the SAE dictionary was fitted on:

    a[t, c] = sum over features f assigned to concept c of  z_f(h_t) * w_f

`z_f(h_t)` is the sparse code of the residual stream at position t. `w_f` is
the feature's MEASURED decision weight, read from the knock-out excess already
on disk (`results/sae/*_fis.json`, `causal.per_feature[f].excess`) rather than
introduced as a new free parameter. Negative excess means the feature moved the
decision less than its firing-rate-matched random control, which is not
evidence of a causal role, so those weights are clipped to zero.

The token score is a[t] = sum_c a[t, c]; the concept the layer would show a
clinician is argmax_c sum_t a[t, c].

A NAMING HAZARD, READ BEFORE EDITING

`sufficiency` already means something else in this project: Aim 2 causal
sufficiency, i.e. feature INJECTION (`patching.py --mode sufficiency`,
`results/sufficiency_medcalc_*.json`). ERASER sufficiency is unrelated -- it is
a rationale-quality metric. This file therefore writes
`results/faithfulness_*.json` and names its keys `eraser_sufficiency` and
`eraser_comprehensiveness`. Never write `results/sufficiency_*.json` from here;
`make_summary.py` and `compare_rerun.py` both read that name as the Aim 2
quantity.

WHAT IS MEASURED AGAINST WHAT

Four attributors, so the layer is measured rather than asserted:

    sae_concept   the section 4.2 layer above
    grad_x_input  gradient of the decision log-probability times the input
                  embedding -- the standard cheap saliency baseline
    occlusion     delete one token, measure the drop. Exact, O(T) forwards per
                  item, so it runs on a subsample and serves as the upper bound
                  a cheap attributor is trying to reach
    random        uniform scores; the floor any of the above must clear

Faithfulness, with m(x) the model's probability of its own decision
renormalised over the two answer-token classes, and r the top-k% of tokens:

    comprehensiveness  = m(x) - m(x \\ r)     higher is better
    eraser_sufficiency = m(x) - m(r)         lower is better

Both are reported over k in {1, 5, 10, 20, 50}% as an AOPC, and both are
reported with the random-attributor value beside them, because on a model that
answers SAFE to nearly everything a rationale can look faithful by deleting
enough of the note to destabilise any answer.

CONCEPT POINTING is the metric this project actually needs, and it is stronger
than the ERASER pair: the fraction of items whose top-attributed concept is one
the family's decision genuinely turns on (creatinine/eGFR for the renal
families, QT for the QT family). Its baseline is picking uniformly among the
concepts the note mentions at all, so a layer that just fires on whatever is
present scores at chance.

EDITED-TOKEN PERCENTILE uses the benchmark's own guarantee. The two arms of a
pair differ in exactly one causal value, so the tokens that differ ARE the
decisive tokens, with no annotation. The percentile rank the attributor gives
them is ground truth for "did it point at the right token", available for every
pair whose arms tokenise to the same length.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sae as sae_mod
from lm_common import answer_token_ids, model_slug, wrap_prompt

# Concepts a family's decision genuinely turns on. The note states the raw
# measurement, not the derived score, so `creatinine` counts for a rule written
# on eGFR -- the model has to read creatinine and compute. `drug` is excluded
# everywhere: every note names the drug, so crediting it would make the metric
# trivially high.
DECISIVE_CONCEPTS = {
    "metformin_renal":            {"creatinine", "egfr"},
    "nitrofurantoin_renal":       {"creatinine", "egfr"},
    "nsaid_renal":                {"creatinine", "egfr"},
    "ondansetron_qt":             {"qt_interval", "heart_rate"},
    "warfarin_inr":               {"inr"},
    "spironolactone_hyperkalaemia": {"potassium"},
    "acei_pregnancy":             {"pregnancy"},
    "betablocker_asthma":         {"asthma"},
    "aspirin_reye":               {"age"},
    "statin_macrolide":           {"drug"},
}

K_GRID = [0.01, 0.05, 0.10, 0.20, 0.50]


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open()]


class Attributor:
    def __init__(self, model_id, sae_path, fis_path, layer=None,
                 dtype="bfloat16", max_input_tokens=1400):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch

        z = np.load(sae_path, allow_pickle=True)
        self.scale = float(z["scale"]) if "scale" in z.files else 1.0
        acts = np.load(str(z["acts"]), allow_pickle=True)
        self.layer = int(acts["layer"]) if layer is None else int(layer)
        self.concepts = [str(c) for c in acts["concepts"]]
        concept_source = (str(acts["concept_source"])
                          if "concept_source" in acts.files else "regex")
        # concept_labels() reads module-level globals in sae.py; bind the same
        # vocabulary the dictionary was scored against or the spans will not
        # line up with S_semantic.
        sae_mod.set_concept_vocabulary(concept_source)
        self.concept_source = concept_source

        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype),
            device_map="cuda").eval()
        # Freeze every weight. Nothing here trains, and `grad_x_input` needs a
        # gradient only with respect to the INPUT embeddings. Left unfrozen,
        # backward allocates a gradient buffer for all 7B parameters -- another
        # ~14.5 GiB on top of the resident weights -- and OOMs a 24 GiB card
        # before it reaches the embedding it actually wanted.
        for prm in self.model.parameters():
            prm.requires_grad_(False)
        self.max_input_tokens = max_input_tokens
        ids = answer_token_ids(self.tok)
        self.safe_ids = sorted(ids["SAFE"])
        self.unsafe_ids = sorted(ids["UNSAFE"])

        d_in = int(z["W_dec"].shape[1])
        self.sae = sae_mod.SAE(d_in, int(z["d_hidden"]), str(z["kind"]),
                               int(z["k"]), torch=torch)
        for k in ("W_enc", "W_dec", "b_enc", "b_dec", "theta"):
            getattr(self.sae, k).data = torch.tensor(z[k]).cuda()

        fis = json.loads(Path(fis_path).read_text())
        per_f = fis.get("causal", {}).get("per_feature", {})
        cidx = {c: i for i, c in enumerate(self.concepts)}
        n_con = len(self.concepts)
        d_hidden = int(z["d_hidden"])
        # [F, C] weight matrix: zero everywhere except the scored features,
        # each placed in its argmax concept column with its knock-out excess.
        W = np.zeros((d_hidden, n_con), dtype=np.float32)
        kept, dropped = 0, 0
        for f in fis["features"]:
            fid, con = int(f["feature"]), f.get("concept")
            if con not in cidx:
                continue
            exc = per_f.get(str(fid), {}).get("excess")
            if exc is None:
                continue
            if exc <= 0:
                dropped += 1        # no evidence over its matched control
                continue
            W[fid, cidx[con]] = float(exc)
            kept += 1
        self.W_fc = torch.tensor(W).cuda()
        # Per-concept feature counts. The top-N-by-S_semantic selection in
        # `sae.py score` is not balanced across concepts -- on the 25-feature
        # topk report, `age` supplies 12 features and `qt_interval` supplies
        # none with a positive excess. Summing over a concept's features then
        # measures the composition of the dictionary rather than the content of
        # the note, so the mean-normalised readout is reported beside the sum,
        # and the concepts the layer cannot express at all are named.
        self.n_per_concept = (W > 0).sum(0).astype(np.float32)
        self.W_fc_mean = torch.tensor(
            W / np.maximum(self.n_per_concept, 1.0)).cuda()
        self.expressible = {c for i, c in enumerate(self.concepts)
                            if self.n_per_concept[i] > 0}
        self.n_weighted = kept
        print(f"[attribution] layer {self.layer}, {kept} features carry a "
              f"positive knock-out excess ({dropped} clipped to zero), "
              f"concept vocabulary '{concept_source}' with {n_con} concepts")
        print(f"[attribution] expressible concepts: "
              f"{ {c: int(self.n_per_concept[i]) for i, c in enumerate(self.concepts) if self.n_per_concept[i] > 0} }")
        print(f"[attribution] NOT expressible (no positively weighted "
              f"feature): {sorted(set(self.concepts) - self.expressible)}")
        if kept == 0:
            raise RuntimeError(
                f"no feature in {fis_path} has a positive knock-out excess; "
                f"the attribution layer would be identically zero. Re-run "
                f"`sae.py score` with --causal_items -1 first.")

    # -- encoding -----------------------------------------------------------

    def encode(self, prompt):
        text = wrap_prompt(self.tok, prompt)
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True,
                       truncation=True, max_length=self.max_input_tokens)
        offs = enc.pop("offset_mapping")[0].tolist()
        return text, {k: v.cuda() for k, v in enc.items()}, offs

    def candidate_positions(self, text, offs, record):
        """
        Tokens the rationale may draw from: the clinical note only.

        The instruction header and the chat template are held fixed. Erasing
        them would measure how badly the model degrades without its
        instructions, which is not a statement about the explanation.
        """
        span = record.get("vignette") or ""
        a = text.find(span) if span else -1
        if a < 0:
            a = text.find("Case:")
            a = 0 if a < 0 else a + len("Case:")
            b = len(text)
        else:
            b = a + len(span)
        return [i for i, (s, e) in enumerate(offs) if e > a and s < b and e > s]

    # -- the decision readout ----------------------------------------------

    def _last_logits(self, batch_ids):
        torch = self.torch
        pad = self.tok.pad_token_id
        n = max(len(x) for x in batch_ids)
        ids = torch.full((len(batch_ids), n), pad, dtype=torch.long)
        att = torch.zeros((len(batch_ids), n), dtype=torch.long)
        for i, x in enumerate(batch_ids):          # left padding
            ids[i, n - len(x):] = torch.tensor(x)
            att[i, n - len(x):] = 1
        with torch.no_grad():
            out = self.model(input_ids=ids.cuda(), attention_mask=att.cuda())
        return out.logits[:, -1, :].float()

    def decision(self, batch_ids):
        """
        (p, label) per row. p is the model's probability of the decision it
        would actually give, renormalised over the two answer classes, so the
        rest of the vocabulary cannot drift the metric around.
        """
        torch = self.torch
        probs = torch.softmax(self._last_logits(batch_ids), dim=-1)
        ps = probs[:, self.safe_ids].sum(-1)
        pu = probs[:, self.unsafe_ids].sum(-1)
        tot = (ps + pu).clamp_min(1e-9)
        ps, pu = ps / tot, pu / tot
        lab = (pu > ps)
        return torch.where(lab, pu, ps).cpu().numpy(), lab.cpu().numpy()

    def held_prob(self, batch_ids, label):
        """Probability of a FIXED decision, for scoring perturbed inputs."""
        torch = self.torch
        probs = torch.softmax(self._last_logits(batch_ids), dim=-1)
        ps = probs[:, self.safe_ids].sum(-1)
        pu = probs[:, self.unsafe_ids].sum(-1)
        tot = (ps + pu).clamp_min(1e-9)
        return ((pu if label else ps) / tot).cpu().numpy()

    # -- attributors --------------------------------------------------------

    def sae_concept_scores(self, enc):
        """
        [T] token scores, plus [T, C] concept scores under both aggregations.

        Token scores come from the sum: a token's importance really is the
        total weighted feature activity on it. Only the CONCEPT readout needs
        the per-concept mean, because there the feature count is a property of
        the dictionary rather than of the note.
        """
        torch = self.torch
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
            h = out.hidden_states[self.layer][0].float() * self.scale
            z = self.sae.encode(h)                       # [T, F]
            a_sum = z @ self.W_fc                        # [T, C]
            a_mean = z @ self.W_fc_mean                  # [T, C]
        return (a_sum.sum(-1).cpu().numpy(),
                a_sum.cpu().numpy(), a_mean.cpu().numpy())

    def grad_x_input(self, enc, label):
        """
        d(decision log-odds)/d(input embedding) . input embedding.

        Two memory measures, both necessary rather than tuning. A 7B model with
        a ~1300-token clinical note does not fit a backward pass in the ~7 GiB
        left on a 24 GiB card once the weights are resident, and this OOMed
        before them:

        `logits_to_keep=1`   the lm_head is only needed at the decision
                             position; materialising [T, 32000] logits and
                             their graph is pure waste here.
        gradient checkpointing  recomputes layer activations in the backward
                             instead of storing 32 layers of them. Enabled only
                             around this call, because every other path in this
                             file is no_grad and would pay the recompute for
                             nothing.
        """
        torch = self.torch
        emb_layer = self.model.get_input_embeddings()
        e = emb_layer(enc["input_ids"]).detach().clone().requires_grad_(True)
        was_ckpt = getattr(self.model, "is_gradient_checkpointing", False)
        if not was_ckpt:
            self.model.gradient_checkpointing_enable()
        try:
            out = self.model(inputs_embeds=e,
                             attention_mask=enc.get("attention_mask"),
                             use_cache=False, logits_to_keep=1)
            lg = out.logits[0, -1, :].float()
            want = self.unsafe_ids if label else self.safe_ids
            other = self.safe_ids if label else self.unsafe_ids
            obj = torch.logsumexp(lg[want], 0) - torch.logsumexp(lg[other], 0)
            obj.backward()
            g = (e.grad[0].float() * e[0].detach().float()).sum(-1)
        finally:
            if not was_ckpt:
                self.model.gradient_checkpointing_disable()
            self.model.zero_grad(set_to_none=True)
        out = g.detach().cpu().numpy()
        del e, lg
        torch.cuda.empty_cache()
        return out

    def occlusion(self, ids, cand, label, clean_p, batch=16):
        s = np.zeros(len(ids), dtype=np.float32)
        for i in range(0, len(cand), batch):
            chunk = cand[i:i + batch]
            variants = [ids[:j] + ids[j + 1:] for j in chunk]
            p = self.held_prob(variants, label)
            for j, pj in zip(chunk, p):
                s[j] = clean_p - float(pj)
        return s


def _rationale(scores, cand, frac):
    """Top-frac% of the candidate positions by score, at least one token."""
    k = max(1, int(round(frac * len(cand))))
    order = sorted(cand, key=lambda i: -scores[i])
    return set(order[:k])


def faithfulness_for_item(A, ids, cand, scores, label, clean_p):
    """ERASER comprehensiveness and sufficiency over the k grid, one item."""
    comp_in, suff_in = [], []
    for frac in K_GRID:
        r = _rationale(scores, cand, frac)
        comp_in.append([t for i, t in enumerate(ids) if i not in r])
        keep = set(range(len(ids))) - set(cand) | r     # scaffolding + rationale
        suff_in.append([t for i, t in enumerate(ids) if i in keep])
    p = A.held_prob(comp_in + suff_in, label)
    comp = clean_p - p[:len(K_GRID)]
    suff = clean_p - p[len(K_GRID):]
    return comp, suff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sae", default="results/sae/sae_topk_L20.npz")
    ap.add_argument("--fis", default="results/sae/sae_topk_L20_fis.json")
    ap.add_argument("--model_id", default="BioMistral/BioMistral-7B")
    ap.add_argument("--layer", type=int, default=None,
                    help="defaults to the layer recorded in the acts npz")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 (default) means every record")
    ap.add_argument("--occlusion_items", type=int, default=30,
                    help="occlusion is O(T) forwards per item; 0 disables it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    records = read_jsonl(args.data)
    if args.limit:
        records = records[:args.limit]
    A = Attributor(args.model_id, args.sae, args.fis, args.layer)
    rng = np.random.default_rng(args.seed)

    names = ["sae_concept", "grad_x_input", "random"]
    if args.occlusion_items:
        names.append("occlusion")
    comp = {n: [] for n in names}
    suff = {n: [] for n in names}
    point_hit = {"sum": [], "mean": []}
    point_hit_expr = {"sum": [], "mean": []}
    point_base = []
    n_inexpressible = 0
    edited_pct = {n: [] for n in names}
    per_item = []
    n_occ = 0
    skipped = defaultdict(int)

    # token ids per record, kept for the paired edited-token check
    enc_cache = {}

    for ri, r in enumerate(records):
        text, enc, offs = A.encode(r["prompt"])
        ids = enc["input_ids"][0].tolist()
        cand = A.candidate_positions(text, offs, r)
        if len(cand) < 20:
            skipped["note span too short to draw a rationale from"] += 1
            continue
        clean_p, clean_lab = A.decision([ids])
        clean_p, clean_lab = float(clean_p[0]), bool(clean_lab[0])

        S = {}
        S["sae_concept"], a_sum, a_mean = A.sae_concept_scores(enc)
        S["grad_x_input"] = A.grad_x_input(enc, clean_lab)
        S["random"] = rng.standard_normal(len(ids)).astype(np.float32)
        if args.occlusion_items and n_occ < args.occlusion_items:
            S["occlusion"] = A.occlusion(ids, cand, clean_lab, clean_p)
            n_occ += 1

        for n in names:
            if n not in S:
                continue
            c, s = faithfulness_for_item(A, ids, cand, S[n], clean_lab, clean_p)
            comp[n].append(c)
            suff[n].append(s)

        # -- concept pointing -----------------------------------------------
        want = DECISIVE_CONCEPTS.get(r.get("family"), set())
        if want:
            lab = sae_mod.concept_labels(text, offs)
            present = {A.concepts[i] for i in range(len(A.concepts))
                       if lab[cand, i].any()}
            if present:
                for agg, arr in (("sum", a_sum), ("mean", a_mean)):
                    tot = arr[cand].sum(0)
                    top = (A.concepts[int(np.argmax(tot))]
                           if tot.max() > 0 else None)
                    point_hit[agg].append(1.0 if top in want else 0.0)
                    # A layer with no feature for the decisive concept cannot
                    # point at it. That is a vocabulary gap, not an attribution
                    # failure, so it is counted separately rather than folded
                    # into the headline as if the layer had tried and missed.
                    if want & A.expressible:
                        point_hit_expr[agg].append(
                            1.0 if top in want else 0.0)
                point_base.append(len(want & present) / len(present))
                if not (want & A.expressible):
                    n_inexpressible += 1

        enc_cache[r["id"]] = (ids, cand, S)
        per_item.append({"id": r["id"], "family": r.get("family"),
                         "clean_p": clean_p,
                         "decision": "UNSAFE" if clean_lab else "SAFE",
                         "n_tokens": len(ids), "n_candidates": len(cand)})
        if (ri + 1) % 20 == 0:
            print(f"  {ri + 1}/{len(records)} items")

    # -- edited-token percentile, from the pairs themselves ------------------
    by_pair = defaultdict(dict)
    for r in records:
        by_pair[r["pair_id"]][r["arm"]] = r["id"]
    for pid, arms in by_pair.items():
        if set(arms) != {"safe", "unsafe"}:
            continue                       # control pairs have no causal edit
        ka, kb = arms["safe"], arms["unsafe"]
        if ka not in enc_cache or kb not in enc_cache:
            continue
        ia, ca, Sa = enc_cache[ka]
        ib, _, _ = enc_cache[kb]
        if len(ia) != len(ib):
            skipped["arms tokenise to different lengths"] += 1
            continue
        diff = [i for i in ca if ia[i] != ib[i]]
        if not diff:
            continue
        for n, sc in Sa.items():
            v = np.asarray([sc[i] for i in ca])
            order = np.argsort(np.argsort(v))          # 0 = lowest score
            pos = {c: k for k, c in enumerate(ca)}
            edited_pct[n].append(float(np.mean(
                [order[pos[i]] / max(1, len(ca) - 1) for i in diff])))

    def block(d):
        return {n: ([float(np.mean(np.asarray(v)[:, i])) for i in
                     range(len(K_GRID))] if v else None)
                for n, v in d.items()}

    out = {
        "layer": A.layer,
        "model_id": args.model_id,
        "model_slug": model_slug(args.model_id),
        "sae": args.sae,
        "fis": args.fis,
        "data": args.data,
        "concept_source": A.concept_source,
        "n_items": len(per_item),
        "n_occlusion_items": n_occ,
        "n_weighted_features": A.n_weighted,
        "k_grid": K_GRID,
        "eraser_comprehensiveness": block(comp),
        "eraser_sufficiency": block(suff),
        "aopc_comprehensiveness": {n: (float(np.mean(np.asarray(v))) if v
                                       else None) for n, v in comp.items()},
        "aopc_sufficiency": {n: (float(np.mean(np.asarray(v))) if v else None)
                             for n, v in suff.items()},
        "concept_pointing": {
            "accuracy_sum": (float(np.mean(point_hit["sum"]))
                             if point_hit["sum"] else None),
            "accuracy_mean": (float(np.mean(point_hit["mean"]))
                              if point_hit["mean"] else None),
            "accuracy_sum_expressible": (float(np.mean(point_hit_expr["sum"]))
                                         if point_hit_expr["sum"] else None),
            "accuracy_mean_expressible": (float(np.mean(point_hit_expr["mean"]))
                                          if point_hit_expr["mean"] else None),
            "random_baseline": (float(np.mean(point_base))
                                if point_base else None),
            "n": len(point_hit["sum"]),
            "n_expressible": len(point_hit_expr["sum"]),
            "n_items_whose_decisive_concept_the_layer_cannot_express":
                n_inexpressible,
            "expressible_concepts": sorted(A.expressible),
            "features_per_concept": {c: int(A.n_per_concept[i]) for i, c
                                     in enumerate(A.concepts)},
        },
        "edited_token_percentile": {
            n: (float(np.mean(v)) if v else None) for n, v in edited_pct.items()},
        "n_edited_pairs": len(edited_pct["sae_concept"]),
        "skipped": dict(skipped),
        "note": ("`eraser_sufficiency` is the ERASER rationale metric and is "
                 "NOT the Aim 2 causal sufficiency in "
                 "results/sufficiency_medcalc_*.json. Lower is better here; "
                 "there, larger |excess| would be evidence of a causal role."),
        "per_item": per_item,
    }
    out_path = args.out or (
        f"results/faithfulness_{Path(args.data).parent.name}_"
        f"{Path(args.data).stem.replace('counterfactual_', '')}_"
        f"{model_slug(args.model_id)}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    for n in names:
        print(f"  {n:14s} AOPC comp={out['aopc_comprehensiveness'][n]} "
              f"suff={out['aopc_sufficiency'][n]} "
              f"edited-token pct={out['edited_token_percentile'][n]}")
    print(f"  concept pointing {out['concept_pointing']}")


if __name__ == "__main__":
    main()
