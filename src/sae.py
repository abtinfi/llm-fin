"""
Aim 1: sparse autoencoders on the residual stream, and the Feature
Interpretability Score.

    x_hat = W_dec . ReLU(W_enc . x + b_enc) + b_dec          (proposal Eq. 1)
    FIS   = a*S_semantic + b*S_causal + c*S_human            (proposal 4.4)

WHY THIS FILE EXISTS
--------------------
Aim 1 is the first aim of the proposal and nothing in this repository
implemented it. `src/probe.py` asks a strictly weaker question -- "is the fact
linearly decodable from the residual stream" -- which is a supervised readout,
not a decomposition. A probe cannot produce features, cannot say what a
direction responds to, and cannot be knocked out one unit at a time. Every
statement about "discovering sparse features" needs an SAE, so here is one.

WHAT IS AND IS NOT CLAIMED
--------------------------
This is a **pilot-scale** SAE. Production SAEs are trained on hundreds of
millions of tokens with 8-64x expansion. This one sees a few hundred thousand
tokens from one narrow corpus, at 2-4x expansion, because that is the corpus
this project has. The consequences are stated rather than hidden:

  - features here describe THIS distribution of clinical notes, not the model
    in general;
  - a feature that fails to appear is not evidence that the model lacks it;
  - the dead-feature count and reconstruction error are reported for every run
    so an under-trained dictionary is visible instead of implied.

Both architectures named in the proposal's fallback are implemented: `topk`
(the default) and `jumprelu`. The proposal says to fall back if fewer than 30%
of features pass expert validation. There are no experts in this pipeline, so
S_human cannot be measured and its weight is FORCED TO ZERO and reported as
such -- an unmeasured term silently weighted at 1/3 would make the FIS look
like an expert-validated number when no expert has seen it.

    python src/sae.py collect --data data/medcalc --split test --layer 20
    python src/sae.py train   --acts results/sae/acts_medcalc_test_L20.npz
    python src/sae.py score   --sae results/sae/sae_medcalc_test_L20.npz

THE CONCEPT VOCABULARY (changed 2026-09-02)
-------------------------------------------
`S_semantic` is only as strong as the thing it matches against. Until
2026-09-02 that was 11 hand-written regexes, so "maps a feature to a biomedical
concept" meant "matches my regex". The default is now the CUI-anchored
groundings from the causal knowledge graph (`--concepts umls`); the regexes are
kept as `LEGACY_CONCEPT_PATTERNS` and reachable with `--concepts regex` so the
earlier numbers stay reproducible.

Two things must travel with any UMLS-grounded number:

  - The matcher is a **union of UMLS atoms and a curated lexical layer**, not
    UMLS alone. UMLS atoms by themselves collapse recall on the surface forms
    notes actually use (eGFR 16->0 hits, pregnancy 20->2, potassium 4->0),
    because UMLS is terminology-normalised and notes write "eGFR"/"pregnant".
    `concept_provenance` in the FIS report gives the split per concept.
  - `age` has **no CUI at all** and is a pure lexical fallback, and
    `qt_interval` matches nothing on UMLS atoms alone. Any feature whose
    concept is one of those is CUI-anchored at best and must not be described
    as UMLS-matched. Run `python src/umls_grounding.py audit` for the numbers.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lm_common import (answer_token_ids, producer_module,
                       wrap_prompt)

# ---------------------------------------------------------------------------
# Concept vocabulary. A feature is "semantic" to the extent that it fires on
# the tokens of exactly one of these and not on the others. Spans are found in
# the prompt text, so the labels are derived from the text the model reads,
# never from the dataset's structured facts.
# ---------------------------------------------------------------------------
NUM = r"\d+(?:\.\d+)?"
LEGACY_CONCEPT_PATTERNS = {
    "creatinine": re.compile(
        r"creatin(?:ine|e)\b[^.\n;]{0,30}?" + NUM +
        r"\s*(?:mg\s*/\s*d[lL]|[µu]mol\s*/\s*L)", re.I),
    "qt_interval": re.compile(
        r"QT\s*(?:interval|duration)?[^.\n;]{0,25}?" + NUM +
        r"\s*(?:ms|msec|milliseconds)\b", re.I),
    "egfr": re.compile(r"eGFR\s*(?:is|of)?\s*" + NUM, re.I),
    "heart_rate": re.compile(
        r"(?:heart rate|pulse)[^.\n;]{0,25}?" + NUM, re.I),
    "potassium": re.compile(r"potassium (?:is |of )?" + NUM, re.I),
    "inr": re.compile(r"INR (?:is |of )?" + NUM, re.I),
    "age": re.compile(NUM + r"[\s-]*(?:year|yr)s?[\s-]*old", re.I),
    "drug": re.compile(
        r"\b(metformin|ibuprofen|lisinopril|propranolol|aspirin|"
        r"spironolactone|warfarin|simvastatin|nitrofurantoin|ondansetron)\b",
        re.I),
    "renal_disease": re.compile(
        r"\b(renal (?:failure|impairment|insufficiency)|kidney (?:disease|"
        r"injury)|dialysis|CKD)\b", re.I),
    "pregnancy": re.compile(
        r"\b(pregnan\w*|hCG|menstrual)\b", re.I),
    "asthma": re.compile(r"\b(asthma|bronchospasm|salbutamol|inhaler)\b", re.I),
}
# The vocabulary actually used is bound at run time by
# `set_concept_vocabulary()`. It defaults to the UMLS groundings; the legacy
# regexes above stay reachable via `--concepts regex` so the pre-2026-09-02
# numbers can still be reproduced exactly.
CONCEPT_PATTERNS = dict(LEGACY_CONCEPT_PATTERNS)
CONCEPTS = list(CONCEPT_PATTERNS)
CONCEPT_SOURCE = "regex"
CONCEPT_PROVENANCE = {}

DEFAULT_GRAPH = "data/umls/causal_graph.json"


def set_concept_vocabulary(source="umls", graph_path=DEFAULT_GRAPH):
    """
    Bind the concept vocabulary the whole file scores against.

    `source="umls"` loads the CUI-anchored groundings from the causal knowledge
    graph; `source="regex"` keeps the hand-written patterns. The graph is read
    from disk and its UMLS cache is warm, so this makes no network calls.

    WHY THIS IS NOT A COSMETIC SWAP: `cmd_collect` keeps every token a concept
    fires on, so the vocabulary decides which activations enter the dictionary.
    Changing it changes X, the trained SAE and every S_semantic downstream. It
    is a full collect -> train -> score re-run, never a re-score.

    WHAT A MATCH MEANS AFTERWARDS: the matcher is a provenance-tracked union of
    UMLS atoms and a curated lexical layer, because UMLS atoms alone collapse
    recall on the surface forms clinical notes actually use (eGFR 16->0 hits,
    pregnancy 20->2, potassium 4->0). `CONCEPT_PROVENANCE` records the split per
    concept so no concept is called "UMLS-matched" without the number that
    qualifies it.
    """
    global CONCEPT_PATTERNS, CONCEPTS, CONCEPT_SOURCE, CONCEPT_PROVENANCE
    if source == "regex":
        CONCEPT_PATTERNS = dict(LEGACY_CONCEPT_PATTERNS)
        CONCEPTS = list(CONCEPT_PATTERNS)
        CONCEPT_SOURCE = "regex"
        CONCEPT_PROVENANCE = {}
        return CONCEPT_PATTERNS

    from umls_grounding import CausalKnowledgeGraph
    gp = Path(graph_path)
    if not gp.exists():
        raise SystemExit(
            f"{gp} not found. Build it first:\n"
            f"  python src/umls_grounding.py build --out data/umls\n"
            f"or score the legacy vocabulary explicitly with --concepts regex.")
    graph = CausalKnowledgeGraph.load(gp)
    pats = graph.concept_patterns()
    if not pats:
        raise SystemExit(f"{gp} carries no concept groundings.")

    # Keep the legacy ordering so a concept's column index is stable across the
    # two vocabularies and the two runs stay column-comparable.
    order = [c for c in LEGACY_CONCEPT_PATTERNS if c in pats]
    order += [c for c in pats if c not in order]
    CONCEPT_PATTERNS = {c: pats[c] for c in order}
    CONCEPTS = list(CONCEPT_PATTERNS)
    CONCEPT_SOURCE = "umls"
    CONCEPT_PROVENANCE = {}
    for slug in CONCEPTS:
        g = graph.groundings.get(slug)
        if g is None:
            continue
        CONCEPT_PROVENANCE[slug] = {
            "cui": g.cui, "preferred_name": g.preferred_name,
            "n_umls_terms": g.n_umls_terms,
            "n_curated_terms": g.n_curated_terms,
            "numeric_tail": bool(g.numeric), "source": g.source}
    missing = [c for c in LEGACY_CONCEPT_PATTERNS if c not in CONCEPT_PATTERNS]
    if missing:
        print(f"  WARNING: no grounding for {missing}; "
              f"these concepts are dropped from the vocabulary")
    return CONCEPT_PATTERNS


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open()]


def concept_labels(text, offsets):
    """
    [T, C] boolean matrix: token t overlaps a span of concept c.

    Offsets come from the tokenizer, so a concept spanning several tokens marks
    all of them. A token belonging to two concepts is marked for both -- the
    scoring treats concepts one at a time and never assumes exclusivity.
    """
    lab = np.zeros((len(offsets), len(CONCEPTS)), dtype=bool)
    for ci, name in enumerate(CONCEPTS):
        for m in CONCEPT_PATTERNS[name].finditer(text):
            a, b = m.span()
            for ti, (s, e) in enumerate(offsets):
                if e > a and s < b and e > s:
                    lab[ti, ci] = True
    return lab


# ---------------------------------------------------------------------------
# 1. collect
# ---------------------------------------------------------------------------

def cmd_collect(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_concept_vocabulary(args.concepts, args.graph)
    print(f"  concept vocabulary: {CONCEPT_SOURCE} "
          f"({len(CONCEPTS)} concepts)")
    if CONCEPT_PROVENANCE:
        for slug, pr in CONCEPT_PROVENANCE.items():
            print(f"    {slug:14s} {pr['cui']:10s} "
                  f"umls_atoms={pr['n_umls_terms']:3d} "
                  f"curated={pr['n_curated_terms']:3d}"
                  + ("  +numeric_tail" if pr["numeric_tail"] else ""))

    records = read_jsonl(Path(args.data) / f"counterfactual_{args.split}.jsonl")
    if args.limit:
        records = records[:args.limit]
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    def wrap(p):
        return wrap_prompt(tok, p)

    X, L, ids, budget = [], [], [], args.max_tokens
    for i, r in enumerate(records):
        text = wrap(r["prompt"])
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True,
                  truncation=True, max_length=args.max_len)
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[args.layer][0]              # [T, D]
        lab = concept_labels(text, offs)

        # Keep every token that carries a concept, plus a random sample of the
        # rest. Keeping only concept tokens would train the dictionary on a
        # distribution the model never sees and make every feature look
        # selective; keeping everything blows the budget on padding-like
        # boilerplate that is identical across items.
        keep = np.where(lab.any(axis=1))[0]
        rng = np.random.default_rng(getattr(args, "seed", 0) * 100003
                                    + 1000 + i)
        others = np.setdiff1d(np.arange(h.shape[0]), keep)
        n_other = min(len(others),
                      max(8, int(args.other_mult * len(keep))))
        keep = np.concatenate([keep, rng.choice(others, n_other,
                                                replace=False)])
        keep = keep[keep < h.shape[0]]
        if len(keep) > budget:
            keep = keep[:budget]
        X.append(h[keep].float().cpu().numpy().astype(np.float16))
        L.append(lab[keep])
        ids.extend([r["id"]] * len(keep))
        budget -= len(keep)
        if budget <= 0:
            print(f"  token budget reached at item {i+1}/{len(records)}")
            break
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(records)} items, "
                  f"{args.max_tokens - budget} tokens", flush=True)

    X = np.concatenate(X)
    L = np.concatenate(L)
    out = Path(args.out or f"results/sae/acts_{Path(args.data).name}_"
                           f"{args.split}_L{args.layer}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, labels=L, concepts=np.array(CONCEPTS),
                        ids=np.array(ids), layer=args.layer,
                        data=str(args.data), split=args.split,
                        concept_source=CONCEPT_SOURCE,
                        concept_provenance=json.dumps(CONCEPT_PROVENANCE))
    print(f"wrote {out}: X{X.shape} labels{L.shape}")
    print("  concept token counts: " +
          ", ".join(f"{c}={int(L[:, i].sum())}"
                    for i, c in enumerate(CONCEPTS)))


# ---------------------------------------------------------------------------
# 2. train
# ---------------------------------------------------------------------------

class SAE:
    """
    Proposal Eq. (1) with the two architectures its fallback names.

    `topk`     keeps the k largest pre-activations and zeroes the rest. Sparsity
               is exact and set by hand, so L0 cannot drift during training and
               there is no L1 coefficient to tune.
    `jumprelu` keeps pre-activations above a learned per-feature threshold, with
               a straight-through estimator for the step. Sparsity is learned,
               which is the honest comparison for the fallback the proposal
               describes.

    The decoder is unit-norm per feature after every step: without it the model
    can shrink activations and grow decoder columns to fake sparsity.
    """

    def __init__(self, d_in, d_hidden, kind="topk", k=32, torch=None,
                 device="cuda", seed=0):
        self.torch = torch
        self.kind, self.k = kind, k
        g = torch.Generator(device="cpu").manual_seed(seed)
        W = torch.randn(d_hidden, d_in, generator=g) / np.sqrt(d_in)
        self.W_dec = (W / W.norm(dim=1, keepdim=True)).to(device).requires_grad_(True)
        self.W_enc = self.W_dec.detach().clone().T.contiguous().to(device).requires_grad_(True)
        self.b_enc = torch.zeros(d_hidden, device=device).requires_grad_(True)
        self.b_dec = torch.zeros(d_in, device=device).requires_grad_(True)
        self.theta = (torch.full((d_hidden,), -3.0, device=device)
                      .requires_grad_(True))      # log-threshold for jumprelu

    def params(self):
        p = [self.W_enc, self.W_dec, self.b_enc, self.b_dec]
        return p + [self.theta] if self.kind == "jumprelu" else p

    def encode(self, x):
        torch = self.torch
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        if self.kind == "topk":
            v, i = torch.topk(pre, self.k, dim=-1)
            z = torch.zeros_like(pre).scatter_(-1, i, torch.relu(v))
            return z
        thr = torch.exp(self.theta)
        act = torch.relu(pre)
        gate = (pre > thr).float()
        # straight-through: gradient flows as if the gate were the identity
        gate = gate + (torch.sigmoid((pre - thr) * 10.0)
                       - torch.sigmoid((pre - thr) * 10.0).detach())
        return act * gate

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def __call__(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def normalise(self):
        with self.torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True) + 1e-8)

    def state(self):
        return {k: v.detach().cpu().numpy() for k, v in
                (("W_enc", self.W_enc), ("W_dec", self.W_dec),
                 ("b_enc", self.b_enc), ("b_dec", self.b_dec),
                 ("theta", self.theta))}


def cmd_train(args):
    import torch
    z = np.load(args.acts, allow_pickle=True)
    X = torch.tensor(z["X"], dtype=torch.float32)
    n, d = X.shape

    # Input scaling. Residual-stream norms in this model grow by ~240x with
    # depth, so raw activations at a middle layer have norms in the tens and a
    # sum-of-squares reconstruction loss in the thousands. The first attempt
    # diverged to NaN on the first optimiser step for exactly this reason, and
    # -- worse -- still printed a plausible-looking FVU and a 100% dead-feature
    # count, which is the failure mode that gets written up as a finding.
    # Rescaling so E||x|| = sqrt(d) is the standard SAE convention and makes
    # the learning rate mean the same thing at any layer. The factor is stored
    # so downstream code can undo it.
    scale = float(np.sqrt(d) / X.norm(dim=1).mean())
    X = X * scale
    print(f"input scale {scale:.4f} -> mean ||x|| = "
          f"{float(X.norm(dim=1).mean()):.2f} (sqrt(d) = {np.sqrt(d):.1f})")
    d_hidden = args.expansion * d
    print(f"{n} tokens x {d} dims -> {d_hidden} features "
          f"({args.kind}, k={args.k})")

    # split so reconstruction is reported on tokens the SAE never fitted
    g = torch.Generator().manual_seed(getattr(args, "seed", 0))
    perm = torch.randperm(n, generator=g)
    # cap the validation split at a fifth of the data. The previous
    # `max(1024, n//10)` took MORE rows than existed on a small collection,
    # leaving the training set empty; the loop then averaged over zero batches,
    # printed `nan`, and still wrote an SAE. Anything trained on nothing must
    # fail loudly, so the size is clamped and the guard below is explicit.
    if n < 64:
        raise SystemExit(f"only {n} activation vectors -- too few to train")
    n_val = max(1, min(n // 5, 8192))
    val, tr = X[perm[:n_val]].cuda(), X[perm[n_val:]]
    print(f"train {tr.shape[0]} / val {n_val} tokens")

    sae = SAE(d, d_hidden, args.kind, args.k, torch=torch,
              seed=getattr(args, "seed", 0))
    opt = torch.optim.Adam(sae.params(), lr=args.lr)
    ntr = tr.shape[0]
    steps_per_epoch = max(1, ntr // args.batch)

    for ep in range(args.epochs):
        idx = torch.randperm(ntr, generator=g)
        tot = 0.0
        for s in range(steps_per_epoch):
            xb = tr[idx[s * args.batch:(s + 1) * args.batch]].cuda()
            xh, zb = sae(xb)
            # mean over dimensions, not sum: keeps the loss scale independent
            # of the model's hidden size
            loss = ((xh - xb) ** 2).mean()
            if args.kind == "jumprelu":
                loss = loss + args.l1 * zb.abs().sum(-1).mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sae.normalise()
            tot += float(loss.detach())
        with torch.no_grad():
            xh, zv = sae(val)
            fvu = float(((xh - val) ** 2).sum() /
                        ((val - val.mean(0)) ** 2).sum())
            l0 = float((zv > 0).float().sum(-1).mean())
            # a feature is dead if it never fires on ANY held-out token
            dead = int(((zv > 0).sum(0) == 0).sum())
        print(f"  epoch {ep+1}/{args.epochs}  train_mse={tot/steps_per_epoch:.5f}"
              f"  val_FVU={fvu:.4f}  val_L0={l0:.1f}  dead={dead}", flush=True)
        if not np.isfinite(tot):
            raise SystemExit("training diverged (non-finite loss) -- refusing "
                             "to write an SAE whose scores would be noise")
    out = Path(args.out or str(args.acts).replace("acts_", "sae_"))
    np.savez_compressed(out, **sae.state(), kind=args.kind, k=args.k,
                        acts=str(args.acts), fvu=fvu, l0=l0, dead=dead,
                        d_hidden=d_hidden, scale=scale)
    print(f"wrote {out}")
    print(f"  final val FVU={fvu:.4f}  L0={l0:.1f}  dead features={dead}"
          f"/{d_hidden} ({dead/d_hidden:.1%})")


# ---------------------------------------------------------------------------
# 3. score: FIS
# ---------------------------------------------------------------------------

def semantic_scores_from_counts(tp, n_active, n_pos):
    """
    S_semantic per feature: best F1 of "this feature is active" against "this
    token belongs to concept c", maximised over concepts.

    F1 rather than a mean-activation difference because the question is whether
    the feature SELECTS the concept: a feature that fires on the creatinine
    token and on a third of everything else is not a creatinine feature, and
    only a metric with a precision term says so.

    Takes accumulated counts instead of the full activation matrix. Scoring a
    150k-token collection against a 16k-feature dictionary materialises 9.8 GB
    of float32 if the matrix is built at once, which is exactly how the first
    attempt died on a 24 GB card that also had the model resident. The counts
    are all the F1 needs, and they stream.
    """
    n_feat, n_con = tp.shape
    best = np.zeros(n_feat)
    which = np.full(n_feat, -1)
    for c in range(n_con):
        if n_pos[c] == 0:
            continue
        tp_c = tp[:, c]
        fp = n_active - tp_c
        fn = n_pos[c] - tp_c
        f1 = np.where(tp_c > 0, 2 * tp_c / np.maximum(2 * tp_c + fp + fn, 1),
                      0.0)
        upd = f1 > best
        best[upd], which[upd] = f1[upd], c
    return best, which


def cmd_score(args):
    import torch
    sae_z = np.load(args.sae, allow_pickle=True)
    acts = np.load(str(sae_z["acts"]), allow_pickle=True)
    scale = float(sae_z["scale"]) if "scale" in sae_z.files else 1.0
    X = torch.tensor(acts["X"], dtype=torch.float32)      # stays on CPU
    labels = acts["labels"]
    concepts = [str(c) for c in acts["concepts"]]
    concept_source = (str(acts["concept_source"])
                      if "concept_source" in acts.files else "regex")
    concept_provenance = (json.loads(str(acts["concept_provenance"]))
                          if "concept_provenance" in acts.files else {})
    d_hidden = int(sae_z["d_hidden"])

    sae = SAE(X.shape[1], d_hidden, str(sae_z["kind"]),
              int(sae_z["k"]), torch=torch)
    for k in ("W_enc", "W_dec", "b_enc", "b_dec", "theta"):
        getattr(sae, k).data = torch.tensor(sae_z[k]).cuda()

    # Stream the collection through the encoder, accumulating only the counts
    # the F1 needs: [features x concepts] true positives, plus per-feature
    # firing counts. Peak memory is one chunk, not the whole matrix.
    n_con = labels.shape[1]
    tp = np.zeros((d_hidden, n_con), dtype=np.int64)
    n_active = np.zeros(d_hidden, dtype=np.int64)
    CH = 4096
    with torch.no_grad():
        for i in range(0, X.shape[0], CH):
            xb = X[i:i + CH].cuda() * scale
            act = (sae.encode(xb) > 0)
            n_active += act.sum(0).cpu().numpy().astype(np.int64)
            yb = torch.tensor(labels[i:i + CH]).cuda()
            tp += (act.float().T @ yb.float()).cpu().numpy().astype(np.int64)
            del xb, act, yb
    torch.cuda.empty_cache()
    n_pos = labels.sum(0)

    s_sem, which = semantic_scores_from_counts(tp, n_active, n_pos)
    order = np.argsort(-s_sem)
    top = order[:args.top]
    Z_active = n_active

    print(f"\ntop {args.top} features by S_semantic:")
    for f in top:
        print(f"  #{f:6d}  S_sem={s_sem[f]:.3f}  concept="
              f"{concepts[which[f]] if which[f] >= 0 else '-'}  "
              f"fires on {int(Z_active[f])}/{X.shape[0]} tokens")

    # Provenance: an artifact that does not name the model it was produced
    # with cannot be audited. Its absence is how an adapter trained on the
    # wrong model went unnoticed until 2026-09-08 -- every model here is
    # 4096-dimensional, so a mismatch loads cleanly and looks plausible.
    report = {"model_id": args.model_id,
              "sae": str(args.sae), "kind": str(sae_z["kind"]),
              "fvu": float(sae_z["fvu"]), "l0": float(sae_z["l0"]),
              "dead": int(sae_z["dead"]), "d_hidden": int(sae_z["d_hidden"]),
              "concepts": concepts,
              "concept_source": concept_source,
              "concept_provenance": concept_provenance,
              "concept_note": (
                  "S_semantic is measured against a UMLS-grounded vocabulary: "
                  "each concept is anchored to a CUI and matched on a "
                  "provenance-tracked union of UMLS atoms and a curated "
                  "lexical layer, plus a regex numeric tail where the concept "
                  "is a measurement (UMLS cannot express '2.1 mg/dL'). "
                  "`concept_provenance` gives the atom-vs-curated split per "
                  "concept; a concept whose n_umls_terms is small rests mostly "
                  "on curation and must not be reported as UMLS-matched."
                  if concept_source == "umls" else
                  "S_semantic is measured against 11 hand-written regexes. "
                  "'Biomedical concept' here means 'matches my regex', not "
                  "'selects a UMLS concept'."),
              "features": [{"feature": int(f), "s_semantic": float(s_sem[f]),
                            "concept": (concepts[which[f]] if which[f] >= 0
                                        else None),
                            "n_active": int(Z_active[f])}
                           for f in top]}

    # ---- S_causal: knock the feature out of the forward pass -------------
    if args.causal_items:
        report["causal"] = causal_knockout(args, sae, top, report, scale,
                                           n_active=n_active)

    # ---- FIS -------------------------------------------------------------
    a, b = args.alpha_sem, args.beta_causal
    for feat in report["features"]:
        sc = feat.get("s_causal", 0.0)
        feat["fis"] = a * feat["s_semantic"] + b * sc
    report["fis_weights"] = {
        "alpha_semantic": a, "beta_causal": b,
        "gamma_human": 0.0,
        "gamma_note": ("S_human is NOT measured: this pipeline has no expert "
                       "annotators. Its weight is forced to zero and the FIS "
                       "reported here is therefore a two-term score. The "
                       "proposal's 30%-expert-validation fallback criterion "
                       "cannot be evaluated without them.")}

    out = Path(args.out or str(args.sae).replace(".npz", "_fis.json"))
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


def matched_control_features(target, n_active, n_controls=5, pool_k=50,
                             rng=None):
    """
    Control features for a knock-out, MATCHED on firing rate.

    Returns `n_controls` feature indices drawn from the `pool_k` features whose
    firing rate is closest to `target`'s in LOG space, excluding dead features
    and `target` itself.

    WHY MATCHING IS NOT OPTIONAL (defect S1, fixed 2026-09-08). This used to be
    `rng.integers(0, n_feat)` -- uniform over the whole dictionary -- while the
    docstring claimed a control "of similar firing rate". 42% of the topk
    dictionary and 67% of the jumprelu one are DEAD, so a uniform draw usually
    knocked out a feature that never fires, measured exactly zero, and left
    `excess = eff - ctrl` equal to the raw effect. It is visible in the
    artifacts that rule produced: 6 of the 12 top features sampled from
    `results/sae/sae_topk_L20_fis.json` have a control effect of exactly
    0.0000. Every S_causal, and so every FIS, was inflated by an uncontrolled
    amount, always in the direction of the claim.

    Log space rather than linear: firing rates span orders of magnitude, so a
    linear band is dominated by the high-frequency tail and matches nothing at
    the sparse end. Nearest-K rather than a multiplicative band because the
    band can be empty for an extreme-frequency feature, and an empty band needs
    a widening fallback whose behaviour is then unstated. Nearest-K is never
    empty as long as any live feature exists, and the realised match quality is
    reported per feature so a poor match is visible rather than silent.
    """
    n_active = np.asarray(n_active)
    rng = rng if rng is not None else np.random.default_rng(0)
    live = np.flatnonzero(n_active > 0)
    live = live[live != int(target)]
    if live.size == 0:
        return np.array([], dtype=int)
    # +1 keeps log finite and monotone; rates are counts, so this is exact.
    d = np.abs(np.log(n_active[live] + 1.0)
               - np.log(n_active[int(target)] + 1.0))
    pool = live[np.argsort(d, kind="stable")[:min(pool_k, live.size)]]
    k = min(n_controls, pool.size)
    return rng.choice(pool, size=k, replace=False)


def causal_knockout(args, sae, feats, report, scale=1.0, n_active=None):
    """
    S_causal: does removing this feature from the residual stream change the
    model's decision?

    The feature's contribution at every position is subtracted:
        h' = h - z_f(h) * W_dec[f]
    and the change in the SAFE-minus-UNSAFE decision logit is measured. This is
    the feature-level version of the necessity test in section 4.5 of the
    proposal (knock-out), run on the units Aim 1 discovers rather than on token
    positions.

    A matched control is mandatory and is run for every feature: the same
    number of items, with a RANDOM feature of similar firing rate knocked out.
    Without it, "the decision moved by 0.4 logits" is unreadable, because
    perturbing the residual stream at all moves it somewhat.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    z = np.load(str(np.load(args.sae, allow_pickle=True)["acts"]),
                allow_pickle=True)
    split = args.causal_split or str(z["split"])
    records = read_jsonl(Path(str(z["data"])) / f"counterfactual_{split}.jsonl")
    # Knock-out is run on a DIFFERENT split from the one the dictionary was
    # fitted on where possible: a feature that only moves the decision on its
    # own training notes has not been shown to matter.
    # causal_items > 0 caps it; -1 means ALL items (0 never reaches here --
    # cmd_score's `if args.causal_items:` gate skips the whole stage on 0).
    if args.causal_items > 0:
        records = records[:args.causal_items]
    print(f"  knock-out on {len(records)} items of split `{split}`")
    layer = int(z["layer"])

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    _aids = answer_token_ids(tok)
    safe_ids, unsafe_ids = _aids["SAFE"], _aids["UNSAFE"]

    def wrap(p):
        return wrap_prompt(tok, p)

    W_dec = sae.W_dec.detach()
    state = {"feature": None}

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        f = state["feature"]
        if f is None:
            return output
        # the SAE was fitted on SCALED activations, so scale in, subtract
        # the feature's contribution, and scale back out
        zf = sae.encode(h[0].float() * scale)[:, f]          # [T]
        delta = (zf[:, None] * W_dec[f][None, :]) / scale    # [T, D]
        h = h.clone()
        h[0] = h[0] - delta.to(h.dtype)
        return (h,) + output[1:] if isinstance(output, tuple) else h

    # LAYER ALIGNMENT (defect B1, fixed 2026-09-02).
    #
    # `cmd_collect` stores `out.hidden_states[layer]`. In transformers,
    # hidden_states[0] is the embedding output and hidden_states[l] is the
    # INPUT to decoder layer l -- i.e. the OUTPUT of decoder layer l-1.
    # Verified empirically against transformers 5.15.0; note also that
    # hidden_states[n_layers] is post-`model.norm`, not the last layer's
    # output, which is defect B3 in patching.py.
    #
    # The hook must therefore fire on the module that PRODUCES
    # hidden_states[layer], which is `layers[layer-1]`, not `layers[layer]`.
    # Before this fix the dictionary fitted on h_20 was used to encode and
    # subtract from h_21, so S_causal was measured with a mismatched
    # dictionary and the knock-out null was not independent evidence.
    #
    # At layer 0 the producing module is the embedding table, which returns a
    # bare tensor rather than a tuple; `hook` already handles both shapes.
    target = producer_module(model, layer)
    print(f"  knock-out hook on {'embed_tokens' if layer == 0 else f'layers[{layer-1}]'}"
          f", which produces hidden_states[{layer}] -- the stream the SAE was "
          f"fitted on")
    handle = target.register_forward_hook(hook)

    def margins():
        out = []
        for r in records:
            enc = tok(wrap(r["prompt"]), return_tensors="pt",
                      truncation=True, max_length=args.max_len).to("cuda")
            with torch.no_grad():
                lg = model(**enc).logits[0, -1].float()
            out.append(max(lg[i].item() for i in safe_ids) -
                       max(lg[i].item() for i in unsafe_ids))
        return np.array(out)

    rng = np.random.default_rng(getattr(args, "seed", 0))
    n_feat = W_dec.shape[0]
    mode = getattr(args, "control_mode", "matched")
    n_ctrl = getattr(args, "n_controls", 5)
    if mode == "matched" and n_active is None:
        raise SystemExit(
            "control_mode='matched' needs per-feature firing rates; "
            "cmd_score must pass n_active into causal_knockout()")
    if mode == "matched":
        n_active = np.asarray(n_active)
        print(f"  controls: {n_ctrl} per feature, matched on firing rate "
              f"(nearest 50 live features in log space); "
              f"{int((n_active == 0).sum())}/{n_feat} dead features excluded")
    else:
        print(f"  controls: 1 per feature, drawn UNIFORMLY over all "
              f"{n_feat} features (legacy pre-2026-09-08 behaviour; "
              f"dead features are NOT excluded and the excess is inflated)")

    results = {}
    # try/finally so an OOM inside margins() cannot leave the knock-out hook
    # attached to a live model (patching.py and steering.py already do this).
    try:
        state["feature"] = None
        clean = margins()

        for f in (feats if not args.causal_features
                 else feats[:args.causal_features]):
            state["feature"] = int(f)
            eff = float(np.abs(margins() - clean).mean())

            if mode == "matched":
                ctrl_fs = matched_control_features(
                    int(f), n_active, n_controls=n_ctrl, rng=rng)
            else:
                ctrl_fs = np.array([int(rng.integers(0, n_feat))])

            ctrl_effects = []
            for cf in ctrl_fs:
                state["feature"] = int(cf)
                ctrl_effects.append(
                    float(np.abs(margins() - clean).mean()))
            ctrl = float(np.mean(ctrl_effects)) if ctrl_effects else 0.0

            rec = {"mean_abs_delta_logit": eff,
                   "control_mean_abs_delta_logit": ctrl,
                   "excess": eff - ctrl,
                   "control_mode": mode,
                   "control_features": [int(c) for c in ctrl_fs],
                   "control_effects": ctrl_effects,
                   "control_sd": (float(np.std(ctrl_effects, ddof=1))
                                  if len(ctrl_effects) > 1 else None),
                   # legacy key: the single control, or the first of the set
                   "control_feature": (int(ctrl_fs[0]) if len(ctrl_fs)
                                       else None)}
            if mode == "matched":
                rec["n_active"] = int(n_active[int(f)])
                rec["control_n_active"] = [int(n_active[c]) for c in ctrl_fs]
                # worst log-ratio in the drawn set: 0 is a perfect match, and
                # a large value means the dictionary had no comparable feature.
                rec["control_match_log_ratio"] = (
                    float(np.max(np.abs(
                        np.log(n_active[ctrl_fs] + 1.0)
                        - np.log(n_active[int(f)] + 1.0))))
                    if len(ctrl_fs) else None)
            results[int(f)] = rec

            sd = rec["control_sd"]
            print(f"  knock-out #{f}: |Δ margin|={eff:.4f} "
                  f"(control mean {ctrl:.4f}"
                  + (f" ± {sd:.4f}" if sd is not None else "")
                  + f" over {len(ctrl_fs)}"
                  + (f", worst log-ratio "
                     f"{rec['control_match_log_ratio']:.2f}"
                     if mode == "matched" and ctrl_fs.size else "")
                  + f", excess {eff-ctrl:+.4f})", flush=True)
    finally:
        handle.remove()

    # normalise into [0, 1] for the FIS: 1 logit of excess effect is a lot
    for feat in report["features"]:
        r = results.get(feat["feature"])
        if r:
            feat["s_causal"] = float(np.clip(r["excess"], 0.0, 1.0))
            feat["causal_detail"] = r
    return {"n_items": len(records), "clean_margin_mean": float(clean.mean()),
            "split": split,
            "control_mode": mode,
            "n_controls": (int(n_ctrl) if mode == "matched" else 1),
            "control_note": (
                "Controls are live features matched on firing rate (nearest "
                "50 in log space), averaged over n_controls draws. Dead "
                "features are excluded."
                if mode == "matched" else
                "LEGACY control: ONE feature drawn uniformly over the whole "
                "dictionary, dead features included. ~42% of this dictionary "
                "never fires, so the control frequently measures exactly zero "
                "and `excess` is inflated toward the claim. Reproduces "
                "pre-2026-09-08 FIS numbers; do not report as a control."),
            "per_feature": {str(k): v for k, v in results.items()}}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--data", default="data/medcalc")
    c.add_argument("--split", default="test")
    c.add_argument("--layer", type=int, default=20)
    c.add_argument("--model_id", default="BioMistral/BioMistral-7B")
    c.add_argument("--max_len", type=int, default=1400)
    c.add_argument("--max_tokens", type=int, default=200000)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--other_mult", type=float, default=2.0,
                   help="non-concept tokens kept per concept token. A large "
                        "value keeps the whole note, which is what a "
                        "dictionary should be fitted on; the default subsample "
                        "exists for quick runs.")
    c.add_argument("--concepts", choices=["umls", "regex"], default="umls",
                   help="concept vocabulary: UMLS groundings from the causal "
                        "knowledge graph (default), or the legacy hand-written "
                        "regexes that produced the pre-2026-09-02 numbers")
    c.add_argument("--graph", default=DEFAULT_GRAPH)
    c.add_argument("--seed", type=int, default=0,
                   help="Seeds the dictionary init, the train/val split, the batch order, the non-concept token subsample and the random control feature -- the four RNGs that were hard-coded. Decoding stays greedy, so this is the only real source of variance in the Aim 1 pipeline.")
    c.add_argument("--out", default=None)
    c.set_defaults(fn=cmd_collect)

    t = sub.add_parser("train")
    t.add_argument("--acts", required=True)
    t.add_argument("--kind", choices=["topk", "jumprelu"], default="topk")
    t.add_argument("--expansion", type=int, default=2)
    t.add_argument("--k", type=int, default=32)
    t.add_argument("--l1", type=float, default=1e-3)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=2048)
    t.add_argument("--seed", type=int, default=0,
                   help="Seeds the dictionary init, the train/val split, the batch order, the non-concept token subsample and the random control feature -- the four RNGs that were hard-coded. Decoding stays greedy, so this is the only real source of variance in the Aim 1 pipeline.")
    t.add_argument("--out", default=None)
    t.set_defaults(fn=cmd_train)

    s = sub.add_parser("score")
    s.add_argument("--sae", required=True)
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--model_id", default="BioMistral/BioMistral-7B")
    s.add_argument("--max_len", type=int, default=1400)
    s.add_argument("--causal_items", type=int, default=-1,
                   help="items to run knock-out on: 0 disables the stage, "
                        "-1 (default) means ALL items of the split, N>0 caps "
                        "it at N")
    s.add_argument("--causal_features", type=int, default=0,
                   help="how many of the --top features to run knock-out on; "
                        "0 (default) means ALL of them")
    s.add_argument("--control_mode", default="matched",
                   choices=["matched", "uniform"],
                   help="how the knock-out negative control is drawn. "
                        "`matched` (default since 2026-09-08) samples live "
                        "features of similar firing rate. `uniform` is the "
                        "legacy single uniform draw over ALL features, "
                        "including dead ones, which produced every FIS number "
                        "reported before that date -- kept so they stay "
                        "reproducible, not because it is correct.")
    s.add_argument("--n_controls", type=int, default=5,
                   help="matched controls averaged per feature. 1 gives the "
                        "control a variance comparable to the signal, which "
                        "is why the single draw was replaced.")
    s.add_argument("--causal_split", default=None,
                   help="split to run the knock-out on; defaults to the split "
                        "the activations came from")
    s.add_argument("--alpha_sem", type=float, default=0.5)
    s.add_argument("--beta_causal", type=float, default=0.5)
    s.add_argument("--seed", type=int, default=0,
                   help="Seeds the dictionary init, the train/val split, the batch order, the non-concept token subsample and the random control feature -- the four RNGs that were hard-coded. Decoding stays greedy, so this is the only real source of variance in the Aim 1 pipeline.")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
