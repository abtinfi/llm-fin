"""
Aim 1/2, first step: is the decisive clinical fact present in the residual
stream at all, even where the model's output ignores it?

WHY A PROBE AND NOT PATCHING, FOR THIS DATASET
----------------------------------------------
Activation patching answers "where does the causal information flow", but it
needs the two arms to produce *different* outputs -- the normalised effect is
(patched - clean) / (counterfactual - clean), and that denominator is the gap
between the arms. On the renal family the model's decision-token margin ranks
SAFE above UNSAFE at AUROC 0.504 [0.451, 0.559]: the arms are output-identical.
Patching between two runs that already agree measures nothing, and the
normalised metric divides by ~zero.

So the prior question has to be answered first: **is the information there?**
A linear probe on the residual stream answers exactly that, and the answer is
decision-relevant either way:

  probe accurate, model not  -> the fact IS represented and simply not read out.
                                Aim 3's constraint layer has something to act on,
                                and this says which layer to attach it to.
  probe at chance too        -> the fact never enters the representation. No
                                intervention on hidden states can recover it,
                                and the symbolic route is not a shortcut but a
                                necessity.

THE CONTROL THAT MAKES THIS INTERPRETABLE
-----------------------------------------
Both arms of a pair share every word except one number. A probe evaluated on
held-out *pairs* therefore cannot score by memorising note content -- to get
both arms of an unseen pair right it must be reading the edited quantity. So
the reported metric is **pair-level consistency**, the same strict metric the
behavioural benchmark uses, not per-item accuracy. Splits are grouped by
`pair_id` so the two arms never straddle the train/test boundary.

A label-shuffled probe is fitted alongside as a floor. If the shuffled probe
scores above chance, the pipeline is leaking and nothing else here is readable.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open()]


def collect_activations(records, model_id, batch_size=4, dtype="bfloat16",
                        max_input_tokens=1400):
    """
    Residual stream at every layer, pooled two ways.

    **final** -- the last prompt position, which is where the answer token is
    predicted from. This is the representation the decision is actually read
    out of, so a probe here answers "is the fact available to the decision?".
    Left padding keeps the position aligned across a batch.

    **mean** -- averaged over all non-pad positions. A probe here answers the
    weaker but different question "is the fact encoded anywhere in the
    sequence?". The distinction matters: a fact can be represented locally at
    the token that states it and never propagate to the readout position. Those
    two outcomes call for opposite conclusions, so both are measured rather
    than one being taken as a proxy for the other.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), device_map="cuda")
    model.eval()

    def wrap(p):
        if tok.chat_template:
            return tok.apply_chat_template([{"role": "user", "content": p}],
                                           tokenize=False,
                                           add_generation_prompt=True)
        return f"[INST] {p} [/INST]"

    out = []
    for i in range(0, len(records), batch_size):
        chunk = [wrap(r["prompt"]) for r in records[i:i + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_input_tokens).to("cuda")
        with torch.no_grad():
            o = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].unsqueeze(-1).float()   # [B, T, 1]
        denom = mask.sum(dim=1).clamp(min=1)                 # [B, 1]
        fin, mea = [], []
        for h in o.hidden_states:                            # each [B, T, D]
            fin.append(h[:, -1, :].float().cpu())
            mea.append(((h.float() * mask).sum(dim=1) / denom).cpu())
        out.append((torch.stack(fin).numpy(), torch.stack(mea).numpy()))
        if (i // batch_size) % 20 == 0:
            print(f"  {min(i+batch_size, len(records))}/{len(records)}",
                  flush=True)
    final = np.concatenate([a for a, _ in out], axis=1)   # [L+1, N, D]
    mean = np.concatenate([b for _, b in out], axis=1)
    del model
    torch.cuda.empty_cache()
    return final, mean


def grouped_folds(pair_ids, n_folds, seed=0):
    """Folds that keep both arms of a pair on the same side of the split."""
    uniq = sorted(set(pair_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    assign = {p: i % n_folds for i, p in enumerate(uniq)}
    return np.array([assign[p] for p in pair_ids])


def probe_layer(X, y, pair_ids, folds, n_folds, C=0.05, n_pca=128):
    """
    Returns (item accuracy, pair-level consistency) cross-validated.

    Pair-level consistency counts a pair only if BOTH arms are predicted
    correctly -- identical definition to Causal Consistency in metrics.py, so
    the probe's number and the model's number are directly comparable.

    The residual stream is 4096-dimensional and a fold has a few hundred
    training items, so an unregularised probe can fit any labelling at all.
    Two things guard against reading noise:

      - PCA to `n_pca` components, **fitted on the training fold only** so no
        test-fold variance leaks into the basis, plus strong L2 (small C).
        This also makes each fit near-instant; at full dimension the search
        does not converge in reasonable time.
      - The shuffled-label probe in main(), fitted through this same function.
        If chance-level labels score above chance, nothing here is readable.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    preds = np.empty(len(y), dtype=object)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if tr.sum() < 10 or te.sum() < 2:
            continue
        k = int(min(n_pca, tr.sum() - 1, X.shape[1]))
        clf = make_pipeline(
            StandardScaler(),
            PCA(n_components=k, random_state=0),
            LogisticRegression(max_iter=5000, C=C))
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])

    ok = np.array([p == t for p, t in zip(preds, y)])
    by_pair = defaultdict(list)
    for pid, good in zip(pair_ids, ok):
        by_pair[pid].append(bool(good))
    cc = np.mean([all(v) for v in by_pair.values()])
    return float(ok.mean()), float(cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="counterfactual_*.jsonl to probe")
    ap.add_argument("--model_id", default="BioMistral/BioMistral-7B")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pca", type=int, default=128,
                    help="PCA components, fitted per training fold")
    ap.add_argument("--C", type=float, default=0.05,
                    help="inverse L2 strength for the probe")
    ap.add_argument("--cache", default=None,
                    help="npz path to save/reuse activations")
    ap.add_argument("--out", default=None, help="json summary path")
    args = ap.parse_args()

    recs = read_jsonl(args.data)
    y = np.array([r["label"] for r in recs])
    pair_ids = [r["pair_id"] for r in recs]
    print(f"{len(recs)} items, {len(set(pair_ids))} pairs, "
          f"labels={dict(zip(*np.unique(y, return_counts=True)))}")

    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists() and "mean" in np.load(cache).files:
        z = np.load(cache)
        pooled = {"final": z["final"], "mean": z["mean"]}
        print(f"loaded cached activations {z['final'].shape} from {cache}")
    else:
        print("collecting activations ...")
        fin, mea = collect_activations(recs, args.model_id, args.batch_size)
        pooled = {"final": fin, "mean": mea}
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, final=fin, mean=mea)
            print(f"cached activations {fin.shape} -> {cache}")

    folds = grouped_folds(pair_ids, args.folds, args.seed)
    rng = np.random.default_rng(args.seed)

    # Null control, structure-preserving.
    #
    # A GLOBAL shuffle of the labels is the wrong null here and reads as a
    # leak when there is none. Every real pair holds exactly one SAFE and one
    # UNSAFE, so a constant predictor scores 0 pair-consistency. Shuffle the
    # labels globally and roughly half the pairs end up with two identical
    # labels, which a constant predictor gets right -- the floor climbs to
    # ~0.23 for reasons that have nothing to do with the activations.
    #
    # Permuting WITHIN each pair -- randomly swapping which arm is called
    # SAFE -- destroys the association with the activations while keeping the
    # one-of-each structure, so the null and the real metric are on the same
    # scale and directly comparable.
    y_shuf = y.copy()
    idx_by_pair = defaultdict(list)
    for i, pid in enumerate(pair_ids):
        idx_by_pair[pid].append(i)
    for idxs in idx_by_pair.values():
        if len(idxs) == 2 and rng.random() < 0.5:
            a, b = idxs
            y_shuf[a], y_shuf[b] = y_shuf[b], y_shuf[a]

    report = {"data": args.data, "pooling": {}}
    for pool_name, acts in pooled.items():
        n_layers = acts.shape[0]
        print(f"\n--- pooling: {pool_name} "
              + ("(the readout position: is the fact AVAILABLE to the "
                 "decision?)" if pool_name == "final"
                 else "(all positions: is the fact ENCODED anywhere?)") + " ---")
        print(f"{'layer':>6} {'item acc':>10} {'pair CC':>10} "
              f"{'shuffled CC':>13}")
        rows = []
        for l in range(n_layers):
            X = acts[l]
            acc, cc = probe_layer(X, y, pair_ids, folds, args.folds,
                                  C=args.C, n_pca=args.pca)
            _, cc_s = probe_layer(X, y_shuf, pair_ids, folds, args.folds,
                                  C=args.C, n_pca=args.pca)
            rows.append({"layer": l, "item_acc": acc, "pair_cc": cc,
                         "shuffled_cc": cc_s})
            print(f"{l:6d} {acc:10.3f} {cc:10.3f} {cc_s:13.3f}", flush=True)

        best = max(rows, key=lambda r: r["pair_cc"])
        floor = max(r["shuffled_cc"] for r in rows)
        print(f"best layer {best['layer']}: pair-level consistency "
              f"{best['pair_cc']:.3f} (item accuracy {best['item_acc']:.3f})")
        # A per-arm-random predictor scores 0.25 pair-consistency on a
        # one-of-each pair, so the null is EXPECTED to sit near 0.25 and a
        # fixed 0.15 alarm would fire on every healthy run. What matters is
        # the margin of the real probe over its own null, not the null's
        # absolute level.
        margin = best["pair_cc"] - floor
        print(f"highest null (within-pair permuted) consistency: {floor:.3f} "
              f"(a per-arm-random predictor scores 0.250)")
        print(f"  signal over null: {margin:+.3f}", end="")
        if margin < 0.10:
            print("   -> no evidence the fact is linearly decodable here.")
        elif margin < 0.25:
            print("   -> weak, report the curve rather than a point.")
        else:
            print("   -> the fact IS linearly decodable from this "
                  "representation.")
        report["pooling"][pool_name] = {"rows": rows, "best": best,
                                        "shuffle_floor": floor}

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
