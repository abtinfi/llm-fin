"""
RQ3 control: do the added components degrade general language capability?

Variants (2)-(4) do not touch the model weights, so perplexity is IDENTICAL by
construction. The question only becomes real for the Constraint-Aware Layer of
Aim 3, which DOES modify hidden states -- and there it is the check that stops
a layer from "winning" the safety task by damaging the model.

  # pre-intervention baseline
  python src/perplexity.py --n 200

  # same model with the trained constraint layer inserted
  python src/perplexity.py --n 200 --adapter results/adapter_qt.npz --layer 5

Both numbers must be reported together. A constraint layer that fixes Causal
Consistency while raising perplexity has not solved the problem, it has traded
one failure for another.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="BioMistral/BioMistral-7B")
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--config", default="wikitext-2-raw-v1")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--adapter", default=None,
                    help="npz saved by constraint_layer.py --save_adapter")
    ap.add_argument("--layer", type=int, default=None,
                    help="layer the adapter was trained at (required with "
                         "--adapter)")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    if args.adapter and args.layer is None:
        raise SystemExit("--adapter requires --layer")

    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.config, split="test")
    text = "\n\n".join(t for t in ds["text"][: args.n * 10] if t.strip())

    tok = AutoTokenizer.from_pretrained(args.model_id)
    # bfloat16, matching every other run in this project. fp16 would put this
    # number on a different footing from the results it is meant to guard.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    handle = None
    if args.adapter:
        import numpy as np
        z = np.load(args.adapter)
        W_down = torch.tensor(z["W_down"], device="cuda").float()
        W_up = torch.tensor(z["W_up"], device="cuda").float()
        b = torch.tensor(z["b"], device="cuda").float()

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            x = h.float()
            delta = (torch.nn.functional.gelu(x @ W_down + b) @ W_up)
            h = h + args.alpha * delta.to(h.dtype)
            return (h,) + output[1:] if isinstance(output, tuple) else h

        handle = model.model.layers[args.layer].register_forward_hook(hook)
        print(f"constraint layer attached at layer {args.layer}, "
              f"alpha={args.alpha}, rank={W_down.shape[1]}")

    ids = tok(text, return_tensors="pt").input_ids
    nlls, prev_end = [], 0
    for begin in range(0, ids.size(1), args.stride):
        end = min(begin + args.max_len, ids.size(1))
        trg_len = end - prev_end
        chunk = ids[:, begin:end].to("cuda")
        target = chunk.clone()
        target[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(chunk, labels=target)
        nlls.append(out.loss.float() * trg_len)
        prev_end = end
        if end == ids.size(1):
            break

    if handle:
        handle.remove()
    ppl = torch.exp(torch.stack(nlls).sum() / prev_end)
    tag = (f"+constraint layer L{args.layer}" if args.adapter
           else "base (no intervention)")
    print(f"model={args.model_id}  {tag}  "
          f"wikitext-2 perplexity = {ppl.item():.4f}")


if __name__ == "__main__":
    main()
