"""
The three pieces of tokenizer/architecture plumbing that every script in this
project needs, in one place.

WHY THIS FILE EXISTS. Prompt wrapping and SAFE/UNSAFE answer-token resolution
were copy-pasted into five modules (`model.py`, `patching.py`, `steering.py`,
`sae.py`, `constraint_layer.py`). Five copies is survivable while there is one
model. It is not survivable across models, because the original resolver was
written against a SentencePiece tokenizer and hard-codes two of its artefacts:
the U+2581 word-boundary marker and the byte-fallback newline literal
`<0x0A>`. Llama-3 uses byte-level BPE and has neither, so the SPM copy would
have silently mis-resolved the decision tokens -- and every margin, every
entropy and every ACE in this project is computed from those ids.

The rule here is tokenizer-agnostic: DECODE a candidate id back to text and
ask whether it carries any non-whitespace character. That is the property the
SPM version was approximating. It covers U+2581, BPE's leading-space tokens,
and byte-fallback newlines without naming any of them.
"""

from typing import Dict, Set

# The leading contexts the model actually sees. A word tokenises differently
# depending on what precedes it, so all three are resolved and unioned.
ANSWER_VARIANTS = {
    "SAFE":   ["SAFE", " SAFE", "\nSAFE"],
    "UNSAFE": ["UNSAFE", " UNSAFE", "\nUNSAFE"],
}


def wrap_prompt(tokenizer, prompt: str) -> str:
    """
    Apply the model's own chat template.

    The BioMistral path is unchanged: it ships a chat_template, so the first
    branch is taken and every prediction already on disk reproduces exactly.

    The fallback is no longer unconditionally Mistral's. `Llama3-OpenBioLLM-8B`
    ships NO chat_template, and wrapping a Llama-3 model in `[INST] ... [/INST]`
    would hand it a prompt format it was never instruction-tuned on -- the
    second model would then look worse than the first for a reason that has
    nothing to do with the science. So when there is no template, pick the
    format from the special tokens the vocabulary actually contains.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)

    vocab = tokenizer.get_vocab()
    if "<|start_header_id|>" in vocab:                      # Llama-3 family
        return ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n")
    return f"[INST] {prompt} [/INST]"


def answer_token_ids(tokenizer, verbose: bool = True) -> Dict[str, Set[int]]:
    """
    First *content* token ids for SAFE / UNSAFE.

    An id whose decoded text is empty or pure whitespace is a boundary marker,
    not the answer, so it is skipped and the next piece is taken. Any id that
    still ends up claimed by both classes is dropped and reported -- leaving it
    in would put the same token on both sides of the margin and make every
    downstream number meaningless (P1 in HANDOFF.md).

    Raises rather than returning a degenerate result: a tokenizer for which
    either class comes out empty cannot support the logit-margin readout, and
    that has to stop a run rather than quietly produce zeros.
    """
    ids = {"SAFE": set(), "UNSAFE": set()}
    for label, texts in ANSWER_VARIANTS.items():
        for t in texts:
            for tok in tokenizer.encode(t, add_special_tokens=False):
                if not tokenizer.decode([tok]).strip():
                    continue            # boundary marker / whitespace / newline
                ids[label].add(tok)
                break

    overlap = ids["SAFE"] & ids["UNSAFE"]
    if overlap:
        if verbose:
            print(f"[lm_common] WARNING: dropping {len(overlap)} token id(s) "
                  f"ambiguous between SAFE and UNSAFE: "
                  f"{[tokenizer.convert_ids_to_tokens(i) for i in overlap]}")
        ids["SAFE"] -= overlap
        ids["UNSAFE"] -= overlap
    if not ids["SAFE"] or not ids["UNSAFE"]:
        raise RuntimeError(
            "could not resolve distinct SAFE/UNSAFE answer tokens; "
            "the logit-margin signal cannot be computed for this tokenizer")
    return ids


def decoder_layers(model):
    """
    The decoder layer ModuleList, whatever the architecture calls it.

    Five call sites indexed `model.model.layers` directly, which holds for
    Llama/Mistral and breaks elsewhere. Failing loudly with the actual module
    names beats an AttributeError from four frames down.
    """
    for path in (("model", "layers"), ("transformer", "h"),
                 ("gpt_neox", "layers"), ("model", "decoder", "layers")):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise RuntimeError(
        f"could not locate the decoder layers on {type(model).__name__}; "
        f"top-level modules are {[n for n, _ in model.named_children()]}")


def producer_module(model, layer: int):
    """
    The module whose OUTPUT is `hidden_states[layer]`.

    This is the off-by-one that defects B1 and B2 were: `hidden_states[l]` is
    the INPUT to decoder layer l, i.e. the OUTPUT of layer l-1. A hook meant to
    read or replace `hidden_states[l]` must therefore fire on `layers[l-1]`,
    and on the embedding module when l == 0.

    `patching.py` deliberately uses the opposite convention -- it hooks
    `layers[l]` and takes donors from `hidden_states[l+1]`, which is
    self-consistent because its donor IS the output of layer l. Do not
    "unify" the two; they are different questions.
    """
    layers = decoder_layers(model)
    if layer == 0:
        emb = getattr(getattr(model, "model", model), "embed_tokens", None)
        if emb is None:
            emb = model.get_input_embeddings()
        return emb
    return layers[layer - 1]


def hidden_size(model) -> int:
    return int(model.config.hidden_size)


def model_slug(model_id: str) -> str:
    """
    Filesystem-safe short name for a model id, used to namespace artifacts so
    a second model cannot overwrite the first's results.
    """
    return model_id.split("/")[-1].replace(".", "-").lower()
