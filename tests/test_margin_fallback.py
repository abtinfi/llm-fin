"""
The margin fallback (model.HFModel.generate) on the case that motivated it:
Llama3-OpenBioLLM-8B writes UNSAFE as `UNS`+`AFE`, which is not a canonical
answer id. Tokenizer only, no model weights; skipped when the tokenizer is not
in the local HF cache.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from lm_common import answer_token_ids, fallback_answer_ids  # noqa: E402
from model import HFModel  # noqa: E402


def tok(mid):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(mid)
    except Exception:
        pytest.skip(f"{mid} tokenizer not cached")


class _Stub:
    _answer_step = HFModel._answer_step

    def __init__(self, t):
        self.tokenizer = t


def test_openbiollm_uns_afe_is_not_canonical_but_is_located():
    t = tok("aaditya/Llama3-OpenBioLLM-8B")
    canon = answer_token_ids(t, verbose=False)
    uns, afe = t.convert_tokens_to_ids(["UNS", "AFE"])
    assert uns not in canon["UNSAFE"] | canon["SAFE"]      # the bug
    assert _Stub(t)._answer_step([uns, afe], "UNSAFE") == 0


def test_prose_answer_step_is_the_answer_word():
    t = tok("aaditya/Llama3-OpenBioLLM-8B")
    ids = t.encode("Yes, it is safe to start.", add_special_tokens=False)
    s = _Stub(t)._answer_step(ids, t.decode(ids))
    assert t.decode(ids[s]).strip().lower() == "safe"


@pytest.mark.parametrize("mid", ["aaditya/Llama3-OpenBioLLM-8B",
                                 "BioMistral/BioMistral-7B",
                                 "mistralai/Mistral-7B-Instruct-v0.2"])
def test_fallback_ids_disjoint_and_contain_canonical(mid):
    t = tok(mid)
    fb, canon = fallback_answer_ids(t), answer_token_ids(t, verbose=False)
    assert not fb["SAFE"] & fb["UNSAFE"]
    assert canon["SAFE"] <= fb["SAFE"] and canon["UNSAFE"] <= fb["UNSAFE"]
