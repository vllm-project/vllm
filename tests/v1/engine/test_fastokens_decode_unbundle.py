# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``_to_hf_tokenizer`` in ``vllm/v1/engine/detokenizer.py``.

The helper exists so ``FastIncrementalDetokenizer`` can keep using HF's
``DecodeStream`` even when the inner tokenizer is a fastokens shim (or any
future shim that exposes the same ``to_str()`` JSON-serialization contract).
We exercise both branches without requiring ``fastokens`` to be installed.
"""

from tokenizers import Tokenizer
from transformers import AutoTokenizer

from vllm.v1.engine.detokenizer import (
    _rebuilt_hf_tokenizer_cache,
    _to_hf_tokenizer,
)

MODEL = "hf-internal-testing/llama-tokenizer"


def test_to_hf_tokenizer_passes_real_tokenizer_through():
    """A real ``tokenizers.Tokenizer`` is returned unchanged."""
    inner = AutoTokenizer.from_pretrained(MODEL)._tokenizer
    assert _to_hf_tokenizer(inner) is inner


def test_to_hf_tokenizer_rebuilds_from_to_str_shim():
    """A shim that only exposes ``to_str()`` is rebuilt into a real Tokenizer."""
    json_str = AutoTokenizer.from_pretrained(MODEL)._tokenizer.to_str()

    class FakeShim:
        """Stand-in for ``fastokens._compat._TokenizerShim``.

        The production code only relies on ``to_str()`` returning a tokenizer
        JSON; any object satisfying that contract should work.
        """

        def to_str(self) -> str:
            return json_str

    shim = FakeShim()
    _rebuilt_hf_tokenizer_cache.clear()

    rebuilt = _to_hf_tokenizer(shim)
    assert isinstance(rebuilt, Tokenizer)
    assert rebuilt.to_str() == json_str
    # Subsequent calls hit the cache (same instance returned).
    assert _to_hf_tokenizer(shim) is rebuilt


def test_to_hf_tokenizer_passes_unknown_object_through():
    """Objects with no ``to_str`` fall through unchanged — informative error
    downstream rather than a silent rebuild."""

    class Mystery:
        pass

    obj = Mystery()
    assert _to_hf_tokenizer(obj) is obj
