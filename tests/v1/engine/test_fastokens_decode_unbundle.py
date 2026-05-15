# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``_to_hf_tokenizer`` in ``vllm/v1/engine/detokenizer.py``."""

import pytest
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


def test_to_hf_tokenizer_rebuilds_from_json_shim():
    """A shim exposing ``_json`` is rebuilt into a real Tokenizer and cached."""
    json_str = AutoTokenizer.from_pretrained(MODEL)._tokenizer.to_str()

    class FakeShim:
        """Stand-in for ``fastokens._compat._TokenizerShim``."""

        def __init__(self, json_str: str) -> None:
            self._json = json_str

    shim = FakeShim(json_str)
    _rebuilt_hf_tokenizer_cache.clear()

    rebuilt = _to_hf_tokenizer(shim)
    assert isinstance(rebuilt, Tokenizer)
    assert rebuilt.to_str() == json_str
    # Subsequent calls hit the cache (same instance returned).
    assert _to_hf_tokenizer(shim) is rebuilt


def test_to_hf_tokenizer_raises_for_object_without_json():
    """Objects without a ``_json`` attribute raise ``TypeError`` rather than
    being silently passed through to ``DecodeStream``, which would surface
    later as a confusing C-extension error."""

    class Mystery:
        pass

    with pytest.raises(TypeError, match="no ``_json`` attribute"):
        _to_hf_tokenizer(Mystery())
