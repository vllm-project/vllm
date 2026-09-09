# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured output with a `MistralCommonBackend` tokenizer."""

import pytest
from transformers import AutoTokenizer, MistralCommonBackend

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.v1.structured_output.backend_outlines import OutlinesBackend
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend
from vllm.v1.structured_output.utils import (
    _reduced_vocabulary,
    maybe_wrap_mistral_common_tokenizer,
)

TOKENIZER = "mistralai/Mistral-Nemo-Instruct-2407"
JSON_SCHEMA = (
    '{"type": "object", "properties": {"x": {"type": "integer"}}, '
    '"required": ["x"], "additionalProperties": false}'
)
VALID_DOCUMENT = '{"x": 1}'

BACKENDS = [("xgrammar", XgrammarBackend), ("outlines", OutlinesBackend)]


@pytest.fixture(scope="module")
def loaded_tokenizer():
    tokenizer = get_tokenizer(tokenizer_name=TOKENIZER, tokenizer_mode="hf")
    assert isinstance(tokenizer, MistralCommonBackend)
    return tokenizer


@pytest.fixture(scope="module")
def wrapped_tokenizer(loaded_tokenizer):
    return maybe_wrap_mistral_common_tokenizer(loaded_tokenizer)


def _encode(tokenizer, text):
    special = set(tokenizer.all_special_ids)
    return [tid for tid in tokenizer.encode(text) if tid not in special]


def _backend(cls, name, tokenizer):
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend=name)
    )
    return cls(vllm_config, tokenizer=tokenizer, vocab_size=tokenizer.vocab_size)


def test_wrapping_produces_a_mistral_tokenizer(loaded_tokenizer, wrapped_tokenizer):
    assert isinstance(wrapped_tokenizer, MistralTokenizer)
    assert len(wrapped_tokenizer.vocab) == loaded_tokenizer.vocab_size


def test_other_tokenizers_are_left_alone():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    assert maybe_wrap_mistral_common_tokenizer(tokenizer) is tokenizer


def test_wrapping_fixes_the_outlines_vocabulary(loaded_tokenizer, wrapped_tokenizer):
    unwrapped_vocab = _reduced_vocabulary(loaded_tokenizer)
    wrapped_vocab = _reduced_vocabulary(wrapped_tokenizer)

    assert unwrapped_vocab.keys() == wrapped_vocab.keys()

    mismapped = {
        token for token, ids in unwrapped_vocab.items() if ids != wrapped_vocab[token]
    }
    assert mismapped

    assert len({tuple(wrapped_vocab[t]) for t in mismapped}) == len(mismapped)
    assert len({tuple(unwrapped_vocab[t]) for t in mismapped}) < len(mismapped)


@pytest.mark.parametrize("name,cls", BACKENDS)
def test_backend_accepts_a_valid_document(wrapped_tokenizer, name, cls):
    backend = _backend(cls, name, wrapped_tokenizer)
    grammar = backend.compile_grammar(StructuredOutputOptions.JSON, JSON_SCHEMA)

    for token in _encode(wrapped_tokenizer, VALID_DOCUMENT):
        assert grammar.accept_tokens("req", [token])


@pytest.mark.parametrize("name,cls", BACKENDS)
def test_backend_rejects_an_invalid_token(wrapped_tokenizer, name, cls):
    backend = _backend(cls, name, wrapped_tokenizer)
    grammar = backend.compile_grammar(StructuredOutputOptions.JSON, JSON_SCHEMA)

    assert not grammar.accept_tokens("req", [_encode(wrapped_tokenizer, "]")[-1]])


def test_xgrammar_vocab_size_matches_the_tokenizer(wrapped_tokenizer):
    backend = _backend(XgrammarBackend, "xgrammar", wrapped_tokenizer)
    assert backend.vocab_size == len(wrapped_tokenizer.vocab)


def test_xgrammar_rejects_the_unwrapped_tokenizer(loaded_tokenizer):
    with pytest.raises(ValueError, match="Unsupported tokenizer type"):
        _backend(XgrammarBackend, "xgrammar", loaded_tokenizer)
