# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured output with a `MistralCommonBackend` tokenizer.

``transformers>=5.5`` returns a ``MistralCommonBackend`` for Mistral repos that
vLLM loads through the HF path, i.e. whenever ``resolve_tokenizer_args`` does not
pick ``tokenizer_mode="mistral"``. That class subclasses
``PreTrainedTokenizerBase`` rather than ``PreTrainedTokenizerFast``, which breaks
the grammar backends in two different ways:

* ``xgrammar`` falls through to ``xgr.TokenizerInfo.from_huggingface()``, which
  cannot classify the class and raises ``ValueError: Unsupported tokenizer type``
  -- an HTTP 500 on the first structured-output request.
* ``outlines`` does not crash, because ``CachedTokenizer`` supplies a
  ``convert_tokens_to_string`` fallback. It silently builds a *wrong* vocabulary
  instead: byte-fallback tokens that decode to U+FFFD collapse onto a handful of
  shared token ids, so the grammar matches against ids that do not correspond to
  the vocabulary entry.

Both are fixed by wrapping the tokenizer in vLLM's ``MistralTokenizer`` once, in
``StructuredOutputManager``.
"""

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

# A Mistral repo that vLLM's HF path loads as a MistralCommonBackend.
TOKENIZER = "mistralai/Mistral-Nemo-Instruct-2407"
JSON_SCHEMA = (
    '{"type": "object", "properties": {"x": {"type": "integer"}}, '
    '"required": ["x"], "additionalProperties": false}'
)
VALID_DOCUMENT = '{"x": 1}'


@pytest.fixture(scope="module")
def loaded_tokenizer():
    """The tokenizer as `StructuredOutputManager` would receive it."""
    tokenizer = get_tokenizer(tokenizer_name=TOKENIZER, tokenizer_mode="hf")
    assert isinstance(tokenizer, MistralCommonBackend), (
        f"expected a MistralCommonBackend, got {type(tokenizer).__name__}"
    )
    return tokenizer


@pytest.fixture(scope="module")
def wrapped_tokenizer(loaded_tokenizer):
    return maybe_wrap_mistral_common_tokenizer(loaded_tokenizer)


def _encode(tokenizer, text):
    """Token ids for `text` with no special tokens, which a grammar rejects."""
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
    """A non-Mistral tokenizer must pass through untouched.

    Loaded while the mistral-common module is imported, so this would catch a
    guard that matches too broadly.
    """
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    assert maybe_wrap_mistral_common_tokenizer(tokenizer) is tokenizer


def test_wrapping_fixes_the_outlines_vocabulary(loaded_tokenizer, wrapped_tokenizer):
    """Unwrapped, byte-fallback tokens collapse onto shared token ids."""
    unwrapped_vocab = _reduced_vocabulary(loaded_tokenizer)
    wrapped_vocab = _reduced_vocabulary(wrapped_tokenizer)

    assert unwrapped_vocab.keys() == wrapped_vocab.keys()

    collapsed = {
        token for token, ids in unwrapped_vocab.items() if ids != wrapped_vocab[token]
    }
    assert collapsed, "expected the unwrapped vocabulary to mismap some tokens"

    # Unwrapped, those tokens share a handful of ids between them; the wrapper
    # resolves each one to its own id.
    unwrapped_ids = {tuple(unwrapped_vocab[token]) for token in collapsed}
    wrapped_ids = {tuple(wrapped_vocab[token]) for token in collapsed}
    assert len(wrapped_ids) == len(collapsed)
    assert len(unwrapped_ids) < len(collapsed)


@pytest.mark.parametrize(
    "name,cls", [("xgrammar", XgrammarBackend), ("outlines", OutlinesBackend)]
)
def test_backend_compiles_and_matches(wrapped_tokenizer, name, cls):
    """The backend must build, and the grammar must accept a matching document."""
    backend = _backend(cls, name, wrapped_tokenizer)
    grammar = backend.compile_grammar(StructuredOutputOptions.JSON, JSON_SCHEMA)

    for token in _encode(wrapped_tokenizer, VALID_DOCUMENT):
        assert grammar.accept_tokens("req", [token])


@pytest.mark.parametrize(
    "name,cls", [("xgrammar", XgrammarBackend), ("outlines", OutlinesBackend)]
)
def test_backend_rejects_invalid_token(wrapped_tokenizer, name, cls):
    """A token that cannot start a JSON object must be rejected."""
    backend = _backend(cls, name, wrapped_tokenizer)
    grammar = backend.compile_grammar(StructuredOutputOptions.JSON, JSON_SCHEMA)

    not_an_object = _encode(wrapped_tokenizer, "]")[-1]
    assert not grammar.accept_tokens("req", [not_an_object])


def test_xgrammar_vocab_size_matches_the_tokenizer(wrapped_tokenizer):
    """The bitmask width has to follow the vocabulary the grammar was built from."""
    backend = _backend(XgrammarBackend, "xgrammar", wrapped_tokenizer)
    assert backend.vocab_size == len(wrapped_tokenizer.vocab)


def test_xgrammar_rejects_the_unwrapped_tokenizer(loaded_tokenizer):
    """Without the wrapper xgrammar cannot classify the tokenizer at all."""
    with pytest.raises(ValueError, match="Unsupported tokenizer type"):
        _backend(XgrammarBackend, "xgrammar", loaded_tokenizer)
