# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for structured output with a `MistralCommonBackend`.

``transformers>=5.5`` returns a ``MistralCommonBackend`` for Mistral repos
under the default ``tokenizer_mode="auto"``. That class subclasses
``PreTrainedTokenizerBase`` rather than ``PreTrainedTokenizerFast`` and does
not implement the tokenizer surface the grammar backends rely on, so each
backend failed on it in its own way:

* ``xgrammar`` fell through to ``xgr.TokenizerInfo.from_huggingface()``, which
  cannot classify the class and raises
  ``ValueError: Unsupported tokenizer type: ...MistralCommonBackend``.
* ``outlines`` called ``tokenizer.convert_tokens_to_string()`` while building
  its reduced vocabulary, which raises ``NotImplementedError``.

Both backends are built lazily on the first structured-output request, so this
surfaced as an HTTP 500 while plain completions kept working.
"""

import pytest

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_outlines import OutlinesBackend
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

# A tekken-backed Mistral repo that transformers loads via MistralCommonBackend.
TOKENIZER = "mistralai/Mistral-Nemo-Instruct-2407"
JSON_SCHEMA = '{"type": "object", "properties": {"x": {"type": "integer"}}}'


@pytest.fixture(scope="module")
def mistral_common_tokenizer():
    transformers_mistral = pytest.importorskip(
        "transformers.tokenization_mistral_common"
    )
    return transformers_mistral.MistralCommonBackend.from_pretrained(TOKENIZER)


@pytest.mark.parametrize(
    "backend_name,backend_cls",
    [("xgrammar", XgrammarBackend), ("outlines", OutlinesBackend)],
)
def test_backend_supports_mistral_common_backend(
    mistral_common_tokenizer, backend_name, backend_cls
):
    """The backend must build and compile a grammar, not raise."""
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend=backend_name)
    )

    backend = backend_cls(
        vllm_config,
        tokenizer=mistral_common_tokenizer,
        vocab_size=mistral_common_tokenizer.vocab_size,
    )

    grammar = backend.compile_grammar(StructuredOutputOptions.JSON, JSON_SCHEMA)
    assert grammar is not None


def test_maybe_wrap_leaves_other_tokenizers_untouched():
    """Non-Mistral tokenizers must pass through unchanged."""
    from vllm.v1.structured_output.utils import maybe_wrap_mistral_common_tokenizer

    sentinel = object()
    assert maybe_wrap_mistral_common_tokenizer(sentinel) is sentinel
