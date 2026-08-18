# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the xgrammar backend with a `MistralCommonBackend`.

``transformers>=5.5`` returns a ``MistralCommonBackend`` for Mistral repos
under the default ``tokenizer_mode="auto"``. That class subclasses
``PreTrainedTokenizerBase`` rather than ``PreTrainedTokenizerFast`` and is not
vLLM's ``MistralTokenizer``, so ``XgrammarBackend.__post_init__`` used to fall
through to ``xgr.TokenizerInfo.from_huggingface()``, which cannot classify it
and raises ``ValueError: Unsupported tokenizer type: ...MistralCommonBackend``.
Because the backend is built lazily on the first structured-output request,
that surfaced as an HTTP 500 while plain completions kept working.
"""

import pytest

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

# A tekken-backed Mistral repo that transformers loads via MistralCommonBackend.
TOKENIZER = "mistralai/Mistral-Nemo-Instruct-2407"


@pytest.fixture(scope="module")
def mistral_common_tokenizer():
    transformers_mistral = pytest.importorskip(
        "transformers.tokenization_mistral_common"
    )
    return transformers_mistral.MistralCommonBackend.from_pretrained(TOKENIZER)


def test_xgrammar_backend_supports_mistral_common_backend(mistral_common_tokenizer):
    """The backend must build and compile a grammar, not raise ValueError."""
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar")
    )

    backend = XgrammarBackend(
        vllm_config,
        tokenizer=mistral_common_tokenizer,
        vocab_size=mistral_common_tokenizer.vocab_size,
    )

    grammar = backend.compile_grammar(
        StructuredOutputOptions.JSON,
        '{"type": "object", "properties": {"x": {"type": "integer"}}}',
    )
    assert grammar is not None
