# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token rejection in ``XgrammarGrammar.accept_tokens`` must not log at ERROR.

Under speculative decoding, the structured output manager replays draft
tokens through ``accept_tokens`` and rolls back on rejection; rejection is
an expected outcome there, not a desync. Logging it at ERROR inside the
backend produced hundreds of "Failed to advance FSM ... Please file an
issue" lines per grammar request while the outputs remained schema-valid.
Severity is owned by the call sites: the scheduler logs at ERROR itself
when non-speculative output tokens are rejected.
"""

import logging

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

TOKENIZER = "openai-community/gpt2"
VOCAB_SIZE = 50257

# gpt2 token ids used to drive a `{"type": "string"}` grammar deterministically.
QUOTE = 1  # standalone `"`; opens then closes the JSON string
LETTER = 55  # `X`: invalid before the string is opened


@pytest.fixture(scope="module")
def backend() -> XgrammarBackend:
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar")
    )
    return XgrammarBackend(
        vllm_config,
        tokenizer=AutoTokenizer.from_pretrained(TOKENIZER),
        vocab_size=VOCAB_SIZE,
    )


def test_token_rejection_does_not_log_error(backend: XgrammarBackend, caplog_vllm):
    grammar = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "string"}'
    )

    with caplog_vllm.at_level(logging.DEBUG):
        # A bare letter before the opening quote is not valid JSON-string
        # grammar content, so the FSM rejects it.
        assert not grammar.accept_tokens("req", [LETTER])

    assert not [r for r in caplog_vllm.records if r.levelno >= logging.ERROR], (
        "accept_tokens rejection is an expected outcome on the speculative "
        "decoding replay path and must not be logged at ERROR"
    )
    # The rejection detail should still be available at DEBUG.
    assert any(
        "Failed to advance FSM" in r.getMessage()
        for r in caplog_vllm.records
        if r.levelno == logging.DEBUG
    )

    # The grammar is still usable after a rejected draft token.
    assert grammar.accept_tokens("req", [QUOTE])
