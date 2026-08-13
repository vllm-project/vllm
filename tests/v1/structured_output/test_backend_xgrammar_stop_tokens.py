# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for xgrammar stop-token handling.

A request's stop-token set (the generation_config eos list plus any
user-supplied ``stop_token_ids``) is invisible to xgrammar, which only knows
the tokenizer's single eos. Such tokens can therefore escape the grammar
bitmask while the FSM is still mid-object and truncate structured output.
``compile_grammar`` now forwards ``all_stop_token_ids`` to the matcher as
``override_stop_tokens`` so xgrammar masks them until the grammar completes.
"""

from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import (
    XgrammarBackend,
    _model_stop_token_ids,
)

TOKENIZER = "openai-community/gpt2"
VOCAB_SIZE = 50257

# gpt2 token ids used to drive a `{"type": "string"}` grammar deterministically.
EOS = 50256  # <|endoftext|> -- the tokenizer's only default stop token
QUOTE = 1  # standalone `"`; opens then closes the JSON string
LETTER = 55  # `X`: valid string content, not a special/stop token by default


def _vllm_config_with_generation_eos(eos_token_id):
    model_config = SimpleNamespace(
        try_get_generation_config=lambda: {"eos_token_id": eos_token_id}
    )
    return SimpleNamespace(model_config=model_config)


def _token_allowed(row, token_id: int) -> bool:
    word = int(row[token_id // 32].item()) & 0xFFFFFFFF
    return bool(word & (1 << (token_id % 32)))


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


def test_request_stop_tokens_gated_to_grammar_terminal(backend: XgrammarBackend):
    schema = '{"type": "string"}'
    default = backend.compile_grammar(StructuredOutputOptions.JSON, schema)
    override = backend.compile_grammar(
        StructuredOutputOptions.JSON, schema, stop_token_ids={EOS, LETTER}
    )

    # Open the string: both grammars are now in a non-terminal state.
    for grammar in (default, override):
        assert grammar.accept_tokens("req", [QUOTE])

    bm_default = backend.allocate_token_bitmask(1)
    bm_override = backend.allocate_token_bitmask(1)
    default.fill_bitmask(bm_default, 0)
    override.fill_bitmask(bm_override, 0)

    # Mid-string, the plain token is valid content, so the default grammar
    # leaves it samplable -- this is the leak. Registering it as a stop token
    # masks it until the grammar can terminate.
    assert _token_allowed(bm_default[0], LETTER)
    assert not _token_allowed(bm_override[0], LETTER)

    # Close the string -> accepting state (grammar complete, not yet terminated).
    for grammar in (default, override):
        assert grammar.accept_tokens("req", [QUOTE])
        assert not grammar.is_terminated()

    default.fill_bitmask(bm_default, 0)
    override.fill_bitmask(bm_override, 0)

    # The extra stop token may now terminate under the override, never under
    # the default grammar -- and the tokenizer's own eos still terminates both,
    # so default termination is preserved.
    assert not _token_allowed(bm_default[0], LETTER)
    assert _token_allowed(bm_override[0], LETTER)
    assert _token_allowed(bm_default[0], EOS)
    assert _token_allowed(bm_override[0], EOS)


@pytest.mark.parametrize(
    ("generation_eos_token_id", "expected"),
    [
        ([EOS, LETTER], [EOS, LETTER]),
        (LETTER, [EOS, LETTER]),
        ([LETTER, EOS, LETTER], [EOS, LETTER]),
        (None, [EOS]),
        ("bad", [EOS]),
        ([LETTER, "bad", None, True], [EOS, LETTER]),
    ],
)
def test_model_stop_token_ids_include_generation_config_eos(
    generation_eos_token_id, expected
):
    tokenizer = SimpleNamespace(eos_token_id=EOS)
    vllm_config = _vllm_config_with_generation_eos(generation_eos_token_id)

    assert _model_stop_token_ids(vllm_config, tokenizer) == expected


def test_model_stop_token_ids_handles_absent_model_config():
    tokenizer = SimpleNamespace(eos_token_id=EOS)
    vllm_config = SimpleNamespace(model_config=None)

    assert _model_stop_token_ids(vllm_config, tokenizer) == [EOS]


def test_model_stop_token_ids_handles_missing_tokenizer_eos():
    tokenizer = SimpleNamespace(eos_token_id=None)
    vllm_config = _vllm_config_with_generation_eos(LETTER)

    assert _model_stop_token_ids(vllm_config, tokenizer) == [LETTER]
