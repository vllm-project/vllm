# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

TOKENIZER = "openai-community/gpt2"
VOCAB_SIZE = 50257

# gpt2 token ids used to drive a `{"type": "string"}` grammar deterministically.
EOS = 50256  # <|endoftext|> -- the tokenizer's only default stop token
QUOTE = 1  # standalone `"`; opens then closes the JSON string
LETTER = 55  # `X`: valid string content, not a special/stop token by default
NEWLINE = 198  # `\n`; a non-stop token used as trailing draft after EOS


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


def _completed_object_grammar(backend: XgrammarBackend):
    """JSON object grammar that has accepted a complete `{"a": "b"}` value."""
    grammar = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "object"}'
    )
    prompt = backend.tokenizer.encode('{"a": "b"}')
    assert grammar.accept_tokens("req", prompt)
    assert not grammar.is_terminated()
    return grammar


def test_request_stop_tokens_gated_to_grammar_terminal(backend: XgrammarBackend):
    """Extra request stop tokens are masked until the grammar can terminate.

    Regression for #42403: a request's stop-token set (generation_config eos
    plus user ``stop_token_ids``) is invisible to xgrammar, which only knows
    the tokenizer's single eos. Such tokens can escape the grammar bitmask
    while the FSM is still mid-object. ``compile_grammar`` forwards
    ``all_stop_token_ids`` as ``override_stop_tokens`` so xgrammar masks them
    until the grammar completes.
    """
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


def test_accept_tokens_stops_at_termination(backend: XgrammarBackend, capfd):
    """Tokens after a terminating EOS do not reach the matcher.

    Regression for #52767 / #52805: speculative batches can include draft
    tokens after a stop token. Those must not be forwarded to xgrammar.
    """
    grammar = _completed_object_grammar(backend)
    processed_before = grammar.num_processed_tokens

    assert grammar.accept_tokens("req", [EOS, NEWLINE])
    assert grammar.is_terminated()
    assert grammar.num_processed_tokens == processed_before + 1
    assert "trying to accept new token" not in capfd.readouterr().err

    processed_after_eos = grammar.num_processed_tokens
    assert grammar.accept_tokens("req", [NEWLINE])
    assert grammar.num_processed_tokens == processed_after_eos
    assert "trying to accept new token" not in capfd.readouterr().err

    grammar.reset()
    assert not grammar.is_terminated()
    assert grammar.num_processed_tokens == 0


def test_validate_tokens_stops_at_termination(backend: XgrammarBackend, capfd):
    """Validation rolls back after reaching a terminating EOS.

    Regression for #52767 / #52805.
    """
    grammar = _completed_object_grammar(backend)

    assert grammar.validate_tokens([EOS, NEWLINE]) == [EOS]
    assert "trying to accept new token" not in capfd.readouterr().err
    # Check matcher state directly to verify validation rolled it back.
    assert not grammar.matcher.is_terminated()

    assert grammar.accept_tokens("req", [EOS])
    assert grammar.is_terminated()

    assert grammar.validate_tokens([NEWLINE]) == []
    assert "trying to accept new token" not in capfd.readouterr().err
