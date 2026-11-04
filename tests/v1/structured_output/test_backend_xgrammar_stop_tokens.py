# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for issue #42403.

A request's stop-token set (the generation_config eos list plus any
user-supplied ``stop_token_ids``) is invisible to xgrammar, which only knows
the tokenizer's single eos. Such tokens can therefore escape the grammar
bitmask while the FSM is still mid-object and truncate structured output.
``compile_grammar`` now forwards ``all_stop_token_ids`` to the matcher as
``override_stop_tokens`` so xgrammar masks them until the grammar completes.

Issue #60580: xgrammar only treats tokens that decode to an empty string as
special, so a tokenizer's control tokens (e.g. ``<|im_start|>``) matched
grammar text and could be sampled inside a JSON string. Structural tags and
Lark grammars, which can name such tokens, must still be able to emit them.
"""

import json

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

QWEN_TOKENIZER = "Qwen/Qwen3-0.6B"
QWEN_VOCAB_SIZE = 151936

# Qwen3 special added tokens.
ENDOFTEXT = 151643  # <|endoftext|>: a generation_config stop token
IM_START = 151644  # <|im_start|>: never a stop token
IM_END = 151645  # <|im_end|>: the tokenizer's eos


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


@pytest.fixture(scope="module")
def qwen_tokenizer():
    return AutoTokenizer.from_pretrained(QWEN_TOKENIZER)


@pytest.fixture(scope="module")
def qwen_backend(qwen_tokenizer) -> XgrammarBackend:
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar")
    )
    return XgrammarBackend(
        vllm_config, tokenizer=qwen_tokenizer, vocab_size=QWEN_VOCAB_SIZE
    )


@pytest.mark.parametrize(
    "request_type, grammar_spec",
    [
        (StructuredOutputOptions.JSON, '{"type": "string"}'),
        (StructuredOutputOptions.REGEX, '"[^"]*"'),
        (StructuredOutputOptions.GRAMMAR, 'root ::= "\\"" [^"]* "\\""'),
    ],
)
def test_special_tokens_masked_until_grammar_ends(
    qwen_backend: XgrammarBackend, qwen_tokenizer, request_type, grammar_spec
):
    grammar = qwen_backend.compile_grammar(
        request_type, grammar_spec, stop_token_ids={IM_END, ENDOFTEXT}
    )
    bitmask = qwen_backend.allocate_token_bitmask(1)

    text = qwen_tokenizer.encode('"The company is', add_special_tokens=False)
    assert grammar.accept_tokens("req", text)
    grammar.fill_bitmask(bitmask, 0)
    for token_id in (IM_START, IM_END, ENDOFTEXT):
        assert not _token_allowed(bitmask[0], token_id)

    quote = qwen_tokenizer.encode('"', add_special_tokens=False)
    assert grammar.accept_tokens("req", quote)
    grammar.fill_bitmask(bitmask, 0)
    assert not _token_allowed(bitmask[0], IM_START)
    assert _token_allowed(bitmask[0], IM_END)
    assert _token_allowed(bitmask[0], ENDOFTEXT)


@pytest.mark.parametrize(
    "request_type, grammar_spec",
    [
        (
            StructuredOutputOptions.STRUCTURAL_TAG,
            json.dumps(
                {
                    "type": "structural_tag",
                    "format": {
                        "type": "tag",
                        "begin": "<|im_start|>",
                        "content": {
                            "type": "json_schema",
                            "json_schema": {"type": "string"},
                        },
                        "end": "\n",
                    },
                }
            ),
        ),
        (StructuredOutputOptions.GRAMMAR, f'start: <[{IM_START}]> "assistant"'),
    ],
)
def test_token_naming_grammars_can_emit_special_tokens(
    qwen_backend: XgrammarBackend, request_type, grammar_spec
):
    grammar = qwen_backend.compile_grammar(request_type, grammar_spec)
    bitmask = qwen_backend.allocate_token_bitmask(1)
    grammar.fill_bitmask(bitmask, 0)
    assert _token_allowed(bitmask[0], IM_START)
    assert grammar.accept_tokens("req", [IM_START])
