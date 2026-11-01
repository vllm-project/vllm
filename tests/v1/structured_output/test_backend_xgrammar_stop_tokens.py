# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for xgrammar stop-token handling."""

import json
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import AutoTokenizer, TokenizersBackend

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import (
    XgrammarBackend,
    _model_stop_token_ids,
)

VOCAB_SIZE = 9

EOS = 1
QUOTE = 2
LETTER = 3


def _vllm_config_with_generation_eos(eos_token_id):
    model_config = SimpleNamespace(
        try_get_generation_config=lambda: {"eos_token_id": eos_token_id}
    )
    return SimpleNamespace(model_config=model_config)


def _backend_config(generation_eos_token_id=None):
    model_config = None
    if generation_eos_token_id is not None:
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="test"),
            try_get_generation_config=lambda: {
                "eos_token_id": generation_eos_token_id,
            },
        )
    return SimpleNamespace(
        model_config=model_config,
        speculative_config=None,
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
    )


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
def tokenizer() -> TokenizersBackend:
    raw_tokenizer = Tokenizer(
        WordLevel(
            vocab={
                "<unk>": 0,
                "<eos>": EOS,
                '"': QUOTE,
                "X": LETTER,
                "{": 4,
                "}": 5,
                ":": 6,
                ",": 7,
                " ": 8,
            },
            unk_token="<unk>",
        )
    )
    raw_tokenizer.pre_tokenizer = WhitespaceSplit()
    return TokenizersBackend(
        tokenizer_object=raw_tokenizer,
        unk_token="<unk>",
        eos_token="<eos>",
    )


@pytest.fixture(scope="module")
def backend(tokenizer: TokenizersBackend) -> XgrammarBackend:
    return XgrammarBackend(
        _backend_config(),
        tokenizer=tokenizer,
        vocab_size=VOCAB_SIZE,
    )


@pytest.fixture(scope="module")
def backend_with_model_eos(tokenizer: TokenizersBackend) -> XgrammarBackend:
    return XgrammarBackend(
        _backend_config([EOS, LETTER]),
        tokenizer=tokenizer,
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


def test_model_eos_tokens_gated_to_grammar_terminal(
    backend_with_model_eos: XgrammarBackend,
):
    schema = '{"type": "string"}'
    grammar = backend_with_model_eos.compile_grammar(
        StructuredOutputOptions.JSON, schema
    )

    # Open the string: LETTER is valid JSON string content, but the model's
    # generation_config also marks it as an EOS id. The backend must pass that
    # model EOS id into xgrammar tokenizer info so it is masked until the JSON
    # string can terminate.
    assert grammar.accept_tokens("req", [QUOTE])

    bitmask = backend_with_model_eos.allocate_token_bitmask(1)
    grammar.fill_bitmask(bitmask, 0)
    assert not _token_allowed(bitmask[0], LETTER)

    assert grammar.accept_tokens("req", [QUOTE])
    assert not grammar.is_terminated()

    grammar.fill_bitmask(bitmask, 0)
    assert _token_allowed(bitmask[0], LETTER)


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
