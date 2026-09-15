# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for xgrammar stop-token handling."""

from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from vllm.config import StructuredOutputsConfig
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
            try_get_generation_config=lambda: {
                "eos_token_id": generation_eos_token_id,
            }
        )
    return SimpleNamespace(
        model_config=model_config,
        speculative_config=None,
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
    )


def _token_allowed(row, token_id: int) -> bool:
    word = int(row[token_id // 32].item()) & 0xFFFFFFFF
    return bool(word & (1 << (token_id % 32)))


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerFast:
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
    return PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="<unk>",
        eos_token="<eos>",
    )


@pytest.fixture(scope="module")
def backend(tokenizer: PreTrainedTokenizerFast) -> XgrammarBackend:
    return XgrammarBackend(
        _backend_config(),
        tokenizer=tokenizer,
        vocab_size=VOCAB_SIZE,
    )


@pytest.fixture(scope="module")
def backend_with_model_eos(tokenizer: PreTrainedTokenizerFast) -> XgrammarBackend:
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
