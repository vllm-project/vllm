# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for ``disable_any_whitespace`` on the xgrammar backend.

xgrammar's ``any_whitespace=False`` selects a *fixed* JSON format rather than a
whitespace-free one, and with ``separators`` unset it falls back to the
json.dumps() defaults of ``(", ", ": ")``. Those *require* a space after every
comma and colon, so the flag used to mandate whitespace instead of removing it -
and, because the space was mandatory, the bare ``"`` that opens a string value
was masked out entirely at the value position.
"""

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

TOKENIZER = "openai-community/gpt2"
VOCAB_SIZE = 50257


def _token_allowed(row, token_id: int) -> bool:
    word = int(row[token_id // 32].item()) & 0xFFFFFFFF
    return bool(word & (1 << (token_id % 32)))


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


def _backend(tokenizer, *, disable_any_whitespace: bool) -> XgrammarBackend:
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(
            backend="xgrammar", disable_any_whitespace=disable_any_whitespace
        )
    )
    return XgrammarBackend(vllm_config, tokenizer=tokenizer, vocab_size=VOCAB_SIZE)


def _mask_after_colon(backend: XgrammarBackend, tokenizer):
    """Bitmask at the value position, i.e. right after ``{"a":``."""
    grammar = backend.compile_grammar(StructuredOutputOptions.JSON_OBJECT, "")
    assert grammar.accept_tokens("req", tokenizer.encode('{"a":'))
    bitmask = backend.allocate_token_bitmask(1)
    grammar.fill_bitmask(bitmask, 0)
    return bitmask[0]


@pytest.mark.parametrize("value_text", ["1", '"'])
def test_compact_value_tokens_survive_disable_any_whitespace(tokenizer, value_text):
    """A compact value must be reachable when whitespace is disabled.

    Before the fix the grammar required ``": "``, so every token legal at this
    position began with a space and both a bare digit and a bare quote were
    masked out - the model could not express a value the way it wanted to.
    """
    row = _mask_after_colon(_backend(tokenizer, disable_any_whitespace=True), tokenizer)
    token_id = tokenizer.encode(value_text)[0]
    assert _token_allowed(row, token_id), (
        f"{value_text!r} must be legal after a colon when whitespace is disabled"
    )


def test_disable_any_whitespace_forbids_the_space(tokenizer):
    """The flag should actually disable whitespace, not merely fix its shape."""
    row = _mask_after_colon(_backend(tokenizer, disable_any_whitespace=True), tokenizer)
    space_digit = tokenizer.encode(" 1")[0]
    assert not _token_allowed(row, space_digit), (
        "' 1' must not be legal when whitespace is disabled"
    )


def test_whitespace_enabled_permits_both_forms(tokenizer):
    """The default must keep allowing whitespace, compact or spaced."""
    row = _mask_after_colon(_backend(tokenizer, disable_any_whitespace=False), tokenizer)
    for text in ("1", " 1"):
        assert _token_allowed(row, tokenizer.encode(text)[0]), (
            f"{text!r} must stay legal when whitespace is allowed"
        )
