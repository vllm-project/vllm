# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import pytest
import torch
from transformers import AutoTokenizer

from vllm.config import DeviceConfig, VllmConfig
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_lm_format_enforcer import (
    LMFormatEnforcerBackend,
)
from vllm.v1.structured_output.backend_outlines import OutlinesBackend
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend
from vllm.v1.structured_output.utils import allocate_token_bitmask

TOKENIZER = "gpt2"
REGEX = "1a"
BACKENDS = (
    pytest.param("xgrammar", id="xgrammar"),
    pytest.param("guidance", id="guidance"),
    pytest.param("outlines", id="outlines"),
    pytest.param(
        "lm-format-enforcer",
        marks=pytest.mark.xfail(
            reason=(
                "lm-format-enforcer 0.11.3 imports the removed "
                "vllm.transformers_utils.tokenizer path"
            ),
            strict=True,
        ),
        id="lm-format-enforcer",
    ),
)
BACKEND_CLASSES: dict[str, Callable[..., StructuredOutputBackend]] = {
    "xgrammar": XgrammarBackend,
    "guidance": GuidanceBackend,
    "outlines": OutlinesBackend,
    "lm-format-enforcer": LMFormatEnforcerBackend,
}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


def _single_token(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text)
    assert len(token_ids) == 1, (text, token_ids)
    return token_ids[0]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "vocab_size_delta",
    [32, -32],
    ids=["model-vocab-larger", "model-vocab-smaller"],
)
def test_vocab_size_is_respected(tokenizer, backend: str, vocab_size_delta: int):
    """Grammar mask must match the logits width (vocab size) required by sampling"""
    vocab_size = len(tokenizer) + vocab_size_delta

    if vocab_size_delta < 0:
        xfail_reason = {
            "guidance": (
                "llguidance requires the model vocabulary size to be at least "
                "the tokenizer vocabulary size"
            ),
            "outlines": (
                "outlines requires a token mask at least as wide as the "
                "tokenizer vocabulary"
            ),
        }.get(backend)
        if xfail_reason is not None:
            pytest.xfail(xfail_reason)

    structured_output_backend = BACKEND_CLASSES[backend](
        VllmConfig(device_config=DeviceConfig(device="cpu")),
        tokenizer,
        vocab_size,
    )

    assert structured_output_backend.vocab_size == vocab_size

    bitmask = allocate_token_bitmask(num_rows=1, vocab_size=vocab_size)
    assert bitmask.shape == (1, (vocab_size + 31) // 32)
    assert bitmask.dtype == torch.int32
    assert bitmask.device.type == "cpu"
    assert bitmask.is_contiguous()

    grammar = structured_output_backend.compile_grammar(
        StructuredOutputOptions.REGEX, REGEX
    )
    grammar.fill_bitmask(bitmask, 0)

    first = _single_token(tokenizer, "1")
    invalid = _single_token(tokenizer, "z")
    assert bitmask[0, first >> 5].item() & (1 << (first & 31))
    assert not bitmask[0, invalid >> 5].item() & (1 << (invalid & 31))


@pytest.mark.parametrize("backend", BACKENDS)
def test_accept_tokens_rejection_is_atomic(tokenizer, backend: str):
    structured_output_backend = BACKEND_CLASSES[backend](
        VllmConfig(device_config=DeviceConfig(device="cpu")),
        tokenizer,
        len(tokenizer),
    )
    grammar = structured_output_backend.compile_grammar(
        StructuredOutputOptions.REGEX, REGEX
    )

    first = _single_token(tokenizer, "1")
    invalid = _single_token(tokenizer, "z")

    assert not grammar.accept_tokens("request", [first, invalid])
    assert grammar.accept_tokens("request", [first])
