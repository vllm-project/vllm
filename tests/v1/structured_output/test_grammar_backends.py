# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import pytest
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
