# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from vllm.v1.spec_decode.vocab_mapping import _detect_space_prefix


@pytest.mark.parametrize(
    "model_name,expected_prefix",
    [
        # BPE tokenizer (GPT-2 family) uses Ġ (U+0120)
        ("HuggingFaceTB/SmolLM2-135M-Instruct", ("Ġ",)),
        # SentencePiece tokenizer (LLaMA family) uses ▁ (U+2581)
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", ("▁",)),
        # BPE tokenizer (Qwen family) uses Ġ (U+0120)
        ("Qwen/Qwen2.5-0.5B-Instruct", ("Ġ",)),
    ],
)
def test_detect_space_prefix_real_tokenizers(model_name, expected_prefix):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    result = _detect_space_prefix(tokenizer)
    assert result == expected_prefix, (
        f"{model_name}: expected {expected_prefix!r}, got {result!r}"
    )


def test_detect_space_prefix_fallback_on_failure():
    """When tokenizer lacks encode(), fall back to both known prefixes."""

    class BrokenTokenizer:
        def encode(self, text, **kwargs):
            raise RuntimeError("broken")

    result = _detect_space_prefix(BrokenTokenizer())
    assert result == ("Ġ", "▁")


def test_detect_space_prefix_empty_encode():
    """When encode returns empty list, fall back."""

    class EmptyTokenizer:
        def encode(self, text, **kwargs):
            return []

    result = _detect_space_prefix(EmptyTokenizer())
    assert result == ("Ġ", "▁")
