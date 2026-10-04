# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MarianMT's multimodal (source-text) preprocessing.

These are GPU-free: they build only the processor (tokenizer + config), never
loading weights or running the model, so they exercise the encoder/decoder
prompt plumbing directly.
"""

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context

MODEL_ID = "Helsinki-NLP/opus-mt-en-de"


def _build_processor():
    ctx = build_model_context(MODEL_ID, limit_mm_per_prompt={"text": 1})
    return MULTIMODAL_REGISTRY.create_processor(ctx.model_config)


@pytest.mark.parametrize("source", ["Hello, world!", "The cat sat on the mat."])
def test_encoder_prompt_matches_tokenized_source(source: str) -> None:
    """The encoder placeholder expands to one slot per source token (including
    the trailing eos), and the encoder mm kwargs carry those exact token ids.
    """
    processor = _build_processor()
    tokenizer = processor.info.get_tokenizer()

    # Marian requires the trailing </s> on the encoder side.
    expected_ids = tokenizer(source, add_special_tokens=True)["input_ids"]

    mm_items = processor.info.parse_mm_data({"text": source})
    # Empty decoder prompt: Marian is bilingual, target language is implied.
    processed = processor("", mm_items=mm_items, hf_processor_mm_kwargs={})

    # Encoder placeholders: one [0] per real source token.
    assert processed["encoder_prompt_token_ids"] == [0] * len(expected_ids)

    # The real source token ids ride along in the mm kwargs.
    encoder_input_ids = processed["mm_kwargs"].get_data()["encoder_input_ids"]
    assert encoder_input_ids.reshape(-1).tolist() == list(expected_ids)


def test_decoder_prompt_is_empty_for_bilingual() -> None:
    """A string decoder prompt (e.g. a target-language code) carries no decoder
    tokens for a bilingual Marian model, so the runtime prepends only
    decoder_start_token_id.
    """
    processor = _build_processor()

    mm_items = processor.info.parse_mm_data({"text": "Hello"})
    processed = processor("", mm_items=mm_items, hf_processor_mm_kwargs={})

    assert processed["prompt_token_ids"] == []


def test_create_prompt_contracts() -> None:
    """Directly exercise the two prompt hooks the engine and endpoint rely on."""
    processor = _build_processor()
    mm_items = processor.info.parse_mm_data({"text": "Hello"})

    # Encoder side is a single placeholder, expanded later by prompt updates.
    assert processor.create_encoder_prompt([0], mm_items) == [0]

    # Bilingual: a target-language string resolves to an empty decoder prompt.
    assert processor.create_decoder_prompt("de", mm_items) == []

    # An explicit token-id list (teacher forcing) passes through unchanged.
    assert processor.create_decoder_prompt([1, 2, 3], mm_items) == [1, 2, 3]
