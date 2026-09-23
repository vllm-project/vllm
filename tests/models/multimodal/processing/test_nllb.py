# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NLLB / M2M100 multimodal (source-text) preprocessing.

These are GPU-free: they build only the processor (tokenizer + config), never
loading weights or running the model, so they exercise the encoder/decoder
prompt plumbing directly. Unlike bilingual MarianMT, these models are
many-to-many, so the decoder prompt resolves the requested target language to a
forced-BOS token id.
"""

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context

MODEL_NLLB = "facebook/nllb-200-distilled-600M"
MODEL_M2M100 = "facebook/m2m100_418M"


def _build_processor(model_id: str):
    ctx = build_model_context(model_id, limit_mm_per_prompt={"text": 1})
    return MULTIMODAL_REGISTRY.create_processor(ctx.model_config)


@pytest.mark.parametrize("model_id", [MODEL_NLLB, MODEL_M2M100])
@pytest.mark.parametrize("source", ["Hello, world!", "The cat sat on the mat."])
def test_encoder_prompt_matches_tokenized_source(model_id: str, source: str) -> None:
    """The encoder placeholder expands to one slot per source token (including
    the src-lang code and trailing </s> added with add_special_tokens=True), and
    the encoder mm kwargs carry those exact token ids.
    """
    processor = _build_processor(model_id)
    tokenizer = processor.info.get_tokenizer()

    expected_ids = tokenizer(source, add_special_tokens=True)["input_ids"]

    mm_items = processor.info.parse_mm_data({"text": source})
    processed = processor("", mm_items=mm_items, hf_processor_mm_kwargs={})

    # Encoder placeholders: one [0] per real source token.
    assert processed["encoder_prompt_token_ids"] == [0] * len(expected_ids)

    # The real source token ids ride along in the mm kwargs.
    encoder_input_ids = processed["mm_kwargs"].get_data()["encoder_input_ids"]
    assert encoder_input_ids.reshape(-1).tolist() == list(expected_ids)


@pytest.mark.parametrize(
    "model_id,aliases",
    [
        # NLLB names languages with the full tag; ISO code and English name alias.
        (MODEL_NLLB, ["deu_Latn", "de", "German"]),
        # M2M100 names languages with the bare ISO code; English name aliases.
        (MODEL_M2M100, ["de", "German"]),
    ],
)
def test_decoder_prompt_resolves_target_language(
    model_id: str, aliases: list[str]
) -> None:
    """Many-to-many: a target-language string resolves to a single forced-BOS
    token id, and all accepted aliases for one language resolve to the same id.
    """
    processor = _build_processor(model_id)
    mm_items = processor.info.parse_mm_data({"text": "Hello"})

    resolved = [processor.create_decoder_prompt(alias, mm_items) for alias in aliases]

    # Each alias resolves to exactly one forced-BOS token.
    assert all(len(r) == 1 for r in resolved)
    # All aliases for the same language resolve to the same token id.
    assert len({tuple(r) for r in resolved}) == 1


@pytest.mark.parametrize("model_id", [MODEL_NLLB, MODEL_M2M100])
def test_decoder_prompt_rejects_unknown_language(model_id: str) -> None:
    """An unsupported target language raises ValueError so the endpoint can turn
    it into a clean 400 rather than emitting a mis-conditioned translation.
    """
    processor = _build_processor(model_id)
    mm_items = processor.info.parse_mm_data({"text": "Hello"})

    with pytest.raises(ValueError):
        processor.create_decoder_prompt("not-a-language", mm_items)


@pytest.mark.parametrize("model_id", [MODEL_NLLB, MODEL_M2M100])
def test_create_prompt_contracts(model_id: str) -> None:
    """Directly exercise the two prompt hooks the engine and endpoint rely on."""
    processor = _build_processor(model_id)
    mm_items = processor.info.parse_mm_data({"text": "Hello"})

    # Encoder side is a single placeholder, expanded later by prompt updates.
    assert processor.create_encoder_prompt([0], mm_items) == [0]

    # An empty target string yields no forced BOS (e.g. profiling); the runtime
    # still prepends the decoder start token.
    assert processor.create_decoder_prompt("", mm_items) == []

    # An explicit token-id list (teacher forcing) passes through unchanged.
    assert processor.create_decoder_prompt([1, 2, 3], mm_items) == [1, 2, 3]
