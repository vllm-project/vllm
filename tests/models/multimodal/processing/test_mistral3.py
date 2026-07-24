# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mistral3's multimodal preprocessing."""

import pytest

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache

from ...registry import HF_EXAMPLE_MODELS


@pytest.mark.parametrize("model_id", ["mistralai/Mistral-Small-3.1-24B-Instruct-2503"])
@pytest.mark.parametrize("use_cache", [True, False])
def test_dummy_mm_inputs_with_mistral_tokenizer(model_id: str, use_cache: bool):
    """
    The HF-format Mistral3 architecture should be usable with
    `--tokenizer-mode mistral`.

    Regression test for https://github.com/vllm-project/vllm/issues/45289
    where building the dummy multi-modal inputs (run at API server startup
    to compute the encoder budget) crashed because the tokenizer handed to
    `PixtralProcessor` did not encode the `[IMG]` placeholder to its
    special token id.
    """
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_available_online(on_fail="skip")

    results = {}
    for tokenizer_mode in ("auto", "mistral"):
        model_config = ModelConfig(
            model_id,
            tokenizer=model_id,
            tokenizer_mode=tokenizer_mode,
            # The HF repo also contains the consolidated (mistral-format)
            # checkpoint; force the HF format to test the Mistral3
            # architecture instead of Pixtral
            config_format="hf",
            max_model_len=8192,
        )

        cache = MultiModalProcessorOnlyCache(model_config) if use_cache else None
        processor = MULTIMODAL_REGISTRY.create_processor(model_config, cache=cache)
        mm_inputs = MULTIMODAL_REGISTRY.get_dummy_mm_inputs(
            model_config,
            mm_counts={"image": 1},
            processor=processor,
        )

        results[tokenizer_mode] = (
            mm_inputs["prompt_token_ids"],
            [
                (placeholder.offset, placeholder.length)
                for placeholder in mm_inputs["mm_placeholders"]["image"]
            ],
        )

    # Both tokenizer modes use the Tekken vocab,
    # so they should produce identical outputs
    assert results["mistral"] == results["auto"]
