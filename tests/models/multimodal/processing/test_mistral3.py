# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mistral3's multimodal preprocessing."""

import pytest

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY


@pytest.mark.parametrize(
    ("model_id", "revision"),
    [
        (
            "unsloth/Mistral-Small-3.2-24B-Instruct-2506",
            "3ed8d341b53ea3d0e4194689905477688a6cf733",
        ),
    ],
)
def test_dummy_mm_inputs_with_dual_format_checkpoint(model_id: str, revision: str):
    """
    HF-format Mistral3 checkpoints that also ship mistral-common tokenizer
    files (tekken.json) resolve to transformers' `MistralCommonBackend`
    under the default tokenizer mode. Building the dummy multi-modal inputs
    (run at engine startup to profile memory usage) must not crash.

    Regression test for https://github.com/vllm-project/vllm/issues/50706
    where the tokenizer handed to `PixtralProcessor` did not encode the
    `[IMG]` placeholder to its special token id, so engine initialization
    died in `_check_special_mm_tokens` with default arguments.
    """
    model_config = ModelConfig(
        model_id,
        tokenizer=model_id,
        revision=revision,
        tokenizer_revision=revision,
        max_model_len=8192,
    )

    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    tokenizer = processor.info.ctx.tokenizer

    hf_processor = processor.info.get_hf_processor()
    image_token_id = tokenizer.convert_tokens_to_ids(hf_processor.image_token)

    mm_inputs = MULTIMODAL_REGISTRY.get_dummy_mm_inputs(
        model_config,
        mm_counts={"image": 1},
        processor=processor,
    )

    # The image placeholder must have been encoded to its special token id
    # and located by the placeholder search.
    assert image_token_id in mm_inputs["prompt_token_ids"]
    assert mm_inputs["mm_placeholders"]["image"]
