# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoModelForImageTextToText

from ....conftest import HfRunner, ImageTestAssets, VllmRunner

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe this image in detail.<|im_end|>\n<|im_start|>assistant\n"
)
MAX_TOKENS = 16


@pytest.mark.core_model
def test_transformers_image_deepstack(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
):
    """DeepStack features must reach the first decoder layers. Without them the
    output stays fluent and within `check_logprobs_close`'s top-k tolerance, so
    greedy tokens are compared exactly in float32 instead."""
    images = [asset.pil_image for asset in image_assets]
    prompts = [PROMPT] * len(images)

    with vllm_runner(
        MODEL_ID,
        model_impl="transformers",
        dtype="float32",
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(prompts, MAX_TOKENS, images=images)

    with hf_runner(
        MODEL_ID, dtype="float32", auto_cls=AutoModelForImageTextToText
    ) as hf_model:
        hf_outputs = hf_model.generate_greedy(prompts, MAX_TOKENS, images=images)

    for (hf_ids, hf_text), (vllm_ids, vllm_text) in zip(hf_outputs, vllm_outputs):
        assert vllm_ids[-MAX_TOKENS:] == hf_ids[-MAX_TOKENS:], (
            f"hf: {hf_text!r}\nvllm: {vllm_text!r}"
        )
