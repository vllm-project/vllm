# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoModelForImageTextToText

from ....conftest import HfRunner, ImageTestAssets, VllmRunner

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
IMAGE_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe this image in detail.<|im_end|>\n<|im_start|>assistant\n"
)
TEXT_PROMPT = (
    "<|im_start|>user\nDescribe a busy street corner in detail.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
MAX_TOKENS = 16
# Keeps the image prefill within the default CUDA graph capture sizes
MAX_IMAGE_SIDE = 384
SMOLVLM_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
SMOLVLM_PROMPT = (
    "<|im_start|>User:<image>Describe this image.<end_of_utterance>\nAssistant:"
)


@pytest.mark.core_model
def test_transformers_image_deepstack(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
):
    """DeepStack features must reach the first decoder layers, and must not leak
    into a later text-only step. Without them the output stays fluent and within
    `check_logprobs_close`'s top-k tolerance, so greedy tokens are compared
    exactly in float32 instead."""
    images = []
    for asset in image_assets:
        image = asset.pil_image.copy()
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        images.append(image)
    image_prompts = [IMAGE_PROMPT] * len(images)

    with vllm_runner(
        MODEL_ID,
        model_impl="transformers",
        dtype="float32",
        max_model_len=4096,
        limit_mm_per_prompt={
            "image": {
                "count": 1,
                "width": MAX_IMAGE_SIDE,
                "height": MAX_IMAGE_SIDE,
            },
        },
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(
            image_prompts, MAX_TOKENS, images=images
        )
        vllm_outputs += vllm_model.generate_greedy([TEXT_PROMPT], MAX_TOKENS)

    with hf_runner(
        MODEL_ID, dtype="float32", auto_cls=AutoModelForImageTextToText
    ) as hf_model:
        hf_outputs = hf_model.generate_greedy(image_prompts, MAX_TOKENS, images=images)
        hf_outputs += hf_model.generate_greedy([TEXT_PROMPT], MAX_TOKENS)

    for (hf_ids, hf_text), (vllm_ids, vllm_text) in zip(hf_outputs, vllm_outputs):
        assert vllm_ids[-MAX_TOKENS:] == hf_ids[-MAX_TOKENS:], (
            f"hf: {hf_text!r}\nvllm: {vllm_text!r}"
        )


@pytest.mark.core_model
def test_transformers_image_request_kwargs(
    vllm_runner: type[VllmRunner], image_assets: ImageTestAssets
):
    """Processor kwargs given per request produce the same prompt and output as the
    same kwargs set for the server."""
    kwargs = {"size": {"longest_edge": 1024}}
    prompts = [
        {"prompt": SMOLVLM_PROMPT, "multi_modal_data": {"image": asset.pil_image}}
        for asset in image_assets
    ]

    with vllm_runner(
        SMOLVLM_ID, model_impl="transformers", max_model_len=8192, enforce_eager=True
    ) as vllm_model:
        request_outputs = vllm_model.generate_greedy(
            [{**prompt, "mm_processor_kwargs": kwargs} for prompt in prompts],
            MAX_TOKENS,
        )
    with vllm_runner(
        SMOLVLM_ID,
        model_impl="transformers",
        max_model_len=8192,
        enforce_eager=True,
        mm_processor_kwargs=kwargs,
    ) as vllm_model:
        server_outputs = vllm_model.generate_greedy(prompts, MAX_TOKENS)

    assert request_outputs == server_outputs
