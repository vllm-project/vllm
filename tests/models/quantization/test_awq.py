# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import pytest
import torch

from vllm.multimodal.image import rescale_image_size

from ...conftest import IMAGE_ASSETS, ImageTestAssets, VllmRunner
from ..utils import check_logprobs_close

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "<|im_start|>User\n<image>\nWhat's the content in the center of the image?<|im_end|>\n<|im_start|>Assistant\n",  # noqa: E501
    "cherry_blossom":
    "<|im_start|>User\n<image>\nWhat is the season?<|im_end|>\n<|im_start|>Assistant\n",  # noqa: E501
})


def run_awq_test(
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
    source_model: str,
    quant_model: str,
    *,
    size_factors: list[float],
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [(
        [prompt for _ in size_factors],
        [rescale_image_size(image, factor) for factor in size_factors],
    ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    # max_model_len should be greater than image_feature_size
    with vllm_runner(source_model,
                     max_model_len=4096,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        source_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs_per_image
        ]

    with vllm_runner(quant_model,
                     quantization="awq",
                     max_model_len=4096,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        quant_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs_per_image
        ]

    for source_outputs, quant_outputs in zip(source_outputs_per_image,
                                             quant_outputs_per_image):
        # TODO: Check whether using original CLIPVisionModel can improve
        # consistency against HF
        check_logprobs_close(
            outputs_0_lst=source_outputs,
            outputs_1_lst=quant_outputs,
            name_0="source",
            name_1="awq",
        )


@pytest.mark.parametrize(
    ("source_model", "quant_model"),
    [("OpenGVLab/InternVL2-2B", "OpenGVLab/InternVL2-2B-AWQ")],
)
@pytest.mark.parametrize(
    "size_factors",
    [
        # No image
        [],
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@torch.inference_mode()
def test_awq_models(vllm_runner, image_assets, source_model, quant_model,
                    size_factors, dtype, max_tokens, num_logprobs,
                    monkeypatch) -> None:

    # Test V1: this test hangs during setup on single-scale input.
    # TODO: fixure out why and re-enable this on V1.
    monkeypatch.setenv("VLLM_USE_V1", "0")
    run_awq_test(
        vllm_runner,
        image_assets,
        source_model,
        quant_model,
        size_factors=size_factors,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
