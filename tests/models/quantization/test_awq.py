# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from unittest.mock import patch

import pytest
import torch

from vllm.multimodal.image import rescale_image_size

from ...conftest import IMAGE_ASSETS, ImageTestAssets, VllmRunner
from ..utils import check_logprobs_close

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts(
    {
        "stop_sign": "<|im_start|>User\n<image>\nWhat's the content in the center of the image?<|im_end|>\n<|im_start|>Assistant\n",  # noqa: E501
        "cherry_blossom": "<|im_start|>User\n<image>\nWhat is the season?<|im_end|>\n<|im_start|>Assistant\n",  # noqa: E501
    }
)


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
    distributed_executor_backend: str | None = None,
    enforce_eager: bool = True,
):
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [
        (
            [prompt for _ in size_factors],
            [rescale_image_size(image, factor) for factor in size_factors],
        )
        for image, prompt in zip(images, HF_IMAGE_PROMPTS)
    ]

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    # max_model_len should be greater than image_feature_size
    with vllm_runner(
        source_model,
        max_model_len=4096,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
        enforce_eager=enforce_eager,
        default_torch_num_threads=1,
    ) as vllm_model:
        source_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(
                prompts, max_tokens, num_logprobs=num_logprobs, images=images
            )
            for prompts, images in inputs_per_image
        ]

    with vllm_runner(
        quant_model,
        quantization="awq",
        max_model_len=4096,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
        enforce_eager=enforce_eager,
        default_torch_num_threads=1,
    ) as vllm_model:
        quant_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(
                prompts, max_tokens, num_logprobs=num_logprobs, images=images
            )
            for prompts, images in inputs_per_image
        ]

    for source_outputs, quant_outputs in zip(
        source_outputs_per_image, quant_outputs_per_image
    ):
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
@pytest.mark.parametrize("enforce_eager", [True, False])
@torch.inference_mode()
def test_awq_models(
    vllm_runner,
    image_assets,
    source_model,
    quant_model,
    size_factors,
    dtype,
    max_tokens,
    num_logprobs,
    enforce_eager,
) -> None:
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
        enforce_eager=enforce_eager,
    )


@pytest.mark.parametrize("num_tokens,expect_gemm", [
    (1, True),
    (128, True),
    (255, True),
    (256, False),
    (512, False),
])
@torch.inference_mode()
def test_awq_linear_kernel_dispatch(num_tokens, expect_gemm):
    """Verify awq_linear dispatches to awq_gemm for small batch sizes
    and to awq_dequantize + matmul for large batch sizes."""
    from vllm._custom_ops import awq_linear

    in_features = 64
    out_features = 128
    group_size = 32
    pack_factor = 8

    input = torch.randn(num_tokens, in_features, dtype=torch.float16,
                         device="cuda")
    qweight = torch.randint(0, 100, (in_features, out_features // pack_factor),
                            dtype=torch.int32, device="cuda")
    scales = torch.randn(in_features // group_size, out_features,
                         dtype=torch.float16, device="cuda")
    qzeros = torch.randint(0, 100,
                           (in_features // group_size,
                            out_features // pack_factor),
                           dtype=torch.int32, device="cuda")

    # Mock both paths and return tensors of the correct shape
    gemm_ret = torch.empty(num_tokens, out_features, dtype=torch.float16,
                           device="cuda")
    deq_ret = torch.empty(in_features, out_features, dtype=torch.float16,
                          device="cuda")

    with patch("vllm._custom_ops.awq_gemm", return_value=gemm_ret) as mock_gemm, \
         patch("vllm._custom_ops.awq_dequantize", return_value=deq_ret) as mock_deq:
        awq_linear(input, qweight, scales, qzeros, pack_factor)

        if expect_gemm:
            mock_gemm.assert_called_once()
            mock_deq.assert_not_called()
        else:
            mock_deq.assert_called_once()
            mock_gemm.assert_not_called()
