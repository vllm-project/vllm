# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from vllm.multimodal.image import rescale_image_size
from vllm.platforms import current_platform

from ...conftest import IMAGE_ASSETS, ImageTestAssets, VllmRunner
from ..utils import check_logprobs_close

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts(
    {
        "stop_sign": "<|im_start|>User\n<image>\nWhat's the content in the center of the image?<|im_end|>\n<|im_start|>Assistant\n",  # noqa: E501
        "cherry_blossom": "<|im_start|>User\n<image>\nWhat is the season?<|im_end|>\n<|im_start|>Assistant\n",  # noqa: E501
    }
)

IMAGE_SIZE_FACTOR_GROUPS = (
    # Single-scale
    (1.0,),
    # Single-scale, batched
    (1.0, 1.0, 1.0),
    # Multi-scale
    (0.25, 0.5, 1.0),
)


def run_awq_test(
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
    source_model: str,
    quant_model: str,
    *,
    size_factor_groups: tuple[tuple[float, ...], ...],
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: str | None = None,
):
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image_and_size_group = [
        (
            [prompt for _ in size_factors],
            [rescale_image_size(image, factor) for factor in size_factors],
        )
        for size_factors in size_factor_groups
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
        enforce_eager=True,
        default_torch_num_threads=1,
    ) as vllm_model:
        source_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(
                prompts, max_tokens, num_logprobs=num_logprobs, images=images
            )
            for prompts, images in inputs_per_image_and_size_group
        ]

    with vllm_runner(
        quant_model,
        quantization="awq",
        max_model_len=4096,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
        enforce_eager=True,
        default_torch_num_threads=1,
    ) as vllm_model:
        quant_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(
                prompts, max_tokens, num_logprobs=num_logprobs, images=images
            )
            for prompts, images in inputs_per_image_and_size_group
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
    ("model", "quantization", "dtype"),
    [
        ("mattbucci/gemma-4-26B-AWQ", "awq", "float16"),
        ("cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit", "compressed-tensors", "bfloat16"),
    ],
    ids=[
        "gemma4-moe-standard-awq-dot-suffix",
        "gemma4-moe-compressed-tensors-underscore-suffix",
    ],
)
@torch.inference_mode()
def test_awq_load(
    vllm_runner: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    quantization: str,
    dtype: str,
) -> None:
    """Regression test: AWQ weight loading must not KeyError."""
    with vllm_runner(
        model,
        quantization=quantization,
        dtype=dtype,
        max_model_len=128,
        enforce_eager=True,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts[:2], max_tokens=32)
    assert len(outputs) == 2


@pytest.mark.parametrize(
    ("source_model", "quant_model"),
    [("OpenGVLab/InternVL2-2B", "OpenGVLab/InternVL2-2B-AWQ")],
)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@torch.inference_mode()
def test_awq_models(
    vllm_runner,
    image_assets,
    source_model,
    quant_model,
    dtype,
    max_tokens,
    num_logprobs,
) -> None:
    run_awq_test(
        vllm_runner,
        image_assets,
        source_model,
        quant_model,
        size_factor_groups=IMAGE_SIZE_FACTOR_GROUPS,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )


def _on_rdna() -> bool:
    """Lazy RDNA check: importing vllm.platforms.rocm reads gcnArchName, so it
    must not run at collection time on non-ROCm hosts."""
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1x

    return on_gfx1x()


@pytest.mark.skipif(
    not _on_rdna(),
    reason="the RDNA W4A16 path is gfx1x-only; CDNA keeps the failing bf16 path",
)
@torch.inference_mode()
def test_awq_bfloat16_on_rocm(
    vllm_runner: type[VllmRunner],
    example_prompts: list[str],
) -> None:
    """AWQ must run in bfloat16 on RDNA.

    Regression test for issue #63455: the AutoAWQ config returned the generic AWQ
    linear method, whose kernel calls tl.dot(..., out_dtype=bfloat16), which Triton
    rejects, so a bfloat16 AWQ model failed at the first forward pass. Selecting a
    mixed-precision W4A16 kernel on RDNA makes bfloat16 work.
    """
    with vllm_runner(
        "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        quantization="awq",
        dtype="bfloat16",
        max_model_len=256,
        enforce_eager=True,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts[:2], max_tokens=16)

    assert len(outputs) == 2
    assert all(text for _, text in outputs), "AWQ model produced empty output"
