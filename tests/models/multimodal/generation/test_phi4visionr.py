# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import pytest
import regex as re
from transformers import AutoTokenizer

from vllm.logprobs import SampleLogprobs
from vllm.multimodal.image import rescale_image_size

from ....conftest import (
    IMAGE_ASSETS,
    HfRunner,
    PromptImageInput,
    VllmRunner,
)
from ....utils import large_gpu_test
from ...utils import check_logprobs_close

MODEL_ID = "microsoft/Phi-4-reasoning-vision-15B"
models = [MODEL_ID]

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts(
    {
        "stop_sign": "<|user|>\n<image>\nWhat's the content of the image?<|end|>\n<|assistant|>\n",  # noqa: E501
        "cherry_blossom": "<|user|>\n<image>\nPlease infer the season with reason in details.<|end|>\n<|assistant|>\n",  # noqa: E501
    }
)
HF_MULTIIMAGE_IMAGE_PROMPT = "<|user|>\n<image>\n<image>\nDescribe these images.<|end|>\n<|assistant|>\n"  # noqa: E501

target_dtype = "half"


def vllm_to_hf_output(
    vllm_output: tuple[list[int], str, SampleLogprobs | None], model: str
):
    """Sanitize vllm output to be comparable with hf output."""
    _, output_str, out_logprobs = vllm_output

    output_str_without_image = re.sub(r"(<image>)+", "", output_str)
    if output_str_without_image and output_str_without_image[0] == " ":
        output_str_without_image = output_str_without_image[1:]

    hf_output_str = output_str_without_image + "<|end|><|endoftext|>"

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    hf_output_ids = tokenizer.encode(output_str_without_image)
    if hf_output_ids and hf_output_ids[0] == tokenizer.bos_token_id:
        hf_output_ids = hf_output_ids[1:]

    return hf_output_ids, hf_output_str, out_logprobs


def run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    inputs: Sequence[tuple[list[str], PromptImageInput]],
    model: str,
    *,
    max_model_len: int,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    mm_limit: int,
    tensor_parallel_size: int,
    distributed_executor_backend: str | None = None,
):
    # NOTE: run vLLM first, then HF. vLLM needs a fresh process without
    # cuda initialization; running HF first would hurt the multiprocessing
    # backend with fork method.
    with vllm_runner(
        model,
        runner="generate",
        max_model_len=max_model_len,
        max_num_seqs=2,
        dtype=dtype,
        limit_mm_per_prompt={"image": mm_limit},
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
        trust_remote_code=True,
        enforce_eager=True,
    ) as vllm_model:
        vllm_outputs_per_case = [
            vllm_model.generate_greedy_logprobs(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                images=images,
            )
            for prompts, images in inputs
        ]

    # TODO: enable HF comparison once the custom model is compatible
    pytest.skip(
        "HF custom model for Phi-4-reasoning-vision is not yet compatible"
    )

    hf_model_kwargs = {"_attn_implementation": "sdpa"}
    with hf_runner(
        model,
        dtype=dtype,
        model_kwargs=hf_model_kwargs,
        auto_cls="AutoModelForCausalLM",
        trust_remote_code=True,
    ) as hf_model:
        hf_outputs_per_case = [
            hf_model.generate_greedy_logprobs_limit(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                images=images,
            )
            for prompts, images in inputs
        ]

    for hf_outputs, vllm_outputs in zip(
        hf_outputs_per_case, vllm_outputs_per_case
    ):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


@large_gpu_test(min_gb=48)
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "size_factors",
    [
        [1.0],
        [1.0, 1.0, 1.0],
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_model_len", [4096])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [10])
def test_models(
    hf_runner,
    vllm_runner,
    image_assets,
    model,
    size_factors,
    dtype: str,
    max_model_len: int,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [
        (
            [prompt for _ in size_factors],
            [rescale_image_size(image, factor) for factor in size_factors],
        )
        for image, prompt in zip(images, HF_IMAGE_PROMPTS)
    ]

    run_test(
        hf_runner,
        vllm_runner,
        inputs_per_image,
        model,
        dtype=dtype,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        mm_limit=1,
        tensor_parallel_size=1,
    )


@large_gpu_test(min_gb=48)
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "size_factors",
    [
        [1.0],
        [1.0, 1.0, 1.0],
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_model_len", [8192])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [10])
def test_multi_images_models(
    hf_runner,
    vllm_runner,
    image_assets,
    model,
    size_factors,
    dtype: str,
    max_model_len: int,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    images = [asset.pil_image for asset in image_assets]

    inputs_per_case = [
        (
            [HF_MULTIIMAGE_IMAGE_PROMPT for _ in size_factors],
            [
                [rescale_image_size(image, factor) for image in images]
                for factor in size_factors
            ],
        ),
    ]

    run_test(
        hf_runner,
        vllm_runner,
        inputs_per_case,
        model,
        dtype=dtype,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        mm_limit=2,
        tensor_parallel_size=1,
    )
