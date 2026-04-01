# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import pytest
import regex as re
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm.logprobs import SampleLogprobs
from vllm.multimodal.image import rescale_image_size

from ....conftest import (
    IMAGE_ASSETS,
    HfRunner,
    PromptImageInput,
    VllmRunner,
)
from ....utils import multi_gpu_test
from ...utils import check_logprobs_close

MODEL_ID = "microsoft/Phi-4-reasoning-vision-15B"

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts(
    {
        "stop_sign": "<|user|>\n<image>\nWhat's the content of the image?<|end|>\n<|assistant|>\n",  # noqa: E501
        "cherry_blossom": "<|user|>\n<image>\nPlease infer the season with reason in details.<|end|>\n<|assistant|>\n",  # noqa: E501
    }
)
HF_MULTIIMAGE_IMAGE_PROMPT = (
    "<|user|>\n<image>\n<image>\nDescribe these images.<|end|>\n<|assistant|>\n"  # noqa: E501
)

DTYPE = "half"
MAX_TOKENS = 128
NUM_LOGPROBS = 10


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


def _build_single_image_inputs(
    image_assets,
) -> list[tuple[list[str], PromptImageInput]]:
    """Build single-image inputs for all size_factors at once."""
    images = [asset.pil_image for asset in image_assets]
    all_inputs: list[tuple[list[str], PromptImageInput]] = []
    for size_factors in [[1.0], [0.25, 0.5, 1.0]]:
        for image, prompt in zip(images, HF_IMAGE_PROMPTS):
            all_inputs.append(
                (
                    [prompt for _ in size_factors],
                    [rescale_image_size(image, f) for f in size_factors],
                )
            )
    return all_inputs


def _build_multi_image_inputs(
    image_assets,
) -> list[tuple[list[str], PromptImageInput]]:
    """Build multi-image inputs for all size_factors at once."""
    images = [asset.pil_image for asset in image_assets]
    all_inputs: list[tuple[list[str], PromptImageInput]] = []
    for size_factors in [[1.0], [1.0, 1.0, 1.0], [0.25, 0.5, 1.0]]:
        all_inputs.append(
            (
                [HF_MULTIIMAGE_IMAGE_PROMPT for _ in size_factors],
                [
                    [rescale_image_size(image, factor) for image in images]
                    for factor in size_factors
                ],
            )
        )
    return all_inputs


def _run_and_compare(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    all_inputs: Sequence[tuple[list[str], PromptImageInput]],
    model: str,
    max_model_len: int,
    max_num_seqs: int,
    mm_limit: int,
    gpu_memory_utilization: float,
):
    """Load each runner once, run all inputs, then compare."""
    # NOTE: run vLLM first, then HF.  vLLM needs a fresh process without
    # cuda initialization; running HF first would break the multiprocessing
    # backend with fork method.
    with vllm_runner(
        model,
        runner="generate",
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=DTYPE,
        limit_mm_per_prompt={"image": mm_limit},
        tensor_parallel_size=2,
        trust_remote_code=True,
        enforce_eager=True,
    ) as vllm_model:
        vllm_outputs_per_case = [
            vllm_model.generate_greedy_logprobs(
                prompts,
                MAX_TOKENS,
                num_logprobs=NUM_LOGPROBS,
                images=images,
            )
            for prompts, images in all_inputs
        ]

    hf_model_kwargs = {"_attn_implementation": "sdpa"}
    with hf_runner(
        model,
        dtype=DTYPE,
        model_kwargs=hf_model_kwargs,
        auto_cls=AutoModelForCausalLM,
        trust_remote_code=True,
    ) as hf_model:
        hf_outputs_per_case = [
            hf_model.generate_greedy_logprobs_limit(
                prompts,
                MAX_TOKENS,
                num_logprobs=NUM_LOGPROBS,
                images=images,
            )
            for prompts, images in all_inputs
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_case, vllm_outputs_per_case):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model", [MODEL_ID])
def test_models(hf_runner, vllm_runner, image_assets, model) -> None:
    all_inputs = _build_single_image_inputs(image_assets)
    _run_and_compare(
        hf_runner,
        vllm_runner,
        all_inputs,
        model,
        max_model_len=8192,
        max_num_seqs=2,
        mm_limit=1,
        gpu_memory_utilization=0.80,
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model", [MODEL_ID])
def test_multi_images_models(hf_runner, vllm_runner, image_assets, model) -> None:
    all_inputs = _build_multi_image_inputs(image_assets)
    _run_and_compare(
        hf_runner,
        vllm_runner,
        all_inputs,
        model,
        max_model_len=8192,
        max_num_seqs=2,
        mm_limit=2,
        gpu_memory_utilization=0.80,
    )
