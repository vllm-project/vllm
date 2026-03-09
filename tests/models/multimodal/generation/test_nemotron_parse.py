# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import pytest
from transformers import AutoModel

from tests.models.utils import check_logprobs_close
from vllm.assets.image import ImageAsset

from ....conftest import HfRunner, PromptImageInput, VllmRunner
from ....utils import create_new_process_for_each_test

IMAGE = ImageAsset("paper-11").pil_image_ext(ext="png").convert("RGB")
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"


def run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    inputs: Sequence[tuple[list[str], PromptImageInput]],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    """Verify that the inference result is the same between hf and vllm."""
    with vllm_runner(
        model,
        dtype=dtype,
        max_num_seqs=64,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
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

    with hf_runner(model, dtype=dtype, auto_cls=AutoModel) as hf_model:
        hf_outputs_per_case = [
            hf_model.generate_greedy_logprobs_limit(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                images=images,
                use_cache=False,  # HF Nemotron Parse crashes here without this
            )
            for prompts, images in inputs
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_case, vllm_outputs_per_case):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


@pytest.mark.core_model
@pytest.mark.parametrize("model", ["nvidia/NVIDIA-Nemotron-Parse-v1.1"])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("num_logprobs", [5])
@create_new_process_for_each_test("spawn")
def test_models(
    hf_runner, vllm_runner, model: str, dtype: str, num_logprobs: int
) -> None:
    run_test(
        hf_runner,
        vllm_runner,
        inputs=[
            (
                [PROMPT] * 10,
                [IMAGE] * 10,
            ),
        ],
        model=model,
        dtype=dtype,
        max_tokens=100,
        num_logprobs=num_logprobs,
    )
