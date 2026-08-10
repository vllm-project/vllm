# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import GenerateModelInfo

from .ppl_utils import vqa_ppl_test

MODELS = [
    GenerateModelInfo("Qwen/Qwen2-VL-2B-Instruct", hf_ppl=41081356.0),
    GenerateModelInfo("Qwen/Qwen2.5-VL-3B-Instruct", hf_ppl=18330016.0),
]


mm_processor_kwargs = {
    "min_pixels": 28 * 28,
    "max_pixels": 1280 * 28 * 28,
}


@pytest.mark.parametrize("model_info", MODELS)
@pytest.mark.parametrize("mm_device_do_normalize", [True, False])
def test_ppl(
    hf_runner, vllm_runner, model_info: GenerateModelInfo, mm_device_do_normalize: bool
):
    vqa_ppl_test(
        hf_runner,
        vllm_runner,
        model_info,
        vllm_extra_kwargs={"mm_device_do_normalize": mm_device_do_normalize},
        mm_processor_kwargs=mm_processor_kwargs,
    )

@pytest.mark.parametrize("model_info", MODELS[:1])
def test_pshm(hf_runner, vllm_runner, model_info: GenerateModelInfo):
    vqa_ppl_test(
        hf_runner,
        vllm_runner,
        model_info,
        vllm_extra_kwargs={"paged_shm_size": 1073741824},
        mm_processor_kwargs=mm_processor_kwargs,
    )