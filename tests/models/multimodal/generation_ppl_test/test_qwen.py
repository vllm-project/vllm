# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import GenerateModelInfo

from .ppl_utils import vqa_ppl_test

MODELS = [
    GenerateModelInfo("Qwen/Qwen2-VL-2B-Instruct"),
    GenerateModelInfo("Qwen/Qwen2.5-VL-3B-Instruct"),
]


mm_processor_kwargs = {
    "min_pixels": 28 * 28,
    "max_pixels": 1280 * 28 * 28,
}


@pytest.mark.parametrize("model_info", MODELS)
def test_ppl(hf_runner, vllm_runner, model_info: GenerateModelInfo):
    vqa_ppl_test(
        hf_runner, vllm_runner, model_info, mm_processor_kwargs=mm_processor_kwargs
    )
