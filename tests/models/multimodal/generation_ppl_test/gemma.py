# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from .ppl_utils import vqa_ppl_test
from tests.models.utils import GenerateModelInfo

MODELS = [
    GenerateModelInfo("google/gemma-3-4b-it", image_token_id=262144),
]


@pytest.mark.parametrize("model_info", MODELS)
@pytest.mark.parametrize("mm_device_do_normalize", [True, False])
def test_ppl(
        hf_runner,
        vllm_runner,
        model_info: GenerateModelInfo,
        mm_device_do_normalize: bool,
):
    vqa_ppl_test(
        hf_runner,
        vllm_runner,
        model_info,
        vllm_extra_kwargs={
            "mm_device_do_normalize": mm_device_do_normalize,
        },
        tol=0.02
    )
