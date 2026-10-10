# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import GenerateModelInfo

from .ppl_utils import vqa_ppl_test

MODELS = [
    (GenerateModelInfo("zai-org/GLM-OCR", hf_ppl=13.885052680969238), 28),
]


def _pixel_size(factor: int):
    # Match max image token size with qwen's PPL test
    return {
        "size": {
            "shortest_edge": 2 * factor**2,
            "longest_edge": 2 * 1280 * factor**2,
        }
    }


@pytest.mark.parametrize("model_info,factor", MODELS)
@pytest.mark.parametrize("mm_device_do_normalize", [True, False])
def test_ppl(
    hf_runner,
    vllm_runner,
    model_info: GenerateModelInfo,
    factor: int,
    mm_device_do_normalize: bool,
):
    vqa_ppl_test(
        hf_runner,
        vllm_runner,
        model_info,
        vllm_extra_kwargs={"mm_device_do_normalize": mm_device_do_normalize},
        mm_processor_kwargs=_pixel_size(factor),
    )
