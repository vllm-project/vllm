# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import GenerateModelInfo

from .ppl_utils import vqa_ppl_test

MODELS = [
    (
        GenerateModelInfo("Qwen/Qwen2-VL-2B-Instruct", hf_ppl=4.666022300720215),
        28,
        {},
    ),
    (
        GenerateModelInfo("Qwen/Qwen2.5-VL-3B-Instruct", hf_ppl=4.172671794891357),
        28,
        {},
    ),
    (
        GenerateModelInfo("Qwen/Qwen3-VL-4B-Instruct", hf_ppl=6.274145603179932),
        32,
        {},
    ),
    (
        GenerateModelInfo("Qwen/Qwen3.5-4B", hf_ppl=4.26973819732666),
        32,
        {"enable_prefix_caching": False},
    ),
]


def _pixel_factor(factor: int):
    """`factor = patch_size * merge_size` for Qwen models"""
    return {"min_pixels": factor**2, "max_pixels": 1280 * factor**2}


@pytest.mark.usefixtures("aiter_config")
@pytest.mark.parametrize("model_info,factor,extra_kwargs", MODELS)
@pytest.mark.parametrize("mm_device_do_normalize", [True, False])
def test_ppl(
    hf_runner,
    vllm_runner,
    model_info: GenerateModelInfo,
    factor: int,
    extra_kwargs: dict,
    mm_device_do_normalize: bool,
):
    vqa_ppl_test(
        hf_runner,
        vllm_runner,
        model_info,
        vllm_extra_kwargs={
            "mm_device_do_normalize": mm_device_do_normalize,
            **extra_kwargs,
        },
        mm_processor_kwargs=_pixel_factor(factor),
    )
