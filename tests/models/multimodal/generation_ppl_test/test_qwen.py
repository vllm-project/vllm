# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import GenerateModelInfo

from .ppl_utils import vqa_ppl_test

MODELS = [
    (
        GenerateModelInfo("Qwen/Qwen2-VL-2B-Instruct", hf_ppl=9.810060501098633),
        28,
        {},
    ),
    (
        GenerateModelInfo("Qwen/Qwen2.5-VL-3B-Instruct", hf_ppl=22.17719078063965),
        28,
        {},
    ),
    (
        GenerateModelInfo("Qwen/Qwen3-VL-4B-Instruct", hf_ppl=11.916606903076172),
        32,
        {},
    ),
    (
        GenerateModelInfo("Qwen/Qwen3.5-4B", hf_ppl=6.478141784667969),
        32,
        {"enable_prefix_caching": False},
    ),
]


def _pixel_factor(factor: int):
    """`factor = patch_size * merge_size` for Qwen models"""
    return {"min_pixels": factor**2, "max_pixels": 1280 * factor**2}


@pytest.mark.parametrize("model_info,factor,extra_kwargs", MODELS)
@pytest.mark.parametrize("mm_device_do_normalize", [True, False])
def test_ppl(
    hf_runner,
    vllm_runner,
    model_info: GenerateModelInfo,
    factor: int,
    extra_kwargs: dict,
    mm_device_do_normalize: bool,
    aiter_config: str,
):
    if aiter_config == "aiter_on":
        # AITER RMSNorm rounds once, unlike HF; keep Inductor's bf16 casts so
        # the surrounding ops round like HF (see ROCm/aiter#2236).
        extra_kwargs = {
            **extra_kwargs,
            "compilation_config": {
                "inductor_compile_config": {"emulate_precision_casts": True}
            },
        }
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
