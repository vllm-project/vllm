# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.models.utils import GenerateModelInfo

from .ppl_utils import vqa_ppl_test

GEMMA_PPL_TOL = 0.02

MODELS = [
    GenerateModelInfo(
        "google/gemma-3-4b-it", image_token_id=262144, hf_ppl=119.61958312988281
    ),
]


@pytest.mark.parametrize("model_info", MODELS)
@pytest.mark.parametrize("mm_device_do_normalize", [False, True])
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
        tol=GEMMA_PPL_TOL,
    )


@pytest.mark.parametrize("model_info", MODELS)
@pytest.mark.parametrize("mm_device_do_normalize", [False, True])
def test_ppl_gc(
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
            "compilation_config": {"cudagraph_mm_encoder": True},
        },
        tol=GEMMA_PPL_TOL,
    )
