# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# flake8: noqa
"""Tests experts_int8 quantization startup and generation,
doesn't test correctness
"""

import pytest

from tests.quantization.utils import is_quant_method_supported

from ..models.registry import HF_EXAMPLE_MODELS

MODELS = ["ai21labs/Jamba-tiny-random", "pfnet/plamo-2-1b"]


@pytest.mark.skipif(
    not is_quant_method_supported("experts_int8"),
    reason="ExpertsInt8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [4])
def test_model_experts_int8_startup(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_transformers_version(on_fail="skip")

    with vllm_runner(
        model, dtype=dtype, enforce_eager=True, quantization="experts_int8"
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
