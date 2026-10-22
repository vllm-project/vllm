# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether unsupported GPTQ group actorder models are rejected.

Run `pytest tests/quantization/test_gptq_dynamic.py --forked`.

Note: Only symmetric GPTQ models are supported after consolidation to Marlin.
"""

import pytest

MODELS = [
    "ModelCloud/Qwen1.5-1.8B-Chat-GPTQ-4bits-dynamic-cfg-with-lm_head-symTrue",
]


@pytest.mark.parametrize("model_id", MODELS)
def test_gptq_with_dynamic_desc_act_rejected(vllm_runner, model_id: str):
    with (
        pytest.raises(ValueError, match="desc_act=True"),
        vllm_runner(model_id, max_model_len=2048, enforce_eager=True),
    ):
        pass
