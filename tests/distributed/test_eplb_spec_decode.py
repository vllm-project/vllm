# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import lm_eval
import pytest


def get_model_args(
    model_name: str,
    spec_model_name: str,
    spec_method: str,
    tp_size: int,
    model_max_len: int,
) -> dict:
    speculative_config = {
        "method": spec_method,
        "model": spec_model_name,
        "num_speculative_tokens": 4,
        "max_model_len": model_max_len,
    }

    model_args = {
        "pretrained": model_name,
        "dtype": "auto",
        "add_bos_token": True,
        "tensor_parallel_size": tp_size,
        "gpu_memory_utilization": 0.7,
        "speculative_config": speculative_config,
        "enable_expert_parallel": True,
        "num_redundant_experts": tp_size,
        "eplb_window_size": 32,
        "eplb_step_interval": 128,
        "eplb_log_balancedness": False,
        "enable_eplb": True,
        "max_model_len": model_max_len,
    }
    return model_args


@pytest.mark.parametrize(
    "model_setup",
    [
        pytest.param(
            ("qwen3_next_mtp", "Qwen/Qwen3-Next-80B-A3B-Instruct", None, 4, 0.86),
        ),
        pytest.param(
            (
                "eagle",
                "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct",
                4,
                0.86,
            ),
            marks=pytest.mark.skip(reason="Skipping due to CI OOM issues"),
        ),
    ],
    ids=["qwen3_next_mtp", "llama4_eagle"],
)
def test_eplb_spec_decode(
    monkeypatch: pytest.MonkeyPatch,
    model_setup: tuple[str, str, str, int, float],
):
    """
    Test the correctness of EPLB speculative decoding with GSM8K dataset.
    Applicable to MoE models with mtp or eagle spec decode.
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_MLA_DISABLE", "1")

        method, model_name, spec_model_name, tp_size, expected_gsm8k_value = model_setup

        TASK = "gsm8k"
        FILTER = "exact_match,strict-match"
        RTOL = 0.03

        model_args = get_model_args(
            model_name=model_name,
            spec_model_name=spec_model_name,
            spec_method=method,
            tp_size=tp_size,
            model_max_len=4096,
        )

        results = lm_eval.simple_evaluate(
            model="vllm",
            model_args=model_args,
            tasks=TASK,
            batch_size=64,
            num_fewshot=8,
        )
        measured_value = results["results"][TASK][FILTER]
        assert (
            measured_value - RTOL < expected_gsm8k_value
            and measured_value + RTOL > expected_gsm8k_value
        ), f"Expected: {expected_gsm8k_value} |  Measured: {measured_value}"
