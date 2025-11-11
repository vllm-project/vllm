# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import lm_eval
import pytest

from tests.utils import large_gpu_mark


def get_model_args(
    model_name: str,
    spec_model_name: str,
    spec_method: str,
    tp_size: int,
    model_max_len: int,
    use_async: bool = False,
) -> dict:
    speculative_config = {
        "method": spec_method,
        "model": spec_model_name,
        "num_speculative_tokens": 1,
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
        "eplb_window_size": 128,
        "eplb_step_interval": 1024,
        "eplb_log_balancedness": False,
        "enable_eplb": True,
        "max_model_len": model_max_len,
    }
    if use_async:
        model_args["eplb_config"] = {"use_async": True}
    return model_args


def _run_mtp_spec_decode_test(use_async: bool) -> None:
    TASK = "gsm8k"
    FILTER = "exact_match,strict-match"
    RTOL = 0.03

    model_args = get_model_args(
        model_name="Qwen/Qwen3-Next-80B-A3B-Instruct",
        spec_model_name=None,
        spec_method="mtp",
        tp_size=4,
        model_max_len=4096,
        use_async=use_async,
    )

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=TASK,
        batch_size=64,
        num_fewshot=8,
    )
    measured_value = results["results"][TASK][FILTER]
    expected_gsm8k_value = 0.86
    assert (
        measured_value - RTOL < expected_gsm8k_value
        and measured_value + RTOL > expected_gsm8k_value
    ), f"Expected: {expected_gsm8k_value} |  Measured: {measured_value}"


@large_gpu_mark(min_gb=80)
def test_eplb_spec_decode_mtp_sync() -> None:
    """Baseline EPLB + MTP run without async load balancing."""

    _run_mtp_spec_decode_test(use_async=False)


@large_gpu_mark(min_gb=80)
def test_eplb_spec_decode_mtp_async() -> None:
    """Same scenario as above but with async EPLB enabled."""

    _run_mtp_spec_decode_test(use_async=True)
