# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import pytest

from tests.evals.gsm8k.gsm8k_eval import (
    assert_gsm8k_result,
    evaluate_gsm8k_lm_eval,
    get_gsm8k_eval_spec,
)
from tests.utils import large_gpu_mark
from vllm.platforms import current_platform


def get_model_args(
    model_name: str,
    spec_model_name: str | None,
    spec_method: str,
    tp_size: int,
    model_max_len: int,
    use_async: bool = True,
) -> dict:
    speculative_config = {
        "method": spec_method,
        "model": spec_model_name,
        "num_speculative_tokens": 1,
        "max_model_len": model_max_len,
    }
    eplb_config = {
        "num_redundant_experts": tp_size,
        "window_size": 128,
        "step_interval": 1024,
        "log_balancedness": False,
        "use_async": use_async,
    }
    model_args = {
        "pretrained": model_name,
        "dtype": "auto",
        "add_bos_token": True,
        "tensor_parallel_size": tp_size,
        "gpu_memory_utilization": 0.7,
        "speculative_config": speculative_config,
        "enable_expert_parallel": True,
        "eplb_config": eplb_config,
        "enable_eplb": True,
        "max_model_len": model_max_len,
    }
    return model_args


pytestmark = pytest.mark.skipif(
    current_platform.is_rocm(),
    reason="EPLB with Spec Decode is a work in progress on ROCm.",
)


@pytest.mark.parametrize(
    "model_setup",
    [
        pytest.param(
            ("mtp", "qwen3-next-mtp", None, 4),
            marks=large_gpu_mark(min_gb=80),
        ),
        pytest.param(
            (
                "eagle",
                "llama4-scout-eagle",
                "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct",
                4,
            ),
            marks=pytest.mark.skip(reason="Skipping due to CI OOM issues"),
        ),
    ],
    ids=["qwen3_next_mtp", "llama4_eagle"],
)
def test_eplb_spec_decode(
    monkeypatch: pytest.MonkeyPatch,
    model_setup: tuple[str, str, str | None, int],
):
    """
    Test the correctness of EPLB speculative decoding with GSM8K dataset.
    Applicable to MoE models with mtp or eagle spec decode.
    """
    method, eval_id, spec_model_name, tp_size = model_setup
    gsm8k_spec = get_gsm8k_eval_spec("eplb_spec_decode", eval_id)
    assert gsm8k_spec.model is not None

    model_args = get_model_args(
        model_name=gsm8k_spec.model,
        spec_model_name=spec_model_name,
        spec_method=method,
        tp_size=tp_size,
        model_max_len=4096,
    )

    result = evaluate_gsm8k_lm_eval(
        model="vllm",
        model_args=model_args,
        batch_size=64,
        **gsm8k_spec.lm_eval_kwargs(),
    )
    assert_gsm8k_result(result, gsm8k_spec)


@large_gpu_mark(min_gb=80)
def test_eplb_spec_decode_qwen3_next_mtp_async() -> None:
    """
    Ensure async EPLB works with MTP speculative decoding for Qwen3-Next.
    """

    gsm8k_spec = get_gsm8k_eval_spec("eplb_spec_decode", "qwen3-next-mtp-async")
    assert gsm8k_spec.model is not None

    model_args = get_model_args(
        model_name=gsm8k_spec.model,
        spec_model_name=None,
        spec_method="mtp",
        tp_size=4,
        model_max_len=4096,
        use_async=True,
    )

    result = evaluate_gsm8k_lm_eval(
        model="vllm",
        model_args=model_args,
        batch_size=64,
        **gsm8k_spec.lm_eval_kwargs(),
    )
    assert_gsm8k_result(result, gsm8k_spec)
