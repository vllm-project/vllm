# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for dense Qwen3.5 sequence parallelism (`VLLM_RESIDUAL_SP_MIN_TOKENS`)."""

import pytest

from tests.models.registry import _HfExamplesInfo
from tests.models.utils import check_logprobs_close
from tests.utils import create_new_process_for_each_test
from vllm.platforms import current_platform

MODEL_ID = "vllm-project/Qwen3.5-0.8B-text-only"


@pytest.mark.skipif(
    not current_platform.is_xpu(), reason="Dense Qwen3.5 SP is XPU-only."
)
@create_new_process_for_each_test()
def test_tp_sp_generation(vllm_runner, monkeypatch, num_gpus_available):
    """SP generation matches the plain TP baseline, on token counts that
    don't divide by TP and on a prefill long enough for the fused kernels."""
    if num_gpus_available < 2:
        pytest.skip("Need at least 2 GPUs")
    model_info = _HfExamplesInfo(MODEL_ID)
    model_info.check_transformers_version(on_fail="skip")
    model_info.check_available_online(on_fail="skip")

    prompts = [
        "Hello, my name is",
        "Summarize: " + "The quick brown fox jumps over the lazy dog. " * 260,
    ]
    outputs = []
    for min_tokens in ("0", "1"):
        monkeypatch.setenv("VLLM_RESIDUAL_SP_MIN_TOKENS", min_tokens)
        with vllm_runner(
            MODEL_ID,
            dtype="bfloat16",
            max_model_len=4096,
            max_num_batched_tokens=4096,
            max_num_seqs=8,
            enforce_eager=True,
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
        ) as llm:
            outputs.append(
                llm.generate_greedy_logprobs(prompts, max_tokens=8, num_logprobs=5)
            )
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="tp",
        name_1="tp+sp",
    )
