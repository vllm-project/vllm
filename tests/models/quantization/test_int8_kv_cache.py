# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end accuracy tests for INT8 KV cache quantization.

Compares logprobs between a baseline bf16 model and the same model with
kv_cache_dtype="int8_per_token" using the Triton attention backend. Since no
pre-calibrated INT8 KV scale checkpoints exist yet, we test with
calculate_kv_scales=True (dynamic per-head scales from the first batch).

Run: pytest tests/models/quantization/test_int8_kv_cache.py -v -s
"""

import pytest

from vllm.platforms import current_platform

from ..utils import check_logprobs_close


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_rocm()),
    reason="INT8 KV cache requires CUDA or ROCm GPU.",
)
@pytest.mark.parametrize(
    "base_model,test_model",
    [
        # BF16 model with dynamic INT8 KV cache quantization
        (
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
        ),
    ],
)
@pytest.mark.parametrize("max_tokens", [4])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("backend", ["TRITON_ATTN"])
@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_int8_kv_cache_accuracy(
    vllm_runner,
    example_prompts,
    base_model: str,
    test_model: str,
    max_tokens: int,
    enforce_eager: bool,
    backend: str,
    tensor_parallel_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare logprobs between bf16 baseline and INT8 KV cache.

    Uses calculate_kv_scales (dynamic scale computation) since there are
    no INT8-calibrated checkpoints available yet.
    """
    with monkeypatch.context() as m:
        m.setenv("TOKENIZERS_PARALLELISM", "true")

        MAX_MODEL_LEN = 1024
        NUM_LOG_PROBS = 8

        with vllm_runner(
            base_model,
            max_model_len=MAX_MODEL_LEN,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            kv_cache_dtype="auto",
            attention_config={"backend": backend},
        ) as vllm_model:
            baseline_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, max_tokens, NUM_LOG_PROBS
            )

        with vllm_runner(
            test_model,
            max_model_len=MAX_MODEL_LEN,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            kv_cache_dtype="int8_per_token",
            calculate_kv_scales=True,
            attention_config={"backend": backend},
        ) as vllm_model:
            test_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, max_tokens, NUM_LOG_PROBS
            )

        check_logprobs_close(
            outputs_0_lst=baseline_outputs,
            outputs_1_lst=test_outputs,
            name_0="bf16_kv_cache",
            name_1="int8_kv_cache",
        )
