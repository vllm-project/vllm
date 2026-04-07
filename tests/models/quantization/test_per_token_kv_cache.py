# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end accuracy tests for per-token-head KV cache quantization.

Compares logprobs between a baseline bf16 model and the same model with
per-token-head quantized KV cache (int8 or fp8) using the Triton attention
backend.

Run: pytest tests/models/quantization/test_per_token_kv_cache.py -v -s
"""

import pytest

from vllm.platforms import current_platform

from ..utils import check_logprobs_close


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Per-token-head KV cache requires CUDA or ROCm GPU.",
)
@pytest.mark.parametrize(
    "base_model,test_model",
    [
        (
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
        ),
    ],
)
@pytest.mark.parametrize(
    "kv_cache_dtype", ["int8_per_token_head", "fp8_per_token_head"]
)
@pytest.mark.parametrize("max_tokens", [4])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("backend", ["TRITON_ATTN"])
@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_per_token_head_kv_cache_accuracy(
    vllm_runner,
    example_prompts,
    base_model: str,
    test_model: str,
    kv_cache_dtype: str,
    max_tokens: int,
    enforce_eager: bool,
    backend: str,
    tensor_parallel_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare logprobs between bf16 baseline and per-token-head quantized KV
    cache.

    Uses calculate_kv_scales (dynamic scale computation) since there are
    no per-token-head calibrated checkpoints available yet.
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
            kv_cache_dtype=kv_cache_dtype,
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
            name_1=f"{kv_cache_dtype}_kv_cache",
        )
