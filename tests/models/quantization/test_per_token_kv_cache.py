# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end accuracy tests for per-token-head KV cache quantization.

Compares logprobs between a baseline bf16 model and the same model with
per-token-head quantized KV cache (int4, int8 or fp8) using the Triton
attention backend, on the prompt tokens and on the greedy continuation.

Run: pytest tests/models/quantization/test_per_token_kv_cache.py -v -s
"""

import pytest

from vllm.platforms import current_platform

from ..utils import check_logprobs_close, check_logprobs_shift

# (mean, largest) logprob shift from bf16 allowed per position, see
# `check_logprobs_shift`. On MI355X the mean is 0.21-0.24 for int4, 0.04-0.05
# for int8 and 0.07-0.08 for fp8 with every request grouping and GEMM kernel
# set; the largest is at most 1.51, 0.28 and 0.70.
LOGPROB_SHIFT_BOUNDS = {
    "int4_per_token_head": (0.3, 2.0),
    "int8_per_token_head": (0.065, 0.45),
    "fp8_per_token_head": (0.105, 1.0),
}


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
    "kv_cache_dtype",
    ["int4_per_token_head", "int8_per_token_head", "fp8_per_token_head"],
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

    The logprob shift is bounded per dtype at every prompt and generated
    position that shares its context with the baseline. int8 and fp8 must also
    keep the baseline's greedy token in their top logprobs at the first
    divergence. int4 cannot: its 16 levels move a flat distribution by more
    than its top-8 spans (#69766), so it is held to the shift bounds alone.
    """
    with monkeypatch.context() as m:
        m.setenv("TOKENIZERS_PARALLELISM", "true")
        # An in-process engine steps only once every prompt is queued, so the
        # request grouping, and with it the logprobs, repeat from run to run.
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

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
                example_prompts,
                max_tokens,
                NUM_LOG_PROBS,
                num_prompt_logprobs=NUM_LOG_PROBS,
            )

        with vllm_runner(
            test_model,
            max_model_len=MAX_MODEL_LEN,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            kv_cache_dtype=kv_cache_dtype,
            attention_config={"backend": backend},
        ) as vllm_model:
            test_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts,
                max_tokens,
                NUM_LOG_PROBS,
                num_prompt_logprobs=NUM_LOG_PROBS,
            )

        max_mean_shift, max_shift = LOGPROB_SHIFT_BOUNDS[kv_cache_dtype]
        check_logprobs_shift(
            outputs_0_lst=baseline_outputs,
            outputs_1_lst=test_outputs,
            name_0="bf16_kv_cache",
            name_1=f"{kv_cache_dtype}_kv_cache",
            num_logprobs=NUM_LOG_PROBS,
            max_mean_shift=max_mean_shift,
            max_shift=max_shift,
        )

        if kv_cache_dtype != "int4_per_token_head":
            check_logprobs_close(
                outputs_0_lst=[outputs[:3] for outputs in baseline_outputs],
                outputs_1_lst=[outputs[:3] for outputs in test_outputs],
                name_0="bf16_kv_cache",
                name_1=f"{kv_cache_dtype}_kv_cache",
            )
