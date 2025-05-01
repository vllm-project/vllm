# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
"""Tests fp8 models against ground truth generation
Note: these tests will only pass on L4 GPU.
"""
import pytest

from tests.quantization.utils import is_quant_method_supported
from vllm.platforms import current_platform
from vllm.utils import STR_BACKEND_ENV_VAR

from ..utils import check_logprobs_close


@pytest.mark.skipif(not is_quant_method_supported("fp8"),
                    reason="fp8 is not supported on this GPU type.")
@pytest.mark.parametrize(
    "kv_cache_dtype,base_model,test_model",
    [
        # Test FP8 checkpoint w. fp8_e4m3 kv-cache scaling factors.
        ("fp8_e4m3", "meta-llama/Llama-3.2-1B-Instruct",
         "nm-testing/Llama-3.2-1B-Instruct-FP8-KV"),
        # Test BF16 checkpoint w. fp8_e5m2 kv-cache.
        ("fp8_e5m2", "meta-llama/Llama-3.2-1B-Instruct",
         "meta-llama/Llama-3.2-1B-Instruct"),
        # Test BF16 checkpoint w. fp8_e4m3 kv-cache scaling factors in json.
        ("fp8_e4m3", "meta-llama/Llama-3.2-1B-Instruct",
         "meta-llama/Llama-3.2-1B-Instruct")
    ])
# Due to low-precision numerical divergence, we only test logprob of 4 tokens
@pytest.mark.parametrize("max_tokens", [4])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("backend", ["FLASH_ATTN", "XFORMERS", "FLASHINFER"])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
@pytest.mark.parametrize("tensor_parallel_size", [1])
# Due to low-precision numerical divergence, this test is too sensitive for
# the async postprocessor
@pytest.mark.parametrize("disable_async_output_proc", [True])
def test_models(
    vllm_runner,
    example_prompts,
    kv_cache_dtype: str,
    base_model: str,
    test_model: str,
    max_tokens: int,
    enforce_eager: bool,
    backend: str,
    tensor_parallel_size: int,
    disable_async_output_proc: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Only checks log probs match to cover the discrepancy in
    numerical sensitive kernels.
    """
    with monkeypatch.context() as m:
        m.setenv("TOKENIZERS_PARALLELISM", 'true')
        m.setenv(STR_BACKEND_ENV_VAR, backend)

        MAX_MODEL_LEN = 1024
        NUM_LOG_PROBS = 8

        with vllm_runner(
                base_model,
                max_model_len=MAX_MODEL_LEN,
                tensor_parallel_size=tensor_parallel_size,
                enforce_eager=enforce_eager,
                kv_cache_dtype="auto",
                disable_async_output_proc=disable_async_output_proc,
        ) as vllm_model:
            baseline_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, max_tokens, NUM_LOG_PROBS)

        with vllm_runner(
                test_model,
                max_model_len=MAX_MODEL_LEN,
                tensor_parallel_size=tensor_parallel_size,
                enforce_eager=enforce_eager,
                kv_cache_dtype=kv_cache_dtype,
                disable_async_output_proc=disable_async_output_proc,
        ) as vllm_model:
            test_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, max_tokens, NUM_LOG_PROBS)

        check_logprobs_close(
            outputs_0_lst=baseline_outputs,
            outputs_1_lst=test_outputs,
            name_0="fp16_kv_cache",
            name_1="fp8_kv_cache",
        )


@pytest.mark.cpu_model
@pytest.mark.skipif(not current_platform.is_cpu(),
                    reason="test for the CPU backend.")
@pytest.mark.parametrize(
    "kv_cache_dtype,base_model,test_model",
    [
        # Test BF16 checkpoint w. fp8_e5m2 kv-cache.
        ("fp8_e5m2", "meta-llama/Llama-3.2-1B-Instruct",
         "meta-llama/Llama-3.2-1B-Instruct"),
    ])
# Due to low-precision numerical divergence, we only test logprob of 4 tokens
@pytest.mark.parametrize("max_tokens", [4])
# Due to low-precision numerical divergence, this test is too sensitive for
# the async postprocessor
@pytest.mark.parametrize("disable_async_output_proc", [True])
def test_cpu_models(
    vllm_runner,
    example_prompts,
    kv_cache_dtype: str,
    base_model: str,
    test_model: str,
    max_tokens: int,
    disable_async_output_proc: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Only checks log probs match to cover the discrepancy in
    numerical sensitive kernels.
    """
    with monkeypatch.context() as m:
        m.setenv("TOKENIZERS_PARALLELISM", 'true')

        MAX_MODEL_LEN = 1024
        NUM_LOG_PROBS = 8

        with vllm_runner(
                base_model,
                max_model_len=MAX_MODEL_LEN,
                dtype="bfloat16",
                kv_cache_dtype="auto",
                disable_async_output_proc=disable_async_output_proc,
        ) as vllm_model:
            baseline_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, max_tokens, NUM_LOG_PROBS)

        with vllm_runner(
                test_model,
                max_model_len=MAX_MODEL_LEN,
                dtype="bfloat16",
                kv_cache_dtype=kv_cache_dtype,
                disable_async_output_proc=disable_async_output_proc,
        ) as vllm_model:
            test_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, max_tokens, NUM_LOG_PROBS)

        check_logprobs_close(
            outputs_0_lst=baseline_outputs,
            outputs_1_lst=test_outputs,
            name_0="bf16_kv_cache",
            name_1="fp8_kv_cache",
        )
