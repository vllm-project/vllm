# flake8: noqa
"""Tests fp8 models against ground truth generation
Note: these tests will only pass on L4 GPU.
"""
import os
from typing import Optional

import pytest

from tests.kernels.utils import override_backend_env_variable
from tests.quantization.utils import is_quant_method_supported

from ...utils import check_logprobs_close

os.environ["TOKENIZERS_PARALLELISM"] = "true"


@pytest.mark.skipif(not is_quant_method_supported("fp8"),
                    reason="fp8 is not supported on this GPU type.")
@pytest.mark.parametrize(
    "kv_cache_dtype,base_model,test_model,scale_path",
    [
        # Test FP8 checkpoint w. fp8_e4m3 kv-cache scaling factors.
        ("fp8_e4m3", "meta-llama/Meta-Llama-3-8B-Instruct",
         "nm-testing/Meta-Llama-3-8B-Instruct-FP8-KV", None),
        # Test FP16 checkpoint w. fp8_e5m2 kv-cache.
        ("fp8_e5m2", "meta-llama/Meta-Llama-3-8B-Instruct",
         "meta-llama/Meta-Llama-3-8B-Instruct", None),
        # Test FP16 checkpoint w. fp8_e4m3 kv-cache scaling factors in json.
        ("fp8_e4m3", "meta-llama/Llama-2-7b-chat-hf",
         "meta-llama/Llama-2-7b-chat-hf",
         "./tests/fp8_kv/llama2-7b-fp8-kv/kv_cache_scales.json")
    ])
# Due to low-precision numerical divergence, we only test logprob of 4 tokens
@pytest.mark.parametrize("max_tokens", [4])
@pytest.mark.parametrize("enforce_eager", [False, True])
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
    scale_path: Optional[str],
    max_tokens: int,
    enforce_eager: bool,
    backend: str,
    tensor_parallel_size: int,
    disable_async_output_proc: bool,
    monkeypatch,
) -> None:
    """
    Only checks log probs match to cover the discrepancy in
    numerical sensitive kernels.
    """
    override_backend_env_variable(monkeypatch, backend)

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

    extra_kwargs = {}
    if scale_path is not None:
        extra_kwargs["quantization_param_path"] = scale_path

    with vllm_runner(
            test_model,
            max_model_len=MAX_MODEL_LEN,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            kv_cache_dtype=kv_cache_dtype,
            disable_async_output_proc=disable_async_output_proc,
            **extra_kwargs,
    ) as vllm_model:
        test_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, NUM_LOG_PROBS)

    check_logprobs_close(
        outputs_0_lst=baseline_outputs,
        outputs_1_lst=test_outputs,
        name_0="fp16_kv_cache",
        name_1="fp8_kv_cache",
    )
