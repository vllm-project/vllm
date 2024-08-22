"""Compare the outputs of HF and vLLM when using greedy sampling.

It tests chunked prefill. Chunked prefill can be enabled by
enable_chunked_prefill=True. If prefill size exceeds max_num_batched_tokens,
prefill requests are chunked.

Run `pytest tests/models/test_chunked_prefill.py`.
"""

import pytest

from ..models.utils import check_logprobs_close, check_outputs_equal

MODELS = [
    "facebook/opt-125m",
    "meta-llama/Llama-2-7b-hf",
]
E5M2_KV_MODELS = [
    "facebook/opt-125m",
    "meta-llama/Llama-2-7b-chat-hf",
]
E4M3_KV_MODELS = [
    "meta-llama/Llama-2-7b-chat-hf", "nm-testing/Qwen2-1.5B-Instruct-FP8-K-V",
    "nm-testing/TinyLlama-1.1B-compressed-tensors-kv-cache-scheme"
]
KV_CACHE_QUANTIZATION_PATHS = {
    "meta-llama/Llama-2-7b-chat-hf":
    "./tests/fp8_kv/llama2-7b-fp8-kv/kv_cache_scales.json"
}


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("chunked_prefill_token_size", [1, 4, 16])
@pytest.mark.parametrize("enforce_eager", [False, True])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    chunked_prefill_token_size: int,
    enforce_eager: bool,
    tensor_parallel_size: int,
) -> None:
    """
    Checks exact match decode between huggingface model and vllm runner with
    chunked prefill.
    """
    max_num_seqs = chunked_prefill_token_size
    max_num_batched_tokens = chunked_prefill_token_size

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    with vllm_runner(
            model,
            dtype=dtype,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=True,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("kv_cache_dtype,model",
                         [("fp8_e5m2", m)
                          for m in E5M2_KV_MODELS] + [("fp8_e4m3", m)
                                                      for m in E4M3_KV_MODELS])
# Due to low-precision numerical divergence, we only test logprob of 4 tokens
@pytest.mark.parametrize("max_tokens", [4])
@pytest.mark.parametrize("chunked_prefill_token_size", [4, 16])
@pytest.mark.parametrize("enforce_eager", [False, True])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_models_with_fp8_kv_cache(
    vllm_runner,
    example_prompts,
    kv_cache_dtype: str,
    model: str,
    max_tokens: int,
    chunked_prefill_token_size: int,
    enforce_eager: bool,
    tensor_parallel_size: int,
) -> None:
    """
    Only checks log probs match between chunked-prefill and
    non-chunked-prefill version of vLLM model runner.
    
    This test is used when there is discrepancy in kernels
    / numerics (e.g. when using lower-precision types like FP8).
    """
    NUM_LOG_PROBS = 8

    if model == "facebook/opt-125m":
        pytest.skip(
            "#7378: CUDA illegal memory access (undiagnosed) facebook/opt-125m"
        )

    max_num_seqs = chunked_prefill_token_size
    max_num_batched_tokens = chunked_prefill_token_size

    extra_kwargs = {}
    if model in KV_CACHE_QUANTIZATION_PATHS:
        extra_kwargs["quantization_param_path"] = KV_CACHE_QUANTIZATION_PATHS[
            model]

    with vllm_runner(
            model,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
            kv_cache_dtype=kv_cache_dtype,
            **extra_kwargs,
    ) as vllm_model:
        no_chunked_prefill_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, NUM_LOG_PROBS)

    with vllm_runner(
            model,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=True,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
            kv_cache_dtype=kv_cache_dtype,
            **extra_kwargs,
    ) as vllm_model:
        chunked_prefill_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, NUM_LOG_PROBS)

    check_logprobs_close(
        outputs_0_lst=no_chunked_prefill_outputs,
        outputs_1_lst=chunked_prefill_outputs,
        name_0="no_chunked_prefill",
        name_1="chunked_prefill",
    )
