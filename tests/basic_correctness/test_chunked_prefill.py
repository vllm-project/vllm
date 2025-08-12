"""Compare the outputs of HF and vLLM when using greedy sampling.

It tests chunked prefill. Chunked prefill can be enabled by
enable_chunked_prefill=True. If prefill size exceeds max_num_batched_tokens,
prefill requests are chunked.

Run `pytest tests/models/test_chunked_prefill.py`.
"""
import os
from contextlib import nullcontext

import pytest

from ..models.utils import check_logprobs_close, check_outputs_equal
from ..utils import multi_gpu_test

MODELS = [
    "facebook/opt-125m",
    "meta-llama/Llama-2-7b-hf",
]


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


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("distributed_executor_backend", ["ray", "mp"])
@pytest.mark.parametrize("model", MODELS)
def test_models_distributed(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    distributed_executor_backend: str,
) -> None:
    if (model == "meta-llama/Llama-2-7b-hf"
            and distributed_executor_backend == "ray"):
        # test ray adag
        os.environ['VLLM_USE_RAY_SPMD_WORKER'] = "1"
        os.environ['VLLM_USE_RAY_COMPILED_DAG'] = "1"

    dtype = "half"
    max_tokens = 5
    chunked_prefill_token_size = 16

    # Add a chunked prefill config.
    max_num_seqs = min(chunked_prefill_token_size, 256)
    assert chunked_prefill_token_size != -1
    enable_chunked_prefill = True
    max_num_batched_tokens = chunked_prefill_token_size

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    with vllm_runner(
            model,
            dtype=dtype,
            tensor_parallel_size=2,
            max_num_seqs=max_num_seqs,
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
            distributed_executor_backend=distributed_executor_backend,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize(
    "kv_cache_dtype,model",
    [("fp8_e4m3",
      "nm-testing/TinyLlama-1.1B-compressed-tensors-kv-cache-scheme")])
# Due to low-precision numerical divergence, we only test logprob of 4 tokens
@pytest.mark.parametrize("max_tokens", [4])
@pytest.mark.parametrize("chunked_prefill_token_size", [4, 16])
@pytest.mark.parametrize("enforce_eager", [False, True])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
@pytest.mark.parametrize("tensor_parallel_size", [1])
# Due to low-precision numerical divergence, this test is too sensitive to
# the async postprocessor
@pytest.mark.parametrize("disable_async_output_proc", [True])
def test_models_with_fp8_kv_cache(
    vllm_runner,
    example_prompts,
    kv_cache_dtype: str,
    model: str,
    max_tokens: int,
    chunked_prefill_token_size: int,
    enforce_eager: bool,
    tensor_parallel_size: int,
    disable_async_output_proc: bool,
) -> None:
    """
    Check output logprobs match between no_chunked_prefill and chunked_prefill
    with fp8 kv cache. General fp8 kv-cache tests are covered in test_fp8.py,
    so here we only check chunked prefill.
    """
    NUM_LOG_PROBS = 8

    max_num_seqs = chunked_prefill_token_size
    max_num_batched_tokens = chunked_prefill_token_size

    with vllm_runner(
            model,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
            kv_cache_dtype=kv_cache_dtype,
            disable_async_output_proc=disable_async_output_proc,
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
            disable_async_output_proc=disable_async_output_proc,
    ) as vllm_model:
        chunked_prefill_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, NUM_LOG_PROBS)

    check_logprobs_close(
        outputs_0_lst=no_chunked_prefill_outputs,
        outputs_1_lst=chunked_prefill_outputs,
        name_0="no_chunked_prefill",
        name_1="chunked_prefill",
    )


@pytest.mark.parametrize("max_tokens", [16])
@pytest.mark.parametrize("enforce_eager", [False])
@pytest.mark.parametrize("chunk_size", [30, 32])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_with_prefix_caching(
    vllm_runner,
    max_tokens: int,
    enforce_eager: bool,
    chunk_size: int,
    tensor_parallel_size: int,
) -> None:
    """
    Checks exact match decode with and without prefix caching
    with chunked prefill enabled.
    """
    model = "meta-llama/Llama-2-7b-chat-hf"
    # The common prompt has 142 tokens with Llama-2 tokenizer.
    common_prompt = "You are a helpful AI assistant " * 20
    unique_prompts = [
        "Question",  # Warmup
        "Question",  # Fully cached
        "Another question",  # Partial cached
    ]
    full_prompts = [f"{common_prompt}\n{p}" for p in unique_prompts]

    max_num_batched_tokens = max_num_seqs = chunk_size
    outputs = {}  # type: ignore
    check_result = True
    for enable in (True, False):
        with vllm_runner(
                model,
                dtype="half",
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=True,
                enable_prefix_caching=enable,
                tensor_parallel_size=tensor_parallel_size,
                enforce_eager=enforce_eager,
                max_num_seqs=max_num_seqs,
        ) as vllm_model:
            # It should fail when prefix caching is enable and chunk
            # size is not a multiple of block size (16).
            should_fail = chunk_size % 16 != 0 and enable
            check_result &= not should_fail
            outputs[enable] = []
            # Send the request one-by-one to ensure the cache is populated.
            with pytest.raises(ValueError) if should_fail else nullcontext():
                for prompt in full_prompts:
                    outputs[enable] += vllm_model.generate_greedy([prompt],
                                                                  max_tokens)

    # Check results only if we did not expect a failure.
    if check_result:
        check_outputs_equal(
            outputs_0_lst=outputs[False],
            outputs_1_lst=outputs[True],
            name_0="w/o prefix caching",
            name_1="with prefix caching",
        )
