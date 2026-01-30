# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import pytest

from tests.models.registry import HF_EXAMPLE_MODELS
from tests.utils import multi_gpu_test
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

from ...utils import check_logprobs_close, check_outputs_equal

# Mark all tests as hybrid
pytestmark = pytest.mark.hybrid_model

# NOTE: The first model in each list is taken as the primary model,
# meaning that it will be used in all tests in this file
# The rest of the models will only be tested by test_models

APC_MULTIPLY_BY = 300

SSM_MODELS = [
    "state-spaces/mamba-130m-hf",
    "tiiuae/falcon-mamba-tiny-dev",
    # mamba2-codestral in transformers is broken pending:
    # https://github.com/huggingface/transformers/pull/40861
    # "yujiepan/mamba2-codestral-v0.1-tiny-random",
]

HYBRID_MODELS = [
    "ai21labs/Jamba-tiny-dev",
    "pfnet/plamo-2-1b",
    "Zyphra/Zamba2-1.2B-instruct",
    "hmellor/tiny-random-BambaForCausalLM",
    "ibm-granite/granite-4.0-tiny-preview",
    "tiiuae/Falcon-H1-0.5B-Base",
    "LiquidAI/LFM2-1.2B",
    "tiny-random/qwen3-next-moe",
]

FULL_CUDA_GRAPH_MODELS = [
    "ai21labs/Jamba-tiny-dev",
    "pfnet/plamo-2-1b",
    "Zyphra/Zamba2-1.2B-instruct",
]

FP32_STATE_MODELS = [
    "state-spaces/mamba-130m-hf",
    "Zyphra/Zamba2-1.2B-instruct",
]

# Avoid OOM
MAX_NUM_SEQS = 4


@pytest.mark.parametrize("model", SSM_MODELS + HYBRID_MODELS)
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pass

    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(model, max_num_seqs=MAX_NUM_SEQS) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_batching(
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pass

    for_loop_outputs = []
    with vllm_runner(model, max_num_seqs=MAX_NUM_SEQS) as vllm_model:
        for prompt in example_prompts:
            (single_output,) = vllm_model.generate_greedy_logprobs(
                [prompt], max_tokens, num_logprobs
            )
            for_loop_outputs.append(single_output)

        batched_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=for_loop_outputs,
        outputs_1_lst=batched_outputs,
        name_0="for_loop_vllm",
        name_1="batched_vllm",
    )


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [10])
def test_chunked_prefill_with_parallel_sampling(
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
) -> None:
    """
    Tests chunked prefill in conjunction with n > 1.

    In this case, prefill is populated with decoding tokens and
    we test that it doesn't fail.

    This test might fail if cache is not allocated correctly for n > 1
    decoding steps inside a chunked prefill forward pass
    (where we have both prefill and decode together)
    """
    sampling_params = SamplingParams(n=3, temperature=1, seed=0, max_tokens=max_tokens)
    with vllm_runner(
        model,
        enable_chunked_prefill=True,
        # forces prefill chunks with decoding
        max_num_batched_tokens=MAX_NUM_SEQS * 3,
        max_num_seqs=MAX_NUM_SEQS,
    ) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [20])
def test_mamba_cache_cg_padding(
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
) -> None:
    """
    This test is for verifying that mamba cache is padded to CG captured
    batch size. If it's not, a torch RuntimeError will be raised because
    tensor dimensions aren't compatible.
    """
    vllm_config = EngineArgs(model=model, trust_remote_code=True).create_engine_config()
    cudagraph_dispatcher = CudagraphDispatcher(vllm_config)
    cudagraph_dispatcher.initialize_cudagraph_keys(
        vllm_config.compilation_config.cudagraph_mode
    )
    while (
        len(example_prompts)
        == cudagraph_dispatcher.dispatch(len(example_prompts))[1].num_tokens
    ):
        example_prompts.append(example_prompts[0])

    try:
        with vllm_runner(model) as vllm_model:
            vllm_model.generate_greedy(example_prompts, max_tokens)
    except RuntimeError:
        pytest.fail(
            "Couldn't run batch size which is not equal to a Cuda Graph "
            "captured batch size. "
            "Could be related to mamba cache not padded correctly"
        )


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
def test_fail_upon_inc_requests_and_finished_requests_lt_available_blocks(
    vllm_runner,
    example_prompts,
    model: str,
) -> None:
    """
    This test is for verifying that the hybrid inner state management doesn't
    collapse in case where the number of incoming requests and
    finished_requests_ids is larger than the maximum mamba block capacity.

    This could generally happen due to the fact that hybrid does support
    statelessness mechanism where it can clean up new incoming requests in
    a single step.
    """
    try:
        with vllm_runner(model, max_num_seqs=MAX_NUM_SEQS) as vllm_model:
            vllm_model.generate_greedy([example_prompts[0]] * 100, 10)
    except ValueError:
        pytest.fail(
            "Hybrid inner state wasn't cleaned up properly between"
            "steps finished requests registered unnecessarily "
        )


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
def test_state_cleanup(
    vllm_runner,
    example_prompts,
    model: str,
) -> None:
    """
    This test is for verifying that the Hybrid state is cleaned up between
    steps.

    If it's not cleaned, an error would be expected.
    """
    try:
        with vllm_runner(model, max_num_seqs=MAX_NUM_SEQS) as vllm_model:
            for _ in range(10):
                vllm_model.generate_greedy([example_prompts[0]] * 100, 1)
    except ValueError:
        pytest.fail(
            "Hybrid inner state wasn't cleaned up between states, "
            "could be related to finished_requests_ids"
        )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_distributed_correctness(
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with vllm_runner(
        model, tensor_parallel_size=1, max_num_seqs=MAX_NUM_SEQS
    ) as vllm_model:
        vllm_outputs_tp_1 = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(
        model, tensor_parallel_size=2, max_num_seqs=MAX_NUM_SEQS
    ) as vllm_model:
        vllm_outputs_tp_2 = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=vllm_outputs_tp_1,
        outputs_1_lst=vllm_outputs_tp_2,
        name_0="vllm_tp_1",
        name_1="vllm_tp_2",
    )


@pytest.mark.parametrize("model", FULL_CUDA_GRAPH_MODELS)
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_full_cuda_graph(
    hf_runner,
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pass

    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(model, max_num_seqs=MAX_NUM_SEQS) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", FP32_STATE_MODELS)
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize(
    "cache_dtype_param", ["mamba_ssm_cache_dtype", "mamba_cache_dtype"]
)
def test_fp32_cache_state(
    hf_runner,
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    num_logprobs: int,
    cache_dtype_param: str,
) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pass

    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(
        model, max_num_seqs=MAX_NUM_SEQS, **{cache_dtype_param: "float32"}
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


# Helper functions for the APC tests
def _get_vllm_runner_params(
    model: str,
    max_model_len: int,
    tensor_parallel_size: int = 1,
):
    return {
        "model_name": model,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": 0.4,
    }


def _get_vLLM_output(
    vllm_runner,
    kwargs,
    prompts,
    max_tokens,
    num_logprobs,
    num_repetitions=1,
    vllm_model=None,
):
    outs = []
    if vllm_model is None:
        vllm_model = vllm_runner(**kwargs)
    for _ in range(num_repetitions):
        if num_logprobs < 0:
            vllm_output = vllm_model.generate_greedy(prompts, max_tokens)
        else:
            vllm_output = vllm_model.generate_greedy_logprobs(
                prompts, max_tokens, num_logprobs
            )
        outs.append(vllm_output)

    return outs, vllm_model


@pytest.mark.parametrize("model", [HYBRID_MODELS[0], HYBRID_MODELS[3]])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("n_repetitions", [2])
# If num_logprobs is set to -1, then the stringent version
# of the test is executed using `check_outputs_equal`
# instead of `check_logprobs_close`
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_apc_single_prompt(
    hf_runner,
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    n_repetitions: int,
    num_logprobs: int,
    tensor_parallel_size: int,
) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pass

    compare_operator: Callable = (
        check_logprobs_close if num_logprobs > 0 else check_outputs_equal  # type: ignore
    )

    # Sample prompts.
    generated_prompts = [APC_MULTIPLY_BY * example_prompts[0]]

    max_model_len = max(len(prompt) + max_tokens for prompt in generated_prompts)
    vllm_runner_kwargs = _get_vllm_runner_params(
        model, max_model_len, tensor_parallel_size=tensor_parallel_size
    )
    vllm_runner_kwargs["mamba_ssm_cache_dtype"] = "float32"
    vllm_outputs_no_cache, _ = _get_vLLM_output(
        vllm_runner, vllm_runner_kwargs, generated_prompts, max_tokens, num_logprobs
    )

    vllm_runner_kwargs["enable_prefix_caching"] = True
    vllm_outputs_cache_rep, _ = _get_vLLM_output(
        vllm_runner,
        vllm_runner_kwargs,
        generated_prompts,
        max_tokens,
        num_logprobs,
        n_repetitions,
    )

    for r_idx, vllm_outputs_cache_itn in enumerate(vllm_outputs_cache_rep):
        # In the first repetition, the caches are filled
        # In the second repetition, these caches are reused

        compare_operator(
            outputs_0_lst=vllm_outputs_no_cache[0],
            outputs_1_lst=vllm_outputs_cache_itn,
            name_0="vllm_no_cache",
            name_1=f"vllm_cache_it_{r_idx + 1}",
        )


@pytest.mark.parametrize("model", [HYBRID_MODELS[0], HYBRID_MODELS[3]])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("n_repetitions", [2])
# If num_logprobs is set to -1, then the stringent version
# of the test is executed using `check_outputs_equal`
# instead of `check_logprobs_close`
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_apc_single_prompt_block_align_alignment(
    hf_runner,
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    n_repetitions: int,
    num_logprobs: int,
    tensor_parallel_size: int,
) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pass

    compare_operator: Callable = (
        check_logprobs_close if num_logprobs > 0 else check_outputs_equal  # type: ignore
    )

    # Sample prompts. This custom prompt is used, as it causes the most issues
    generated_prompts = ["The president of the United States is " * APC_MULTIPLY_BY]

    max_model_len = max(len(prompt) + max_tokens for prompt in generated_prompts)
    vllm_runner_kwargs = _get_vllm_runner_params(
        model, max_model_len, tensor_parallel_size=tensor_parallel_size
    )
    vllm_runner_kwargs["mamba_ssm_cache_dtype"] = "float32"

    vllm_outputs_no_cache, _ = _get_vLLM_output(
        vllm_runner, vllm_runner_kwargs, generated_prompts, max_tokens, num_logprobs
    )

    vllm_runner_kwargs["enable_prefix_caching"] = True
    with vllm_runner(**vllm_runner_kwargs) as vllm_model:
        # Retrieve the default mamba state block size
        mamba_block_size = vllm_model.llm.llm_engine.cache_config.mamba_block_size

    # In case the hybrid model does not have the
    # "mamba_block_size" assume a fixed constant
    if mamba_block_size is None:
        mamba_block_size = 512

    mamba_block_size_multiplier = 10
    for offsets in [-3, 3, mamba_block_size // 4 + 3, mamba_block_size // 2 - 3]:
        vllm_runner_kwargs["max_num_batched_tokens"] = (
            mamba_block_size_multiplier * mamba_block_size - offsets
        )
        vllm_outputs_cache_rep, _ = _get_vLLM_output(
            vllm_runner,
            vllm_runner_kwargs,
            generated_prompts,
            max_tokens,
            num_logprobs,
            n_repetitions,
        )

        # Check alignment of the output logits when using APC
        for r_idx, vllm_outputs_cache_itn in enumerate(vllm_outputs_cache_rep):
            # In the first repetition, the caches are filled
            # In the second repetition, these caches are reused

            compare_operator(
                outputs_0_lst=vllm_outputs_no_cache[0],
                outputs_1_lst=vllm_outputs_cache_itn,
                name_0="vllm_no_cache",
                name_1=f"vllm_cache_it_{r_idx + 1}",
            )


@pytest.mark.parametrize("model", [HYBRID_MODELS[0], HYBRID_MODELS[3]])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("n_repetitions", [2])
# If num_logprobs is set to -1, then the stringent version
# of the test is executed using `check_outputs_equal`
# instead of `check_logprobs_close`
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_apc_multiple_prompts_all_cached_outputs(
    hf_runner,
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    n_repetitions: int,
    num_logprobs: int,
    tensor_parallel_size: int,
) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pass

    compare_operator: Callable = (
        check_logprobs_close if num_logprobs > 0 else check_outputs_equal  # type: ignore
    )

    # Sample prompts.
    generated_prompts = [APC_MULTIPLY_BY * prompt for prompt in example_prompts]

    max_model_len = max(len(prompt) + max_tokens for prompt in generated_prompts)
    vllm_runner_kwargs = _get_vllm_runner_params(
        model, max_model_len, tensor_parallel_size=tensor_parallel_size
    )
    vllm_runner_kwargs["mamba_ssm_cache_dtype"] = "float32"

    vllm_outputs_no_cache, _ = _get_vLLM_output(
        vllm_runner, vllm_runner_kwargs, generated_prompts, max_tokens, num_logprobs
    )

    vllm_runner_kwargs["enable_prefix_caching"] = True
    vllm_outputs_cache_rep, _ = _get_vLLM_output(
        vllm_runner,
        vllm_runner_kwargs,
        generated_prompts,
        max_tokens,
        num_logprobs,
        n_repetitions,
    )

    for r_idx, vllm_outputs_cache_itn in enumerate(vllm_outputs_cache_rep):
        # In the first repetition, the caches are filled
        # In the second repetition, these caches are reused

        compare_operator(
            outputs_0_lst=vllm_outputs_no_cache[0],
            outputs_1_lst=vllm_outputs_cache_itn,
            name_0="vllm_no_cache",
            name_1=f"vllm_cache_it_{r_idx + 1}",
        )


@pytest.mark.parametrize("model", [HYBRID_MODELS[0], HYBRID_MODELS[3]])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("n_repetitions", [2])
# If num_logprobs is set to -1, then the stringent version
# of the test is executed using `check_outputs_equal`
# instead of `check_logprobs_close`
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_apc_multiple_prompts_block_align_alignment(
    hf_runner,
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    n_repetitions: int,
    num_logprobs: int,
    tensor_parallel_size: int,
) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pass

    compare_operator: Callable = (
        check_logprobs_close if num_logprobs > 0 else check_outputs_equal  # type: ignore
    )

    # Sample prompts. This custom prompt is used, as it causes the most issues
    prompt_text = "The president of the United States is "
    prompt_offsets = [0, 3, 7, 13, 17, 22, 25, 31]
    generated_prompts = [
        prompt_text[offset:] * APC_MULTIPLY_BY for offset in prompt_offsets
    ]

    max_model_len = max(len(prompt) + max_tokens for prompt in generated_prompts)
    vllm_runner_kwargs = _get_vllm_runner_params(
        model, max_model_len, tensor_parallel_size
    )
    vllm_runner_kwargs["mamba_ssm_cache_dtype"] = "float32"

    vllm_outputs_no_cache, _ = _get_vLLM_output(
        vllm_runner, vllm_runner_kwargs, generated_prompts, max_tokens, num_logprobs
    )

    vllm_runner_kwargs["enable_prefix_caching"] = True
    with vllm_runner(**vllm_runner_kwargs) as vllm_model:
        # Retrieve the default mamba state block size
        mamba_block_size = vllm_model.llm.llm_engine.cache_config.mamba_block_size

    # In case the hybrid model does not have the
    # "mamba_block_size" assume a fixed constant
    if mamba_block_size is None:
        mamba_block_size = 512

    mamba_block_size_multiplier = 10
    for offsets in [-3, 3, mamba_block_size // 4 + 3, mamba_block_size // 2 - 3]:
        vllm_runner_kwargs["max_num_batched_tokens"] = (
            mamba_block_size_multiplier * mamba_block_size - offsets
        )
        vllm_outputs_cache_rep, _ = _get_vLLM_output(
            vllm_runner,
            vllm_runner_kwargs,
            generated_prompts,
            max_tokens,
            num_logprobs,
            n_repetitions,
        )

        # Check alignment of the output logits when using APC
        for r_idx, vllm_outputs_cache_itn in enumerate(vllm_outputs_cache_rep):
            # In the first repetition, the caches are filled
            # In the second repetition, these caches are reused

            compare_operator(
                outputs_0_lst=vllm_outputs_no_cache[0],
                outputs_1_lst=vllm_outputs_cache_itn,
                name_0="vllm_no_cache",
                name_1=f"vllm_cache_it_{r_idx + 1}",
            )


@pytest.mark.parametrize("model", [HYBRID_MODELS[0], HYBRID_MODELS[3]])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("n_repetitions", [2])
# If num_logprobs is set to -1, then the stringent version
# of the test is executed using `check_outputs_equal`
# instead of `check_logprobs_close`
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_apc_multiple_prompts_partial_cached_outputs(
    hf_runner,
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    n_repetitions: int,
    num_logprobs: int,
    tensor_parallel_size: int,
) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pass

    compare_operator: Callable = (
        check_logprobs_close if num_logprobs > 0 else check_outputs_equal  # type: ignore
    )

    # Sample prompts.
    generated_prompts = [APC_MULTIPLY_BY * prompt for prompt in example_prompts]

    max_model_len = max(len(prompt) + max_tokens for prompt in generated_prompts)
    vllm_runner_kwargs = _get_vllm_runner_params(
        model, max_model_len, tensor_parallel_size=tensor_parallel_size
    )
    vllm_runner_kwargs["mamba_ssm_cache_dtype"] = "float32"

    vllm_outputs_no_cache, _ = _get_vLLM_output(
        vllm_runner, vllm_runner_kwargs, generated_prompts, max_tokens, num_logprobs
    )

    # Cache only part of all the prompts
    vllm_runner_kwargs["enable_prefix_caching"] = True
    vllm_outputs_partial_cache, vllm_model = _get_vLLM_output(
        vllm_runner, vllm_runner_kwargs, generated_prompts[:3], max_tokens, num_logprobs
    )

    compare_operator(
        outputs_0_lst=vllm_outputs_no_cache[0][:3],
        outputs_1_lst=vllm_outputs_partial_cache[0],
        name_0="vllm_no_cache",
        name_1="vllm_partial_cache",
    )

    vllm_outputs_cache_rep, _ = _get_vLLM_output(
        vllm_runner,
        vllm_runner_kwargs,
        generated_prompts,
        max_tokens,
        num_logprobs,
        n_repetitions,
        vllm_model=vllm_model,
    )

    for r_idx, vllm_outputs_cache_itn in enumerate(vllm_outputs_cache_rep):
        # In the first repetition, the caches are filled
        # In the second repetition, these caches are reused

        compare_operator(
            outputs_0_lst=vllm_outputs_no_cache[0],
            outputs_1_lst=vllm_outputs_cache_itn,
            name_0="vllm_no_cache",
            name_1=f"vllm_cache_it_{r_idx + 1}",
        )
