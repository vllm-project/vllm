# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Callable
from contextlib import contextmanager, nullcontext

import pytest
import torch

from tests.models.registry import HF_EXAMPLE_MODELS
from tests.utils import multi_gpu_test
from vllm import LLM
from vllm.config import CUDAGraphMode
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.metrics.reader import Counter

from ...utils import check_logprobs_close, check_outputs_equal

# Mark all tests as hybrid
pytestmark = pytest.mark.hybrid_model

# NOTE: The first model in each list is taken as the primary model,
# meaning that it will be used in all tests in this file
# The rest of the models will only be tested by test_models

APC_MULTIPLY_BY = 300

SSM_MODELS = [
    "vllm/mamba2-130m-hf",
    "vllm/mamba2-codestral-v0.1-tiny-random",
]

HYBRID_MODELS = [
    "tiiuae/Falcon-H1-Tiny-90M-Instruct",
    "Zyphra/Zamba2-1.2B-instruct",
    "ibm-granite/granite-4.0-tiny-preview",
    "tiiuae/Falcon-H1-0.5B-Base",
    "LiquidAI/LFM2-1.2B",
    "tiny-random/qwen3-next-moe",
]

FULL_CUDA_GRAPH_MODELS = [
    "Zyphra/Zamba2-1.2B-instruct",
]

FP32_STATE_MODELS = [
    "vllm/mamba2-130m-hf",
    "Zyphra/Zamba2-1.2B-instruct",
]

# Avoid OOM
MAX_NUM_SEQS = 4

ATTN_BACKEND = "TRITON_ATTN" if current_platform.is_rocm() else "auto"


def _set_conv_state_layout(monkeypatch, layout: str) -> None:
    """Set conv state layout env var and clear cache to pick up new value."""
    from vllm.model_executor.layers.mamba import mamba_utils

    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", layout)
    mamba_utils.get_conv_state_layout.cache_clear()


@pytest.mark.parametrize("model", SSM_MODELS + HYBRID_MODELS)
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    hf_runner,
    vllm_runner,
    gpu_memory_cleared,
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

    with vllm_runner(
        model,
        max_num_seqs=MAX_NUM_SEQS,
        attention_backend=ATTN_BACKEND,
        enable_chunked_prefill=True,
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


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("conv_state_layout", ["SD", "DS"])
def test_batching(
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    num_logprobs: int,
    conv_state_layout: str,
) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pass

    _set_conv_state_layout(monkeypatch, conv_state_layout)

    for_loop_outputs = []
    with vllm_runner(
        model, max_num_seqs=MAX_NUM_SEQS, enable_chunked_prefill=True
    ) as vllm_model:
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
@pytest.mark.parametrize("conv_state_layout", ["SD", "DS"])
def test_chunked_prefill_with_parallel_sampling(
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    conv_state_layout: str,
) -> None:
    """Tests chunked prefill in conjunction with n > 1.

    In this case, prefill is populated with decoding tokens and
    we test that it doesn't fail.

    This test might fail if cache is not allocated correctly for n > 1
    decoding steps inside a chunked prefill forward pass
    (where we have both prefill and decode together)
    """
    _set_conv_state_layout(monkeypatch, conv_state_layout)

    sampling_params = SamplingParams(n=3, temperature=1, seed=0, max_tokens=max_tokens)
    with vllm_runner(
        model,
        enable_chunked_prefill=True,
        # forces prefill chunks with decoding
        max_num_batched_tokens=MAX_NUM_SEQS * 3,
        max_num_seqs=MAX_NUM_SEQS,
        attention_backend=ATTN_BACKEND,
    ) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("conv_state_layout", ["SD", "DS"])
def test_mamba_cache_cg_padding(
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    conv_state_layout: str,
) -> None:
    """This test is for verifying that mamba cache is padded to CG captured
    batch size. If it's not, a torch RuntimeError will be raised because
    tensor dimensions aren't compatible.
    """
    _set_conv_state_layout(monkeypatch, conv_state_layout)

    vllm_config = EngineArgs(model=model).create_engine_config()
    compilation_config = vllm_config.compilation_config
    if compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        pytest.skip("CUDA/XPU graph is disabled. Please enable it to run this test.")
    while len(example_prompts) in (
        compilation_config.decode_cudagraph_capture_request_sizes or ()
    ):
        example_prompts.append(example_prompts[0])

    try:
        with vllm_runner(model, enable_chunked_prefill=True) as vllm_model:
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
    """This test is for verifying that the hybrid inner state management doesn't
    collapse in case where the number of incoming requests and
    finished_requests_ids is larger than the maximum mamba block capacity.

    This could generally happen due to the fact that hybrid does support
    statelessness mechanism where it can clean up new incoming requests in
    a single step.
    """
    try:
        with vllm_runner(
            model, max_num_seqs=MAX_NUM_SEQS, enable_chunked_prefill=True
        ) as vllm_model:
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
    """This test is for verifying that the Hybrid state is cleaned up between
    steps.

    If it's not cleaned, an error would be expected.
    """
    try:
        with vllm_runner(
            model, max_num_seqs=MAX_NUM_SEQS, enable_chunked_prefill=True
        ) as vllm_model:
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
        model,
        tensor_parallel_size=1,
        max_num_seqs=MAX_NUM_SEQS,
        enable_chunked_prefill=True,
    ) as vllm_model:
        vllm_outputs_tp_1 = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(
        model,
        tensor_parallel_size=2,
        max_num_seqs=MAX_NUM_SEQS,
        enable_chunked_prefill=True,
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


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model", ["ibm-granite/granite-4.0-tiny-preview"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_pipeline_parallel_correctness(
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    """Each pipeline stage must load and run only its own layers. A MoE
    checkpoint also covers the router-to-gate weight rename."""
    with vllm_runner(
        model,
        max_num_seqs=MAX_NUM_SEQS,
        enable_chunked_prefill=True,
    ) as vllm_model:
        vllm_outputs_pp_1 = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(
        model,
        pipeline_parallel_size=2,
        max_num_seqs=MAX_NUM_SEQS,
        enable_chunked_prefill=True,
    ) as vllm_model:
        vllm_outputs_pp_2 = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=vllm_outputs_pp_1,
        outputs_1_lst=vllm_outputs_pp_2,
        name_0="vllm_pp_1",
        name_1="vllm_pp_2",
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model", ["ibm-granite/granite-4.0-tiny-preview"])
def test_pipeline_parallel_mamba_only_stage_prefix_caching(
    vllm_runner, monkeypatch, model: str
) -> None:
    """A pipeline stage that holds only Mamba layers uses the KV cache block
    size of the stages with attention layers, so prefix caching reuses the
    same tokens as without pipeline parallelism.

    The model's first attention layer is layer 5, so a 5/35 split leaves the
    first stage with Mamba layers only. Each prompt extends the previous one
    and is generated alone, which makes the cache hits deterministic.
    """
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "5,35")
    lines = [
        f"Record {i}: the code for item {i} is {i * 7919 % 10007}.\n"
        for i in range(160)
    ]
    prompts = [
        "".join(lines[:n]) + "The code for item 3 is" for n in (40, 80, 120, 160)
    ]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16, logprobs=5)
    outputs = {}
    num_cached_tokens = {}
    for pp_size in (1, 2):
        with vllm_runner(
            model,
            pipeline_parallel_size=pp_size,
            max_num_seqs=MAX_NUM_SEQS,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            max_model_len=4096,
            distributed_executor_backend="mp",
        ) as vllm_model:
            results = [
                vllm_model.llm.generate(prompt, sampling_params)[0]
                for prompt in prompts
            ]
        num_cached_tokens[pp_size] = [result.num_cached_tokens for result in results]
        outputs[pp_size] = [
            (
                list(result.outputs[0].token_ids),
                result.outputs[0].text,
                result.outputs[0].logprobs,
            )
            for result in results
        ]

    assert any(num_cached_tokens[1]), num_cached_tokens
    assert num_cached_tokens[2] == num_cached_tokens[1], num_cached_tokens
    check_logprobs_close(
        outputs_0_lst=outputs[1],
        outputs_1_lst=outputs[2],
        name_0="vllm_pp_1",
        name_1="vllm_pp_2",
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

    # Leave headroom for the HF model that ran earlier in this process on a
    # 32.5 GiB MIG.
    with vllm_runner(
        model,
        max_num_seqs=MAX_NUM_SEQS,
        attention_backend=ATTN_BACKEND,
        gpu_memory_utilization=0.9,
        enable_chunked_prefill=True,
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

    # Leave enough headroom for repeated engine initialization on a
    # 32.5 GiB MIG.
    with vllm_runner(
        model,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=0.9,
        enable_chunked_prefill=True,
        **{cache_dtype_param: "float32"},
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
        "attention_backend": ATTN_BACKEND,
    }


@contextmanager
def _owned_vllm_runner(vllm_runner, kwargs):
    with vllm_runner(**kwargs) as runner:
        yield runner


def _get_vLLM_output(
    vllm_runner,
    kwargs,
    prompts,
    max_tokens,
    num_logprobs,
    num_repetitions=1,
    vllm_model=None,
):
    runner_context = (
        _owned_vllm_runner(vllm_runner, kwargs)
        if vllm_model is None
        else nullcontext(vllm_model)
    )
    with runner_context as runner:
        outs = []
        for _ in range(num_repetitions):
            if num_logprobs < 0:
                vllm_output = runner.generate_greedy(prompts, max_tokens)
            else:
                vllm_output = runner.generate_greedy_logprobs(
                    prompts, max_tokens, num_logprobs
                )
            outs.append(vllm_output)

    return outs, vllm_model


@pytest.mark.parametrize("model", [HYBRID_MODELS[0]])
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


@pytest.mark.parametrize("model", [HYBRID_MODELS[0]])
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
        vllm_config = vllm_model.llm.llm_engine.vllm_config
        mamba_block_size = vllm_config.cache_config.mamba_block_size

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


@pytest.mark.parametrize("model", [HYBRID_MODELS[0]])
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
    # Reduce the effects of batch variance on ROCm since batch invariance is not
    # yet supported. See: https://github.com/vllm-project/vllm/issues/27433
    if current_platform.is_rocm():
        vllm_runner_kwargs["max_num_seqs"] = 4

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


@pytest.mark.parametrize("model", [HYBRID_MODELS[0]])
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
        vllm_config = vllm_model.llm.llm_engine.vllm_config
        mamba_block_size = vllm_config.cache_config.mamba_block_size

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


@pytest.mark.parametrize("model", [HYBRID_MODELS[0]])
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
    with _owned_vllm_runner(vllm_runner, vllm_runner_kwargs) as vllm_model:
        vllm_outputs_partial_cache, _ = _get_vLLM_output(
            vllm_runner,
            vllm_runner_kwargs,
            generated_prompts[:3],
            max_tokens,
            num_logprobs,
            vllm_model=vllm_model,
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


# Test that outputs match whether prefix caching is enabled or not for mamba.
@pytest.mark.parametrize("model", [SSM_MODELS[0]])
def test_same_mamba_output_apc_on_vs_off(
    vllm_runner,
    model: str,
) -> None:
    num_logprobs = 5
    prompts = [
        "hello what is one plus one what is one plus one what is one plus one the answer is",  # noqa: E501
        "hello what is one plus one what is one plus one what is one plus one the answer is",  # noqa: E501
    ]
    max_tokens = 20
    max_model_len = max(len(p) for p in prompts) + max_tokens + 64

    base_kwargs = _get_vllm_runner_params(model, max_model_len)
    base_kwargs.update(
        enforce_eager=True, block_size=16, seed=42, gpu_memory_utilization=0.8
    )

    # No prefix caching
    kwargs_no_apc = {**base_kwargs, "enable_prefix_caching": False}
    with vllm_runner(**kwargs_no_apc) as vllm_model:
        outputs_no_apc, _ = _get_vLLM_output(
            vllm_runner,
            kwargs_no_apc,
            prompts,
            max_tokens,
            num_logprobs=num_logprobs,
            vllm_model=vllm_model,
        )
    # With prefix caching
    kwargs_with_apc = {
        **base_kwargs,
        "enable_prefix_caching": True,
        "mamba_block_size": 16,
    }
    with vllm_runner(**kwargs_with_apc) as vllm_model:
        outputs_with_apc, _ = _get_vLLM_output(
            vllm_runner,
            kwargs_with_apc,
            prompts,
            max_tokens,
            num_logprobs=num_logprobs,
            vllm_model=vllm_model,
        )

    check_logprobs_close(
        outputs_0_lst=outputs_no_apc[0],
        outputs_1_lst=outputs_with_apc[0],
        name_0="vllm_no_apc",
        name_1="vllm_with_apc",
    )


# we have to use a real large model to get reasonable results
# the model can't be a hybrid model as we need block_size 16
@pytest.mark.parametrize("model", ["vllm/mamba2-1.3b-hf"])
def test_apc_common_prefix_same_batch(
    model: str,
    monkeypatch,
) -> None:
    # Required to put the two requests in the same batch
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    llm = LLM(
        model=model,
        enforce_eager=True,
        block_size=16,
        mamba_block_size=16,
        enable_prefix_caching=True,
        seed=42,
        attention_backend=ATTN_BACKEND,
    )
    prompts = [
        "hello what is one plus one what is one plus one what is one plus one the answer is",  # noqa: E501
        "hello what is one plus one what is one plus one what is one plus one the answer is",  # noqa: E501
    ]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        assert "two" in output.outputs[0].text


def _uses_flashinfer_prefill(model: torch.nn.Module) -> set[bool]:
    return {
        module.use_flashinfer_prefill
        for module in model.modules()
        if isinstance(module, MambaMixer2)
    }


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="FlashInfer SSD needs SM10x",
)
def test_flashinfer_ssd_prefill_matches_triton_with_prefix_caching(
    vllm_runner, example_prompts, monkeypatch
) -> None:
    """FlashInfer SSD prefill matches Triton, including prompts resumed from the
    prefix-cache checkpoints it wrote."""
    # In-process, so the mixers' prefill backend can be checked.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    max_tokens, num_logprobs = 32, 5
    prompts = [APC_MULTIPLY_BY * prompt for prompt in example_prompts]
    max_model_len = max(len(prompt) + max_tokens for prompt in prompts)
    kwargs = _get_vllm_runner_params(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4", max_model_len
    )
    kwargs.update(mamba_ssm_cache_dtype="float32", trust_remote_code=True)

    (baseline,), _ = _get_vLLM_output(
        vllm_runner,
        {**kwargs, "mamba_prefill_backend": "triton"},
        prompts,
        max_tokens,
        num_logprobs,
    )
    kwargs.update(
        enable_prefix_caching=True,
        mamba_prefill_backend="flashinfer",
        disable_log_stats=False,
    )
    with vllm_runner(**kwargs) as vllm_model:
        assert vllm_model.apply_model(_uses_flashinfer_prefill) == [{True}]
        filled = vllm_model.generate_greedy_logprobs(prompts, max_tokens, num_logprobs)
        hits_before = _prefix_cache_hits(vllm_model.llm)
        cached = vllm_model.generate_greedy_logprobs(prompts, max_tokens, num_logprobs)
        assert _prefix_cache_hits(vllm_model.llm) > hits_before

    for name, outputs in (("filled", filled), ("cached", cached)):
        check_logprobs_close(
            outputs_0_lst=baseline,
            outputs_1_lst=outputs,
            name_0="triton",
            name_1=f"flashinfer_{name}",
        )


def _prefix_cache_hits(llm: LLM) -> int:
    return sum(
        metric.value
        for metric in llm.get_metrics()
        if isinstance(metric, Counter) and metric.name == "vllm:prefix_cache_hits"
    )


@pytest.fixture(scope="module")
def zaya_tiny_random(tmp_path_factory) -> str:
    """A random tiny ZAYA1 checkpoint with both full and sliding window layers.

    The published tiny ZAYA1 checkpoints predate the released config format.
    """
    import torch
    from transformers import AutoTokenizer, ZayaConfig, ZayaForCausalLM

    config = ZayaConfig(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=4,
        moe_intermediate_size=64,
        router_hidden_size=32,
        layer_types=["hybrid", "hybrid_sliding"] * 2,
        sliding_window=16,
        initializer_range=0.2,
    )
    torch.manual_seed(0)
    model = ZayaForCausalLM(config).to(torch.bfloat16)
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if name.endswith("qk_norm.temp"):
                tensor.fill_(3.0)
            elif name.endswith("balancing_biases"):
                tensor.normal_(0, 0.1)
                # Transformers pins the skip expert's bias, as in the checkpoints
                tensor[-1] = -1.0
    path = tmp_path_factory.mktemp("zaya_tiny_random")
    model.save_pretrained(path)
    AutoTokenizer.from_pretrained("Zyphra/ZAYA1-8B").save_pretrained(path)
    # The checkpoint names a `processor_class` it has no image processor for
    tokenizer_config = path / "tokenizer_config.json"
    tokenizer_config.write_text(
        json.dumps(
            {
                k: v
                for k, v in json.loads(tokenizer_config.read_text()).items()
                if k != "processor_class"
            }
        )
    )
    return str(path)


def test_zaya_cca_state(
    hf_runner,
    vllm_runner,
    example_prompts,
    zaya_tiny_random,
) -> None:
    """ZAYA1's CCA carries the previous tokens' q/k and delayed value in
    per-request state, which must survive decode, chunked prefill and prefix
    cache hits. Runs in fp32 so only the hard router's near-ties differ."""
    max_tokens, num_logprobs = 32, 5
    with hf_runner(zaya_tiny_random, dtype="float32") as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(
        zaya_tiny_random,
        dtype="float32",
        moe_backend="triton",
        max_num_seqs=MAX_NUM_SEQS,
        enable_chunked_prefill=True,
        max_num_batched_tokens=32,
        enable_prefix_caching=True,
    ) as vllm_model:
        # The first pass prefills cold, the second hits the prefix cache
        for _ in range(2):
            vllm_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, max_tokens, num_logprobs
            )
            check_logprobs_close(
                outputs_0_lst=hf_outputs,
                outputs_1_lst=vllm_outputs,
                name_0="hf",
                name_1="vllm",
            )


def test_zaya_cca_exports_state_at_checkpoint() -> None:
    """A checkpoint holds the CCA state of the prefill stopped at its offset.

    A decode leads the batch, so the prefills' tokens start after its token.
    Nothing else in the cache changes.
    """
    import torch
    from torch import nn

    from vllm.model_executor.layers.mamba.checkpoint import (
        MambaPrefillCheckpointMetadata,
    )
    from vllm.model_executor.models.zaya import ZayaCCA
    from vllm.v1.attention.backends.short_conv_attn import (
        ShortConvAttentionMetadata,
    )

    channels, groups, delayed_dim, num_slots = 12, 3, 4, 8
    cca = object.__new__(ZayaCCA)
    nn.Module.__init__(cca)
    cca.history_len = 2
    cca.conv_qk_depthwise = nn.Conv1d(channels, channels, 2, groups=channels)
    cca.conv_qk_grouped = nn.Conv1d(channels, channels, 2, groups=groups)
    torch.manual_seed(0)
    initial = (
        torch.randn(num_slots, channels, cca.history_len),
        torch.randn(num_slots, delayed_dim),
    )

    def run(qk, delayed, num_decodes, prefill_lens, slots, checkpoint=None):
        cca.kv_cache = tuple(t.clone() for t in initial)
        query_start_loc = torch.tensor([0, *prefill_lens]).cumsum(0)
        metadata = ShortConvAttentionMetadata(
            num_prefills=len(prefill_lens),
            num_prefill_tokens=sum(prefill_lens),
            num_decodes=num_decodes,
            num_decode_tokens=num_decodes,
            num_reqs=num_decodes + len(prefill_lens),
            query_start_loc_p=query_start_loc,
            has_initial_states_p=torch.zeros(len(prefill_lens), dtype=torch.bool),
            state_indices_tensor_p=torch.tensor(slots[num_decodes:]),
            state_indices_tensor_d=torch.tensor(slots[:num_decodes]).view(-1, 1),
            query_start_loc_d=None,
            num_accepted_tokens=None,
            seq_lens=torch.tensor([1] * num_decodes + prefill_lens),
            checkpoint=checkpoint,
        )
        with torch.no_grad():
            cca._mix_with_state(metadata, qk, delayed, torch.zeros(delayed_dim))
        return cca.kv_cache

    prefill_lens, offsets, checkpoint_slots = [40, 24], [32, 16], [5, 6]
    num_tokens = 1 + sum(prefill_lens)
    qk, delayed = (
        torch.randn(num_tokens, channels),
        torch.randn(num_tokens, delayed_dim),
    )
    checkpoint = MambaPrefillCheckpointMetadata(
        checkpoint_offsets=torch.tensor(offsets),
        state_indices=torch.tensor(checkpoint_slots),
        offsets=offsets,
    )
    without_checkpoint = run(qk, delayed, 1, prefill_lens, [1, 2, 3])
    with_checkpoint = run(qk, delayed, 1, prefill_lens, [1, 2, 3], checkpoint)

    starts = [1, 1 + prefill_lens[0]]
    for start, offset, slot in zip(starts, offsets, checkpoint_slots):
        rows = slice(start, start + offset)
        stopped = run(qk[rows], delayed[rows], 0, [offset], [7])
        for state, stopped_state, plain in zip(
            with_checkpoint, stopped, without_checkpoint
        ):
            torch.testing.assert_close(state[slot], stopped_state[7], rtol=0, atol=0)
            state[slot] = plain[slot]
    for state, plain in zip(with_checkpoint, without_checkpoint):
        torch.testing.assert_close(state, plain, rtol=0, atol=0)
