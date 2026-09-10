# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared, non-collected helpers for the hybrid model test slices.

This module holds the body of the original ``test_hybrid.py::test_models``
plus the constants and APC helpers shared by the split files under
``hybrid/`` and ``hybrid_granite/``. The leading underscore keeps it out of
pytest's ``test_*.py`` collection; ``tests/tools/test_language_layout.py``
proves it is never collected on its own.
"""

from contextlib import contextmanager, nullcontext

from tests.models.registry import HF_EXAMPLE_MODELS
from vllm.platforms import current_platform

from ...utils import check_logprobs_close

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
    "Zyphra/Zamba2-1.2B-instruct",
    "ibm-granite/granite-4.0-tiny-preview",
    "tiiuae/Falcon-H1-0.5B-Base",
    "LiquidAI/LFM2-1.2B",
    "tiny-random/qwen3-next-moe",
]

FULL_CUDA_GRAPH_MODELS = [
    "ai21labs/Jamba-tiny-dev",
    "Zyphra/Zamba2-1.2B-instruct",
]

FP32_STATE_MODELS = [
    "state-spaces/mamba-130m-hf",
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


def _test_models(
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
