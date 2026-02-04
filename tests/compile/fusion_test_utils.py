# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared utilities for fusion tests (e.g. test_fusion_attn.py)."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any, NamedTuple

from tests.v1.attention.utils import AttentionBackendEnum
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.platforms import current_platform

is_blackwell = lambda: current_platform.is_device_capability_family(100)
"""Are we running on Blackwell, a lot of tests depend on it"""


def has_cuda_graph_wrapper_metadata() -> bool:
    from importlib import import_module

    try:
        module = import_module("torch._inductor.utils")
        module.CUDAGraphWrapperMetadata  # noqa B018
    except AttributeError:
        return False
    return True


class Matches(NamedTuple):
    attention_fusion: int = 0
    allreduce_fusion: int = 0
    sequence_parallel: int = 0
    async_tp: int = 0
    rms_quant_norm_fusion: int = 0


class ModelBackendTestCase(NamedTuple):
    model_name: str
    model_kwargs: dict[str, Any]
    backend: AttentionBackendEnum
    matches: Matches


# E2E model test cases
DUMMY_MODELS_FP8: list[ModelBackendTestCase] = []
DUMMY_MODELS_FP4: list[ModelBackendTestCase] = []
DUMMY_MODELS: list[ModelBackendTestCase] = []  # tp-only (unquantized)
MODELS_GROUP_FP8: list[ModelBackendTestCase] = []

if current_platform.is_cuda():
    DUMMY_MODELS_FP8 = [
        ModelBackendTestCase(
            # Use smaller model for L40s in CI
            model_name="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
            model_kwargs=dict(
                max_model_len=1024,
                kv_cache_dtype="fp8",
                hf_overrides={"num_hidden_layers": 4},
                load_format="dummy",
            ),
            backend=AttentionBackendEnum.TRITON_ATTN,
            matches=Matches(
                attention_fusion=4,
                allreduce_fusion=9,
                sequence_parallel=9,
                async_tp=16,
            ),
        ),
        ModelBackendTestCase(
            model_name="nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
            model_kwargs=dict(
                max_model_len=1024,
                kv_cache_dtype="fp8",
                hf_overrides={"text_config": {"num_hidden_layers": 4}},
                load_format="dummy",
            ),
            # TODO FlashInfer attn broken on Hopper with kvcache=fp8:
            # https://github.com/vllm-project/vllm/issues/28568
            backend=AttentionBackendEnum.FLASHINFER
            if is_blackwell()
            else AttentionBackendEnum.TRITON_ATTN,
            matches=Matches(
                attention_fusion=4,
                allreduce_fusion=8,
                sequence_parallel=8,
                async_tp=8,  # mlp is moe, no fusion there
            ),
        ),
    ]

    DUMMY_MODELS_FP4 = [
        ModelBackendTestCase(
            model_name="nvidia/Llama-3.1-8B-Instruct-FP4",
            model_kwargs=dict(
                max_model_len=1024,
                kv_cache_dtype="fp8",
                hf_overrides={"num_hidden_layers": 4},
                load_format="dummy",
            ),
            backend=AttentionBackendEnum.FLASHINFER,
            matches=Matches(
                attention_fusion=4,
                allreduce_fusion=9,
                sequence_parallel=9,
                async_tp=16,
            ),
        ),
    ]

    # TP only
    DUMMY_MODELS = [
        ModelBackendTestCase(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            model_kwargs=dict(
                max_model_len=1024,
                hf_overrides={"num_hidden_layers": 4},
                load_format="dummy",
            ),
            backend=AttentionBackendEnum.TRITON_ATTN,
            matches=Matches(
                attention_fusion=0,
                allreduce_fusion=9,
                sequence_parallel=9,
                async_tp=16,
            ),
        ),
        ModelBackendTestCase(
            model_name="Qwen/Qwen3-30B-A3B",
            model_kwargs=dict(
                max_model_len=1024,
                hf_overrides={"num_hidden_layers": 4},
                load_format="dummy",
            ),
            backend=AttentionBackendEnum.TRITON_ATTN,
            matches=Matches(
                attention_fusion=0,
                allreduce_fusion=9,
                sequence_parallel=9,
                async_tp=8,  # MLP is MoE, half the fusions of dense
            ),
        ),
    ]

    MODELS_GROUP_FP8 = [
        ModelBackendTestCase(
            model_name="Qwen/Qwen3-30B-A3B-FP8",
            model_kwargs=dict(max_model_len=1024, kv_cache_dtype="fp8"),
            backend=AttentionBackendEnum.TRITON_ATTN,
            matches=Matches(
                rms_quant_norm_fusion=48,
            ),
        ),
    ]

elif current_platform.is_rocm():
    DUMMY_MODELS_FP8 = [
        ModelBackendTestCase(
            model_name="amd/Llama-3.1-8B-Instruct-FP8-KV",
            model_kwargs=dict(
                max_model_len=1024,
                hf_overrides={"num_hidden_layers": 4},
                load_format="dummy",
            ),
            backend=AttentionBackendEnum.TRITON_ATTN,
            matches=Matches(attention_fusion=4),
        ),
        ModelBackendTestCase(
            model_name="amd/Llama-3.1-8B-Instruct-FP8-KV",
            model_kwargs=dict(
                max_model_len=1024,
                hf_overrides={"num_hidden_layers": 4},
                load_format="dummy",
            ),
            backend=AttentionBackendEnum.ROCM_ATTN,
            matches=Matches(attention_fusion=4),
        ),
        ModelBackendTestCase(
            model_name="amd/Llama-3.1-8B-Instruct-FP8-KV",
            model_kwargs=dict(
                max_model_len=1024,
                hf_overrides={"num_hidden_layers": 4},
                load_format="dummy",
            ),
            backend=AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN,
            matches=Matches(attention_fusion=4),
        ),
    ]


# Custom ops toggle lists for parametrization
CUSTOM_OPS_FP8 = ["-quant_fp8", "+quant_fp8"]
CUSTOM_OPS_RMS_NORM = ["-rms_norm", "+rms_norm"]
CUSTOM_OPS_QUANT_RMS_NORM = ["+quant_fp8,+rms_norm"]


def custom_ops_product(*custom_ops_lists: list[str]) -> Iterable[str]:
    """Generate all combinations of custom ops for parametrization."""
    for op_list in itertools.product(*custom_ops_lists):
        yield ",".join(op_list)


def run_model(compile_config: int | CompilationConfig, model: str, **model_kwargs):
    """Run a model with the given compilation config for E2E fusion tests."""
    compilation_config = (
        compile_config
        if isinstance(compile_config, CompilationConfig)
        else CompilationConfig(mode=compile_config)
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0)
    # Allow override from model_kwargs
    model_kwargs = {"tensor_parallel_size": 1, **model_kwargs}
    model_kwargs = {"disable_custom_all_reduce": True, **model_kwargs}

    # No cudagraphs by default
    if compilation_config.cudagraph_mode is None:
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    llm = LLM(
        model=model,
        compilation_config=compilation_config,
        **model_kwargs,
    )
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Get the compile ranges split points after vllm config post init
    # in order to compute compile ranges correctly
    compilation_config.compile_ranges_split_points = (
        llm.llm_engine.vllm_config.compilation_config.compile_ranges_split_points
    )
