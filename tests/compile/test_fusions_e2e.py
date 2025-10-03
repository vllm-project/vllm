# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable
from typing import Any, Optional, Union

import pytest
import regex as re

from tests.v1.attention.utils import _Backend
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel, CUDAGraphMode, PassConfig
from vllm.platforms import current_platform
from vllm.utils import is_torch_equal_or_newer
from vllm.utils.flashinfer import has_flashinfer

from ..utils import flat_product, multi_gpu_test

MODELS_FP8: list[tuple[str, dict[str, Any], _Backend]] = []
MODELS_FP4: list[tuple[str, dict[str, Any], _Backend]] = []
MODELS: list[tuple[str, dict[str, Any], _Backend]] = []  # tp-only

if current_platform.is_cuda():
    MODELS_FP8 += [
        (
            "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
            {"max_model_len": 1024},
            _Backend.TRITON_ATTN,
        )
    ]

    if current_platform.is_device_capability((10, 0)) and has_flashinfer():
        MODELS_FP8 += [
            (
                "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
                {"kv_cache_dtype": "fp8", "max_model_len": 1024},
                _Backend.FLASHINFER,
            )
        ]

        MODELS_FP4 += [
            (
                "nvidia/Llama-4-Scout-17B-16E-Instruct-FP4",
                {"kv_cache_dtype": "fp8", "max_model_len": 1024},
                _Backend.FLASHINFER,
            )
        ]

        MODELS += [
            (
                "meta-llama/Llama-3.1-8B-Instruct",
                {"max_model_len": 1024},
                _Backend.FLASHINFER,
            )
        ]

elif current_platform.is_rocm():
    MODELS_FP8 += [("amd/Llama-3.1-8B-Instruct-FP8-KV", {}, _Backend.TRITON_ATTN)]

INDUCTOR_GRAPH_PARTITION = (
    [True, False] if (is_torch_equal_or_newer("2.9.0.dev")) else [False]
)

# TODO(luka) test both in nightly
CUSTOM_OPS_FP8 = ["-quant_fp8"]  # , "+quant_fp8"]


@pytest.mark.parametrize(
    "model_name, model_kwargs, backend, custom_ops",
    # Test attention+quant_fp8 fusion with custom and torch impls of QuantFP8
    list(flat_product(MODELS_FP8, CUSTOM_OPS_FP8))
    # quant_fp4 only has the custom impl
    + list(flat_product(MODELS_FP4, [""])),
)
@pytest.mark.parametrize("inductor_graph_partition", INDUCTOR_GRAPH_PARTITION)
def test_attn_quant(
    model_name: str,
    model_kwargs: dict[str, Any],
    backend: _Backend,
    custom_ops: str,
    inductor_graph_partition: bool,
    caplog_mp_spawn,
    monkeypatch,
):
    custom_ops_list = custom_ops.split(",") if custom_ops else []

    if inductor_graph_partition:
        mode = CUDAGraphMode.FULL_AND_PIECEWISE
        splitting_ops: Optional[list[str]] = None
    else:
        mode = CUDAGraphMode.FULL_DECODE_ONLY
        splitting_ops = []

    # Disable, compile cache to make sure custom passes run.
    # Otherwise, we can't verify fusion happened through the logs.
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    # To capture subprocess logs, we need to know whether spawn or fork is used.
    # Force spawn as it is more general.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", backend.name)

    compilation_config = CompilationConfig(
        # Testing properties
        custom_ops=custom_ops_list,
        use_inductor_graph_partition=inductor_graph_partition,
        cudagraph_mode=mode,
        splitting_ops=splitting_ops,
        # Common
        level=CompilationLevel.PIECEWISE,
        pass_config=PassConfig(enable_attn_fusion=True, enable_noop=True),
        # Inductor caches custom passes by default as well via uuid
        inductor_compile_config={"force_disable_caches": True},
    )

    with caplog_mp_spawn(logging.DEBUG) as log_holder:
        run_model(compilation_config, model_name, **model_kwargs)

    assert "Fused quant onto 48 attention nodes" in log_holder.text, log_holder.text


# TODO(luka) test both in nightly
# TODO(luka) change to -
CUSTOM_OPS_RMS_NORM = ["+rms_norm"]  # , "+rms_norm"]


def custom_ops_product(*custom_ops_lists: list[str]) -> Iterable[str]:
    for op_list in itertools.product(*custom_ops_lists):
        yield ",".join(op_list)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model_name, model_kwargs, backend, custom_ops",
    # Toggle RMSNorm and QuantFP8 for FP8 models
    list(flat_product(MODELS_FP8, ["+quant_fp8,+rms_norm"]))
    # custom_ops_product(CUSTOM_OPS_FP8, CUSTOM_OPS_RMS_NORM))) # TODO
    # Toggle RMSNorm for FP4 models and unquant models
    + list(flat_product(MODELS_FP4 + MODELS, CUSTOM_OPS_RMS_NORM)),
)
@pytest.mark.parametrize("inductor_graph_partition", INDUCTOR_GRAPH_PARTITION)
@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not has_flashinfer()
    or not current_platform.has_device_capability(90),
    reason="allreduce+rmsnorm fusion requires flashinfer",
)
def test_tp2_attn_quant_allreduce_rmsnorm(
    model_name,
    model_kwargs,
    backend,
    custom_ops: str,
    inductor_graph_partition: bool,
    caplog_mp_spawn,
    monkeypatch,
):
    custom_ops_list = custom_ops.split(",") if custom_ops else []

    if inductor_graph_partition:
        mode = CUDAGraphMode.FULL_AND_PIECEWISE
        splitting_ops: Optional[list[str]] = None
    else:
        mode = CUDAGraphMode.FULL_DECODE_ONLY
        splitting_ops = []

    # Disable, compile cache to make sure custom passes run.
    # Otherwise, we can't verify fusion happened through the logs.
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    # To capture subprocess logs, we need to know whether spawn or fork is used.
    # Force spawn as it is more general.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", backend.name)

    compilation_config = CompilationConfig(
        # Testing properties
        use_inductor_graph_partition=inductor_graph_partition,
        cudagraph_mode=mode,
        custom_ops=custom_ops_list,
        splitting_ops=splitting_ops,
        # Common
        level=CompilationLevel.PIECEWISE,
        pass_config=PassConfig(
            enable_attn_fusion=True,
            enable_noop=True,
            enable_fi_allreduce_fusion=True,
        ),
        # Inductor caches custom passes by default as well via uuid
        inductor_compile_config={"force_disable_caches": True},
    )

    with caplog_mp_spawn(logging.DEBUG) as log_holder:
        run_model(
            compilation_config, model_name, tensor_parallel_size=2, **model_kwargs
        )

    assert "Fused quant onto 48 attention nodes" in log_holder.text, log_holder.text

    matches = re.findall(
        r"\[collective_fusion.py:\d+] Replaced 96 patterns", log_holder.text
    )
    assert len(matches) == 2, log_holder.text


def run_model(
    compile_config: Union[int, CompilationConfig], model: str, **model_kwargs
):
    compilation_config = (
        compile_config
        if isinstance(compile_config, CompilationConfig)
        else CompilationConfig(level=compile_config)
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
