# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable
from typing import Any, NamedTuple

import pytest
import regex as re

from tests.v1.attention.utils import AttentionBackendEnum
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode, PassConfig
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.torch_utils import is_torch_equal_or_newer

from ...utils import flat_product, multi_gpu_test

is_blackwell = lambda: current_platform.is_device_capability(100)
"""Are we running on Blackwell, a lot of tests depend on it"""


class Matches(NamedTuple):
    attention_fusion: int = 0
    allreduce_fusion: int = 0
    sequence_parallel: int = 0
    async_tp: int = 0


class ModelBackendTestCase(NamedTuple):
    model_name: str
    model_kwargs: dict[str, Any]
    backend: AttentionBackendEnum
    matches: Matches


MODELS_FP8: list[ModelBackendTestCase] = []
MODELS_FP4: list[ModelBackendTestCase] = []
MODELS: list[ModelBackendTestCase] = []  # tp-only

if current_platform.is_cuda():
    MODELS_FP8 = [
        ModelBackendTestCase(
            # Use smaller model for L40s in CI
            model_name="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
            model_kwargs=dict(max_model_len=1024, kv_cache_dtype="fp8"),
            backend=AttentionBackendEnum.TRITON_ATTN,
            matches=Matches(
                attention_fusion=32,
                allreduce_fusion=65,
                sequence_parallel=65,
                async_tp=128,
            ),
        ),
        ModelBackendTestCase(
            model_name="nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
            model_kwargs=dict(max_model_len=1024, kv_cache_dtype="fp8"),
            # TODO FlashInfer attn broken on Hopper with kvcache=fp8:
            # https://github.com/vllm-project/vllm/issues/28568
            backend=AttentionBackendEnum.FLASHINFER
            if is_blackwell()
            else AttentionBackendEnum.TRITON_ATTN,
            matches=Matches(
                attention_fusion=48,
                allreduce_fusion=96,
                sequence_parallel=96,
                async_tp=95,  # mlp is moe, no fusion there
            ),
        ),
    ]

    MODELS_FP4 = [
        ModelBackendTestCase(
            model_name="nvidia/Llama-3.1-8B-Instruct-FP4",
            model_kwargs=dict(max_model_len=1024, kv_cache_dtype="fp8"),
            backend=AttentionBackendEnum.FLASHINFER,
            matches=Matches(
                attention_fusion=32,
                allreduce_fusion=65,
                sequence_parallel=65,
                async_tp=128,
            ),
        ),
    ]

    # TP only
    MODELS = [
        ModelBackendTestCase(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            model_kwargs=dict(max_model_len=1024),
            backend=AttentionBackendEnum.TRITON_ATTN,
            matches=Matches(
                attention_fusion=0,
                allreduce_fusion=65,
                sequence_parallel=65,
                async_tp=128,
            ),
        ),
        ModelBackendTestCase(
            model_name="Qwen/Qwen3-30B-A3B",
            model_kwargs=dict(max_model_len=1024),
            backend=AttentionBackendEnum.TRITON_ATTN,
            matches=Matches(
                attention_fusion=0,
                allreduce_fusion=97,
                sequence_parallel=97,
                async_tp=96,  # MLP is MoE, half the fusions of dense
            ),
        ),
    ]

elif current_platform.is_rocm():
    MODELS_FP8 = [
        ModelBackendTestCase(
            model_name="amd/Llama-3.1-8B-Instruct-FP8-KV",
            model_kwargs=dict(max_model_len=1024),
            backend=AttentionBackendEnum.TRITON_ATTN,
            matches=Matches(attention_fusion=32),
        ),
        ModelBackendTestCase(
            model_name="amd/Llama-3.1-8B-Instruct-FP8-KV",
            model_kwargs=dict(max_model_len=1024),
            backend=AttentionBackendEnum.ROCM_ATTN,
            matches=Matches(attention_fusion=32),
        ),
        ModelBackendTestCase(
            model_name="amd/Llama-3.1-8B-Instruct-FP8-KV",
            model_kwargs=dict(max_model_len=1024),
            backend=AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN,
            matches=Matches(attention_fusion=32),
        ),
    ]

CUSTOM_OPS_FP8 = ["-quant_fp8", "+quant_fp8"]


@pytest.mark.parametrize(
    "model_name, model_kwargs, backend, matches, custom_ops",
    # Test attention+quant_fp8 fusion with custom and torch impls of QuantFP8
    list(flat_product(MODELS_FP8, CUSTOM_OPS_FP8))
    # quant_fp4 only has the custom impl
    + list(flat_product(MODELS_FP4, [""])),
)
@pytest.mark.parametrize("inductor_graph_partition", [True, False])
def test_attn_quant(
    model_name: str,
    model_kwargs: dict[str, Any],
    backend: AttentionBackendEnum,
    matches: Matches,
    custom_ops: str,
    inductor_graph_partition: bool,
    caplog_mp_spawn,
    monkeypatch,
):
    if backend == AttentionBackendEnum.FLASHINFER and (
        not is_blackwell() or not has_flashinfer()
    ):
        pytest.skip("FlashInfer attn fusion requires Blackwell and flashinfer")
    if inductor_graph_partition and not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("Inductor graph partition requires torch>=2.9")

    custom_ops_list = custom_ops.split(",") if custom_ops else []

    if inductor_graph_partition:
        mode = CUDAGraphMode.FULL_AND_PIECEWISE
        splitting_ops: list[str] | None = None
    else:
        # FIXME: Llama-4-Scout-17B-16E-Instruct-FP8 + FlashInfer + Blackwell end at
        # CUDAGraphMode.NONE here because it derives an attention backend that
        # does not support full cudagraphs
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
        mode=CompilationMode.VLLM_COMPILE,
        pass_config=PassConfig(fuse_attn_quant=True, eliminate_noops=True),
        # Inductor caches custom passes by default as well via uuid
        inductor_compile_config={"force_disable_caches": True},
    )

    with caplog_mp_spawn(logging.DEBUG) as log_holder:
        run_model(compilation_config, model_name, **model_kwargs)

    log_matches = re.findall(
        r"fusion_attn.py:\d+] Fused quant onto (\d+) attention nodes",
        log_holder.text,
    )
    assert len(log_matches) == 1, log_holder.text
    assert int(log_matches[0]) == matches.attention_fusion


CUSTOM_OPS_RMS_NORM = ["-rms_norm", "+rms_norm"]


def custom_ops_product(*custom_ops_lists: list[str]) -> Iterable[str]:
    for op_list in itertools.product(*custom_ops_lists):
        yield ",".join(op_list)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model_name, model_kwargs, backend, matches, custom_ops",
    # Toggle RMSNorm and QuantFP8 for FP8 models
    list(
        flat_product(
            MODELS_FP8, custom_ops_product(CUSTOM_OPS_FP8, CUSTOM_OPS_RMS_NORM)
        )
    )
    # Toggle RMSNorm for FP4 models and unquant models
    + list(flat_product(MODELS_FP4 + MODELS, CUSTOM_OPS_RMS_NORM)),
)
@pytest.mark.parametrize("inductor_graph_partition", [True, False])
@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not has_flashinfer()
    or not current_platform.has_device_capability(90),
    reason="allreduce+rmsnorm fusion requires flashinfer",
)
def test_tp2_attn_quant_allreduce_rmsnorm(
    model_name: str,
    model_kwargs: dict,
    backend: AttentionBackendEnum,
    matches: Matches,
    custom_ops: str,
    inductor_graph_partition: bool,
    caplog_mp_spawn,
    monkeypatch,
):
    if inductor_graph_partition and not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("Inductor graph partition requires torch>=2.9")

    if "fp4" in model_name.lower() and not is_blackwell():
        pytest.skip("NVFP4 quant requires Blackwell")

    if backend == AttentionBackendEnum.FLASHINFER and not is_blackwell():
        # FlashInfer attn fusion requires Blackwell
        matches = matches._replace(attention_fusion=0)

    custom_ops_list = custom_ops.split(",") if custom_ops else []

    if inductor_graph_partition:
        mode = CUDAGraphMode.FULL_AND_PIECEWISE
        splitting_ops: list[str] | None = None
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
        mode=CompilationMode.VLLM_COMPILE,
        pass_config=PassConfig(
            fuse_attn_quant=True,
            eliminate_noops=True,
            fuse_allreduce_rms=True,
        ),
        # Inductor caches custom passes by default as well via uuid
        inductor_compile_config={"force_disable_caches": True},
    )

    with caplog_mp_spawn(logging.DEBUG) as log_holder:
        run_model(
            compilation_config, model_name, tensor_parallel_size=2, **model_kwargs
        )
    log_matches = re.findall(
        r"fusion_attn.py:\d+] Fused quant onto (\d+) attention nodes",
        log_holder.text,
    )
    assert len(log_matches) == 2, log_holder.text

    assert int(log_matches[0]) == matches.attention_fusion
    assert int(log_matches[1]) == matches.attention_fusion

    log_matches = re.findall(
        r"collective_fusion.py:\d+] Replaced (\d+) patterns",
        log_holder.text,
    )
    assert len(log_matches) == 2, log_holder.text

    assert int(log_matches[0]) == matches.allreduce_fusion
    assert int(log_matches[1]) == matches.allreduce_fusion


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model_name, model_kwargs, backend, matches, custom_ops",
    # Toggle RMSNorm and QuantFP8 for FP8 models
    list(
        flat_product(
            MODELS_FP8, custom_ops_product(CUSTOM_OPS_FP8, CUSTOM_OPS_RMS_NORM)
        )
    )
    # Toggle RMSNorm for FP4 models and unquant models
    + list(flat_product(MODELS_FP4 + MODELS, CUSTOM_OPS_RMS_NORM)),
)
@pytest.mark.parametrize("inductor_graph_partition", [True, False])
@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="sequence parallel only tested on CUDA",
)
def test_tp2_attn_quant_async_tp(
    model_name: str,
    model_kwargs: dict,
    backend: AttentionBackendEnum,
    matches: Matches,
    custom_ops: str,
    inductor_graph_partition: bool,
    caplog_mp_spawn,
    monkeypatch,
):
    if is_blackwell():
        # TODO: https://github.com/vllm-project/vllm/issues/27893
        pytest.skip("Blackwell is not supported for AsyncTP pass")

    if inductor_graph_partition and not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("Inductor graph partition requires torch>=2.9")

    if "fp4" in model_name.lower() and not is_blackwell():
        pytest.skip("NVFP4 quant requires Blackwell")

    if backend == AttentionBackendEnum.FLASHINFER:
        if not has_flashinfer():
            pytest.skip("FlashInfer backend requires flashinfer installed")
        if not is_blackwell():
            # FlashInfer attn fusion requires Blackwell
            matches = matches._replace(attention_fusion=0)

    custom_ops_list = custom_ops.split(",") if custom_ops else []

    if inductor_graph_partition:
        mode = CUDAGraphMode.FULL_AND_PIECEWISE
        splitting_ops: list[str] | None = None
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
        level=CompilationMode.VLLM_COMPILE,
        pass_config=PassConfig(
            fuse_attn_quant=True,
            eliminate_noops=True,
            enable_sp=True,
            fuse_gemm_comms=True,
        ),
        # Inductor caches custom passes by default as well via uuid
        inductor_compile_config={"force_disable_caches": True},
    )

    with caplog_mp_spawn(logging.DEBUG) as log_holder:
        run_model(
            compilation_config, model_name, tensor_parallel_size=2, **model_kwargs
        )
    log_matches = re.findall(
        r"fusion_attn.py:\d+] Fused quant onto (\d+) attention nodes",
        log_holder.text,
    )
    assert len(log_matches) == 2, log_holder.text

    assert int(log_matches[0]) == matches.attention_fusion
    assert int(log_matches[1]) == matches.attention_fusion

    log_matches = re.findall(
        r"sequence_parallelism.py:\d+] Replaced (\d+) patterns",
        log_holder.text,
    )
    assert len(log_matches) == 2, log_holder.text

    assert int(log_matches[0]) == matches.sequence_parallel
    assert int(log_matches[1]) == matches.sequence_parallel

    log_matches = re.findall(
        r"collective_fusion.py:\d+] Replaced (\d+) patterns",
        log_holder.text,
    )
    assert len(log_matches) == 2, log_holder.text

    assert int(log_matches[0]) == matches.async_tp
    assert int(log_matches[1]) == matches.async_tp


def run_model(compile_config: int | CompilationConfig, model: str, **model_kwargs):
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
