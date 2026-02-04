# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from typing import Any

import pytest
import regex as re

from tests.compile.fusion_test_utils import (
    CUSTOM_OPS_FP8,
    CUSTOM_OPS_QUANT_RMS_NORM,
    CUSTOM_OPS_RMS_NORM,
    DUMMY_MODELS,
    DUMMY_MODELS_FP4,
    DUMMY_MODELS_FP8,
    MODELS_GROUP_FP8,
    Matches,
    custom_ops_product,
    is_blackwell,
    run_model,
)
from tests.v1.attention.utils import AttentionBackendEnum
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode, PassConfig
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.torch_utils import is_torch_equal_or_newer

from ...utils import flat_product, multi_gpu_test


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model_name, model_kwargs, backend, matches, custom_ops",
    # Toggle RMSNorm and QuantFP8 for FP8 models
    list(
        flat_product(
            DUMMY_MODELS_FP8, custom_ops_product(CUSTOM_OPS_FP8, CUSTOM_OPS_RMS_NORM)
        )
    )
    # Toggle RMSNorm for FP4 models and unquant models
    + list(flat_product(DUMMY_MODELS_FP4 + DUMMY_MODELS, CUSTOM_OPS_RMS_NORM)),
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

    model_kwargs["attention_config"] = {"backend": backend.name}

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
    # 2 for each compile range
    # (global compile range can be split due to fuse_allreduce_rmsnorm)
    num_compile_ranges = len(compilation_config.get_compile_ranges())
    assert num_compile_ranges in [1, 2]

    assert len(log_matches) == 2 * num_compile_ranges, log_holder.text

    assert all(int(log_match) == matches.attention_fusion for log_match in log_matches)

    log_matches = re.findall(
        r"collective_fusion.py:\d+] Replaced (\d+) patterns",
        log_holder.text,
    )
    assert len(log_matches) == 2, log_holder.text

    assert int(log_matches[0]) == matches.allreduce_fusion
    assert int(log_matches[1]) == matches.allreduce_fusion

    log_matches = re.findall(
        r"pass_manager.py:\d+] Skipping .*AllReduceFusionPass.* with compile range",
        log_holder.text,
    )
    assert len(log_matches) == 2 * (num_compile_ranges - 1), log_holder.text


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model_name, model_kwargs, backend, matches, custom_ops",
    # Toggle RMSNorm and QuantFP8 for FP8 models
    list(
        flat_product(
            DUMMY_MODELS_FP8, custom_ops_product(CUSTOM_OPS_FP8, CUSTOM_OPS_RMS_NORM)
        )
    )
    # Toggle RMSNorm for FP4 models and unquant models
    + list(flat_product(DUMMY_MODELS_FP4 + DUMMY_MODELS, CUSTOM_OPS_RMS_NORM)),
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

    model_kwargs["attention_config"] = {"backend": backend.name}

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


@pytest.mark.parametrize(
    "model_name, model_kwargs, backend, matches, custom_ops",
    # Test rms norm+group quant_fp8 fusion
    list[tuple[Any, ...]](flat_product(MODELS_GROUP_FP8, CUSTOM_OPS_QUANT_RMS_NORM)),
)
@pytest.mark.parametrize("inductor_graph_partition", [True, False])
# TODO: remove skip after we fix the fusion thoroughly
@pytest.mark.skipif(is_blackwell(), reason="Temporarily disabled on Blackwell")
def test_rms_group_quant(
    model_name: str,
    model_kwargs: dict[str, Any],
    backend: AttentionBackendEnum,
    matches: Matches,
    custom_ops: str,
    inductor_graph_partition: bool,
    caplog_mp_spawn,
    monkeypatch,
):
    if inductor_graph_partition and not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("Inductor graph partition requires torch>=2.9")

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

    # TODO: remove this after fusion is fixed
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES", "0")

    model_kwargs["attention_config"] = {"backend": backend.name}

    compilation_config = CompilationConfig(
        # Testing properties
        custom_ops=custom_ops_list,
        use_inductor_graph_partition=inductor_graph_partition,
        cudagraph_mode=mode,
        splitting_ops=splitting_ops,
        # Common
        mode=CompilationMode.VLLM_COMPILE,
        pass_config=PassConfig(
            fuse_norm_quant=True, fuse_act_quant=True, eliminate_noops=True
        ),
        # Inductor caches custom passes by default as well via uuid
        inductor_compile_config={"force_disable_caches": True},
    )

    with caplog_mp_spawn(logging.DEBUG) as log_holder:
        run_model(compilation_config, model_name, **model_kwargs)

    log_matches = re.findall(
        r"\[fusion.py:\d+] Replaced (\d+) patterns",
        log_holder.text,
    )
    assert len(log_matches) == 1, log_holder.text
    assert int(log_matches[0]) == matches.rms_quant_norm_fusion
