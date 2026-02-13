# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark for FlashInfer fused collective operations vs standard operations.

This benchmark compares:
1. FlashInfer's allreduce_fusion (fused allreduce + rmsnorm + optional quant)
2. Standard tensor_model_parallel_all_reduce + separate rmsnorm/quant operations

Usage with torchrun:
    torchrun --nproc_per_node=2 benchmark_fused_collective.py

"""

import argparse
import itertools
import os
import time

import pandas as pd
import torch  # type: ignore
import torch.distributed as dist  # type: ignore

from vllm.config.vllm import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import (
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm  # noqa
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8  # noqa
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape  # noqa
from vllm.platforms import current_platform  # noqa

RMS_NORM_OP = torch.ops._C.rms_norm
FUSED_ADD_RMS_NORM_OP = torch.ops._C.fused_add_rms_norm
RMS_NORM_STATIC_FP8_QUANT_OP = torch.ops._C.rms_norm_static_fp8_quant
FUSED_ADD_RMS_NORM_STATIC_FP8_QUANT_OP = (
    torch.ops._C.fused_add_rms_norm_static_fp8_quant
)
SCALED_FP4_QUANT_OP = torch.ops._C.scaled_fp4_quant

logger = init_logger(__name__)

# Try to import FlashInfer
try:
    import flashinfer.comm as flashinfer_comm  # type: ignore

    if not (
        hasattr(flashinfer_comm, "allreduce_fusion")
        and hasattr(flashinfer_comm, "create_allreduce_fusion_workspace")
    ):
        flashinfer_comm = None
        logger.warning("FlashInfer comm module found but missing allreduce_fusion API")
except ImportError:
    flashinfer_comm = None
    logger.warning("FlashInfer not found, only benchmarking standard operations")

# Constants
FP8_DTYPE = current_platform.fp8_dtype()
MiB = 1024 * 1024

# FlashInfer max sizes per world size
# Enable 64MB for 2, 4, 8 world sizes to verify large input sizes
# use --disable-oneshot to disable oneshot mode for very large input sizes
_FI_MAX_SIZES = {
    2: 64 * MiB,  # 64MB
    4: 64 * MiB,  # 64MB
    8: 64 * MiB,  # 64MB
}

# Global workspace tensor for FlashInfer
_FI_WORKSPACE = None


def setup_flashinfer_workspace(
    world_size: int,
    rank: int,
    hidden_dim: int,
    max_token_num: int,
    dtype: torch.dtype,
):
    """Setup FlashInfer workspace for fused allreduce operations."""
    global _FI_WORKSPACE

    if flashinfer_comm is None:
        return None, None

    if world_size not in _FI_MAX_SIZES:
        logger.warning("FlashInfer not supported for world size %s", world_size)
        return None, None

    try:
        workspace = flashinfer_comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
        )

        _FI_WORKSPACE = workspace
        return workspace
    except Exception as e:
        logger.error("Failed to setup FlashInfer workspace: %s", e)
        return None


def cleanup_flashinfer_workspace(workspace):
    """Cleanup FlashInfer workspace."""
    if flashinfer_comm is None or workspace is None:
        return

    try:
        workspace.destroy()
    except Exception as e:
        logger.error("Failed to cleanup FlashInfer workspace: %s", e)


class FlashInferFusedAllReduceParams:
    """Parameters for FlashInfer fused allreduce operations."""

    def __init__(
        self,
        max_token_num: int = 1024,
    ):
        self.launch_with_pdl = True
        self.fp32_acc = True
        self.max_token_num = max_token_num

    def get_trtllm_fused_allreduce_kwargs(self):
        return {
            "launch_with_pdl": self.launch_with_pdl,
            "fp32_acc": self.fp32_acc,
        }


def flashinfer_fused_allreduce_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor | None,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    allreduce_params: "FlashInferFusedAllReduceParams",
    use_oneshot: bool,
    norm_out: torch.Tensor | None = None,
):
    """FlashInfer fused allreduce + rmsnorm operation."""
    if flashinfer_comm is None or _FI_WORKSPACE is None:
        raise RuntimeError("FlashInfer not available or workspace not initialized")

    if norm_out is None:
        norm_out = input_tensor
        residual_out = residual
    else:
        residual_out = input_tensor

    flashinfer_comm.allreduce_fusion(
        input=input_tensor,
        workspace=_FI_WORKSPACE,
        pattern=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        quant_out=None,
        scale_out=None,
        layout_code=flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4,
        scale_factor=None,
        use_oneshot=use_oneshot,
        **allreduce_params.get_trtllm_fused_allreduce_kwargs(),
    )


def flashinfer_fused_allreduce_rmsnorm_fp8_quant(
    input_tensor: torch.Tensor,
    residual: torch.Tensor | None,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    scale_factor: torch.Tensor,
    allreduce_params: FlashInferFusedAllReduceParams,
    use_oneshot: bool = True,
    norm_out: torch.Tensor | None = None,
    quant_out: torch.Tensor | None = None,
):
    """FlashInfer fused allreduce + rmsnorm + FP8 quantization."""
    if flashinfer_comm is None or _FI_WORKSPACE is None:
        raise RuntimeError("FlashInfer not available or workspace not initialized")

    if norm_out is None:
        norm_out = input_tensor
        residual_out = residual
    else:
        residual_out = input_tensor

    flashinfer_comm.allreduce_fusion(
        input=input_tensor,
        workspace=_FI_WORKSPACE,
        pattern=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        quant_out=quant_out,
        scale_out=None,
        layout_code=flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4,
        scale_factor=scale_factor,
        use_oneshot=use_oneshot,
        **allreduce_params.get_trtllm_fused_allreduce_kwargs(),
    )


def flashinfer_fused_allreduce_rmsnorm_fp4_quant(
    input_tensor: torch.Tensor,
    residual: torch.Tensor | None,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    input_global_scale: torch.Tensor,
    allreduce_params: FlashInferFusedAllReduceParams,
    quant_out: torch.Tensor,
    use_oneshot: bool,
    output_scale: torch.Tensor,
    norm_out: torch.Tensor | None = None,
):
    """FlashInfer fused allreduce + rmsnorm + FP4 quantization."""
    if flashinfer_comm is None or _FI_WORKSPACE is None:
        raise RuntimeError("FlashInfer not available or workspace not initialized")

    if norm_out is None:
        norm_out = input_tensor
        residual_out = residual
    else:
        residual_out = input_tensor

    flashinfer_comm.allreduce_fusion(
        input=input_tensor,
        workspace=_FI_WORKSPACE,
        pattern=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        quant_out=quant_out,
        scale_out=output_scale,
        layout_code=flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4,
        scale_factor=input_global_scale,
        use_oneshot=use_oneshot,
        **allreduce_params.get_trtllm_fused_allreduce_kwargs(),
    )


class VllmFusedAllreduce:
    def __init__(self, hidden_dim, dtype):
        self.rms_eps = 1e-6
        self.rms_norm = RMSNorm(hidden_dim, eps=self.rms_eps, dtype=dtype)
        self.fp8_quant = QuantFP8(
            static=True,
            group_shape=GroupShape.PER_TENSOR,
        )

    def allreduce_rmsnorm(
        self, input_tensor: torch.Tensor, residual: torch.Tensor | None
    ):
        allreduce_out = tensor_model_parallel_all_reduce(input_tensor)
        return self.rms_norm(allreduce_out, residual)

    def allreduce_rmsnorm_fp8_quant(
        self,
        input_tensor: torch.Tensor,
        residual: torch.Tensor | None,
        scale_factor: torch.Tensor,
    ):
        allreduce_out = tensor_model_parallel_all_reduce(input_tensor)
        rms_out = self.rms_norm(allreduce_out, residual)
        if residual is None:
            quant_out = self.fp8_quant(rms_out, scale_factor)
            return quant_out
        else:
            rms_out, residual_out = rms_out
            quant_out = self.fp8_quant(rms_out, scale_factor)
            return quant_out, residual_out

    def allreduce_rmsnorm_fp4_quant(
        self,
        input_tensor: torch.Tensor,
        residual: torch.Tensor | None,
        input_global_scale: torch.Tensor,
        quant_out: torch.Tensor,
        output_scale: torch.Tensor,
    ):
        allreduce_out = tensor_model_parallel_all_reduce(input_tensor)
        rms_out = self.rms_norm(allreduce_out, residual)
        if residual is None:
            SCALED_FP4_QUANT_OP(quant_out, rms_out, output_scale, input_global_scale)
            return quant_out, output_scale
        else:
            rms_out, residual_out = rms_out
            SCALED_FP4_QUANT_OP(quant_out, rms_out, output_scale, input_global_scale)
            return quant_out, residual_out, output_scale


def create_test_tensors(
    num_tokens: int, hidden_dim: int, dtype: torch.dtype, use_residual: bool = True
):
    """Create test tensors for benchmarking."""
    input_tensor = torch.randn(num_tokens, hidden_dim, dtype=dtype)
    residual = (
        torch.randn_like(input_tensor)
        if use_residual
        else torch.zeros_like(input_tensor)
    )
    rms_gamma = torch.ones(hidden_dim, dtype=dtype)
    norm_out = None if use_residual else torch.empty_like(input_tensor)

    # Quantization scales
    scale_fp8 = torch.tensor(1.0, dtype=torch.float32)
    scale_fp4 = torch.tensor(1.0, dtype=torch.float32)
    quant_out_fp8 = torch.empty_like(input_tensor, dtype=FP8_DTYPE)
    # Pre-allocate FP4 output tensors (to avoid allocation overhead in benchmarks)
    fp4_quant_out = torch.empty((num_tokens, hidden_dim // 2), dtype=torch.uint8)
    fp4_output_scale = torch.empty((128, 4), dtype=torch.int32)

    return (
        input_tensor,
        norm_out,
        residual,
        rms_gamma,
        scale_fp8,
        quant_out_fp8,
        scale_fp4,
        fp4_quant_out,
        fp4_output_scale,
    )


def benchmark_operation(
    operation_func, *args, warmup: int = 5, trials: int = 20, **kwargs
):
    """Benchmark a single operation using CUDA graphs."""
    # Warmup before graph capture
    for _ in range(warmup):
        operation_func(*args, **kwargs)
    torch.cuda.synchronize()

    # Create CUDA graph
    graph = torch.cuda.CUDAGraph()
    num_op_per_cudagraph = 10

    # Use vLLM's graph_capture to make tensor_model_parallel_all_reduce graph-safe
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    with graph_capture(device=device), torch.cuda.graph(graph):
        for _ in range(num_op_per_cudagraph):
            operation_func(*args, **kwargs)

    # Graph warmup
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()

    # Benchmark with CUDA graph
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(trials // num_op_per_cudagraph):
        # operation_func(*args, **kwargs)
        graph.replay()

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_time_ms = ((end_time - start_time) / trials) * 1000
    return avg_time_ms


def run_benchmarks(
    num_tokens: int,
    hidden_dim: int,
    dtype: torch.dtype,
    use_residual: bool,
    allreduce_params: FlashInferFusedAllReduceParams | None,
    quant_modes: set[str],
    no_oneshot: bool,
):
    """Run all benchmarks for given configuration.

    Args:
        quant_mode: "none", "fp8_only", "fp4_only", or "all"
    """
    (
        input_tensor,
        norm_out,
        residual,
        rms_gamma,
        scale_fp8,
        quant_out_fp8,
        scale_fp4,
        fp4_quant_out,
        fp4_output_scale,
    ) = create_test_tensors(num_tokens, hidden_dim, dtype, use_residual)

    rms_eps = 1e-6
    results = {}
    vllm_fused_allreduce = VllmFusedAllreduce(hidden_dim, dtype)
    use_oneshot_options = [False] if no_oneshot else [True, False]

    # Create RMSNorm and QuantFP8 layers once for native benchmarks

    if "none" in quant_modes:
        # Standard AllReduce + RMSNorm
        for custom_op in ["-rms_norm", "+rms_norm"]:
            with set_current_vllm_config(
                VllmConfig(compilation_config=CompilationConfig(custom_ops=[custom_op]))
            ):
                try:
                    suffix = (
                        "_custom_rms_norm" if "+" in custom_op else "_native_rms_norm"
                    )
                    time_ms = benchmark_operation(
                        vllm_fused_allreduce.allreduce_rmsnorm,
                        input_tensor,
                        residual=residual,
                    )
                    results[f"standard_allreduce_{suffix}"] = time_ms
                except Exception as e:
                    logger.error("Standard AllReduce+RMSNorm failed: %s", e)
                    results[f"standard_allreduce_{suffix}"] = float("inf")

        # Standard AllReduce + RMSNorm Native Compiled
        with set_current_vllm_config(
            VllmConfig(compilation_config=CompilationConfig(custom_ops=["-rms_norm"]))
        ):
            try:
                standard_allreduce_rmsnorm_native_compiled = torch.compile(
                    vllm_fused_allreduce.allreduce_rmsnorm,
                    fullgraph=True,
                    dynamic=False,
                )
                time_ms = benchmark_operation(
                    standard_allreduce_rmsnorm_native_compiled,
                    input_tensor,
                    residual=residual,
                )
                results["standard_allreduce_rmsnorm_native_compiled"] = time_ms
            except Exception as e:
                logger.error("Standard AllReduce+RMSNorm Native Compiled failed: %s", e)
                results["standard_allreduce_rmsnorm_native_compiled"] = float("inf")

        # FlashInfer Fused AllReduce + RMSNorm Oneshot/Twoshot
        if flashinfer_comm is not None and allreduce_params is not None:
            for use_oneshot in use_oneshot_options:
                suffix = "_oneshot" if use_oneshot else "_twoshot"
                try:
                    time_ms = benchmark_operation(
                        flashinfer_fused_allreduce_rmsnorm,
                        input_tensor,
                        residual=residual,
                        norm_out=norm_out,
                        rms_gamma=rms_gamma,
                        rms_eps=rms_eps,
                        allreduce_params=allreduce_params,
                        use_oneshot=use_oneshot,
                    )
                    results[f"flashinfer_fused_allreduce_rmsnorm{suffix}"] = time_ms
                except Exception as e:
                    logger.error("FlashInfer Fused AllReduce+RMSNorm failed: %s", e)
                    results[f"flashinfer_fused_allreduce_rmsnorm{suffix}"] = float(
                        "inf"
                    )

    if "fp8" in quant_modes:
        # Standard AllReduce + RMSNorm + FP8 Quant
        for rms_norm_custom_op in ["-rms_norm", "+rms_norm"]:
            suffix = (
                "_custom_rms_norm" if "+" in rms_norm_custom_op else "_native_rms_norm"
            )
            for quant_fp8_custom_op in ["-quant_fp8", "+quant_fp8"]:
                suffix += (
                    "_custom_quant_fp8"
                    if "+" in quant_fp8_custom_op
                    else "_native_quant_fp8"
                )
                with set_current_vllm_config(
                    VllmConfig(
                        compilation_config=CompilationConfig(
                            custom_ops=[rms_norm_custom_op, quant_fp8_custom_op]
                        )
                    )
                ):
                    try:
                        time_ms = benchmark_operation(
                            vllm_fused_allreduce.allreduce_rmsnorm_fp8_quant,
                            input_tensor,
                            residual=residual,
                            scale_factor=scale_fp8,
                        )
                        results[f"standard_allreduce{suffix}"] = time_ms
                    except Exception as e:
                        logger.error("Standard AllReduce+RMSNorm+FP8 failed: %s", e)
                        results[f"standard_allreduce{suffix}"] = float("inf")

        # Standard AllReduce + RMSNorm + FP8 Quant Native Compiled
        with set_current_vllm_config(
            VllmConfig(
                compilation_config=CompilationConfig(
                    custom_ops=["-rms_norm", "-quant_fp8"]
                )
            )
        ):
            try:
                standard_allreduce_rmsnorm_fp8_quant_native_compiled = torch.compile(
                    vllm_fused_allreduce.allreduce_rmsnorm_fp8_quant,
                    fullgraph=True,
                    dynamic=False,
                )
                time_ms = benchmark_operation(
                    standard_allreduce_rmsnorm_fp8_quant_native_compiled,
                    input_tensor,
                    residual=residual,
                    scale_factor=scale_fp8,
                )
                results["standard_allreduce_rmsnorm_fp8_quant_native_compiled"] = (
                    time_ms
                )
            except Exception as e:
                logger.error(
                    "Standard AllReduce+RMSNorm+FP8 Native Compiled failed: %s", e
                )
                results["standard_allreduce_rmsnorm_fp8_quant_native_compiled"] = float(
                    "inf"
                )

        # FlashInfer Fused AllReduce + RMSNorm + FP8 Quant Oneshot
        if flashinfer_comm is not None and allreduce_params is not None:
            for use_oneshot in use_oneshot_options:
                suffix = "_oneshot" if use_oneshot else "_twoshot"
                try:
                    time_ms = benchmark_operation(
                        flashinfer_fused_allreduce_rmsnorm_fp8_quant,
                        input_tensor,
                        norm_out=norm_out,
                        residual=residual,
                        rms_gamma=rms_gamma,
                        rms_eps=rms_eps,
                        scale_factor=scale_fp8,
                        quant_out=quant_out_fp8,
                        allreduce_params=allreduce_params,
                        use_oneshot=use_oneshot,
                    )
                    results[f"flashinfer_fused_allreduce_rmsnorm_fp8_quant{suffix}"] = (
                        time_ms
                    )
                except Exception as e:
                    logger.error(
                        "FlashInfer Fused AllReduce+RMSNorm+FP8 Oneshot failed: %s",
                        e,
                    )
                    results[f"flashinfer_fused_allreduce_rmsnorm_fp8_quant{suffix}"] = (
                        float("inf")
                    )

    if "fp4" in quant_modes and current_platform.has_device_capability(100):
        # Standard AllReduce + RMSNorm + FP4 Quant
        for rms_norm_custom_op in ["-rms_norm", "+rms_norm"]:
            suffix = (
                "_custom_rms_norm" if "+" in rms_norm_custom_op else "_native_rms_norm"
            )
            with set_current_vllm_config(
                VllmConfig(
                    compilation_config=CompilationConfig(
                        custom_ops=[rms_norm_custom_op]
                    )
                )
            ):
                try:
                    time_ms = benchmark_operation(
                        vllm_fused_allreduce.allreduce_rmsnorm_fp4_quant,
                        input_tensor,
                        residual=residual,
                        input_global_scale=scale_fp4,
                        quant_out=fp4_quant_out,
                        output_scale=fp4_output_scale,
                    )
                    results[f"standard_allreduce_{suffix}_fp4_quant"] = time_ms
                except Exception as e:
                    logger.error("Standard AllReduce+RMSNorm+FP4 failed: %s", e)
                    results[f"standard_allreduce_{suffix}_fp4_quant"] = float("inf")

        # Standard AllReduce + RMSNorm + FP4 Quant Native Compiled
        with set_current_vllm_config(
            VllmConfig(compilation_config=CompilationConfig(custom_ops=["-rms_norm"]))
        ):
            try:
                standard_allreduce_rmsnorm_fp4_quant_native_compiled = torch.compile(
                    vllm_fused_allreduce.allreduce_rmsnorm_fp4_quant,
                    fullgraph=True,
                    dynamic=False,
                )
                time_ms = benchmark_operation(
                    standard_allreduce_rmsnorm_fp4_quant_native_compiled,
                    input_tensor,
                    residual=residual,
                    quant_out=fp4_quant_out,
                    input_global_scale=scale_fp4,
                    output_scale=fp4_output_scale,
                )
                results["standard_allreduce_rmsnorm_fp4_quant_native_compiled"] = (
                    time_ms
                )
            except Exception as e:
                logger.error(
                    "Standard AllReduce+RMSNorm+FP4 Native Compiled failed: %s", e
                )
                results["standard_allreduce_rmsnorm_fp4_quant_native_compiled"] = float(
                    "inf"
                )

        # FlashInfer Fused AllReduce + RMSNorm + FP4 Quant Oneshot
        if flashinfer_comm is not None and allreduce_params is not None:
            for use_oneshot in use_oneshot_options:
                suffix = "_oneshot" if use_oneshot else "_twoshot"
                try:
                    time_ms = benchmark_operation(
                        flashinfer_fused_allreduce_rmsnorm_fp4_quant,
                        input_tensor,
                        residual=residual,
                        norm_out=norm_out,
                        rms_gamma=rms_gamma,
                        rms_eps=rms_eps,
                        input_global_scale=scale_fp4,
                        allreduce_params=allreduce_params,
                        quant_out=fp4_quant_out,
                        output_scale=fp4_output_scale,
                        use_oneshot=use_oneshot,
                    )
                    results[f"flashinfer_fused_allreduce_rmsnorm_fp4_quant{suffix}"] = (
                        time_ms
                    )
                except Exception as e:
                    logger.error(
                        "FlashInfer Fused AllReduce+RMSNorm+FP4 Oneshot failed: %s",
                        e,
                    )
                    results[f"flashinfer_fused_allreduce_rmsnorm_fp4_quant{suffix}"] = (
                        float("inf")
                    )

        # FlashInfer Fused AllReduce + RMSNorm + FP4 Quant Two-shot
        if flashinfer_comm is not None and allreduce_params is not None:
            try:
                time_ms = benchmark_operation(
                    flashinfer_fused_allreduce_rmsnorm_fp4_quant,
                    input_tensor,
                    residual=residual,
                    norm_out=norm_out,
                    rms_gamma=rms_gamma,
                    rms_eps=rms_eps,
                    input_global_scale=scale_fp4,
                    allreduce_params=allreduce_params,
                    quant_out=fp4_quant_out,
                    output_scale=fp4_output_scale,
                    use_oneshot=False,
                )
                results["flashinfer_fused_allreduce_rmsnorm_fp4_quant_twoshot"] = (
                    time_ms
                )
            except Exception as e:
                logger.error(
                    "FlashInfer Fused AllReduce+RMSNorm+FP4 Two-shot failed: %s",
                    e,
                )
                results["flashinfer_fused_allreduce_rmsnorm_fp4_quant_twoshot"] = float(
                    "inf"
                )

    return results


def prepare_results_with_speedups(results_dict):
    """Prepare results with speedup calculations based on dynamic baseline selection."""
    prepared_results = []

    # Determine the fastest baseline for each operation type
    def get_fastest_baseline(op_name, results_dict):
        """Get the fastest baseline between standard and native_compiled versions."""
        if "fp8_quant" in op_name:
            candidates = [
                "standard_allreduce_rmsnorm_fp8_quant",
                "standard_allreduce_rmsnorm_fp8_quant_native_compiled",
            ]
        elif "fp4_quant" in op_name:
            candidates = [
                "standard_allreduce_rmsnorm_fp4_quant",
                "standard_allreduce_rmsnorm_fp4_quant_native_compiled",
            ]
        else:
            candidates = [
                "standard_allreduce_rmsnorm",
                "standard_allreduce_rmsnorm_native_compiled",
            ]

        # Find the fastest among available candidates
        fastest_time = float("inf")
        fastest_baseline = None

        for candidate in candidates:
            if (
                candidate in results_dict
                and results_dict[candidate] != float("inf")
                and results_dict[candidate] < fastest_time
            ):
                fastest_time = results_dict[candidate]
                fastest_baseline = candidate

        return fastest_baseline

    # Create dynamic baseline mapping
    dynamic_baseline_mapping = {}
    for op_name in results_dict:
        if (
            op_name.startswith("flashinfer_")
            or op_name.startswith("standard_")
            and not op_name.endswith("_native_compiled")
        ):
            dynamic_baseline_mapping[op_name] = get_fastest_baseline(
                op_name, results_dict
            )

    for op_name, time_ms in results_dict.items():
        if time_ms == float("inf"):
            speedup_str = "FAILED"
            time_str = "FAILED"
        else:
            time_str = f"{time_ms:.3f}"
            # Find the appropriate baseline for this operation
            baseline_op = dynamic_baseline_mapping.get(op_name)
            if baseline_op and baseline_op in results_dict:
                baseline_time = results_dict[baseline_op]
                if baseline_time != float("inf") and baseline_time > 0:
                    speedup = baseline_time / time_ms
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = "N/A"
            else:
                # For baseline operations, determine if this is the fastest baseline
                if op_name.endswith("_native_compiled") or (
                    op_name.startswith("standard_")
                    and not op_name.endswith("_native_compiled")
                ):
                    fastest_baseline = get_fastest_baseline(op_name, results_dict)
                    if fastest_baseline == op_name:
                        speedup_str = "baseline"
                    else:
                        if fastest_baseline and fastest_baseline in results_dict:
                            baseline_time = results_dict[fastest_baseline]
                            if baseline_time != float("inf") and baseline_time > 0:
                                speedup = baseline_time / time_ms
                                speedup_str = f"{speedup:.2f}x"
                            else:
                                speedup_str = "N/A"
                        else:
                            speedup_str = "N/A"
                else:
                    speedup_str = "N/A"

        prepared_results.append(
            {
                "operation": op_name,
                "time_ms": time_ms,
                "time_str": time_str,
                "speedup_str": speedup_str,
            }
        )

    return prepared_results


def print_results(
    results_dict,
    num_tokens,
    hidden_dim,
    dtype,
    use_residual,
    quant_modes,
    input_size_mb,
):
    """Print benchmark results in a formatted table."""
    print(f"\n{'=' * 80}")
    print(
        f"Results: num_tokens={num_tokens}, hidden_dim={hidden_dim} "
        f"(input size: {input_size_mb:.2f} MB)"
    )
    print(
        f"dtype={dtype}, residual={'yes' if use_residual else 'no'}, "
        f"quant_modes={','.join(sorted(list(quant_modes)))}"
    )
    print(f"{'=' * 80}")
    print(f"{'Operation':<50} {'Time (ms)':<12} {'Speedup':<10}")
    print(f"{'-' * 80}")

    # Prepare results with speedup calculations
    prepared_results = prepare_results_with_speedups(results_dict)

    for result in prepared_results:
        if result["time_ms"] == float("inf"):
            time_display = result["time_str"]
        else:
            time_display = f"{result['time_ms']:.3f}"

        print(
            f"{result['operation']:<50} {time_display:<12} {result['speedup_str']:<10}"
        )


def format_results_markdown(
    all_results: list[dict], world_size: int, args: argparse.Namespace
) -> str:
    """Format all benchmark results as markdown."""
    lines: list[str] = []
    lines.append("# FlashInfer Fused Collective Operations Benchmark Results")
    lines.append("")
    lines.append(f"**World Size:** {world_size}  ")
    lines.append(f"**Hidden Dimension:** {args.hidden_dim}  ")
    lines.append(f"**Warmup Iterations:** {args.warmup}  ")
    lines.append(f"**Benchmark Trials:** {args.trials}  ")
    modes = ",".join(all_results[0]["quant_modes"]) if all_results else "N/A"
    lines.append(f"**Quantization Modes:** {modes}  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    for entry in all_results:
        num_tokens = entry["num_tokens"]
        dtype = entry["dtype"]
        use_residual = entry["use_residual"]
        results_dict = entry["results"]
        input_size_mb = entry["input_size_mb"]
        residual_str = "with residual" if use_residual else "no residual"

        lines.append(
            f"## Configuration: num_tokens={num_tokens}, dtype={dtype}, {residual_str}"
        )
        lines.append(f"**Input Size:** {input_size_mb:.2f} MB")
        lines.append("")

        prepared = prepare_results_with_speedups(results_dict)
        # Build DataFrame for markdown export
        rows = [
            {
                "Operation": r["operation"].replace("_", " ").title(),
                "Time (ms)": r["time_str"],
                "Speedup": r["speedup_str"],
            }
            for r in prepared
        ]
        df = pd.DataFrame(rows)
        if df.empty:
            lines.append("No results.")
        else:
            lines.append(df.to_markdown(index=False))
        lines.append("")

    return "\n".join(lines)


def save_results_to_file(
    all_results: list[dict], world_size: int, args: argparse.Namespace, rank: int
):
    """Save benchmark results to markdown file (only on rank 0)."""
    if rank != 0:
        return

    if not all_results:
        logger.warning("No results to save")
        return

    output_path = args.output_file

    try:
        markdown_content = format_results_markdown(all_results, world_size, args)

        with open(output_path, "a") as f:
            f.write(markdown_content)

    except Exception as e:
        logger.error("Failed to save results to file: %s", e)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fused collective operations"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 2048],
        help="Numbers of tokens to test",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=8192, help="Hidden dimension size"
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        default=["bfloat16"],
        choices=["float16", "bfloat16", "float32"],
        help="Data types to test",
    )
    parser.add_argument(
        "--no-residual",
        action="store_true",
        help="Skip residual connection tests",
    )

    parser.add_argument(
        "--quant-modes",
        type=str,
        default="none,fp8,fp4",
        help=(
            "Comma-separated quantization modes to run: none, fp8, fp4. "
            "Default: none,fp8,fp4"
        ),
    )

    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="Number of benchmark trials"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="""Output file path for markdown results 
                (default: benchmark_results_<timestamp>.md)
        """,
    )

    parser.add_argument(
        "--no-oneshot",
        action="store_true",
        help="Skip oneshot benchmarks",
    )

    args = parser.parse_args()

    # Check if running with torchrun (required for collective operations)
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Must run with torchrun for distributed benchmarking. "
            "Example: torchrun --nproc_per_node=2 benchmark_fused_collective.py"
        )

    # Initialize distributed environment
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # Validate world size (must be > 1 for collective operations)
    if world_size <= 1:
        raise ValueError(
            "World size must be > 1 for collective operations benchmarking. "
            f"Current world size: {world_size}. Use torchrun with --nproc_per_node > 1."
        )

    # Parse quantization modes
    valid_quant_modes = {"none", "fp8", "fp4"}
    raw_modes = [
        m.strip().lower() for m in (args.quant_modes or "").split(",") if m.strip()
    ]
    quant_modes = set(raw_modes) if raw_modes else {"none", "fp8", "fp4"}
    invalid = sorted(list(quant_modes - valid_quant_modes))
    if invalid:
        raise ValueError(
            f"Invalid --quant-modes entries: {','.join(invalid)}. "
            f"Valid options are: {','.join(sorted(valid_quant_modes))}."
        )

    if rank == 0:
        logger.info("Running benchmark with world_size=%s, rank=%s", world_size, rank)
        logger.info("Quantization modes: %s", ",".join(sorted(list(quant_modes))))
        if flashinfer_comm is not None:
            logger.info(
                "FlashInfer available - will benchmark fused operations",
            )
        else:
            logger.info(
                "FlashInfer not available - only benchmarking standard operations"
            )

    # Convert dtype strings to torch dtypes
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtypes = [dtype_map[dt] for dt in args.dtypes]

    # Test configurations
    residual_options = [True] if not args.no_residual else [False]

    configs = list(itertools.product(args.num_tokens, dtypes, residual_options))

    # Setup FlashInfer workspace if available
    workspace = None
    allreduce_params = None

    if flashinfer_comm is not None:
        # Use the largest hidden dimension for workspace setup
        max_element_size = max(torch.finfo(dt).bits // 8 for dt in dtypes)
        workspace_dtype = (
            torch.float32
            if max_element_size == 4
            else (torch.bfloat16 if torch.bfloat16 in dtypes else torch.float16)
        )
        max_num_token = _FI_MAX_SIZES.get(world_size) // (
            args.hidden_dim * max_element_size
        )

        workspace = setup_flashinfer_workspace(
            world_size,
            rank,
            args.hidden_dim,
            max_num_token,
            dtype=workspace_dtype,
        )

        if workspace is not None:
            allreduce_params = FlashInferFusedAllReduceParams(
                max_token_num=max_num_token,
            )

    # Collect all results for markdown export
    all_results = []

    try:
        # Run benchmarks
        for num_tokens, dtype, use_residual in configs:
            if rank == 0:
                logger.info(
                    "\nTesting:  num_tokens=%s, hidden_dim=%s, dtype=%s, residual=%s",
                    num_tokens,
                    args.hidden_dim,
                    dtype,
                    use_residual,
                )

            results = run_benchmarks(
                num_tokens,
                args.hidden_dim,
                dtype,
                use_residual,
                allreduce_params,
                quant_modes=quant_modes,
                no_oneshot=args.no_oneshot,
            )

            # Store results for markdown export
            if rank == 0:
                # Calculate input size in MB
                input_size_mb = (
                    num_tokens * args.hidden_dim * torch.finfo(dtype).bits
                ) / (8 * 1024 * 1024)
                all_results.append(
                    {
                        "num_tokens": num_tokens,
                        "hidden_dim": args.hidden_dim,
                        "dtype": str(dtype).replace("torch.", ""),
                        "use_residual": use_residual,
                        "quant_modes": sorted(list(quant_modes)),
                        "input_size_mb": input_size_mb,
                        "results": results,
                    }
                )

                print_results(
                    results,
                    num_tokens,
                    args.hidden_dim,
                    dtype,
                    use_residual,
                    quant_modes,
                    input_size_mb,
                )

        # Save results to markdown file
        if args.output_file and rank == 0:
            save_results_to_file(all_results, world_size, args, rank)

    finally:
        # Cleanup
        if workspace is not None:
            cleanup_flashinfer_workspace(workspace)

        dist.barrier()


if __name__ == "__main__":
    main()
