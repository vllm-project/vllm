# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark for FlashInfer fused collective operations vs standard operations.

This benchmark compares:
1. FlashInfer's trtllm_allreduce_fusion (fused allreduce + rmsnorm + optional quant)
2. Standard tensor_model_parallel_all_reduce + separate rmsnorm/quant operations

Usage with torchrun:
    torchrun --nproc_per_node=2 benchmark_fused_collective.py

"""

import argparse
import itertools
import os
import time

import torch  # type: ignore
import torch.distributed as dist  # type: ignore

from vllm.distributed import (
    get_tp_group,
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

    if not hasattr(flashinfer_comm, "trtllm_allreduce_fusion"):
        flashinfer_comm = None
        logger.warning(
            "FlashInfer comm module found but missing trtllm_allreduce_fusion"
        )
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
_FI_WORKSPACE_TENSOR = None


def setup_flashinfer_workspace(
    world_size: int,
    rank: int,
    hidden_dim: int,
    max_token_num: int,
    use_fp32_lamport: bool = False,
):
    """Setup FlashInfer workspace for fused allreduce operations."""
    global _FI_WORKSPACE_TENSOR

    if flashinfer_comm is None:
        return None, None

    if world_size not in _FI_MAX_SIZES:
        logger.warning("FlashInfer not supported for world size %s", world_size)
        return None, None

    try:
        # Create IPC workspace
        ipc_handles, workspace_tensor = (
            flashinfer_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                tp_rank=rank,
                tp_size=world_size,
                max_token_num=max_token_num,
                hidden_dim=hidden_dim,
                group=get_tp_group().device_group,
                use_fp32_lamport=use_fp32_lamport,
            )
        )

        _FI_WORKSPACE_TENSOR = workspace_tensor
        return ipc_handles, workspace_tensor
    except Exception as e:
        logger.error("Failed to setup FlashInfer workspace: %s", e)
        return None, None


def cleanup_flashinfer_workspace(ipc_handles):
    """Cleanup FlashInfer workspace."""
    if flashinfer_comm is None or ipc_handles is None:
        return

    try:
        group = get_tp_group().device_group
        flashinfer_comm.trtllm_destroy_ipc_workspace_for_all_reduce(ipc_handles, group)
    except Exception as e:
        logger.error("Failed to cleanup FlashInfer workspace: %s", e)


class FlashInferFusedAllReduceParams:
    """Parameters for FlashInfer fused allreduce operations."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        use_fp32_lamport: bool = False,
        max_token_num: int = 1024,
    ):
        self.rank = rank
        self.world_size = world_size
        self.use_fp32_lamport = use_fp32_lamport
        self.trigger_completion_at_end = True
        self.launch_with_pdl = True
        self.fp32_acc = True
        self.max_token_num = max_token_num

    def get_trtllm_fused_allreduce_kwargs(self):
        return {
            "world_rank": self.rank,
            "world_size": self.world_size,
            "launch_with_pdl": self.launch_with_pdl,
            "trigger_completion_at_end": self.trigger_completion_at_end,
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
    if flashinfer_comm is None or _FI_WORKSPACE_TENSOR is None:
        raise RuntimeError("FlashInfer not available or workspace not initialized")

    if norm_out is None:
        norm_out = input_tensor
        residual_out = residual
    else:
        residual_out = input_tensor

    flashinfer_comm.trtllm_allreduce_fusion(
        allreduce_in=input_tensor,
        token_num=input_tensor.shape[0],
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        hidden_dim=input_tensor.shape[-1],
        workspace_ptrs=_FI_WORKSPACE_TENSOR,
        pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
        allreduce_out=None,
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
    if flashinfer_comm is None or _FI_WORKSPACE_TENSOR is None:
        raise RuntimeError("FlashInfer not available or workspace not initialized")

    if norm_out is None:
        norm_out = input_tensor
        residual_out = residual
    else:
        residual_out = input_tensor

    flashinfer_comm.trtllm_allreduce_fusion(
        allreduce_in=input_tensor,
        token_num=input_tensor.shape[0],
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        hidden_dim=input_tensor.shape[-1],
        workspace_ptrs=_FI_WORKSPACE_TENSOR,
        pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
        allreduce_out=None,
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
    if flashinfer_comm is None or _FI_WORKSPACE_TENSOR is None:
        raise RuntimeError("FlashInfer not available or workspace not initialized")

    if norm_out is None:
        norm_out = input_tensor
        residual_out = residual
    else:
        residual_out = input_tensor

    flashinfer_comm.trtllm_allreduce_fusion(
        allreduce_in=input_tensor,
        token_num=input_tensor.shape[0],
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        hidden_dim=input_tensor.shape[-1],
        workspace_ptrs=_FI_WORKSPACE_TENSOR,
        pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
        allreduce_out=None,
        quant_out=quant_out,
        scale_out=output_scale,
        layout_code=flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4,
        scale_factor=input_global_scale,
        use_oneshot=use_oneshot,
        **allreduce_params.get_trtllm_fused_allreduce_kwargs(),
    )


def standard_allreduce_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor | None,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    norm_out: torch.Tensor | None = None,
):
    """Standard allreduce + rmsnorm operations."""
    # All-reduce first
    allreduce_out = tensor_model_parallel_all_reduce(input_tensor)
    # Then RMS norm
    if residual is not None:
        # Fused add + RMS norm
        FUSED_ADD_RMS_NORM_OP(allreduce_out, residual, rms_gamma, rms_eps)
    else:
        # Just RMS norm
        if norm_out is None:
            norm_out = torch.empty_like(allreduce_out)
        RMS_NORM_OP(norm_out, allreduce_out, rms_gamma, rms_eps)


def standard_allreduce_rmsnorm_fp8_quant(
    input_tensor: torch.Tensor,
    residual: torch.Tensor | None,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    scale_factor: torch.Tensor,
    norm_out: torch.Tensor | None = None,
    quant_out: torch.Tensor | None = None,
):
    """Standard allreduce + rmsnorm + FP8 quantization."""
    if quant_out is None:
        quant_out = torch.empty_like(input_tensor, dtype=FP8_DTYPE)

    # All-reduce first
    allreduce_out = tensor_model_parallel_all_reduce(input_tensor)

    # Then fused RMS norm + FP8 quantization
    if residual is not None:
        FUSED_ADD_RMS_NORM_STATIC_FP8_QUANT_OP(
            quant_out, allreduce_out, residual, rms_gamma, scale_factor, rms_eps
        )
        return quant_out, residual
    else:
        RMS_NORM_STATIC_FP8_QUANT_OP(
            quant_out, allreduce_out, rms_gamma, scale_factor, rms_eps
        )
        return quant_out


def standard_allreduce_rmsnorm_fp4_quant(
    input_tensor: torch.Tensor,
    residual: torch.Tensor | None,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    input_global_scale: torch.Tensor,
    quant_out: torch.Tensor,
    output_scale: torch.Tensor,
    norm_out: torch.Tensor | None = None,
):
    """Standard allreduce + rmsnorm + FP4 quantization."""

    # All-reduce first
    allreduce_out = tensor_model_parallel_all_reduce(input_tensor)

    # Then RMS norm
    if residual is not None:
        FUSED_ADD_RMS_NORM_OP(allreduce_out, residual, rms_gamma, rms_eps)
        quant_input = allreduce_out
        residual_out = residual
    else:
        if norm_out is None:
            norm_out = torch.empty_like(allreduce_out)
        RMS_NORM_OP(norm_out, allreduce_out, rms_gamma, rms_eps)
        quant_input = norm_out
        residual_out = allreduce_out

    # Finally FP4 quantization
    SCALED_FP4_QUANT_OP(quant_out, quant_input, output_scale, input_global_scale)
    if residual is not None:
        return quant_out, residual_out, output_scale
    else:
        return quant_out, norm_out


def standard_allreduce_rmsnorm_native(
    input_tensor: torch.Tensor,
    residual: torch.Tensor | None,
    rmsnorm_layer: RMSNorm,
    norm_out: torch.Tensor | None = None,
):
    """Standard allreduce + rmsnorm operations using native RMSNorm forward."""
    # All-reduce first
    allreduce_out = tensor_model_parallel_all_reduce(input_tensor)
    # Apply native RMSNorm
    if residual is not None:
        result = rmsnorm_layer.forward_native(allreduce_out, residual)
        return result  # Returns (norm_out, residual_out)
    else:
        result = rmsnorm_layer.forward_native(allreduce_out)
        return result  # Returns norm_out


def standard_allreduce_rmsnorm_fp8_quant_native(
    input_tensor: torch.Tensor,
    residual: torch.Tensor | None,
    rmsnorm_layer: RMSNorm,
    quant_fp8_layer: QuantFP8,
    scale_factor: torch.Tensor,
    norm_out: torch.Tensor | None = None,
    quant_out: torch.Tensor | None = None,
):
    """Standard allreduce + rmsnorm + FP8 quantization using native implementations."""
    # All-reduce first
    allreduce_out = tensor_model_parallel_all_reduce(input_tensor)

    # Apply native RMSNorm
    if residual is not None:
        norm_out, residual_out = rmsnorm_layer.forward_native(allreduce_out, residual)
    else:
        norm_out = rmsnorm_layer.forward_native(allreduce_out)
        residual_out = allreduce_out

    # Apply native FP8 quantization
    quant_out, _ = quant_fp8_layer.forward_native(norm_out, scale=scale_factor)

    if residual is not None:
        return quant_out, residual_out
    else:
        return quant_out


def standard_allreduce_rmsnorm_fp4_quant_native(
    input_tensor: torch.Tensor,
    residual: torch.Tensor | None,
    rmsnorm_layer: RMSNorm,
    input_global_scale: torch.Tensor,
    quant_out: torch.Tensor,
    output_scale: torch.Tensor,
    norm_out: torch.Tensor | None = None,
):
    """Standard allreduce + rmsnorm + FP4 quantization using native RMSNorm."""
    # All-reduce first
    allreduce_out = tensor_model_parallel_all_reduce(input_tensor)

    # Apply native RMSNorm
    if residual is not None:
        norm_out, residual_out = rmsnorm_layer.forward_native(allreduce_out, residual)
        quant_input = norm_out
    else:
        norm_out = rmsnorm_layer.forward_native(allreduce_out)
        quant_input = norm_out
        residual_out = allreduce_out

    # Apply FP4 quantization (still using fused CUDA op as there's no native FP4)
    SCALED_FP4_QUANT_OP(quant_out, quant_input, output_scale, input_global_scale)

    if residual is not None:
        return quant_out, residual_out, output_scale
    else:
        return quant_out, norm_out


# Compiled versions of native functions
@torch.compile
def standard_allreduce_rmsnorm_native_compiled(
    input_tensor: torch.Tensor,
    residual: torch.Tensor | None,
    rmsnorm_layer: RMSNorm,
    norm_out: torch.Tensor | None = None,
):
    """Compiled version of standard allreduce + rmsnorm."""
    return standard_allreduce_rmsnorm_native(
        input_tensor, residual, rmsnorm_layer, norm_out
    )


@torch.compile
def standard_allreduce_rmsnorm_fp8_quant_native_compiled(
    input_tensor: torch.Tensor,
    residual: torch.Tensor | None,
    rmsnorm_layer: RMSNorm,
    quant_fp8_layer: QuantFP8,
    scale_factor: torch.Tensor,
    norm_out: torch.Tensor | None = None,
    quant_out: torch.Tensor | None = None,
):
    """Compiled version of standard allreduce + rmsnorm + FP8 quantization."""
    return standard_allreduce_rmsnorm_fp8_quant_native(
        input_tensor,
        residual,
        rmsnorm_layer,
        quant_fp8_layer,
        scale_factor,
        norm_out,
        quant_out,
    )


@torch.compile
def standard_allreduce_rmsnorm_fp4_quant_native_compiled(
    input_tensor: torch.Tensor,
    residual: torch.Tensor | None,
    rmsnorm_layer: RMSNorm,
    input_global_scale: torch.Tensor,
    quant_out: torch.Tensor,
    output_scale: torch.Tensor,
    norm_out: torch.Tensor | None = None,
):
    """Compiled version of standard allreduce + rmsnorm + FP4 quantization."""
    return standard_allreduce_rmsnorm_fp4_quant_native(
        input_tensor,
        residual,
        rmsnorm_layer,
        input_global_scale,
        quant_out,
        output_scale,
        norm_out,
    )


def create_test_tensors(
    seq_len: int, hidden_dim: int, dtype: torch.dtype, use_residual: bool = True
):
    """Create test tensors for benchmarking."""
    input_tensor = torch.randn(seq_len, hidden_dim, dtype=dtype)
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
    fp4_quant_out = torch.empty((seq_len, hidden_dim // 2), dtype=torch.uint8)
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
    seq_len: int,
    hidden_dim: int,
    dtype: torch.dtype,
    use_residual: bool,
    allreduce_params: FlashInferFusedAllReduceParams | None,
    quant_mode: str = "all",
    disable_oneshot: bool = False,
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
    ) = create_test_tensors(seq_len, hidden_dim, dtype, use_residual)

    rms_eps = 1e-6
    results = {}

    # Create RMSNorm and QuantFP8 layers once for native benchmarks
    rmsnorm_layer = RMSNorm(hidden_dim, eps=rms_eps, dtype=dtype)
    rmsnorm_layer.weight.data = rms_gamma
    quant_fp8_layer = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)

    if quant_mode in ["all", "none"]:
        # Standard AllReduce + RMSNorm
        try:
            time_ms = benchmark_operation(
                standard_allreduce_rmsnorm,
                input_tensor,
                norm_out=norm_out,
                residual=residual,
                rms_gamma=rms_gamma,
                rms_eps=rms_eps,
            )
            results["standard_allreduce_rmsnorm"] = time_ms
        except Exception as e:
            logger.error("Standard AllReduce+RMSNorm failed: %s", e)
            results["standard_allreduce_rmsnorm"] = float("inf")

        # Standard AllReduce + RMSNorm Native Compiled
        try:
            time_ms = benchmark_operation(
                standard_allreduce_rmsnorm_native_compiled,
                input_tensor,
                residual=residual,
                rmsnorm_layer=rmsnorm_layer,
                norm_out=norm_out,
            )
            results["standard_allreduce_rmsnorm_native_compiled"] = time_ms
        except Exception as e:
            logger.error("Standard AllReduce+RMSNorm Native Compiled failed: %s", e)
            results["standard_allreduce_rmsnorm_native_compiled"] = float("inf")

        # FlashInfer Fused AllReduce + RMSNorm Oneshot
        if flashinfer_comm is not None and allreduce_params is not None:
            try:
                if not disable_oneshot:
                    time_ms = benchmark_operation(
                        flashinfer_fused_allreduce_rmsnorm,
                        input_tensor,
                        residual=residual,
                        norm_out=norm_out,
                        rms_gamma=rms_gamma,
                        rms_eps=rms_eps,
                        allreduce_params=allreduce_params,
                        use_oneshot=True,
                    )
                    results["flashinfer_fused_allreduce_rmsnorm_oneshot"] = time_ms
            except Exception as e:
                logger.error("FlashInfer Fused AllReduce+RMSNorm Oneshot failed: %s", e)
                results["flashinfer_fused_allreduce_rmsnorm_oneshot"] = float("inf")

            # FlashInfer Fused AllReduce + RMSNorm Two-shot
            try:
                time_ms = benchmark_operation(
                    flashinfer_fused_allreduce_rmsnorm,
                    input_tensor,
                    residual=residual,
                    norm_out=norm_out,
                    rms_gamma=rms_gamma,
                    rms_eps=rms_eps,
                    allreduce_params=allreduce_params,
                    use_oneshot=False,
                )
                results["flashinfer_fused_allreduce_rmsnorm_twoshot"] = time_ms
            except Exception as e:
                logger.error(
                    "FlashInfer Fused AllReduce+RMSNorm Two-shot failed: %s", e
                )
                results["flashinfer_fused_allreduce_rmsnorm_twoshot"] = float("inf")

    if quant_mode in ["all", "fp8_only"]:
        # Standard AllReduce + RMSNorm + FP8 Quant
        try:
            time_ms = benchmark_operation(
                standard_allreduce_rmsnorm_fp8_quant,
                input_tensor,
                norm_out=norm_out,
                residual=residual,
                rms_gamma=rms_gamma,
                rms_eps=rms_eps,
                scale_factor=scale_fp8,
                quant_out=quant_out_fp8,
            )
            results["standard_allreduce_rmsnorm_fp8_quant"] = time_ms
        except Exception as e:
            logger.error("Standard AllReduce+RMSNorm+FP8 failed: %s", e)
            results["standard_allreduce_rmsnorm_fp8_quant"] = float("inf")

        # Standard AllReduce + RMSNorm + FP8 Quant Native Compiled
        try:
            time_ms = benchmark_operation(
                standard_allreduce_rmsnorm_fp8_quant_native_compiled,
                input_tensor,
                residual=residual,
                rmsnorm_layer=rmsnorm_layer,
                quant_fp8_layer=quant_fp8_layer,
                scale_factor=scale_fp8,
                norm_out=norm_out,
                quant_out=quant_out_fp8,
            )
            results["standard_allreduce_rmsnorm_fp8_quant_native_compiled"] = time_ms
        except Exception as e:
            logger.error("Standard AllReduce+RMSNorm+FP8 Native Compiled failed: %s", e)
            results["standard_allreduce_rmsnorm_fp8_quant_native_compiled"] = float(
                "inf"
            )

        # FlashInfer Fused AllReduce + RMSNorm + FP8 Quant Oneshot
        if flashinfer_comm is not None and allreduce_params is not None:
            try:
                if not disable_oneshot:
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
                        use_oneshot=True,
                    )
                    results["flashinfer_fused_allreduce_rmsnorm_fp8_quant_oneshot"] = (
                        time_ms
                    )
            except Exception as e:
                logger.error(
                    "FlashInfer Fused AllReduce+RMSNorm+FP8 Oneshot failed: %s",
                    e,
                )
                results["flashinfer_fused_allreduce_rmsnorm_fp8_quant_oneshot"] = float(
                    "inf"
                )
            # FlashInfer Fused AllReduce + RMSNorm + FP8 Quant Two-shot
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
                    use_oneshot=False,
                )
                results["flashinfer_fused_allreduce_rmsnorm_fp8_quant_twoshot"] = (
                    time_ms
                )
            except Exception as e:
                logger.error(
                    "FlashInfer Fused AllReduce+RMSNorm+FP8 Two-shot failed: %s",
                    e,
                )
                results["flashinfer_fused_allreduce_rmsnorm_fp8_quant_twoshot"] = float(
                    "inf"
                )

    if quant_mode in ["all", "fp4_only"]:
        # Standard AllReduce + RMSNorm + FP4 Quant
        try:
            time_ms = benchmark_operation(
                standard_allreduce_rmsnorm_fp4_quant,
                input_tensor,
                norm_out=norm_out,
                residual=residual,
                rms_gamma=rms_gamma,
                rms_eps=rms_eps,
                input_global_scale=scale_fp4,
                quant_out=fp4_quant_out,
                output_scale=fp4_output_scale,
            )
            results["standard_allreduce_rmsnorm_fp4_quant"] = time_ms
        except Exception as e:
            logger.error("Standard AllReduce+RMSNorm+FP4 failed: %s", e)
            results["standard_allreduce_rmsnorm_fp4_quant"] = float("inf")

        # Standard AllReduce + RMSNorm + FP4 Quant Native Compiled
        try:
            time_ms = benchmark_operation(
                standard_allreduce_rmsnorm_fp4_quant_native_compiled,
                input_tensor,
                residual=residual,
                rmsnorm_layer=rmsnorm_layer,
                input_global_scale=scale_fp4,
                quant_out=fp4_quant_out,
                output_scale=fp4_output_scale,
                norm_out=norm_out,
            )
            results["standard_allreduce_rmsnorm_fp4_quant_native_compiled"] = time_ms
        except Exception as e:
            logger.error("Standard AllReduce+RMSNorm+FP4 Native Compiled failed: %s", e)
            results["standard_allreduce_rmsnorm_fp4_quant_native_compiled"] = float(
                "inf"
            )

        # FlashInfer Fused AllReduce + RMSNorm + FP4 Quant Oneshot
        if flashinfer_comm is not None and allreduce_params is not None:
            try:
                if not disable_oneshot:
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
                        use_oneshot=True,
                    )
                    results["flashinfer_fused_allreduce_rmsnorm_fp4_quant_oneshot"] = (
                        time_ms
                    )
            except Exception as e:
                logger.error(
                    "FlashInfer Fused AllReduce+RMSNorm+FP4 Oneshot failed: %s",
                    e,
                )
                results["flashinfer_fused_allreduce_rmsnorm_fp4_quant_oneshot"] = float(
                    "inf"
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
    results_dict, seq_len, hidden_dim, dtype, use_residual, quant_mode, input_size_mb
):
    """Print benchmark results in a formatted table."""
    print(f"\n{'=' * 80}")
    print(
        f"Results: seq_len={seq_len}, hidden_dim={hidden_dim} "
        f"(input size: {input_size_mb:.2f} MB)"
    )
    print(
        f"dtype={dtype}, residual={'yes' if use_residual else 'no'}, "
        f"quant_mode={quant_mode}"
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
    markdown = f"""# FlashInfer Fused Collective Operations Benchmark Results

**World Size:** {world_size}  
**Hidden Dimension:** {args.hidden_dim}  
**Warmup Iterations:** {args.warmup}  
**Benchmark Trials:** {args.trials}  
**Quantization Mode:** {all_results[0]["quant_mode"] if all_results else "N/A"}  

---

"""

    for result in all_results:
        seq_len = result["seq_len"]
        dtype = result["dtype"]
        use_residual = result["use_residual"]
        results_dict = result["results"]
        input_size_mb = result["input_size_mb"]
        residual_str = "with residual" if use_residual else "no residual"

        markdown += f"""
## Configuration: seq_len={seq_len}, dtype={dtype}, {residual_str}
**Input Size:** {input_size_mb:.2f} MB

| Operation | Time (ms) | Speedup |
|-----------|-----------|---------|
"""

        # Prepare results with speedup calculations
        prepared_results = prepare_results_with_speedups(results_dict)

        for result in prepared_results:
            # Format operation name for better readability
            formatted_op_name = result["operation"].replace("_", " ").title()
            markdown += f"| {formatted_op_name} | {result['time_str']} |"
            markdown += f"{result['speedup_str']} |\n"

        markdown += "\n"

    return markdown


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

        with open(output_path, "w") as f:
            f.write(markdown_content)

    except Exception as e:
        logger.error("Failed to save results to file: %s", e)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fused collective operations"
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 2048],
        help="Sequence lengths to test",
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

    # Quantization mode options (mutually exclusive with --no-quant)
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument(
        "--no-quant", action="store_true", help="Skip all quantization tests"
    )
    quant_group.add_argument(
        "--quant-fp8", action="store_true", help="Only run FP8 quantization tests"
    )
    quant_group.add_argument(
        "--quant-fp4", action="store_true", help="Only run FP4 quantization tests"
    )
    quant_group.add_argument(
        "--quant-all",
        action="store_true",
        help="Run all quantization tests (default)",
    )

    parser.add_argument(
        "--disable-oneshot",
        action="store_true",
        help="Disable oneshot mode for FlashInfer operations",
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

    # Determine quantization mode
    if args.no_quant:
        quant_mode = "none"
    elif args.quant_fp8:
        quant_mode = "fp8_only"
    elif args.quant_fp4:
        quant_mode = "fp4_only"
    else:  # args.quant_all or default
        quant_mode = "all"

    if rank == 0:
        logger.info("Running benchmark with world_size=%s, rank=%s", world_size, rank)
        logger.info("Quantization mode: %s", quant_mode)
        if flashinfer_comm is not None:
            oneshot_status = "enabled" if not args.disable_oneshot else "disabled"
            logger.info(
                "FlashInfer available - will benchmark fused operations (oneshot: %s)",
                oneshot_status,
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
    if not args.no_residual:
        residual_options.append(False)

    configs = list(itertools.product(args.seq_lens, dtypes, residual_options))

    # Setup FlashInfer workspace if available
    ipc_handles = None
    allreduce_params = None

    if flashinfer_comm is not None:
        # Use the largest hidden dimension for workspace setup
        max_num_token = _FI_MAX_SIZES.get(world_size) // (
            args.hidden_dim * world_size * 2
        )

        ipc_handles, workspace_tensor = setup_flashinfer_workspace(
            world_size, rank, args.hidden_dim, max_num_token
        )

        if workspace_tensor is not None:
            allreduce_params = FlashInferFusedAllReduceParams(
                rank=rank,
                world_size=world_size,
                max_token_num=max_num_token,
            )

    # Collect all results for markdown export
    all_results = []

    try:
        # Run benchmarks
        for seq_len, dtype, use_residual in configs:
            if rank == 0:
                logger.info(
                    "\nTesting:  seq_len=%s, hidden_dim=%s, dtype=%s, residual=%s",
                    seq_len,
                    args.hidden_dim,
                    dtype,
                    use_residual,
                )

            results = run_benchmarks(
                seq_len,
                args.hidden_dim,
                dtype,
                use_residual,
                allreduce_params,
                quant_mode=quant_mode,
                disable_oneshot=args.disable_oneshot,
            )

            # Store results for markdown export
            if rank == 0:
                # Calculate input size in MB
                input_size_mb = (
                    seq_len * args.hidden_dim * torch.finfo(dtype).bits
                ) / (8 * 1024 * 1024)
                all_results.append(
                    {
                        "seq_len": seq_len,
                        "hidden_dim": args.hidden_dim,
                        "dtype": str(dtype).replace("torch.", ""),
                        "use_residual": use_residual,
                        "quant_mode": quant_mode,
                        "input_size_mb": input_size_mb,
                        "results": results,
                    }
                )

                print_results(
                    results,
                    seq_len,
                    args.hidden_dim,
                    dtype,
                    use_residual,
                    quant_mode,
                    input_size_mb,
                )

        # Save results to markdown file
        if args.output_file and rank == 0:
            save_results_to_file(all_results, world_size, args, rank)

    finally:
        # Cleanup
        if ipc_handles is not None:
            cleanup_flashinfer_workspace(ipc_handles)

        dist.barrier()


if __name__ == "__main__":
    main()
