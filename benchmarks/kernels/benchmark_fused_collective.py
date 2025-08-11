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
from typing import Optional

import torch
import torch.distributed as dist

from vllm.distributed import (
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

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
    import flashinfer.comm as flashinfer_comm

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

# FlashInfer max sizes per world size (from collective_fusion.py)
_FI_MAX_SIZES = {
    2: 64 * MiB,  # 64MB
    4: 32 * MiB,  # 32MB
    6: 32 * MiB,  # 32MB
    8: 32 * MiB,  # 32MB
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
                group=get_tp_group().device_group if world_size > 1 else None,
                use_fp32_lamport=use_fp32_lamport,
            )
        )

        _FI_WORKSPACE_TENSOR = workspace_tensor
        return ipc_handles, workspace_tensor
    except Exception as e:
        logger.error("Failed to setup FlashInfer workspace: %s", e)
        return None, None


def cleanup_flashinfer_workspace(ipc_handles, world_size: int):
    """Cleanup FlashInfer workspace."""
    if flashinfer_comm is None or ipc_handles is None:
        return

    try:
        group = get_tp_group().device_group if world_size > 1 else None
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
        use_oneshot: bool = True,
    ):
        self.rank = rank
        self.world_size = world_size
        self.use_fp32_lamport = use_fp32_lamport
        self.trigger_completion_at_end = True
        self.launch_with_pdl = True
        self.fp32_acc = True
        self.max_token_num = max_token_num
        self.use_oneshot = use_oneshot

    def get_trtllm_fused_allreduce_kwargs(self):
        return {
            "world_rank": self.rank,
            "world_size": self.world_size,
            "launch_with_pdl": self.launch_with_pdl,
            "trigger_completion_at_end": self.trigger_completion_at_end,
            "fp32_acc": self.fp32_acc,
            "use_oneshot": self.use_oneshot,
        }


def flashinfer_fused_allreduce_rmsnorm(
    input_tensor: torch.Tensor,
    residual: Optional[torch.Tensor],
    rms_gamma: torch.Tensor,
    rms_eps: float,
    allreduce_params: FlashInferFusedAllReduceParams,
    norm_out: Optional[torch.Tensor] = None,
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
        token_num=input_tensor.shape[1],
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
        layout_code=flashinfer_comm.FP4QuantizationSFLayout.SWIZZLED,
        scale_factor=None,
        **allreduce_params.get_trtllm_fused_allreduce_kwargs(),
    )


def flashinfer_fused_allreduce_rmsnorm_fp8_quant(
    input_tensor: torch.Tensor,
    residual: Optional[torch.Tensor],
    rms_gamma: torch.Tensor,
    rms_eps: float,
    scale_factor: torch.Tensor,
    allreduce_params: FlashInferFusedAllReduceParams,
    norm_out: Optional[torch.Tensor] = None,
    quant_out: Optional[torch.Tensor] = None,
):
    """FlashInfer fused allreduce + rmsnorm + FP8 quantization."""
    if flashinfer_comm is None or _FI_WORKSPACE_TENSOR is None:
        raise RuntimeError("FlashInfer not available or workspace not initialized")

    if norm_out is None:
        norm_out = input_tensor
        residual_out = residual
    else:
        # return residual_out as allreduce_out with zeroed residual_in
        # as flashinfer does not support rms_norm
        # and allreduce_out together
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
        use_oneshot=allreduce_params.use_oneshot,
        pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
        allreduce_out=None,
        quant_out=quant_out,
        scale_out=None,
        layout_code=flashinfer_comm.FP4QuantizationSFLayout.SWIZZLED,
        scale_factor=scale_factor,
        **allreduce_params.get_trtllm_fused_allreduce_kwargs(),
    )


def flashinfer_fused_allreduce_rmsnorm_fp4_quant(
    input_tensor: torch.Tensor,
    residual: Optional[torch.Tensor],
    rms_gamma: torch.Tensor,
    rms_eps: float,
    input_global_scale: torch.Tensor,
    allreduce_params: FlashInferFusedAllReduceParams,
    quant_out: torch.Tensor,
    output_scale: torch.Tensor,
    norm_out: Optional[torch.Tensor] = None,
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
        use_oneshot=allreduce_params.use_oneshot,
        pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
        allreduce_out=None,
        quant_out=quant_out,
        scale_out=output_scale,
        layout_code=flashinfer_comm.FP4QuantizationSFLayout.SWIZZLED,
        scale_factor=input_global_scale,
        **allreduce_params.get_trtllm_fused_allreduce_kwargs(),
    )


def standard_allreduce_rmsnorm(
    input_tensor: torch.Tensor,
    residual: Optional[torch.Tensor],
    rms_gamma: torch.Tensor,
    rms_eps: float,
    norm_out: Optional[torch.Tensor] = None,
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
    residual: Optional[torch.Tensor],
    rms_gamma: torch.Tensor,
    rms_eps: float,
    scale_factor: torch.Tensor,
    norm_out: Optional[torch.Tensor] = None,
    quant_out: Optional[torch.Tensor] = None,
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
    residual: Optional[torch.Tensor],
    rms_gamma: torch.Tensor,
    rms_eps: float,
    input_global_scale: torch.Tensor,
    quant_out: torch.Tensor,
    output_scale: torch.Tensor,
    norm_out: Optional[torch.Tensor] = None,
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


def create_test_tensors(
    seq_len: int, hidden_dim: int, dtype: torch.dtype, use_residual: bool = True
):
    """Create test tensors for benchmarking."""
    input_tensor = torch.randn(seq_len, hidden_dim, dtype=dtype)
    residual = torch.randn_like(input_tensor) if use_residual else None
    rms_gamma = torch.ones(hidden_dim, dtype=dtype)
    norm_out = torch.empty_like(input_tensor)

    # Quantization scales
    scale_fp8 = torch.tensor(1.0, dtype=torch.float32)
    scale_fp4 = torch.tensor(1.0, dtype=torch.float32)

    # Pre-allocate FP4 output tensors (to avoid allocation overhead in benchmarks)
    fp4_quant_out = torch.empty((seq_len, hidden_dim // 2), dtype=torch.uint8)
    fp4_output_scale = torch.empty((128, 4), dtype=torch.int32)

    return (
        input_tensor,
        norm_out,
        residual,
        rms_gamma,
        scale_fp8,
        scale_fp4,
        fp4_quant_out,
        fp4_output_scale,
    )


def benchmark_operation(
    operation_func, *args, warmup: int = 5, trials: int = 20, **kwargs
):
    """Benchmark a single operation."""
    # Warmup
    for _ in range(warmup):
        operation_func(*args, **kwargs)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(trials):
        operation_func(*args, **kwargs)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_time_ms = ((end_time - start_time) / trials) * 1000
    return avg_time_ms


def run_benchmarks(
    seq_len: int,
    hidden_dim: int,
    dtype: torch.dtype,
    use_residual: bool,
    allreduce_params: Optional[FlashInferFusedAllReduceParams],
    quant_mode: str = "all",
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
        scale_fp4,
        fp4_quant_out,
        fp4_output_scale,
    ) = create_test_tensors(seq_len, hidden_dim, dtype, use_residual)

    rms_eps = 1e-6
    results = {}

    # 1. Standard AllReduce + RMSNorm (always run as baseline)
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

    # 2. FlashInfer Fused AllReduce + RMSNorm (always run if available)
    if flashinfer_comm is not None and allreduce_params is not None:
        try:
            time_ms = benchmark_operation(
                flashinfer_fused_allreduce_rmsnorm,
                input_tensor,
                residual=residual,
                norm_out=norm_out,
                rms_gamma=rms_gamma,
                rms_eps=rms_eps,
                allreduce_params=allreduce_params,
            )
            results["flashinfer_fused_allreduce_rmsnorm"] = time_ms
        except Exception as e:
            logger.error("FlashInfer Fused AllReduce+RMSNorm failed: %s", e)
            results["flashinfer_fused_allreduce_rmsnorm"] = float("inf")

    # Quantization tests (only for bfloat16 and if requested)
    if quant_mode != "none" and dtype == torch.bfloat16:
        # FP8 Quantization tests
        if quant_mode in ["fp8_only", "all"]:
            # 3. Standard AllReduce + RMSNorm + FP8 Quant
            try:
                time_ms = benchmark_operation(
                    standard_allreduce_rmsnorm_fp8_quant,
                    input_tensor,
                    norm_out=norm_out,
                    residual=residual,
                    rms_gamma=rms_gamma,
                    rms_eps=rms_eps,
                    scale_factor=scale_fp8,
                )
                results["standard_allreduce_rmsnorm_fp8_quant"] = time_ms
            except Exception as e:
                logger.error("Standard AllReduce+RMSNorm+FP8 failed: %s", e)
                results["standard_allreduce_rmsnorm_fp8_quant"] = float("inf")

            # 4. FlashInfer Fused AllReduce + RMSNorm + FP8 Quant
            if flashinfer_comm is not None and allreduce_params is not None:
                try:
                    time_ms = benchmark_operation(
                        flashinfer_fused_allreduce_rmsnorm_fp8_quant,
                        input_tensor,
                        norm_out=norm_out,
                        residual=residual,
                        rms_gamma=rms_gamma,
                        rms_eps=rms_eps,
                        scale_factor=scale_fp8,
                        allreduce_params=allreduce_params,
                    )
                    results["flashinfer_fused_allreduce_rmsnorm_fp8_quant"] = time_ms
                except Exception as e:
                    logger.error(
                        "FlashInfer Fused AllReduce+RMSNorm+FP8 failed: %s",
                        e,
                    )
                    results["flashinfer_fused_allreduce_rmsnorm_fp8_quant"] = float(
                        "inf"
                    )

        # FP4 Quantization tests (if supported and requested)
        if quant_mode in [
            "fp4_only",
            "all",
        ] and current_platform.has_device_capability(100):
            # 5. Standard AllReduce + RMSNorm + FP4 Quant
            try:
                time_ms = benchmark_operation(
                    standard_allreduce_rmsnorm_fp4_quant,
                    input_tensor,
                    residual=residual,
                    norm_out=norm_out,
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

            # 6. FlashInfer Fused AllReduce + RMSNorm + FP4 Quant
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
                    )
                    results["flashinfer_fused_allreduce_rmsnorm_fp4_quant"] = time_ms
                except Exception as e:
                    logger.error(
                        "FlashInfer Fused AllReduce+RMSNorm+FP4 failed: %s",
                        e,
                    )
                    results["flashinfer_fused_allreduce_rmsnorm_fp4_quant"] = float(
                        "inf"
                    )

    return results


def print_results(results_dict, seq_len, hidden_dim, dtype, use_residual, quant_mode):
    """Print benchmark results in a formatted table."""
    print(f"\n{'=' * 80}")
    print(f"Results: seq_len={seq_len}, hidden_dim={hidden_dim}")
    print(
        f"dtype={dtype}, residual={'yes' if use_residual else 'no'}, "
        f"quant_mode={quant_mode}"
    )
    print(f"{'=' * 80}")
    print(f"{'Operation':<50} {'Time (ms)':<12} {'Speedup':<10}")
    print(f"{'-' * 80}")

    # Find baseline time (standard allreduce+rmsnorm)
    baseline_time = results_dict.get("standard_allreduce_rmsnorm", float("inf"))

    for op_name, time_ms in results_dict.items():
        if time_ms == float("inf"):
            speedup_str = "FAILED"
        else:
            speedup = baseline_time / time_ms if baseline_time != float("inf") else 1.0
            speedup_str = f"{speedup:.2f}x"

        print(f"{op_name:<50} {time_ms:<12.3f} {speedup_str:<10}")


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
        "--quant-fp8-only",
        action="store_true",
        help="Only run FP8 quantization tests",
    )
    quant_group.add_argument(
        "--quant-fp4-only",
        action="store_true",
        help="Only run FP4 quantization tests",
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
    elif args.quant_fp8_only:
        quant_mode = "fp8_only"
    elif args.quant_fp4_only:
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
            args.hidden_dim * world_size * 4
        )

        ipc_handles, workspace_tensor = setup_flashinfer_workspace(
            world_size, rank, args.hidden_dim, max_num_token
        )

        if workspace_tensor is not None:
            allreduce_params = FlashInferFusedAllReduceParams(
                rank=rank,
                world_size=world_size,
                max_token_num=max_num_token,
                use_oneshot=not args.disable_oneshot,
            )

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
            )

            if rank == 0:
                print_results(
                    results,
                    seq_len,
                    args.hidden_dim,
                    dtype,
                    use_residual,
                    quant_mode,
                )

    finally:
        # Cleanup
        if ipc_handles is not None:
            cleanup_flashinfer_workspace(ipc_handles, world_size)

        dist.barrier()


if __name__ == "__main__":
    main()
