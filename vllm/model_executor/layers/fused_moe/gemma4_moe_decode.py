# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 MoE decode GEMV optimization.

Provides access to optimized CUDA GEMV kernels for Gemma4 MoE expert
computation during decode (small batch sizes, T <= 8).

During decode, each token independently routes to top-k experts. With
E=128 experts and K=8, each expert invocation is a GEMV (matrix-vector
multiply). The default Triton fused_experts kernel is tuned for batched GEMM.
For T<=8, launching many small CUDA blocks (one per assignment x column_group)
achieves much higher SM utilization.

Kernel crossover:
    T <= 8:  CUDA GEMV is 1.4-5.3x faster than Triton fused_experts
    T > 8:   Triton fused_experts is faster (amortizes weight loads)

Kernel loading strategy:
    1. Try pre-built ops via ``torch.ops.vllm`` (CMake-built ``_moe_C``).
    2. Fall back to JIT compilation via ``torch.utils.cpp_extension.load()``.
"""

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Dispatch mode: "cmake" if using pre-built ops, "jit" if JIT-compiled
_dispatch_mode: str | None = None
_jit_routing_module = None
_jit_expert_module = None
_init_attempted = False


def _try_cmake_ops() -> bool:
    """Check if the ops are available via pre-built _moe_C extension."""
    try:
        torch.ops.vllm.gemma4_moe_decode_forward  # noqa: B018
        torch.ops.vllm.gemma4_routing  # noqa: B018
        return True
    except (AttributeError, RuntimeError):
        return False


def _get_kernel_source_dir() -> Path:
    """Locate the CUDA kernel source directory for JIT compilation."""
    # Try relative to vllm package root first
    vllm_root = Path(__file__).resolve().parents[4]
    candidate = vllm_root / "csrc" / "moe" / "gemma4_decode"
    if candidate.exists():
        return candidate
    # Fallback: check VLLM_SOURCE_DIR env var
    src_dir = os.environ.get("VLLM_SOURCE_DIR", "")
    if src_dir:
        candidate = Path(src_dir) / "csrc" / "moe" / "gemma4_decode"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Cannot find csrc/moe/gemma4_decode/ CUDA kernel sources. "
        "Set VLLM_SOURCE_DIR to the vLLM repository root."
    )


def _try_jit_compile() -> bool:
    """JIT-compile the CUDA kernels."""
    global _jit_routing_module, _jit_expert_module

    try:
        from torch.utils.cpp_extension import load

        base_dir = _get_kernel_source_dir()
        cache_dir = base_dir / ".build_cache"
        cache_dir.mkdir(exist_ok=True)

        common_cuda_flags = [
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "-arch=sm_90a",
            "-lineinfo",
            "--expt-relaxed-constexpr",
        ]
        common_cxx_flags = ["-O3", "-std=c++17"]

        _jit_routing_module = load(
            name="gemma4_routing",
            sources=[str(base_dir / "gemma4_routing.cu")],
            extra_cuda_cflags=common_cuda_flags
            + ["-DTORCH_EXTENSION_NAME=gemma4_routing"],
            extra_cflags=common_cxx_flags,
            verbose=False,
            build_directory=str(cache_dir),
        )

        _jit_expert_module = load(
            name="gemma4_moe_decode",
            sources=[str(base_dir / "gemma4_moe_decode.cu")],
            extra_cuda_cflags=common_cuda_flags
            + ["-DTORCH_EXTENSION_NAME=gemma4_moe_decode"],
            extra_cflags=common_cxx_flags,
            verbose=False,
            build_directory=str(cache_dir),
        )

        logger.info("Gemma4 decode GEMV kernels JIT-compiled successfully")
        return True

    except Exception as e:
        logger.warning("Gemma4 decode GEMV kernel JIT compilation failed: %s", e)
        _jit_routing_module = None
        _jit_expert_module = None
        return False


def _ensure_initialized() -> bool:
    """Initialize kernel dispatch (CMake-built or JIT).

    Returns True if kernels are available, False otherwise.
    """
    global _dispatch_mode, _init_attempted

    if _dispatch_mode is not None:
        return True
    if _init_attempted:
        return False

    _init_attempted = True

    # Strategy 1: pre-built via CMake
    if _try_cmake_ops():
        _dispatch_mode = "cmake"
        logger.info("Using pre-built Gemma4 decode GEMV kernels")
        return True

    # Strategy 2: JIT compilation
    if _try_jit_compile():
        _dispatch_mode = "jit"
        return True

    return False


def gemma4_decode_routing(
    router_logits: torch.Tensor,
    per_expert_scale: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Gemma4 routing via optimized CUDA kernel.

    Args:
        router_logits: [T, E] fp32 router output logits.
        per_expert_scale: [E] fp32 per-expert scaling factors.
        top_k: Number of experts to select per token.

    Returns:
        Tuple of (topk_weights [T, K] fp32, topk_ids [T, K] int32).
    """
    logits = router_logits.contiguous().float()
    scale = per_expert_scale.contiguous().float()

    if _dispatch_mode == "cmake":
        return torch.ops.vllm.gemma4_routing(logits, scale, top_k)
    else:
        assert _jit_routing_module is not None, "Kernels not initialized"
        return _jit_routing_module.routing(logits, scale, top_k)


def gemma4_decode_expert_forward(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    intermediate_size: int,
) -> torch.Tensor:
    """Run Gemma4 expert GEMV forward via optimized CUDA kernel.

    Args:
        hidden_states: [T, H] bf16 input activations.
        w13: [E, 2*N, H] bf16 packed gate+up expert weights.
        w2: [E, H, N] bf16 down-projection expert weights.
        topk_ids: [T, K] int32 selected expert indices.
        topk_weights: [T, K] fp32 routing weights.
        intermediate_size: N (per TP shard).

    Returns:
        [T, H] bf16 output tensor.
    """
    hs = hidden_states.contiguous().bfloat16()
    w13_c = w13.contiguous().bfloat16()
    w2_c = w2.contiguous().bfloat16()

    if _dispatch_mode == "cmake":
        return torch.ops.vllm.gemma4_moe_decode_forward(
            hs, w13_c, w2_c, topk_ids, topk_weights, intermediate_size
        )
    else:
        assert _jit_expert_module is not None, "Kernels not initialized"
        return _jit_expert_module.forward(
            hs, w13_c, w2_c, topk_ids, topk_weights, intermediate_size
        )


def is_available() -> bool:
    """Check whether the optimized kernels can be loaded."""
    return _ensure_initialized()
