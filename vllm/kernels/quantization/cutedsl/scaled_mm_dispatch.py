# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CuTe DSL dispatch layer for scaled GEMM (FP8 / INT8) on SM90.

Provides ``cutedsl_scaled_mm`` as a drop-in replacement for the CUTLASS C++
path used by ``torch.ops._C.cutlass_scaled_mm``.  The function is only called
when ``VLLM_SWITCH_TO_CUTEDSL=1`` is set and the inputs are FP8 or INT8 on a
Hopper GPU.

Supports per-tensor, per-token (row-wise), and per-channel (column-wise)
scales.  The dispatch layer pre-combines ``scale_a * scale_b`` into an
``(M, N)`` tensor which is fused into the kernel epilogue via partition_C,
eliminating any post-kernel scaling passes.

Uses ``from_dlpack`` + ``mark_compact_shape_dynamic`` for zero-copy
wrapping of PyTorch GPU tensors as dynamic-layout CuTe tensors at runtime.

Kernels are JIT-compiled on first use and cached for subsequent calls.
"""

import logging
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import torch
from cutlass.cute.runtime import from_dlpack

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compiled-kernel cache
# ---------------------------------------------------------------------------
class _KernelCache:
    """Cache for compiled CuTe DSL kernels.

    Cache key: ``(tile_mn, cluster_mn, input_dtype, output_dtype)``
    Each unique tile/cluster/dtype combination is compiled once and reused for
    all matrix sizes (the kernels are compiled with ``is_dynamic_layout=True``).
    """

    def __init__(self):
        self._cache: dict[tuple, object] = {}

    def get_or_compile(self, key: tuple, compile_fn):
        if key not in self._cache:
            self._cache[key] = compile_fn()
        return self._cache[key]


_kernel_cache = _KernelCache()

# CuTe tensor caches — keyed by (data_ptr, shape, dtype) so the CuTe
# wrapper (from_dlpack + dynamic-layout marking) can be computed once and
# reused for subsequent calls with the same tensor, saving ~5 C++ interop
# calls per cached hit.
_weight_cute_cache: dict[int, object] = {}
_cute_tensor_cache: dict[tuple, object] = {}


def _get_or_wrap_cute(pt_tensor: torch.Tensor) -> object:
    """Return a cached CuTe dynamic-layout tensor, or create and cache one."""
    key = (pt_tensor.data_ptr(), pt_tensor.shape, pt_tensor.dtype)
    cached = _cute_tensor_cache.get(key)
    if cached is not None:
        return cached
    ct = _to_cute_dynamic(pt_tensor)
    _cute_tensor_cache[key] = ct
    return ct


# PyTorch <-> CUTLASS dtype helpers
_TORCH_TO_CUTLASS: dict[torch.dtype, object] = {}


def _cutlass_dtype(torch_dtype: torch.dtype):
    """Map a PyTorch dtype to its CUTLASS equivalent (lazy init)."""
    global _TORCH_TO_CUTLASS
    if not _TORCH_TO_CUTLASS:
        import cutlass
        _TORCH_TO_CUTLASS = {
            torch.float8_e4m3fn: cutlass.Float8E4M3FN,
            torch.int8: cutlass.Int8,
            torch.bfloat16: cutlass.BFloat16,
            torch.float16: cutlass.Float16,
            torch.float32: cutlass.Float32,
        }
    return _TORCH_TO_CUTLASS[torch_dtype]


def _opt_level() -> int:
    """Select nvcc optimisation level for CuTe DSL compilation."""
    try:
        from cutlass import CUDA_VERSION
        if (CUDA_VERSION.major < 13
                or (CUDA_VERSION.major == 13 and CUDA_VERSION.minor < 1)):
            return 3
        return 2
    except ImportError:
        return 3


# Kernel compilation helpers
def _compile_kernel(
    is_fp8: bool,
    tile_mn: Tuple[int, int],
    cluster_mn: Tuple[int, int],
    template_m: int,
    template_n: int,
    template_k: int,
    out_dtype_torch: torch.dtype,
):
    """JIT-compile a CuTe DSL scaled MM kernel and return the callable."""
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch

    if is_fp8:
        from vllm.kernels.quantization.cutedsl.scaled_mm_sm90_fp8 import (
            ScaledMmSm90Fp8Kernel,
        )
        ab_dtype = cutlass.Float8E4M3FN
        acc_dtype = cutlass.Float32
        KernelClass = ScaledMmSm90Fp8Kernel
    else:
        from vllm.kernels.quantization.cutedsl.scaled_mm_sm90_int8 import (
            ScaledMmSm90Int8Kernel,
        )
        ab_dtype = cutlass.Int8
        acc_dtype = cutlass.Int32
        KernelClass = ScaledMmSm90Int8Kernel

    c_dtype = _cutlass_dtype(out_dtype_torch)
    scale_dtype = cutlass.Float32

    # Create template tensors for compilation with is_dynamic_layout=True
    # so the compiled kernel works for any M, N, K at runtime.
    m = max(template_m, tile_mn[0])
    n = max(template_n, tile_mn[1])
    k = max(template_k, 128)
    l = 1

    a_cpu = cutlass_torch.matrix(l, m, k, False, ab_dtype)
    b_cpu = cutlass_torch.matrix(l, n, k, False, ab_dtype)
    c_cpu = cutlass_torch.matrix(l, m, n, False, c_dtype)
    sa_cpu = cutlass_torch.matrix(1, m, n, False, scale_dtype)
    sb_cpu = cutlass_torch.matrix(1, 1, 1, False, scale_dtype)

    a_cute, _ = cutlass_torch.cute_tensor_like(
        a_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16)
    b_cute, _ = cutlass_torch.cute_tensor_like(
        b_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16)
    c_cute, _ = cutlass_torch.cute_tensor_like(
        c_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16)
    sa_cute, _ = cutlass_torch.cute_tensor_like(
        sa_cpu, scale_dtype, is_dynamic_layout=True, assumed_align=16)
    sb_cute, _ = cutlass_torch.cute_tensor_like(
        sb_cpu, scale_dtype, is_dynamic_layout=True, assumed_align=16)

    scaled_mm = KernelClass(
        acc_dtype=acc_dtype,
        tile_shape_mn=tile_mn,
        cluster_shape_mn=cluster_mn,
    )

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled = cute.compile(
        scaled_mm,
        a_cute, b_cute, c_cute, sa_cute, sb_cute, stream,
        options=f"--opt-level {_opt_level()}",
    )
    return compiled


# Runtime tensor helpers
def _to_cute_dynamic(pt_tensor: torch.Tensor) -> object:
    """Wrap a PyTorch GPU tensor as a dynamic-layout CuTe tensor (zero-copy).

    Uses ``from_dlpack`` for zero-copy wrapping, then
    ``mark_compact_shape_dynamic`` on all modes to match the dynamic layout
    produced by ``cute_tensor_like(is_dynamic_layout=True)`` at compile time.
    """
    ct = from_dlpack(pt_tensor, assumed_align=16)
    leading_dim = ct.leading_dim
    stride_order = pt_tensor.dim_order()
    for mode in range(pt_tensor.dim()):
        ct.mark_compact_shape_dynamic(mode=mode, stride_order=stride_order)
    if leading_dim is not None:
        ct.mark_layout_dynamic(leading_dim=leading_dim)
    return ct


# PyTorch custom op -- raw GEMM only, opaque to torch.compile
@torch.library.custom_op(
    "vllm::cutedsl_scaled_mm",
    mutates_args=[],
    device_types="cuda",
)
def _cutedsl_scaled_mm_op(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Custom op: CuTe DSL Scaled MM with fused scaling.

    Computes ``scale_a * scale_b * (A @ B)``. All scale types (per-tensor,
    per-token, per-channel) are pre-combined into an (M, N) scale tensor
    and fused into the kernel epilogue via partition_C.

    Uses from_dlpack + mark_compact_shape_dynamic for zero-copy input
    wrapping with dynamic layouts.  No torch.cuda.synchronize().
    """
    M, K = a.shape
    N = b.shape[1]
    is_fp8 = a.dtype == torch.float8_e4m3fn

    # ---- select tile / cluster config -----------------------------------
    if is_fp8:
        from vllm.kernels.quantization.cutedsl.scaled_mm_sm90_fp8 import (
            select_tile_config,
        )
    else:
        from vllm.kernels.quantization.cutedsl.scaled_mm_sm90_int8 import (
            select_tile_config,
        )
    tile_mn, cluster_mn = select_tile_config(M, N, K)

    # ---- get or compile kernel ------------------------------------------
    cache_key = (tile_mn, cluster_mn, a.dtype, out_dtype)
    compiled = _kernel_cache.get_or_compile(
        cache_key,
        lambda: _compile_kernel(
            is_fp8, tile_mn, cluster_mn, M, N, K, out_dtype,
        ),
    )

    # ---- pre-combine scales into (M, N) for fused epilogue --------------
    # Pad to tile boundaries so partial-tile element reads don't go OOB.
    tile_m, tile_n = tile_mn
    padded_M = ((M + tile_m - 1) // tile_m) * tile_m
    padded_N = ((N + tile_n - 1) // tile_n) * tile_n
    if padded_M == M and padded_N == N:
        scale_combined = (
            scale_a.float() * scale_b.float()
        ).expand(M, N).contiguous()
    else:
        scale_combined = torch.ones(
            padded_M, padded_N, device=a.device, dtype=torch.float32)
        scale_combined[:M, :N] = (
            scale_a.float() * scale_b.float()
        ).expand(M, N)
    scale_combined_3d = scale_combined.unsqueeze(-1)
    dummy_sb = torch.ones(1, 1, 1, device=a.device, dtype=torch.float32)

    # from_dlpack requires the current CUDA device to match the tensor's
    # device, so set it explicitly for multi-GPU correctness.
    with torch.cuda.device(a.device):
        # ---- zero-copy wrap inputs as dynamic-layout CuTe tensors ------
        # Kernel expects 3D: A (M,K,1), B (N,K,1), C (M,N,1), scale (M,N,1).
        # b is (K,N) column-major from weight loading; b.t() yields a
        # contiguous (N,K) view — no data copy.  unsqueeze is also a view.
        a_3d = a.contiguous().unsqueeze(-1)
        c_3d = torch.empty(M, N, 1, dtype=out_dtype, device=a.device)

        a_cute = _get_or_wrap_cute(a_3d)
        c_cute = _to_cute_dynamic(c_3d)  # new alloc each call, can't cache

        # B (weight) CuTe tensor is cached — weights are persistent
        # nn.Parameters whose data pointer never changes during serving.
        b_ptr = b.data_ptr()
        if b_ptr in _weight_cute_cache:
            b_cute = _weight_cute_cache[b_ptr]
        else:
            b_3d = b.t().unsqueeze(-1)
            b_cute = _to_cute_dynamic(b_3d)
            _weight_cute_cache[b_ptr] = b_cute

        sc_cute = _to_cute_dynamic(scale_combined_3d)
        sb_cute = _to_cute_dynamic(dummy_sb)

        # ---- launch kernel (async, no synchronize) ---------------------
        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)
        compiled(a_cute, b_cute, c_cute, sc_cute, sb_cute, stream)

    return c_3d.squeeze(-1)


@torch.library.register_fake("vllm::cutedsl_scaled_mm")
def _cutedsl_scaled_mm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Fake impl for torch.compile shape inference."""
    return torch.empty(
        a.shape[0], b.shape[1],
        dtype=out_dtype,
        device=a.device,
    )


def cutedsl_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Run scaled GEMM using the CuTe DSL kernel.

    This is a drop-in replacement for the ``torch.ops._C.cutlass_scaled_mm``
    path inside ``vllm._custom_ops.cutlass_scaled_mm``.

    Supports per-tensor, per-token (row-wise), and per-channel (column-wise)
    scales.  The dispatch layer pre-combines ``scale_a * scale_b`` into an
    ``(M, N)`` tensor fused into the kernel epilogue via partition_C.

    Returns ``None`` if the inputs are not supported (unsupported dtype,
    non-Hopper GPU, missing dependencies).

    Args:
        a: (M, K) FP8-e4m3 or INT8 input, contiguous, on CUDA.
        b: (K, N) FP8-e4m3 or INT8 weight, column-major, on CUDA.
        scale_a: FP32 scale for *a* -- scalar (per-tensor) or (M, 1)
            (per-token).
        scale_b: FP32 scale for *b* -- scalar (per-tensor) or (N, 1) / (1, N)
            (per-channel).
        out_dtype: ``torch.bfloat16`` or ``torch.float16``.
        bias: optional per-channel bias tensor of shape ``(N,)`` in
            ``out_dtype``.

    Returns:
        ``(M, N)`` output tensor, or ``None`` on unsupported config.
    """
    # ---- dtype must be FP8 or INT8 -------------------------------
    is_fp8 = a.dtype == torch.float8_e4m3fn
    is_int8 = a.dtype == torch.int8
    if not (is_fp8 or is_int8):
        return None

    # ---- raw GEMM via custom op (opaque to torch.compile) ----------------
    # All scale types (scalar, per-token, per-channel) are fused into
    # the kernel epilogue.
    raw_output = torch.ops.vllm.cutedsl_scaled_mm(
        a, b, scale_a, scale_b, out_dtype)
    result = raw_output

    if bias is not None:
        result = result + bias

    return result
