# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTeDSL fused QK-norm for MuseSpark1x attention."""

import operator
from typing import Any

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import Float32, Int32
    from cutlass.cute.runtime import make_ptr

    HAS_CUTEDSL = True
except ImportError:  # pragma: no cover - exercised only without nvidia-cutlass-dsl
    HAS_CUTEDSL = False


_TORCH_TO_CUTLASS_DTYPE = (
    {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
    if HAS_CUTEDSL
    else {}
)

COPY_BITS = 128


if HAS_CUTEDSL:

    def _cross_warp_sum(warp_sum, reduction_buffer, warps_per_row, row_idx):
        """Reduce ``warps_per_row`` warp partials through shared memory."""
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        if lane_idx == 0:
            reduction_buffer[row_idx, warp_idx % warps_per_row] = warp_sum
        cute.arch.barrier()
        block_val = Float32(0.0)
        if lane_idx < warps_per_row:
            block_val = reduction_buffer[row_idx, lane_idx]
        return cute.arch.warp_reduction(block_val, operator.add)

    class _DirectConfig:
        """Launch shape for the direct-indexing schedule."""

        def __init__(self, dtype: Any, head_dim: int):
            self.dtype = dtype
            self.N = head_dim
            self.vec_size = COPY_BITS // dtype.width
            self.threads_per_row = self._threads_per_row(head_dim)
            self.num_threads = 128 if head_dim <= 16384 else 256
            self.rows_per_block = self.num_threads // self.threads_per_row
            self.warps_per_row = max(self.threads_per_row // 32, 1)

            # No tail handling: a mismatch silently crosses head boundaries.
            covered = self.threads_per_row * self.vec_size
            if covered != head_dim:
                raise ValueError(
                    f"qk_norm cannot handle head_dim={head_dim} at "
                    f"{dtype.width}-bit: threads_per_row={self.threads_per_row} * "
                    f"vec_size={self.vec_size} = {covered}, which must equal "
                    "head_dim."
                )

        @staticmethod
        def _threads_per_row(N: int) -> int:
            if N <= 64:
                return 8
            if N <= 128:
                return 16
            if N <= 3072:
                return 32
            if N <= 6144:
                return 64
            if N <= 16384:
                return 128
            return 256

        def smem_size_in_bytes(self) -> int:
            return self.rows_per_block * self.warps_per_row * 4

    class QKNormDirectKernel:
        """RMS-norm each ``head_dim`` group, reading a strided input directly.

        Direct indexing avoids a tiled copy's alignment checks on strided input.
        """

        def __init__(self, dtype: Any, head_dim: int, has_scale: bool = False):
            self.cfg = _DirectConfig(dtype, head_dim)
            self.has_scale = has_scale

        @cute.jit
        def __call__(
            self,
            x_ptr: cute.Pointer,
            o_ptr: cute.Pointer,
            M: Int32,
            num_heads: Int32,
            input_row_stride: Int32,
            q_size: Int32,
            eps: Float32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            cfg = self.cfg
            mX = cute.make_tensor(
                x_ptr, cute.make_layout((M, q_size), stride=(input_row_stride, 1))
            )
            mO = cute.make_tensor(
                o_ptr, cute.make_layout((M, q_size), stride=(q_size, 1))
            )
            self.kernel(mX, mO, eps, scale).launch(
                grid=[cute.ceil_div(M, cfg.rows_per_block), num_heads, 1],
                block=[cfg.num_threads, 1, 1],
                smem=cfg.smem_size_in_bytes(),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self, mX: cute.Tensor, mO: cute.Tensor, eps: Float32, scale: Float32
        ):
            cfg = self.cfg
            tidx = cute.arch.thread_idx()[0]
            bidx_m = cute.arch.block_idx()[0]
            bidx_h = cute.arch.block_idx()[1]

            threads_per_row = cfg.threads_per_row
            vec_size = cfg.vec_size
            head_dim = cfg.N
            warps_per_row = cfg.warps_per_row
            rows_per_block = cfg.rows_per_block
            out_dtype = cfg.dtype

            thread_row = tidx // threads_per_row
            thread_col = tidx % threads_per_row
            row = bidx_m * rows_per_block + thread_row
            col_base = bidx_h * head_dim + thread_col * vec_size
            row_in_bounds = row < mX.shape[0]

            smem = utils.SmemAllocator()
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )

            x_reg = cute.make_rmem_tensor(cute.make_layout(vec_size), Float32)
            local_sum_sq = Float32(0.0)
            for i in cutlass.range_constexpr(vec_size):
                val = Float32(0.0)
                if row_in_bounds:
                    val = Float32(mX[row, col_base + i])
                x_reg[i] = val
                local_sum_sq = local_sum_sq + val * val

            warp_sum = cute.arch.warp_reduction(
                local_sum_sq, operator.add, threads_in_group=min(threads_per_row, 32)
            )
            if cutlass.const_expr(warps_per_row > 1):
                sum_sq = _cross_warp_sum(
                    warp_sum,
                    reduction_buffer,
                    warps_per_row,
                    cute.arch.warp_idx() // warps_per_row,
                )
            else:
                sum_sq = warp_sum

            rstd = cute.math.rsqrt(sum_sq / head_dim + eps, fastmath=True)

            for i in cutlass.range_constexpr(vec_size):
                if row_in_bounds:
                    # fp32 through the scale; the store is the only rounding.
                    y = x_reg[i] * rstd
                    if cutlass.const_expr(self.has_scale):
                        y = y * scale
                    mO[row, col_base + i] = out_dtype(y)

    _compile_cache: dict = {}

    def _compiled_kernel(dtype: Any, head_dim: int, has_scale: bool, stream):
        key = (dtype, head_dim, has_scale)
        compiled = _compile_cache.get(key)
        if compiled is None:
            compiled = cute.compile(
                QKNormDirectKernel(dtype, head_dim, has_scale),
                make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
                Int32(1),
                Int32(1),
                Int32(1),
                Int32(1),
                Float32(1e-5),
                Float32(1.0),
                stream,
            )
            _compile_cache[key] = compiled
        return compiled


def qk_norm(
    x: torch.Tensor,
    head_dim: int,
    eps: float = 1e-5,
    scale: float = 1.0,
) -> torch.Tensor:
    """Unweighted RMSNorm over each ``head_dim`` group, plus an output scale."""
    assert HAS_CUTEDSL, "fused QK-norm requires nvidia-cutlass-dsl"
    assert x.dim() == 2, "x must be [num_tokens, num_heads * head_dim]"
    assert x.stride(1) == 1, "x must be contiguous along its last dim"
    M, N = x.shape
    assert N % head_dim == 0, f"{N} is not a multiple of head_dim={head_dim}"

    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    if M == 0:
        return out

    dtype = _TORCH_TO_CUTLASS_DTYPE[x.dtype]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = _compiled_kernel(dtype, head_dim, scale != 1.0, stream)

    compiled(
        make_ptr(dtype, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(dtype, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        Int32(M),
        Int32(N // head_dim),
        Int32(x.stride(0)),
        Int32(N),
        Float32(eps),
        Float32(scale),
        stream,
    )
    return out


def supports_qk_norm(dtype: torch.dtype, head_dim: int) -> bool:
    """Whether the fused kernel (no tail path) can serve this dtype and head."""
    if not HAS_CUTEDSL or dtype not in _TORCH_TO_CUTLASS_DTYPE:
        return False
    try:
        _DirectConfig(_TORCH_TO_CUTLASS_DTYPE[dtype], head_dim)
    except ValueError:
        return False
    return True
