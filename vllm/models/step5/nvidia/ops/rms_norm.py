# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RMSNorm with independent input, weight and output dtypes."""

import operator
from functools import cache

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32
from cutlass.cute.runtime import make_ptr

from vllm.utils.torch_utils import current_stream

_DTYPES = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: Float32,
}


class RMSNormKernel:
    def __init__(self, hidden_size: int, add_residual: bool):
        self.hidden_size = hidden_size
        self.add_residual = add_residual
        self.threads = 128 if hidden_size >= 1024 else 32
        while hidden_size % self.threads:
            self.threads //= 2
        self.vec = 4
        while hidden_size % (self.threads * self.vec):
            self.vec //= 2

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        r_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        rows: Int32,
        eps: Float32,
        weight_bias: Float32,
        stream: CUstream,
    ):
        layout = cute.make_layout(
            (rows, self.hidden_size), stride=(self.hidden_size, 1)
        )
        x = cute.make_tensor(x_ptr, layout)
        residual = cute.make_tensor(r_ptr, layout)
        y = cute.make_tensor(y_ptr, layout)
        weight = cute.make_tensor(
            w_ptr, cute.make_layout((1, self.hidden_size), stride=(0, 1))
        )
        self.kernel(x, weight, residual, y, eps, weight_bias).launch(
            grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        residual: cute.Tensor,
        y: cute.Tensor,
        eps: Float32,
        weight_bias: Float32,
    ):
        tid = cute.arch.thread_idx()[0]
        row = cute.arch.block_idx()[0]
        tile = (1, self.hidden_size)
        vectors = self.hidden_size // (self.threads * self.vec)
        tv = cute.make_layout(
            ((self.threads, 1), (self.vec, vectors)),
            stride=((self.vec, 1), (1, self.threads * self.vec)),
        )
        x_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            x.element_type,
            num_bits_per_copy=self.vec * x.element_type.width,
        )
        thread_copy = cute.make_tiled_copy(x_atom, tv, tile).get_slice(tid)
        tx = thread_copy.partition_S(cute.local_tile(x, tile, (row, 0)))
        tw = thread_copy.partition_S(cute.local_tile(weight, tile, (0, 0)))
        ty = thread_copy.partition_D(cute.local_tile(y, tile, (row, 0)))
        xf = cute.make_fragment_like(tx)
        wf = cute.make_fragment_like(tw)
        cute.copy(x_atom, tx, xf)
        cute.copy(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                weight.element_type,
                num_bits_per_copy=self.vec * weight.element_type.width,
            ),
            tw,
            wf,
        )
        values = xf.load().to(Float32)
        if cutlass.const_expr(self.add_residual):
            tr = thread_copy.partition_S(cute.local_tile(residual, tile, (row, 0)))
            rf = cute.make_fragment_like(tr)
            r_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=self.vec * 32
            )
            cute.copy(r_atom, tr, rf)
            values = values + rf.load()
            rf.store(values)
            cute.copy(r_atom, rf, tr)

        partial = (values * values).reduce(
            cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0
        )
        total = cute.arch.warp_reduction(partial, operator.add)
        if cutlass.const_expr(self.threads > 32):
            smem = utils.SmemAllocator()
            sums = smem.allocate_tensor(
                Float32, cute.make_layout(self.threads // 32), byte_alignment=4
            )
            if tid % 32 == 0:
                sums[tid // 32] = total
            cute.arch.sync_threads()
            total = Float32(0.0)
            for warp in cutlass.range_constexpr(self.threads // 32):
                total = total + sums[warp]

        inverse_rms = cute.math.rsqrt(total / self.hidden_size + eps, fastmath=True)
        normalized = (values * inverse_rms) * (wf.load().to(Float32) + weight_bias)
        yf = cute.make_fragment_like(ty)
        yf.store(normalized.to(y.element_type))
        cute.copy(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                y.element_type,
                num_bits_per_copy=self.vec * y.element_type.width,
            ),
            yf,
            ty,
        )


@cache
def _compile(input_dtype, weight_dtype, output_dtype, hidden_size, add_residual):
    def pointer(dtype):
        return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16)

    return cute.compile(
        RMSNormKernel(hidden_size, add_residual),
        pointer(_DTYPES[input_dtype]),
        pointer(_DTYPES[weight_dtype]),
        pointer(Float32),
        pointer(_DTYPES[output_dtype]),
        Int32(1),
        Float32(1e-5),
        Float32(0.0),
        CUstream(0),
    )


def supports_rms_norm(x, weight, residual=None) -> bool:
    tensors = (x, weight) if residual is None else (x, weight, residual)
    return (
        x.ndim >= 2
        and 0 < x.shape[-1] <= 16384
        and x.shape[-1] % 32 == 0
        and weight.ndim == 1
        and weight.numel() == x.shape[-1]
        and all(
            t.is_cuda
            and t.device == x.device
            and t.dtype in _DTYPES
            and t.is_contiguous()
            and t.data_ptr() % 16 == 0
            for t in tensors
        )
        and (
            residual is None
            or (residual.shape == x.shape and residual.dtype == torch.float32)
        )
    )


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    out_dtype: torch.dtype | None = None,
    weight_bias: float = 0.0,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute in FP32 and round at the output store.

    When supplied, the FP32 residual is updated in place with x + residual.
    """
    if not supports_rms_norm(x, weight, residual):
        raise ValueError("Unsupported RMSNorm inputs")
    out_dtype = out_dtype or x.dtype
    if out_dtype not in _DTYPES:
        raise ValueError(f"Unsupported output dtype: {out_dtype}")
    if out is None:
        out = torch.empty_like(x, dtype=out_dtype)
    elif (
        out.shape != x.shape
        or out.dtype != out_dtype
        or out.device != x.device
        or not out.is_contiguous()
        or out.data_ptr() % 16
    ):
        raise ValueError("Output must match shape/device/dtype and be contiguous")
    rows = x.numel() // x.shape[-1]
    if rows == 0:
        return out
    compiled = _compile(
        x.dtype, weight.dtype, out_dtype, x.shape[-1], residual is not None
    )
    r_ptr = residual.data_ptr() if residual is not None else x.data_ptr()
    compiled(
        make_ptr(_DTYPES[x.dtype], x.data_ptr(), assumed_align=16),
        make_ptr(_DTYPES[weight.dtype], weight.data_ptr(), assumed_align=16),
        make_ptr(Float32, r_ptr, assumed_align=16),
        make_ptr(_DTYPES[out_dtype], out.data_ptr(), assumed_align=16),
        Int32(rows),
        Float32(eps),
        Float32(weight_bias),
        CUstream(current_stream().cuda_stream),
    )
    return out
