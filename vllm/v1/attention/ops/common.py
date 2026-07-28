# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import (
    VllmJitKernel,
    WarmupIntRange,
)
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_scalar_specialization_rep,
)
from vllm.triton_utils import tl, triton


class PackSeqTritonKernel(VllmJitKernel["PackSeqTritonKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        pad_value: float | int
        pad_is_uint8: bool
        block_t: int
        block_d: int

    @staticmethod
    @triton.jit(do_not_specialize=["N", "D", "Lmax"])
    def kernel(
        x_ptr,  # [N, D]
        out_ptr,  # [B, Lmax, D]
        lengths_ptr,  # *i32, [B]
        N,
        D,
        Lmax,
        PAD_VALUE: tl.constexpr,
        PAD_IS_UINT8: tl.constexpr,
        BLOCK_T: tl.constexpr,  # timesteps per program
        BLOCK_D: tl.constexpr,  # features per program
    ):
        pid_b = tl.program_id(0)  # batch id
        pid_t = tl.program_id(1)  # block over time dimension
        pid_d = tl.program_id(2)  # block over feature dimension
        off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
        off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

        # Compute start index and sequence length from cumulative lengths
        in_start = 0
        for i in range(pid_b):
            in_start += tl.load(lengths_ptr + i)
        seq_len = tl.load(lengths_ptr + pid_b)

        # valid time positions for this block
        t_mask = off_t < Lmax

        # compute input row indices for valid (b, t)
        in_row = in_start + off_t
        valid_row = (off_t < seq_len) & t_mask

        # Pointers
        # x_ptr: row-major [N, D]
        x_row_ptr = x_ptr + in_row[:, None] * D + off_d[None, :]

        # out_ptr: row-major [B, Lmax, D]
        out_row_ptr = out_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

        # Initialize with PAD. PAD_IS_UINT8 selects the pad tensor's dtype so
        # integer-typed outputs (e.g. MXFP4 packed nibbles, ue8m0 scale bytes)
        # get an exact-byte pad rather than going through an fp32→uint8 cast
        # that's implementation-defined outside of value 0.
        d_mask = off_d[None, :] < D
        if PAD_IS_UINT8:
            pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.uint8)
        else:
            pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.float32)
        tl.store(out_row_ptr, pad_vals, mask=t_mask[:, None] & d_mask)

        # Load & write only where within seq_len
        x_vals = tl.load(x_row_ptr, mask=valid_row[:, None] & d_mask)
        tl.store(out_row_ptr, x_vals, mask=valid_row[:, None] & d_mask)

    def dispatch(  # type: ignore[override]
        self,
        *,
        dtype: torch.dtype,
        pad_value: float | int,
        block_t: int,
        block_d: int,
    ) -> CompileKey:
        is_uint8 = dtype == torch.uint8
        return self.CompileKey(
            dtype=dtype,
            pad_value=int(pad_value) if is_uint8 else float(pad_value),
            pad_is_uint8=is_uint8,
            block_t=block_t,
            block_d=block_d,
        )

    def get_warmup_keys(self, _vllm_config: Any) -> list[CompileKey]:
        # SparseAttnIndexer uses this for packed decode Q. FP8 uses the current
        # platform FP8 dtype; FP4 uses uint8 values/scales with zero padding.
        return self._trace_dispatch(self.dispatch)(
            dtype=(torch.float8_e4m3fn, torch.uint8),
            pad_value=(-float("inf"), 0),
            block_t=64,
            block_d=64,
            _when=lambda *, dtype, pad_value: (
                (dtype == torch.uint8 and pad_value == 0)
                or (dtype != torch.uint8 and pad_value != 0)
            ),
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        data_ptr = TritonWarmupTensor(compile_key.dtype)
        lengths_ptr = TritonWarmupTensor(torch.int32)
        warmup(
            data_ptr,
            data_ptr,
            lengths_ptr,
            1,  # do not specialize N
            1,  # do not specialize D
            1,  # do not specialize Lmax
            PAD_VALUE=compile_key.pad_value,
            PAD_IS_UINT8=compile_key.pad_is_uint8,
            BLOCK_T=compile_key.block_t,
            BLOCK_D=compile_key.block_d,
            grid=(1, 1, 1),
            num_warps=4,
            num_stages=2,
        )

    def __call__(
        self,
        x_reshaped: torch.Tensor,
        out: torch.Tensor,
        lengths: torch.Tensor,
        *,
        N: int,
        D: int,
        Lmax: int,
        pad_value: float | int,
        pad_is_uint8: bool,
        block_t: int,
        block_d: int,
    ) -> None:
        grid = (lengths.numel(), triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
        self.kernel[grid](
            x_reshaped,
            out,
            lengths.int(),
            N,
            D,
            Lmax,
            PAD_VALUE=pad_value,
            PAD_IS_UINT8=pad_is_uint8,
            BLOCK_T=block_t,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=2,
        )


def pack_seq_triton(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float | int = -float("inf"),
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    """Pack sequences of different lengths into a batched tensor.

    Supports float dtypes (any, via fp32 pad) and ``torch.uint8`` (exact-byte
    pad — e.g. MXFP4 packed nibbles or ue8m0 scale bytes). For uint8 inputs
    ``pad_value`` must be an integer in ``[0, 255]``.

    Args:
        x: [N, ...] — input tensor where N is total number of tokens.
        lengths: [B] — sequence lengths for each batch.
        pad_value: value to use for padding. Defaults to ``-inf`` which is
            only sensible for float dtypes; pass ``0`` (or any byte) for
            uint8 inputs.
        block_t: block size for time dimension.
        block_d: block size for feature dimension.

    Returns:
        packed: [B, Lmax, ...] — packed tensor.
    """
    is_uint8 = x.dtype == torch.uint8
    if is_uint8:
        assert isinstance(pad_value, int) and 0 <= pad_value <= 255, (
            f"uint8 pack requires an integer pad in [0, 255], got {pad_value!r}"
        )
        pad_constexpr: int | float = int(pad_value)
    else:
        pad_constexpr = float(pad_value)

    # Handle multi-dimensional input by reshaping to (N, -1)
    original_shape = x.shape
    if len(original_shape) > 2:
        N = original_shape[0]
        x_reshaped = x.reshape(N, -1)
        D = x_reshaped.shape[1]
    else:
        N, D = x.shape
        x_reshaped = x

    B = lengths.numel()
    Lmax = int(lengths.max().item())

    out = torch.empty((B, Lmax, D), device=x.device, dtype=x.dtype)

    _PACK_SEQ_TRITON_KERNEL(
        x_reshaped,
        out,
        lengths,
        N=N,
        D=D,
        Lmax=Lmax,
        pad_value=pad_constexpr,
        pad_is_uint8=is_uint8,
        block_t=block_t,
        block_d=block_d,
    )

    if len(original_shape) > 2:
        out = out.reshape((B, Lmax) + original_shape[1:])

    return out


class UnpackSeqTritonKernel(VllmJitKernel["UnpackSeqTritonKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        block_t: int
        block_d: int

    @staticmethod
    @triton.jit(do_not_specialize=["B", "Lmax", "D"])
    def kernel(
        packed_ptr,  # [B, Lmax, D]
        out_ptr,  # [N, D]
        lengths_ptr,  # *i32, [B]
        B,
        Lmax,
        D,
        BLOCK_T: tl.constexpr,  # timesteps per program
        BLOCK_D: tl.constexpr,  # features per program
    ):
        pid_b = tl.program_id(0)  # batch id
        pid_t = tl.program_id(1)  # block over time dimension
        pid_d = tl.program_id(2)  # block over feature dimension
        off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
        off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

        # bounds: compute start from cumulative lengths
        in_start = 0
        for i in range(pid_b):
            in_start += tl.load(lengths_ptr + i)
        seq_len = tl.load(lengths_ptr + pid_b)

        # valid time positions for this block
        t_mask = off_t < Lmax
        valid_row = (off_t < seq_len) & t_mask

        # compute output row indices for valid (b, t)
        out_row = in_start + off_t

        # Pointers
        # packed_ptr: row-major [B, Lmax, D]
        packed_row_ptr = (
            packed_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]
        )

        # out_ptr: row-major [N, D]
        out_row_ptr = out_ptr + out_row[:, None] * D + off_d[None, :]

        # Load from packed tensor and store to output
        d_mask = off_d[None, :] < D
        packed_vals = tl.load(packed_row_ptr, mask=valid_row[:, None] & d_mask)
        tl.store(out_row_ptr, packed_vals, mask=valid_row[:, None] & d_mask)

    def dispatch(  # type: ignore[override]
        self,
        *,
        dtype: torch.dtype,
        block_t: int,
        block_d: int,
    ) -> CompileKey:
        return self.CompileKey(dtype=dtype, block_t=block_t, block_d=block_d)

    def get_warmup_keys(self, _vllm_config: Any) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            dtype=torch.int32,
            block_t=64,
            block_d=64,
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        data_ptr = TritonWarmupTensor(compile_key.dtype)
        lengths_ptr = TritonWarmupTensor(torch.int32)
        warmup(
            data_ptr,
            data_ptr,
            lengths_ptr,
            1,  # do not specialize B
            1,  # do not specialize Lmax
            1,  # do not specialize D
            BLOCK_T=compile_key.block_t,
            BLOCK_D=compile_key.block_d,
            grid=(1, 1, 1),
            num_warps=4,
            num_stages=2,
        )

    def __call__(
        self,
        packed_reshaped: torch.Tensor,
        out: torch.Tensor,
        lengths: torch.Tensor,
        *,
        B: int,
        Lmax: int,
        D: int,
        block_t: int,
        block_d: int,
    ) -> None:
        grid = (B, triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
        self.kernel[grid](
            packed_reshaped,
            out,
            lengths.int(),
            B,
            Lmax,
            D,
            BLOCK_T=block_t,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=2,
        )


def unpack_seq_triton(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    """
    Unpack a packed decode query tensor back to the original format.
    Efficient Triton implementation.

    Args:
        packed_tensor: [B, Lmax, ...] - packed tensor from pack_seq_triton
        lengths: [B] - sequence lengths for each batch
        block_t: block size for time dimension
        block_d: block size for feature dimension

    Returns:
        unpacked_tensor: [N, ...] where N = sum(lengths)
    """

    # Handle multi-dimensional input by reshaping to (B, Lmax, -1)
    original_shape = packed_tensor.shape
    if len(original_shape) > 3:
        B, Lmax = original_shape[:2]
        packed_reshaped = packed_tensor.reshape(B, Lmax, -1)
        D = packed_reshaped.shape[2]
    else:
        B, Lmax, D = packed_tensor.shape
        packed_reshaped = packed_tensor

    # Calculate total number of elements
    N = int(lengths.sum().item())

    out = torch.empty((N, D), device=packed_tensor.device, dtype=packed_tensor.dtype)

    _UNPACK_SEQ_TRITON_KERNEL(
        packed_reshaped,
        out,
        lengths,
        B=B,
        Lmax=Lmax,
        D=D,
        block_t=block_t,
        block_d=block_d,
    )

    # Reshape output back to original dimensions (except first dimension)
    if len(original_shape) > 3:
        output_shape = (N,) + original_shape[2:]
        out = out.reshape(output_shape)

    return out


_PACK_SEQ_TRITON_KERNEL = PackSeqTritonKernel()
_UNPACK_SEQ_TRITON_KERNEL = UnpackSeqTritonKernel()
