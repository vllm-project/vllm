"""Dequantize release block-FP8 tensors into BF16 master weights."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only contract environments do not require Triton.
    triton = None
    tl = None


BLOCK_SHAPE = (128, 128)


if triton is not None:

    @triton.jit
    def _block_fp8_to_bf16_kernel(
        qweight,
        scales,
        output,
        columns,
        scale_columns,
        elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements
        rows = offsets // columns
        columns_in_row = offsets - rows * columns
        scale_offsets = (rows // 128) * scale_columns + columns_in_row // 128
        values = tl.load(qweight + offsets, mask=mask).to(tl.float32)
        block_scales = tl.load(scales + scale_offsets, mask=mask)
        tl.store(output + offsets, values * block_scales, mask=mask)

else:
    _block_fp8_to_bf16_kernel = None


class BlockFP8CheckpointDequantAdapter:
    """Strict checkpoint-only block dequantization.

    Release weights use one scale per ``128 x 128`` block.  The resulting BF16
    tensor becomes model state; deployment FP8 packing remains a separate
    forward-time operation.
    """

    def __call__(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        if qweight.ndim != 2:
            raise ValueError("checkpoint block-FP8 weight must be 2-D")
        if qweight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "checkpoint block-FP8 weight must use torch.float8_e4m3fn"
            )
        expected = tuple(
            (size + block - 1) // block
            for size, block in zip(qweight.shape, BLOCK_SHAPE, strict=True)
        )
        if scales.ndim != 2 or tuple(scales.shape) != expected:
            raise ValueError(
                f"checkpoint scales must have shape {expected}, got "
                f"{tuple(scales.shape)}"
            )
        if scales.device != qweight.device:
            raise ValueError("checkpoint weight and scales must share a device")
        if scales.dtype == torch.uint8:
            scale_fp32 = (scales.to(torch.int32) << 23).view(torch.float32)
        elif scales.dtype == torch.float8_e8m0fnu:
            scale_fp32 = (scales.view(torch.uint8).to(torch.int32) << 23).view(
                torch.float32
            )
        elif scales.dtype == torch.float32:
            scale_fp32 = scales
        else:
            raise TypeError(
                "checkpoint block scales must be float32, float8_e8m0fnu, or uint8"
            )
        if qweight.is_cuda and _block_fp8_to_bf16_kernel is not None:
            output = torch.empty_like(qweight, dtype=torch.bfloat16)
            block_size = 256
            grid = (triton.cdiv(qweight.numel(), block_size),)
            _block_fp8_to_bf16_kernel[grid](
                qweight,
                scale_fp32.contiguous(),
                output,
                qweight.shape[1],
                scale_fp32.shape[1],
                qweight.numel(),
                BLOCK_SIZE=block_size,
            )
            return output
        expanded = scale_fp32.repeat_interleave(BLOCK_SHAPE[0], dim=0)
        expanded = expanded.repeat_interleave(BLOCK_SHAPE[1], dim=1)
        expanded = expanded[: qweight.shape[0], : qweight.shape[1]]
        return (qweight.float() * expanded).to(torch.bfloat16)


__all__ = ["BLOCK_SHAPE", "BlockFP8CheckpointDequantAdapter"]
