# Copyright 2026, The FlagOS Contributors.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row I/O kernels for the Qwen4 PLE short-conv state.

The short-conv cache is exposed as a non-contiguous logical
``[rows, hidden, width]`` view of physical ``[rows, width, hidden]`` storage.
FlagGems 5.3/5.4 materializes that whole multi-GiB view for ``index_select``.
These public entries therefore use the compact stride-aware Triton kernel on
an accelerator satisfying the guard and fail closed elsewhere. Only NVIDIA
H100 has current correctness/performance evidence for this extraction.

The scatter write mask supplied by PLE prevents NULL/padded rows from writing
the reserved row zero; out-of-range rows are ignored.  The caller must supply
that mask explicitly so the original ``-1`` index remains distinguishable from
valid row zero.  Duplicate destinations are supported when the mask keeps only
the desired (normally last) writer.  There is no Torch compute fallback.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

HAS_TRITON = True


_BLOCK_SIZE = 1024
_NUM_WARPS = 8


def _is_triton_device(*tensors: torch.Tensor) -> bool:
    return bool(
        HAS_TRITON
        and tensors
        and len({tensor.device for tensor in tensors}) == 1
        and all(tensor.device.type not in ("cpu", "meta") for tensor in tensors)
    )


if HAS_TRITON:

    @triton.jit
    def _ple_state_gather_kernel_3d(
        state_ptr,
        indices_ptr,
        output_ptr,
        indices_stride,
        state_stride0,
        state_stride1,
        state_stride2,
        output_stride0,
        output_stride1,
        output_stride2,
        num_cache_rows,
        hidden_size,
        state_width,
        HIDDEN_FASTEST: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < hidden_size * state_width
        index = tl.load(indices_ptr + row * indices_stride).to(tl.int64)
        valid_index = (index >= 0) & (index < num_cache_rows)
        safe_index = tl.minimum(tl.maximum(index, 0), num_cache_rows - 1)
        if HIDDEN_FASTEST:
            hidden = offsets % hidden_size
            width = offsets // hidden_size
        else:
            hidden = offsets // state_width
            width = offsets % state_width
        source = (
            state_ptr
            + safe_index * state_stride0
            + hidden * state_stride1
            + width * state_stride2
        )
        destination = (
            output_ptr
            + row * output_stride0
            + hidden * output_stride1
            + width * output_stride2
        )
        values = tl.load(source, mask=valid & valid_index, other=0)
        tl.store(destination, values, mask=valid)

    @triton.jit
    def _ple_state_scatter_kernel_3d(
        state_ptr,
        indices_ptr,
        rows_ptr,
        indices_stride,
        write_mask_ptr,
        write_mask_stride,
        state_stride0,
        state_stride1,
        state_stride2,
        rows_stride0,
        rows_stride1,
        rows_stride2,
        num_cache_rows,
        hidden_size,
        state_width,
        HIDDEN_FASTEST: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_elements = hidden_size * state_width
        valid = offsets < row_elements
        valid &= tl.load(write_mask_ptr + row * write_mask_stride)
        index = tl.load(indices_ptr + row * indices_stride).to(tl.int64)
        valid &= (index >= 0) & (index < num_cache_rows)
        safe_index = tl.minimum(tl.maximum(index, 0), num_cache_rows - 1)
        if HIDDEN_FASTEST:
            hidden = offsets % hidden_size
            width = offsets // hidden_size
        else:
            hidden = offsets // state_width
            width = offsets % state_width
        source = (
            rows_ptr + row * rows_stride0 + hidden * rows_stride1 + width * rows_stride2
        )
        destination = (
            state_ptr
            + safe_index * state_stride0
            + hidden * state_stride1
            + width * state_stride2
        )
        values = tl.load(source, mask=valid, other=0)
        tl.store(destination, values, mask=valid)

else:  # pragma: no cover - exercised by import-only/CPU environments
    _ple_state_gather_kernel_3d = None
    _ple_state_scatter_kernel_3d = None


def _validate_state_tensor(state: torch.Tensor, name: str) -> None:
    if state.ndim != 3:
        raise ValueError(f"PLE {name} must be a rank-3 [rows, hidden, width] tensor")
    if state.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"PLE {name} must use float16, bfloat16, or float32; got {state.dtype}"
        )


def _empty_state_rows_like(state: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Allocate compact rows with the same dense inner layout as ``state``."""

    if state.stride(1) <= state.stride(2):
        return torch.empty(
            (num_rows, state.shape[2], state.shape[1]),
            dtype=state.dtype,
            device=state.device,
        ).transpose(1, 2)
    return torch.empty(
        (num_rows, state.shape[1], state.shape[2]),
        dtype=state.dtype,
        device=state.device,
    )


def ple_state_gather(
    state: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor | None = None,
    *,
    indices_are_safe: bool = False,
) -> torch.Tensor:
    """Gather rows; original ``-1``/out-of-range indices produce zero rows.

    ``indices_are_safe`` is retained for call-site compatibility; it never
    bypasses the original-index validity check.
    """

    _validate_state_tensor(state, "state")
    if not _is_triton_device(state, indices):
        raise RuntimeError("Qwen4 PLE gather requires a Triton accelerator")
    if indices.ndim != 1 or indices.device != state.device:
        raise ValueError(
            "PLE gather indices must be a one-dimensional same-device tensor"
        )
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("PLE gather indices must be int32 or int64")
    if state.shape[0] == 0:
        raise ValueError("PLE gather requires at least one cache row")
    expected_shape = (indices.numel(),) + tuple(state.shape[1:])
    if output is not None:
        _validate_state_tensor(output, "state output")
        if output.shape != expected_shape:
            raise ValueError(
                "PLE state gather output has an invalid shape: "
                f"got {tuple(output.shape)}, expected {expected_shape}"
            )
        if output.device != state.device or output.dtype != state.dtype:
            raise ValueError(
                "PLE state gather output must match state device and dtype"
            )

    if not indices.numel() or state.shape[1] == 0 or state.shape[2] == 0:
        if output is None:
            output = _empty_state_rows_like(state, indices.numel())
        return output
    if output is None:
        output = _empty_state_rows_like(state, indices.numel())
    row_elements = state.shape[1] * state.shape[2]
    _ple_state_gather_kernel_3d[
        (indices.numel(), triton.cdiv(row_elements, _BLOCK_SIZE))
    ](
        state,
        indices,
        output,
        indices.stride(0),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        state.shape[0],
        state.shape[1],
        state.shape[2],
        HIDDEN_FASTEST=state.stride(1) <= state.stride(2),
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=_NUM_WARPS,
    )
    return output


def ple_state_scatter_(
    state: torch.Tensor,
    indices: torch.Tensor,
    rows: torch.Tensor,
    *,
    write_mask: torch.Tensor | None = None,
    indices_are_safe: bool = False,
) -> torch.Tensor:
    """Write rows through an explicit NULL/padding/duplicate write mask.

    ``indices_are_safe`` is retained for call-site compatibility; it never
    bypasses the original-index validity check.
    """

    _validate_state_tensor(state, "state")
    _validate_state_tensor(rows, "state rows")
    if not _is_triton_device(state, indices, rows):
        raise RuntimeError("Qwen4 PLE scatter requires a Triton accelerator")
    if indices.ndim != 1 or indices.device != state.device:
        raise ValueError(
            "PLE scatter indices must be a one-dimensional same-device tensor"
        )
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("PLE scatter indices must be int32 or int64")
    if state.shape[0] == 0:
        raise ValueError("PLE scatter requires at least one cache row")
    if rows.shape[0] != indices.numel() or tuple(rows.shape[1:]) != tuple(
        state.shape[1:]
    ):
        raise ValueError(
            "PLE state scatter rows and indices have incompatible shapes: "
            f"rows={tuple(rows.shape)}, indices={indices.numel()}, "
            f"state={tuple(state.shape)}"
        )
    if rows.device != state.device or rows.dtype != state.dtype:
        raise ValueError("PLE state scatter rows must match state device and dtype")

    if write_mask is None:
        raise NotImplementedError(
            "Qwen4 PLE scatter requires an explicit write_mask; "
            "derive duplicate/null semantics in the caller"
        )
    if write_mask.ndim != 1 or write_mask.numel() != indices.numel():
        raise ValueError("PLE state scatter write_mask must match indices")
    if write_mask.device != state.device or write_mask.dtype != torch.bool:
        raise ValueError("PLE state scatter write_mask must be same-device bool")

    if not indices.numel() or state.shape[1] == 0 or state.shape[2] == 0:
        return state

    row_elements = state.shape[1] * state.shape[2]
    _ple_state_scatter_kernel_3d[
        (indices.numel(), triton.cdiv(row_elements, _BLOCK_SIZE))
    ](
        state,
        indices,
        rows,
        indices.stride(0),
        write_mask,
        write_mask.stride(0),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        rows.stride(0),
        rows.stride(1),
        rows.stride(2),
        state.shape[0],
        state.shape[1],
        state.shape[2],
        HIDDEN_FASTEST=state.stride(1) <= state.stride(2),
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=_NUM_WARPS,
    )
    return state


__all__ = ["ple_state_gather", "ple_state_scatter_"]
