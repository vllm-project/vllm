# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin wrapper around the DeepSelect top-k CUDA extension (vllm._deepselect_C).

DeepSelect (https://github.com/deepseek-ai/DeepSelect, MIT license) provides
high-performance per-row top-k selection kernels for SM100a/SM103a.
"""

import functools

import torch

# Below this row count the existing cooperative/persistent top-k kernels are
# faster than DeepSelect's per-SM kernels.
DEEP_SELECT_MIN_ROWS = 32

# Matches the -1 fill convention used for topk_indices_buffer elsewhere.
IDX_OOB_FILL_VALUE = -1

try:
    import vllm._deepselect_C as _ds
except ImportError:
    _ds = None


def is_available() -> bool:
    """Whether the DeepSelect CUDA extension was built and can be imported."""
    return _ds is not None


@functools.lru_cache(maxsize=1)
def get_stride_requirement() -> tuple[int, int]:
    """Stride alignment requirement (input, output) in bytes."""
    return _ds.get_alignment_requirement()


def supports(input: torch.Tensor, topk: int) -> bool:
    """Kernel-level input constraints (availability checked separately)."""
    return (
        topk <= 4096
        and input.dtype in (torch.float32, torch.bfloat16)
        and input.stride(1) == 1
        and input.stride(0) * input.element_size() % get_stride_requirement()[0] == 0
    )


def _get_empty_and_aligned_tensor(
    dim0: int, dim1: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Tensor with shape (dim0, dim1) whose stride(0) is 32B-aligned."""
    output_stride_requirement = get_stride_requirement()[1] // dtype.itemsize
    assert output_stride_requirement > 0
    dim1_rounded = (
        (dim1 + output_stride_requirement - 1)
        // output_stride_requirement
        * output_stride_requirement
    )
    return torch.empty((dim0, dim1_rounded), device=device, dtype=dtype)[:, :dim1]


def topk(
    input: torch.Tensor,
    topk: int,
    end: torch.Tensor | None = None,
    output_idx: torch.Tensor | None = None,
    indices_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Select the top-k indices per row of `input`.

    Args:
        input: (num_rows, vocab_size), bf16 or fp32. stride(1) must be 1 and
            stride(0) must be 1024B-aligned.
        topk: Number of elements to select per row; must be <= 4096.
        end: Optional (num_rows,) int32 tensor with the exclusive right
            boundary of each row. Rows with `end[i] < topk` get their
            remaining indices filled with -1.
        output_idx: Optional preallocated (num_rows, topk) output tensor whose
            stride(0) is 32B-aligned (e.g. a slice of a wider buffer).
        indices_dtype: Output dtype when `output_idx` is not provided.

    Returns:
        The (num_rows, topk) indices tensor.
    """
    assert _ds is not None, "DeepSelect extension is not available"
    assert input.dim() == 2 and input.stride(1) == 1
    assert input.stride(0) * input.element_size() % get_stride_requirement()[0] == 0

    num_rows = input.shape[0]
    if output_idx is None:
        output_idx = _get_empty_and_aligned_tensor(
            num_rows, topk, input.device, indices_dtype
        )
    else:
        assert output_idx.dtype == indices_dtype

    _ds.topk(
        input,
        topk,
        None,  # begin is not supported
        end,
        False,  # sorted_value
        False,  # sorted_index
        None,  # output_value
        output_idx,
        None,  # output_idx_offset
        IDX_OOB_FILL_VALUE,
        float("-inf"),  # value_oob_fill_value
        False,  # return_value
        True,  # abort_when_nan_found
    )
    return output_idx
