# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Some utilities for logprobs, including logits."""

import torch

from vllm.triton_utils import tl, triton, HAS_TRITON


@triton.jit
def batched_count_greater_than_triton_kernel(
    x_ptr,           # Pointer to the 2D input tensor
    values_ptr,      # Pointer to the 2D values tensor
    output_ptr,      # Pointer to the 1D output tensor
    x_stride_batch,  # Stride to move between rows in x
    x_stride_elem,   # Stride to move between elements in a row of x
    values_stride_batch, # Stride to move between rows in values
    n_elements,      # Number of elements in the second dimension
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel is launched with a 2D grid.
    # axis 0: batch dimension
    # axis 1: element dimension (split into blocks)
    pid_batch = tl.program_id(axis=0)
    pid_elem_block = tl.program_id(axis=1)

    # 1. Load the single comparison value for the current batch item.
    # Each program on axis 0 gets a different batch item.
    value_offset = pid_batch * values_stride_batch
    value = tl.load(values_ptr + value_offset)

    # 2. Create pointers for the current row in the input tensor `x`.
    x_row_start_ptr = x_ptr + pid_batch * x_stride_batch

    # 3. Calculate element offsets for the current block.
    block_start = pid_elem_block * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 4. Create a mask to avoid reading out of bounds.
    mask = offsets < n_elements

    # 5. Load a block of data from the correct row of `x`.
    # We use the mask to safely handle rows whose size is not a multiple of BLOCK_SIZE.
    x_block = tl.load(x_row_start_ptr + offsets * x_stride_elem, mask=mask)

    # 6. Perform the comparison and sum the result within this block.
    # `x_block > value` creates a boolean vector, and tl.sum treats True as 1.
    local_count = tl.sum(x_block > value)

    # 7. Atomically add the local count to the correct slot in the output tensor.
    # This ensures that writes from different blocks to the same output element are safe.
    output_batch_ptr = output_ptr + pid_batch
    tl.atomic_add(output_batch_ptr, local_count)


# Python wrapper function to launch the batched kernel
def batched_count_greater_than_triton(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    # --- Shape and Device Validation ---
    if not (x.ndim == 2 and values.ndim == 2):
        raise ValueError("Inputs `x` and `values` must be 2D tensors.")
    if x.shape[0] != values.shape[0] or values.shape[1] != 1:
        raise ValueError(f"Shape mismatch: x is {x.shape}, values is {values.shape}. "
                         "Expected x=(B, N), values=(B, 1).")

    batch_size, n_elements = x.shape
    device = x.device

    # --- Prepare Output Tensor ---
    output = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # --- Configure and Launch Kernel ---
    def grid(meta):
        # Create a 2D grid.
        # Dimension 0 has `batch_size` programs, one for each batch item.
        # Dimension 1 has enough programs to cover all `n_elements` for each row.
        return (batch_size, triton.cdiv(n_elements, meta['BLOCK_SIZE']))

    # Launch the kernel on the grid.
    # Triton automatically passes pointers and strides for tensor arguments.
    batched_count_greater_than_triton_kernel[grid](
        x,
        values,
        output,
        x.stride(0),
        x.stride(1),
        values.stride(0),
        n_elements,
        BLOCK_SIZE=1024, # This can be tuned for performance
    )

    return output

def batched_count_greater_than(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Counts elements in each row of x that are greater than the corresponding value in values.

    Args:
        x (torch.Tensor): A 2D tensor of shape (batch_size, n_elements).
        values (torch.Tensor): A 2D tensor of shape (batch_size, 1).

    Returns:
        torch.Tensor: A 1D tensor of shape (batch_size,) with the counts.
    """
    if HAS_TRITON:
        return batched_count_greater_than_triton(x, values)
    else:
        return (x >= values).sum(-1)
