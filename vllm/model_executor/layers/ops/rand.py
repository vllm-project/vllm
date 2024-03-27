from typing import Optional, Union

import torch
import triton
import triton.language as tl


def seeded_uniform(
    *size,
    seeds: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None,
    pin_memory: Optional[bool] = False,
) -> torch.Tensor:
    """Similar to torch.rand, but allows for seeds to be set per row.

    seeds must be a 1d tensor. The output tensor may be 1d, 2d, or 3d.
    If it is 3d, the additional seeds needed will be derived automatically
    in a deterministic fashion:
    [
        row 0: [columns_with_seed_0], [columns_with_seed0^1], ...
    ]
    """
    n_dims = len(size)

    if n_dims > 3:
        raise ValueError("seeded_uniform only supports up to 3D tensors")

    if out is None:
        out = torch.empty(*size,
                          dtype=dtype,
                          device=device,
                          pin_memory=pin_memory)
    elif out.shape != size:
        raise ValueError("shape of out and size must be the same")

    if n_dims == 3:
        n_rows, n_3d, n_cols = out.shape
        stride_row = out.stride(0)
        stride_3d = out.stride(1)
    elif n_dims == 2:
        n_rows, n_cols = out.shape
        n_3d = 1
        stride_row = out.stride(0)
        stride_3d = 1
    else:
        n_cols = out.shape[0]
        n_rows = 1
        n_3d = 1
        stride_row = 1
        stride_3d = 1

    if seeds.ndim != 1:
        raise ValueError("seeds must be a 1D tensor")

    if seeds.numel() != n_rows:
        raise ValueError(
            "seeds must have the same number of elements as out has rows")

    # The philox PRNG Triton uses generates 4 random numbers at once.
    # Therefore, the most efficient use of it is to divide the
    # block size by 4, and then save the generated random numbers to
    # each of the 4 slices of the tensor.
    full_block_size = triton.next_power_of_2(n_cols)
    philox_block_size = max(full_block_size // 4, 1)
    n_slices = full_block_size // philox_block_size
    num_warps = 4
    # Manual tuning. This seems to give best performance on A100 for
    # simple kernels like this.
    if philox_block_size >= 8192:
        num_warps = 32
    elif philox_block_size >= 4096:
        num_warps = 16
    elif philox_block_size >= 2048:
        num_warps = 8

    _seeded_uniform_triton[(n_rows, n_3d)](
        out,
        seeds,
        stride_row,
        stride_3d,
        seeds.stride(0),
        n_rows,
        n_3d,
        n_cols,
        n_slices=n_slices,
        num_warps=num_warps,
        block_size=philox_block_size,
    )
    return out


@triton.jit
def _seeded_uniform_triton(
    out_ptr: torch.Tensor,
    seed_ptr: torch.Tensor,
    out_row_stride: int,
    out_3d_stride: int,
    seed_row_stride: int,
    n_rows: int,
    n_3d: int,
    n_cols: int,
    n_slices: tl.constexpr,
    block_size: tl.constexpr,
):
    """
    Generate a random float32 number in [0, 1) for each element in the output
    tensor. The random numbers in a row generated using the seed for that row.

    Args:
        out_ptr: The output tensor.
        seed_ptr: The per-row seeds to use for random number generation.
        out_row_stride: The stride between rows of the output tensor.
        out_3d_stride: The stride between 3D slices of the output tensor.
        seed_row_stride: The stride between rows of the seed tensor.
        n_rows: The number of rows in the output tensor.
        n_3d: The size of second dimension of the output tensor,
            if output tensor is 3D.
        n_cols: The number of columns in the output tensor.
        n_slices: The number of philox outputs to use.
    """
    tl.static_assert(n_slices > 0 and n_slices <= 4, "0 < n_slices <= 4")

    # Get the row index.
    row_idx = tl.program_id(axis=0)
    three_d_idx = tl.program_id(axis=1)

    philox_offsets = tl.arange(0, block_size)
    # Get the seed for the current element.
    seed = tl.load(seed_ptr + row_idx * seed_row_stride)
    if three_d_idx > 0:
        seed ^= three_d_idx
    # Generate random numbers in [0, 1).
    out1, out2, out3, out4 = tl.rand4x(seed, philox_offsets)

    output_row_start_ptr = (out_ptr + row_idx * out_row_stride +
                            three_d_idx * out_3d_stride)
    out1_offsets = philox_offsets
    tl.store(output_row_start_ptr + out1_offsets,
             out1,
             mask=out1_offsets < n_cols)
    if n_slices > 1:
        out2_offsets = tl.arange(block_size, block_size * 2)
        tl.store(output_row_start_ptr + out2_offsets,
                 out2,
                 mask=out2_offsets < n_cols)
    if n_slices > 2:
        out3_offsets = tl.arange(block_size * 2, block_size * 3)
        tl.store(output_row_start_ptr + out3_offsets,
                 out3,
                 mask=out3_offsets < n_cols)
    if n_slices > 3:
        out4_offsets = tl.arange(block_size * 3, block_size * 4)
        tl.store(output_row_start_ptr + out4_offsets,
                 out4,
                 mask=out4_offsets < n_cols)
