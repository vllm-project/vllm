# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.triton_utils import tl, triton


@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    n_ranks,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
    IS_BASE_E: tl.constexpr,
):
    """
    Apply the all-gathered lses to correct each local rank's attention
    output. we still need perform a cross-rank reduction to obtain the
    final attention output.

    Args:
        outputs_ptr (triton.PointerType):
            Pointer to input tensor of shape [ B, H, D ]
        lses_ptr (triton.PointerType):
            Pointer to input tensor of shape [ N, B, H ]
        new_output_ptr (triton.PointerType):
            Pointer to output tensor of shape [ B, H, D ]
        vlse_ptr (triton.PointerType):
            Pointer to output tensor of shape [ B, H ]
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)
    num_n_offsets = tl.arange(0, N_ROUNDED)

    # shape = [N]
    lse_offsets = (
        num_n_offsets * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )

    # calc final lse (mask the power-of-2 padding with -inf, which is
    # neutral for both the max and the exp-sum)
    lse = tl.load(
        lses_ptr + lse_offsets, mask=num_n_offsets < n_ranks, other=-float("inf")
    )
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    if IS_BASE_E:
        lse_exp = tl.exp(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log(lse_acc)
    else:
        lse_exp = tl.exp2(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log2(lse_acc)
    lse += lse_max

    lse_offsets = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets, lse)

    # shape = [D]
    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )

    # correct output
    lse_offset = (
        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    )
    lse_tmp = tl.load(lses_ptr + lse_offset)
    lse_finally = lse_tmp - lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally,
    )
    factor = tl.exp(lse_finally) if IS_BASE_E else tl.exp2(lse_finally)
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor
    output = tl.where(factor == 0.0, 0.0, output)

    tl.store(new_output_ptr + output_offsets, output)


class CPTritonContext:
    """The CPTritonContext is used to avoid recompilation of the Triton JIT."""

    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


def correct_attn_out(
    out: torch.Tensor,
    lses: torch.Tensor,
    cp_rank: int,
    ctx: CPTritonContext,
    is_lse_base_on_e: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct the attention output using the all-gathered lses.

    Args:
        out: Tensor of shape [ B, H, D ]
        lses: Tensor of shape [ N, B, H ]
        cp_rank: Current rank in the context-parallel group
        ctx: Triton context to avoid recompilation

    Returns:
        Tuple of (out, lse) with corrected attention and final log-sum-exp.
    """
    if ctx is None:
        ctx = CPTritonContext()

    # --- Normalize to 3D views ---
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D] or [B,1,H,D], got {tuple(out.shape)}"

    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, (
        f"expected lses [N,B,H] (optionally with a 1-sized extra dim), "
        f"got {tuple(lses.shape)}"
    )

    B, H, D = out.shape
    N = lses.shape[0]

    # Strides after we normalized shapes to 3-D views.  The kernel computes
    # offsets for `vlse_ptr` using lses_stride_B/H, so the output buffer must
    # have the same B/H stride layout as a slice of `lses`.
    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()

    # Allocate LSE with the same B/H strides as `lses` so writes land correctly
    # even when `lses` is a non-contiguous view (e.g., 4-D to 3-D squeeze).
    lse = torch.empty_strided(
        (B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype
    )

    # Kernel launch config
    grid = (B, H, 1)

    regular_args = (
        out,
        out,
        lses,
        lse,
        o_sB,
        o_sH,
        o_sD,
        l_sN,
        l_sB,
        l_sH,
        cp_rank,
        N,
    )
    # tl.arange needs a power-of-2 extent; pad and mask (relevant for
    # non-power-of-2 CP group sizes, e.g. dcp=3 under uneven DCP).
    const_args = {
        "HEAD_DIM": D,
        "N_ROUNDED": triton.next_power_of_2(N),
        "IS_BASE_E": is_lse_base_on_e,
    }
    ctx.call_kernel(_correct_attn_cp_out_kernel, grid, *regular_args, **const_args)
    return out, lse


def _cp_lse_common(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    is_lse_base_on_e=True,
):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    if cp_group.world_size == 1:
        return cp_attn_out

    if ctx is None:
        ctx = CPTritonContext()

    cp_attn_lse = cp_attn_lse.contiguous()
    lses = cp_group.all_gather(cp_attn_lse, dim=0).reshape(
        (cp_group.world_size,) + cp_attn_lse.shape
    )
    out, lse = correct_attn_out(
        cp_attn_out,
        lses,
        cp_group.rank_in_group,
        ctx,
        is_lse_base_on_e=is_lse_base_on_e,
    )
    return out, lse


def cp_lse_ag_out_rs(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e=True,
):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    out, lse = _cp_lse_common(
        cp_attn_out, cp_attn_lse, cp_group, ctx=ctx, is_lse_base_on_e=is_lse_base_on_e
    )
    out = cp_group.reduce_scatter(out, dim=1)

    if return_lse:
        cp_num_heads = lse.shape[1] // cp_group.world_size
        cp_rank = cp_group.rank_in_group
        lse = lse[:, cp_num_heads * cp_rank : cp_num_heads * (cp_rank + 1)]
        return out, lse
    return out


def cp_lse_ag_out_ar(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e=True,
):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    out, lse = _cp_lse_common(
        cp_attn_out, cp_attn_lse, cp_group, ctx=ctx, is_lse_base_on_e=is_lse_base_on_e
    )
    out = cp_group.all_reduce(out)

    if return_lse:
        return out, lse
    return out


def cp_all_gather_heads_uneven(
    x: torch.Tensor, cp_group: GroupCoordinator, head_counts: list[int]
) -> torch.Tensor:
    """All-gather along the head dim (-2) when per-rank head counts
    differ (--rank-tp-ratio): torch's all_gather needs equal shapes, so
    pad each rank's heads to the maximum count, gather, and
    re-concatenate the true head slices in rank order (which is the
    global head order - head shards are contiguous prefix-sum slices).
    head_counts is the per-rank q-head partition (cp_q_head_counts) -
    under v4 free partitioning it need not be proportional to the token
    vector. Shapes are static per rank, so this stays CUDA-graph
    friendly; the padding garbage is sliced away before any compute.
    """
    counts = head_counts
    local_heads = x.shape[-2]
    assert counts[cp_group.rank_in_group] == local_heads, (counts, local_heads)
    max_heads = max(counts)
    pad_shape = list(x.shape)
    pad_shape[-2] = max_heads
    padded = x.new_empty(pad_shape)
    padded[..., :local_heads, :].copy_(x)
    gathered = cp_group.all_gather(padded, dim=-2)
    parts = [
        gathered[..., r * max_heads : r * max_heads + counts[r], :]
        for r in range(cp_group.world_size)
    ]
    return torch.cat(parts, dim=-2)


def cp_local_head_bounds(
    cp_group: GroupCoordinator, head_counts: list[int]
) -> tuple[int, int]:
    """(start, stop) of this rank's head slice in the gathered full head
    set under uneven DCP (head_counts = per-rank q-head partition)."""
    rank = cp_group.rank_in_group
    start = sum(head_counts[:rank])
    return start, start + head_counts[rank]


def cp_lse_ag_out_ar_uneven(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e=True,
    head_counts: list[int] | None = None,
):
    """Uneven-DCP combine: like cp_lse_ag_out_rs, but per-rank head
    shards differ (--rank-tp-ratio), so the equal-shard reduce-scatter
    cannot be used. All-reduce the corrected partial outputs and slice
    this rank's head shard (from head_counts = per-rank q-head
    partition) instead.

    cp_attn_out: [ B, H_total, D ]  (H_total = sum of all ranks' q heads)
    cp_attn_lse: [ B, H_total ]
    """
    out, lse = _cp_lse_common(
        cp_attn_out, cp_attn_lse, cp_group, ctx=ctx, is_lse_base_on_e=is_lse_base_on_e
    )
    out = cp_group.all_reduce(out)

    assert head_counts is not None, (
        "cp_lse_ag_out_ar_uneven requires the per-rank head partition"
    )
    start, stop = cp_local_head_bounds(cp_group, head_counts)
    out = out[:, start:stop].contiguous()
    if return_lse:
        return out, lse[:, start:stop].contiguous()
    return out


@triton.jit
def _pack_seq_kernel(
    x_ptr,  # [N, D]
    out_ptr,  # [B, Lmax, D]
    lengths_ptr,  # *i32, [B]
    N: tl.constexpr,
    D: tl.constexpr,
    Lmax: tl.constexpr,
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

    grid = (B, triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
    _pack_seq_kernel[grid](
        x_reshaped,
        out,
        lengths.int(),
        N,
        D,
        Lmax,
        PAD_VALUE=pad_constexpr,
        PAD_IS_UINT8=is_uint8,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

    if len(original_shape) > 2:
        out = out.reshape((B, Lmax) + original_shape[1:])

    return out


@triton.jit
def _unpack_seq_triton_kernel(
    packed_ptr,  # [B, Lmax, D]
    out_ptr,  # [N, D]
    lengths_ptr,  # *i32, [B]
    B: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
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
    packed_row_ptr = packed_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

    # out_ptr: row-major [N, D]
    out_row_ptr = out_ptr + out_row[:, None] * D + off_d[None, :]

    # Load from packed tensor and store to output
    d_mask = off_d[None, :] < D
    packed_vals = tl.load(packed_row_ptr, mask=valid_row[:, None] & d_mask)
    tl.store(out_row_ptr, packed_vals, mask=valid_row[:, None] & d_mask)


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

    grid = (B, triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
    _unpack_seq_triton_kernel[grid](
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

    # Reshape output back to original dimensions (except first dimension)
    if len(original_shape) > 3:
        output_shape = (N,) + original_shape[2:]
        out = out.reshape(output_shape)

    return out
