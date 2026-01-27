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

    # calc final lse
    lse = tl.load(lses_ptr + lse_offsets)
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
    )
    const_args = {"HEAD_DIM": D, "N_ROUNDED": N, "IS_BASE_E": is_lse_base_on_e}
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


@triton.jit
def _pack_seq_kernel(
    x_ptr,  # [N, D]
    out_ptr,  # [B, Lmax, D]
    lengths_ptr,  # *i32, [B]
    N: tl.constexpr,
    D: tl.constexpr,
    Lmax: tl.constexpr,
    PAD_VALUE: tl.constexpr,
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

    # Initialize with PAD (cast will occur as needed based on out_ptr dtype)
    d_mask = off_d[None, :] < D
    pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.float32)
    tl.store(out_row_ptr, pad_vals, mask=t_mask[:, None] & d_mask)

    # Load & write only where within seq_len
    x_vals = tl.load(x_row_ptr, mask=valid_row[:, None] & d_mask)
    tl.store(out_row_ptr, x_vals, mask=valid_row[:, None] & d_mask)


def pack_seq_triton(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float = -float("inf"),
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    """
    Pack sequences of different lengths into a batched tensor.

    Args:
        x: [N, ...] - input tensor where N is total number of tokens
        lengths: [B] - sequence lengths for each batch
        pad_value: value to use for padding
        block_t: block size for time dimension
        block_d: block size for feature dimension

    Returns:
        packed: [B, Lmax, ...] - packed tensor
    """

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

    # Starts are computed inside the kernel from lengths

    out = torch.empty((B, Lmax, D), device=x.device, dtype=x.dtype)

    grid = (B, triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
    _pack_seq_kernel[grid](
        x_reshaped,
        out,
        lengths.int(),
        N,
        D,
        Lmax,
        PAD_VALUE=float(pad_value),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

    # Reshape output back to original dimensions (except first dimension)
    if len(original_shape) > 2:
        output_shape = (B, Lmax) + original_shape[1:]
        out = out.reshape(output_shape)

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


@triton.jit
def _fused_pcp_qkv_select_kernel(
    q_ptr,
    q_stride_B,
    q_stride_H,
    k_ptr,
    k_stride_B,
    k_stride_H,
    v_ptr,
    v_stride_B,
    v_stride_H,
    query_start_ptr,
    out_q_head_ptr,
    out_q_tail_ptr,
    out_k_head_ptr,
    out_k_tail_ptr,
    out_v_head_ptr,
    out_v_tail_ptr,
    pcp_world_size: tl.constexpr,
    pcp_rank: tl.constexpr,
    n_head: tl.constexpr,
    q_head_dim: tl.constexpr,
    k_head_dim: tl.constexpr,
    v_head_dim: tl.constexpr,
    SEQ_BLOCK_SIZE: tl.constexpr,
    DIM_BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0) // (2 * pcp_world_size)
    seq_block_id = tl.program_id(0) % (2 * pcp_world_size)
    head_id = tl.program_id(1)
    dim_block_id = tl.program_id(2)
    dim_off = tl.arange(0, DIM_BLOCK_SIZE) + dim_block_id * DIM_BLOCK_SIZE

    q_start_loc = tl.load(query_start_ptr + req_id)
    q_end_loc = tl.load(query_start_ptr + req_id + 1)
    q_select_len = (q_end_loc - q_start_loc) // 2

    # Select Q
    if seq_block_id < 2:
        block_q_start_loc = q_start_loc + seq_block_id * q_select_len
        out_ptr = out_q_head_ptr if seq_block_id == 0 else out_q_tail_ptr
        for qi in range(tl.cdiv(q_select_len, SEQ_BLOCK_SIZE)):
            q_offset = tl.arange(0, SEQ_BLOCK_SIZE) + qi * SEQ_BLOCK_SIZE
            mask = (dim_off[None, :] < q_head_dim) & (q_offset[:, None] < q_select_len)
            q_src_idx = block_q_start_loc + q_offset[:, None]
            q_dst_idx = q_start_loc // 2 + q_offset[:, None]
            q_val = tl.load(
                q_ptr
                + q_src_idx * q_stride_B
                + head_id * q_stride_H
                + dim_off[None, :],
                mask=mask,
            )
            tl.store(
                out_ptr
                + q_dst_idx * n_head * q_head_dim
                + head_id * q_head_dim
                + dim_off[None, :],
                q_val,
                mask=mask,
            )

    # Select KV
    kv_start_loc = q_start_loc * pcp_world_size
    kv_select_len = q_select_len
    k_d_mask = dim_off[None, :] < k_head_dim
    v_d_mask = dim_off[None, :] < v_head_dim
    block_src_kv_start_loc = kv_start_loc + seq_block_id * kv_select_len
    block_dst_kv_head_start_loc = (
        kv_start_loc // 2 // pcp_world_size * (pcp_rank + 1)
        + seq_block_id * kv_select_len
    )
    block_dst_kv_tail_start_loc = (
        kv_start_loc // 2 // pcp_world_size * (2 * pcp_world_size - pcp_rank)
        + seq_block_id * kv_select_len
    )
    for ki in range(tl.cdiv(kv_select_len, SEQ_BLOCK_SIZE)):
        kv_offset = tl.arange(0, SEQ_BLOCK_SIZE) + ki * SEQ_BLOCK_SIZE
        kv_block_mask = kv_offset[:, None] < kv_select_len
        kv_src_idx = block_src_kv_start_loc + kv_offset[:, None]
        kv_dst_idx_head = block_dst_kv_head_start_loc + kv_offset[:, None]
        kv_dst_idx_tail = block_dst_kv_tail_start_loc + kv_offset[:, None]
        k_val = tl.load(
            k_ptr + kv_src_idx * k_stride_B + head_id * k_stride_H + dim_off[None, :],
            mask=k_d_mask & kv_block_mask,
        )
        v_val = tl.load(
            v_ptr + kv_src_idx * v_stride_B + head_id * v_stride_H + dim_off[None, :],
            mask=v_d_mask & kv_block_mask,
        )
        if seq_block_id < pcp_rank + 1:
            tl.store(
                out_k_head_ptr
                + kv_dst_idx_head * n_head * k_head_dim
                + head_id * k_head_dim
                + dim_off[None, :],
                k_val,
                mask=k_d_mask & kv_block_mask,
            )
            tl.store(
                out_v_head_ptr
                + kv_dst_idx_head * n_head * v_head_dim
                + head_id * v_head_dim
                + dim_off[None, :],
                v_val,
                mask=v_d_mask & kv_block_mask,
            )
        if seq_block_id < 2 * pcp_world_size - pcp_rank:
            tl.store(
                out_k_tail_ptr
                + kv_dst_idx_tail * n_head * k_head_dim
                + head_id * k_head_dim
                + dim_off[None, :],
                k_val,
                mask=k_d_mask & kv_block_mask,
            )
            tl.store(
                out_v_tail_ptr
                + kv_dst_idx_tail * n_head * v_head_dim
                + head_id * v_head_dim
                + dim_off[None, :],
                v_val,
                mask=v_d_mask & kv_block_mask,
            )


def fused_pcp_qkv_select(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_start_loc: torch.Tensor,
    pcp_world_size: int,
    pcp_rank: int,
):
    """
    Select the query and kv tensors for PCP. Instead of calling
    `torch.index_select` multiple times, this function fuses the
    selection for Q, K, and V into a single kernel to reduce
    kernel launch overhead.
    Args:
        q: query tensor on the current PCP rank.
        k: key tensor across PCP ranks.
        v: value tensor across PCP ranks.
        query_start_loc: start location of each query.
        pcp_world_size: number of PCP ranks.
        pcp_rank: rank of the current PCP rank.
    Returns:
        q_head: selected query tensor for pcp head.
        k_head: selected key tensor for pcp head.
        v_head: selected value tensor for pcp head.
        q_tail: selected query tensor for pcp tail.
        k_tail: selected key tensor for pcp tail.
        v_tail: selected value tensor for pcp tail.

    """
    q_head = torch.empty(
        (q.size(0) // 2,) + q.shape[1:], device=q.device, dtype=q.dtype
    )
    q_tail = torch.empty_like(q_head)
    k_head = torch.empty(
        (q.size(0) // 2 * (pcp_rank + 1),) + k.shape[1:], device=k.device, dtype=k.dtype
    )
    v_head = torch.empty(
        (q.size(0) // 2 * (pcp_rank + 1),) + v.shape[1:], device=v.device, dtype=v.dtype
    )
    k_tail = torch.empty(
        (q.size(0) // 2 * (2 * pcp_world_size - pcp_rank),) + k.shape[1:],
        device=k.device,
        dtype=k.dtype,
    )
    v_tail = torch.empty(
        (q.size(0) // 2 * (2 * pcp_world_size - pcp_rank),) + v.shape[1:],
        device=v.device,
        dtype=v.dtype,
    )
    BS = len(query_start_loc) - 1
    DIM_BLOCK_SIZE: int = 64
    SEQ_BLOCK_SIZE: int = 256
    assert q.shape[1] == k.shape[1] == v.shape[1]
    n_head = q.shape[1]
    n_dim_block = (
        max(q.shape[2], k.shape[2], v.shape[2]) + DIM_BLOCK_SIZE
    ) // DIM_BLOCK_SIZE
    grid = (
        2 * pcp_world_size * BS,
        n_head,
        n_dim_block,
    )
    _fused_pcp_qkv_select_kernel[grid](
        q,
        q.stride(0),
        q.stride(1),
        k,
        k.stride(0),
        k.stride(1),
        v,
        v.stride(0),
        v.stride(1),
        query_start_loc,
        q_head,
        q_tail,
        k_head,
        k_tail,
        v_head,
        v_tail,
        pcp_world_size,
        pcp_rank,
        n_head,
        q.shape[2],
        k.shape[2],
        v.shape[2],
        SEQ_BLOCK_SIZE,
        DIM_BLOCK_SIZE,
    )
    return q_head, k_head, v_head, q_tail, k_tail, v_tail
