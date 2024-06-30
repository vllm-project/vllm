import torch
import triton
import triton.language as tl

# generally faster than 16, but can be lowered to 16 to reduce the
# shared memory required by the kernel.
MAX_REPEATS_PER_BLOCK = 32


@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE_H_IN': 32}, num_warps=1),
    triton.Config({'BLOCK_SIZE_H_IN': 32}, num_warps=4),
    triton.Config({'BLOCK_SIZE_H_IN': 32}, num_warps=8),
    triton.Config({'BLOCK_SIZE_H_IN': 64}, num_warps=1),
    triton.Config({'BLOCK_SIZE_H_IN': 64}, num_warps=4),
    triton.Config({'BLOCK_SIZE_H_IN': 64}, num_warps=8),
    triton.Config({'BLOCK_SIZE_H_IN': 128}, num_warps=1),
    triton.Config({'BLOCK_SIZE_H_IN': 128}, num_warps=4),
    triton.Config({'BLOCK_SIZE_H_IN': 128}, num_warps=8),
],
                 key=['R', 'H', 'BLOCK_SIZE_INPUT_PER_LORA'],
                 restore_value=['o_ptr'])
@triton.jit
def sgmv_shrink_multi_lora_rank(
        # Same arguments as below, some renamed
        x_ptr,
        w_ptr,
        o_ptr,
        ranks,
        indices,
        repeats,
        S,
        R: tl.constexpr,
        H,
        stride_xs,
        stride_xh,
        stride_wl,
        stride_wr,
        stride_wh,
        stride_os,
        stride_or,
        # Meta-parameters
        BLOCK_SIZE_INPUT_PER_LORA: tl.constexpr,
        BLOCK_SIZE_H_IN: tl.constexpr):
    """
    The shrink side of the lora, very similar implementation to expand, but 
    uses the split-k strategy as in punica.
    """
    # grid will be [num_unique, h out // block size h out]
    h_id, lora_id = tl.program_id(axis=0), tl.program_id(axis=1)
    idx = tl.load(indices + lora_id)
    if idx < 0:
        return
    rank = tl.load(ranks + idx)
    repeats_0, repeats_1 = (tl.load(repeats + lora_id),
                            tl.load(repeats + lora_id + 1))

    n_inputs = repeats_1 - repeats_0
    input_range = tl.arange(0, BLOCK_SIZE_INPUT_PER_LORA)
    offs_xs = repeats_0 + input_range
    rank_range = tl.arange(0, R)
    offs_h = h_id * BLOCK_SIZE_H_IN + tl.arange(0, BLOCK_SIZE_H_IN)
    offs_os = offs_xs

    w_ptrs = (w_ptr + idx * stride_wl + offs_h[:, None] * stride_wh +
              rank_range[None, :] * stride_wr)
    w = tl.load(w_ptrs,
                mask=(offs_h[:, None] < H) & (rank_range[None, :] < rank),
                other=0.0).to(dtype=tl.float32)  # [H_OUT, R]

    # tl.dot works only on sizes >= 16
    if BLOCK_SIZE_INPUT_PER_LORA >= 16 and R >= 16:
        x_ptrs = (x_ptr + offs_xs[:, None] * stride_xs +
                  offs_h[None, :] * stride_xh)
        # [next pow 2 inputs for this lora, R]
        x = tl.load(x_ptrs,
                    mask=(input_range[:, None] < n_inputs) &
                    (offs_h[None, :] < H),
                    other=0.0).to(dtype=tl.float32)

        o_ptrs = (o_ptr + offs_os[:, None] * stride_os +
                  rank_range[None, :] * stride_or)
        tl.atomic_add(o_ptrs,
                      tl.dot(x, w),
                      mask=(input_range[:, None] < n_inputs) &
                      (rank_range[None, :] < rank))
    else:
        for i in range(n_inputs):
            x_ptrs = x_ptr + (repeats_0 + i) * stride_xs + offs_h * stride_xh
            o_ptrs = (o_ptr + (repeats_0 + i) * stride_os +
                      rank_range * stride_or)
            x = tl.load(x_ptrs, mask=offs_h < H,
                        other=0.0).to(dtype=tl.float32)
            tl.atomic_add(o_ptrs,
                          tl.sum(x[:, None] * w, axis=0),
                          mask=rank_range < rank)


@torch.inference_mode()
def sgmv_shrink(x, weights, out, ranks, indices, repeats, max_repeats):
    '''
    weights shape: (max_loras, 1, out, in)
    Tokens for a LoRA (repeats) should be split into groups of 
    MAX_REPEATS_PER_BLOCK for load balancing and shared memory constraints.
    This should be done at the beginning of the forward pass, so it isn't
    repeated every call.

    max rank in ranks should not be greater than buffer.shape[2]
    weights.shape[-2] shouldn't be larger than out.shape[-1] (hidden dim)
    buffer.shape[0] == out.shape[0] (sequence length)

    buffer, weights and out should be contiguous
    '''
    S, H = x.shape
    R = out.shape[-1]

    BLOCK_SIZE_INPUT_PER_LORA = triton.next_power_of_2(max_repeats)
    grid = lambda META: (triton.cdiv(H, META['BLOCK_SIZE_H_IN']), len(repeats)
                         - 1)
    sgmv_shrink_multi_lora_rank[grid](
        x,
        weights,
        out,
        ranks,
        indices,
        repeats,
        S,
        R,
        H,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(2),
        weights.stride(3),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_INPUT_PER_LORA=BLOCK_SIZE_INPUT_PER_LORA,
    )
    return out


@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE_H_OUT': 32}, num_warps=1),
    triton.Config({'BLOCK_SIZE_H_OUT': 32}, num_warps=4),
    triton.Config({'BLOCK_SIZE_H_OUT': 32}, num_warps=8),
    triton.Config({'BLOCK_SIZE_H_OUT': 64}, num_warps=1),
    triton.Config({'BLOCK_SIZE_H_OUT': 64}, num_warps=4),
    triton.Config({'BLOCK_SIZE_H_OUT': 64}, num_warps=8),
    triton.Config({'BLOCK_SIZE_H_OUT': 128}, num_warps=1),
    triton.Config({'BLOCK_SIZE_H_OUT': 128}, num_warps=4),
    triton.Config({'BLOCK_SIZE_H_OUT': 128}, num_warps=8),
],
                 key=['R', 'H', 'BLOCK_SIZE_INPUT_PER_LORA'],
                 restore_value=['o_ptr'])
@triton.jit
def sgmv_expand_multi_lora_rank(
        # NOTE: Inputs MUST be grouped by lora
        # Pointers to buffer, weight page and output respectively
        b_ptr,
        w_ptr,
        o_ptr,
        # indices a tensor of [num unique loras in seq]
        # repeats [num unique loras in seq + 1]
        # indices contains, for each group of inputs, the unique lora idx
        # repeats, r such that sum(r)=seq_length, repeats=cumsum(r).
        # Cumulative sum of how many inputs are using the same lora,
        # starting at 0
        ranks,
        indices,
        repeats,
        # optional output column offset
        out_col_offset,
        scale,
        # Dimensions, sequence length/batch, max rank, hidden out
        S,
        R: tl.constexpr,
        H,
        # row, col stride for each
        stride_bs,
        stride_br,
        stride_wl,
        stride_wh,
        stride_wr,
        stride_os,
        stride_oh,
        # Meta-parameters
        BLOCK_SIZE_INPUT_PER_LORA: tl.constexpr,
        BLOCK_SIZE_H_OUT: tl.constexpr):
    """
    The punica expand kernel in Triton. Can take advantage of tl.dot() for
    increased speed when the rank and number of inputs are larger than 16.
    i.e. prefill or grouped
    """
    # grid will be [num_unique, h out // block size h out]
    h_id, lora_id = tl.program_id(axis=0), tl.program_id(axis=1)
    idx = tl.load(indices + lora_id)
    if idx < 0:
        return
    rank = tl.load(ranks + idx)
    repeats_0, repeats_1 = (tl.load(repeats + lora_id),
                            tl.load(repeats + lora_id + 1))

    n_inputs = repeats_1 - repeats_0
    input_range = tl.arange(0, BLOCK_SIZE_INPUT_PER_LORA)
    offs_bs = repeats_0 + input_range
    rank_range = tl.arange(0, R)
    offs_wh = h_id * BLOCK_SIZE_H_OUT + tl.arange(0, BLOCK_SIZE_H_OUT)

    # compare transpose after vs transpose ptrs
    w_ptrs = (w_ptr + idx * stride_wl + rank_range[:, None] * stride_wr +
              offs_wh[None, :] * stride_wh)

    offs_os = offs_bs
    offs_oh = offs_wh

    w = tl.load(w_ptrs,
                mask=(rank_range[:, None] < rank) & (offs_wh[None, :] < H),
                other=0.0).to(dtype=tl.float32)  # [R, H_OUT]

    # tl.dot works only on sizes >= 16
    if BLOCK_SIZE_INPUT_PER_LORA >= 16 and R >= 16:
        b_ptrs = (b_ptr + offs_bs[:, None] * stride_bs +
                  rank_range[None, :] * stride_br)
        buffer = tl.load(b_ptrs,
                         mask=(input_range[:, None] < n_inputs) &
                         (rank_range[None, :] < rank),
                         other=0.0)  # [next pow 2 inputs for this lora, R]
        buffer *= scale

        o_ptrs = (o_ptr + offs_os[:, None] * stride_os +
                  (offs_oh[None, :] + out_col_offset) * stride_oh)
        accumulator = tl.load(o_ptrs,
                              mask=(input_range[:, None] < n_inputs) &
                              (offs_oh[None, :] < H),
                              other=0.0).to(dtype=tl.float32)
        accumulator += tl.dot(buffer, w)

        tl.store(o_ptrs,
                 accumulator,
                 mask=(input_range[:, None] < n_inputs) &
                 (offs_oh[None, :] < H))
    else:
        for i in range(n_inputs):
            b_ptrs = b_ptr + (repeats_0 +
                              i) * stride_bs + rank_range * stride_br
            o_ptrs = (o_ptr + (repeats_0 + i) * stride_os +
                      (offs_oh + out_col_offset) * stride_oh)
            out = tl.load(o_ptrs, mask=offs_oh < H,
                          other=0.0).to(dtype=tl.float32)
            buffer = tl.load(b_ptrs, mask=rank_range < rank,
                             other=0.0).to(dtype=tl.float32)
            buffer *= scale

            out += tl.sum(buffer[:, None] * w, axis=0)
            tl.store(o_ptrs, out, mask=offs_oh < H)


@torch.inference_mode()
def sgmv_expand(buffer,
                weights,
                out,
                ranks,
                indices,
                repeats,
                max_repeats,
                out_col_offset=0,
                scale=1.0):
    '''
    weights shape: (max_loras, 1, out, in)
    Tokens for a LoRA (repeats) should be split into groups of 
    MAX_REPEATS_PER_BLOCK for load balancing and shared memory constraints.
    This should be done at the beginning of the forward pass, so it isn't
    repeated every call.

    max rank in ranks should not be greater than buffer.shape[2]
    buffer.shape[0] == out.shape[0] (sequence length)
    out_col_offset + weights.shape[-2] can't be greater than out.shape[-1]

    buffer, weights and out should be contiguous
    '''
    assert out_col_offset + weights.shape[-1] <= out.shape[-1], (
        f"Output column offset {out_col_offset} with output dim " +
        f"{weights.shape[-1]} is too high for given output tensor {out.shape}")
    S, R = buffer.shape
    H = weights.shape[-2]

    BLOCK_SIZE_INPUT_PER_LORA = triton.next_power_of_2(max_repeats)
    grid = lambda META: (triton.cdiv(H, META['BLOCK_SIZE_H_OUT']), len(repeats)
                         - 1)
    sgmv_expand_multi_lora_rank[grid](
        buffer,
        weights,
        out,
        ranks,
        indices,
        repeats,
        out_col_offset,
        scale,
        S,
        R,
        H,
        buffer.stride(0),
        buffer.stride(1),
        weights.stride(0),
        weights.stride(2),
        weights.stride(3),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_INPUT_PER_LORA=BLOCK_SIZE_INPUT_PER_LORA,
    )
    return out
