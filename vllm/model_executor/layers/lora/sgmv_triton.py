import triton
import triton.language as tl

# generally faster than 16, but can be lowered to 16 to reduce the
# shared memory required by the kernel.
MAX_REPEATS_PER_BLOCK = 32


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H_OUT': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE_H_OUT': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_H_OUT': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H_OUT': 64}, num_warps=1),
        triton.Config({'BLOCK_SIZE_H_OUT': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_H_OUT': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H_OUT': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE_H_OUT': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE_H_OUT': 128}, num_warps=4),
    ],
    key=['R', 'H'],
)
@triton.jit
def sgmv_shrink_multi_lora_rank(
        # Same arguments as below, some renamed
        x_ptr,
        w_ptr,
        o_ptr,
        w_start,
        ranks,
        w_locs,
        indices,
        repeats,
        S,
        R: tl.constexpr,
        H,
        stride_xs,
        stride_xh,
        stride_wp,
        stride_wh,
        stride_os,
        stride_or,
        # Meta-parameters
        BLOCK_SIZE_INPUT_PER_LORA: tl.constexpr,
        BLOCK_SIZE_H_OUT: tl.constexpr):
    """
    The shrink side of the lora, very similar implementation to expand, but 
    uses the split-k strategy as in punica.
    """
    # grid will be [num_unique, h out // block size h out]
    lora_id, h_id = tl.program_id(axis=0), tl.program_id(axis=1)
    idx = tl.load(indices + lora_id)
    if idx < 0:
        return
    rank = tl.load(ranks + idx)
    w_start_ = tl.load(w_start + idx)
    repeats_0, repeats_1 = (tl.load(repeats + lora_id),
                            tl.load(repeats + lora_id + 1))

    n_inputs = repeats_1 - repeats_0
    input_range = tl.arange(0, BLOCK_SIZE_INPUT_PER_LORA)
    offs_xs = repeats_0 + input_range
    rank_range = tl.arange(0, R)
    offs_wp = tl.load(w_locs + w_start_ + rank_range)
    offs_h = h_id * BLOCK_SIZE_H_OUT + tl.arange(0, BLOCK_SIZE_H_OUT)
    offs_os = offs_xs

    w_ptrs = w_ptr + offs_wp[:, None] * stride_wp + offs_h[None, :] * stride_wh
    w = tl.load(w_ptrs,
                mask=(offs_h[None, :] < H) & (rank_range[:, None] < rank),
                other=0.0).to(dtype=tl.float32)  # [R, H_OUT]

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
                      tl.dot(x, tl.trans(w)),
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
                          tl.sum(x[None, :] * w, axis=1),
                          mask=rank_range < rank)


def sgmv_shrink(x, weights, out, w_start, ranks, w_locs, indices, repeats,
                max_rank):
    # Check constraints.
    assert weights.shape[-1] == x.shape[-1], (
        "weight hidden dim is greater than the output tensor hidden dim: " +
        f"weight shape {weights.shape}, out shape {out.shape}")
    assert x.shape[0] == out.shape[0], (
        "x shape at 0 differs from out shape at 0: x shape " +
        f"{x.shape}, out shape {out.shape}")
    assert max_rank >= ranks.max(), (
        "ranks tensor includes a rank that is higher than the given max_rank")
    assert x.is_contiguous(), "x must be contiguous"
    assert weights.is_contiguous(), "Weights must be contiguous"
    assert out.is_contiguous(), "Out must be contiguous"
    S, H = x.shape
    R = max_rank
    assert triton.next_power_of_2(R) == R

    BLOCK_SIZE_INPUT_PER_LORA = triton.next_power_of_2(
        (repeats[1:] - repeats[:-1]).max().item())
    # for load balancing and shared memory limitations
    assert BLOCK_SIZE_INPUT_PER_LORA <= MAX_REPEATS_PER_BLOCK, (
        "Exceeded the maximum number of repeats for a single lora. " +
        "Repeats should be split into groups of size at most " +
        f"{MAX_REPEATS_PER_BLOCK}")
    grid = lambda META: (len(repeats) - 1,
                         triton.cdiv(H, META['BLOCK_SIZE_H_OUT']))
    sgmv_shrink_multi_lora_rank[grid](
        x,
        weights,
        out,
        w_start,
        ranks,
        w_locs,
        indices,
        repeats,
        S,
        R,
        H,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_INPUT_PER_LORA=BLOCK_SIZE_INPUT_PER_LORA)
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H_OUT': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE_H_OUT': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_H_OUT': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H_OUT': 64}, num_warps=1),
        triton.Config({'BLOCK_SIZE_H_OUT': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_H_OUT': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H_OUT': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE_H_OUT': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE_H_OUT': 128}, num_warps=4),
    ],
    key=['R', 'H'],
)
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
        # w_locs holds the row indices for a [page_size, hidden]
        # tensor in which the weights are stored
        # rows of weights for a lora are not necessarily contiguous
        # lora is w_ptr[
        # w_locs[
        #   w_start[indices[lora_id]] :
        #   w_start[indices[lora_id]] + ranks[indices[lora_id]]]
        # ]
        w_start,
        ranks,
        w_locs,
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
        stride_wp,
        stride_wh,
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
    lora_id, h_id = tl.program_id(axis=0), tl.program_id(axis=1)
    idx = tl.load(indices + lora_id)
    if idx < 0:
        return
    rank = tl.load(ranks + idx)
    w_start_ = tl.load(w_start + idx)
    repeats_0, repeats_1 = tl.load(repeats + lora_id), tl.load(repeats +
                                                               lora_id + 1)

    n_inputs = repeats_1 - repeats_0
    input_range = tl.arange(0, BLOCK_SIZE_INPUT_PER_LORA)
    offs_bs = repeats_0 + input_range
    rank_range = tl.arange(0, R)
    offs_wp = tl.load(w_locs + w_start_ + rank_range)
    offs_wh = h_id * BLOCK_SIZE_H_OUT + tl.arange(0, BLOCK_SIZE_H_OUT)
    offs_r = rank_range

    w_ptrs = (w_ptr + offs_wp[:, None] * stride_wp +
              offs_wh[None, :] * stride_wh)

    offs_os = offs_bs
    offs_oh = offs_wh

    w = tl.load(w_ptrs,
                mask=(offs_wh[None, :] < H) & (rank_range[:, None] < rank),
                other=0.0).to(dtype=tl.float32)  # [R, H_OUT]

    # tl.dot works only on sizes >= 16
    if BLOCK_SIZE_INPUT_PER_LORA >= 16 and R >= 16:
        b_ptrs = (b_ptr + offs_bs[:, None] * stride_bs +
                  offs_r[None, :] * stride_br)
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
            b_ptrs = b_ptr + (repeats_0 + i) * stride_bs + offs_r * stride_br
            o_ptrs = (o_ptr + (repeats_0 + i) * stride_os +
                      (offs_oh + out_col_offset) * stride_oh)
            out = tl.load(o_ptrs, mask=offs_oh < H,
                          other=0.0).to(dtype=tl.float32)
            buffer = tl.load(b_ptrs, mask=rank_range < rank,
                             other=0.0).to(dtype=tl.float32)
            buffer *= scale

            out += tl.sum(buffer[:, None] * w, axis=0)
            tl.store(o_ptrs, out, mask=offs_oh < H)


def sgmv_expand(buffer,
                weights,
                out,
                w_start,
                ranks,
                w_locs,
                indices,
                repeats,
                out_col_offset=0,
                scale=1.0):
    # Check constraints.
    assert ranks.max() <= buffer.shape[1], (
        "Ranks argument includes a higher rank than the buffer's " +
        f"second dim: max rank {ranks.max()}, buffer shape {buffer.shape}")
    assert weights.shape[-1] <= out.shape[-1], (
        "Weight hidden dim is greater than the output tensor hidden " +
        f"dim: weight shape {weights.shape}, out shape {out.shape}")
    assert buffer.shape[0] == out.shape[0], (
        "Buffer shape at 0 differs from out shape at 0: " +
        f"buffer shape {buffer.shape}, out shape {out.shape}")
    assert out_col_offset + weights.shape[-1] <= out.shape[-1], (
        f"Output column offset {out_col_offset} with output dim " +
        f"{weights.shape[-1]} is too high for given output tensor {out.shape}")
    assert buffer.is_contiguous(), "Buffer must be contiguous"
    assert weights.is_contiguous(), "Weights must be contiguous"
    assert out.is_contiguous(), "Out must be contiguous"
    S, R = buffer.shape
    H = weights.shape[-1]
    assert triton.next_power_of_2(R) == R

    BLOCK_SIZE_INPUT_PER_LORA = triton.next_power_of_2(
        (repeats[1:] - repeats[:-1]).max().item())
    # for load balancing and shared memory limitations
    assert BLOCK_SIZE_INPUT_PER_LORA <= MAX_REPEATS_PER_BLOCK, (
        "Exceeded the maximum number of repeats for a single lora. " +
        "Repeats should be split into groups of size at most " +
        f"{MAX_REPEATS_PER_BLOCK}")
    grid = lambda META: (len(repeats) - 1,
                         triton.cdiv(H, META['BLOCK_SIZE_H_OUT']))
    sgmv_expand_multi_lora_rank[grid](
        buffer,
        weights,
        out,
        w_start,
        ranks,
        w_locs,
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
        weights.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_INPUT_PER_LORA=BLOCK_SIZE_INPUT_PER_LORA)
    return out
