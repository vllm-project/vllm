# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hidden-stream short conv fused with the AttnRes mix + RMSNorm of the next
sublayer, for large batches of an already reduced input (CUDA)."""

import torch

from .seq_parallel import SeqParallel


def conv_mix(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    stream_offset: int,
    stream_width: int,
    conv_out: torch.Tensor,
    residual: torch.Tensor | None,
    addend: torch.Tensor | None,
    stack: torch.Tensor,
    last: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    eps: float = 1e-6,
    sp: SeqParallel | None = None,
    halo: int = 0,
    prefix_align: int = 0,
) -> torch.Tensor:
    """`short_conv` the reduced bf16 `x` into `conv_out` (with the
    optional `(residual + addend) + y` update), then `attn_res` over
    `[*stack, (last +) conv_out]` with each query. Returns the `(Q, T, D)`
    bf16 mix outputs.

    With `sp`, the tokens are this rank's shard of the batch, `x` starts
    with the `halo` rows before it (or is `(world, halo + T, D)`, the
    partial rows every rank pushed, summed in rank order), and the mix
    outputs of all shards land in the all-gather buffer of every rank
    (returned once all have landed).
    Every rank writes the conv-state cache rows that later steps read (see
    `SeqParallel`) given the batch's `prefix_align`.
    """
    num_tokens, d = conv_out.shape
    assert num_tokens > 0 and conv_out.is_contiguous()
    assert x.shape[-2:] == (halo + num_tokens, d)
    assert x.stride(-1) == 1 and x.stride(-2) == d
    assert x.dtype == torch.bfloat16
    for t in (residual, addend, last):
        assert t is None or (t.is_contiguous() and t.shape == conv_out.shape)
    assert residual is None or addend is not None
    assert stack.stride(1) == d and stack.stride(2) == 1
    assert queries.dtype == torch.float32 and queries.is_contiguous()
    assert norm_weight.shape == queries.shape and norm_weight.is_contiguous()
    assert conv_weight.shape == (d, 4) and conv_weight.is_contiguous()
    assert d == cache.shape[1] * stream_width and cache.stride(3) == 1
    assert positions.dtype == slot_mapping.dtype == torch.int64
    args = (
        x,
        cache,
        conv_weight,
        positions,
        seq_idx,
        slot_mapping,
        block_table,
        query_start,
        stream_offset,
        stream_width,
        5**-0.5,
        residual,
        addend,
        conv_out,
        stack,
        last,
        queries,
        norm_weight,
        eps,
        norm_eps,
    )
    if sp is None:
        assert halo == 0 and positions.shape[0] == num_tokens and x.dim() == 2
        out = conv_out.new_empty((queries.shape[0], num_tokens, d))
        torch.ops._C.inkling2_conv_mix(*args, out)
        return out
    batch_tokens = positions.shape[0]
    token_base, end, _ = sp.shard(batch_tokens)
    assert end - token_base == num_tokens
    assert x.dim() == 2 or x.shape[0] == sp.world_size
    torch.ops._C.inkling2_seq_parallel_conv_mix(
        *args,
        token_base,
        halo,
        sp.push_bases.data_ptr(),
        sp.gather_offset,
        sp.world_size,
        sp.max_tokens,
        sp.rank,
        sp.gather_slot,
        sp.gather_epochs,
        sp.staging_offset,
        sp.tail_rows,
        prefix_align,
    )
    return sp.gathered(queries.shape[0], batch_tokens)
