# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocab-parallel greedy selection for MiniMax's identity draft vocabulary."""

import torch

from vllm.distributed import get_tp_group
from vllm.triton_utils import tl, triton


@triton.jit
def _argmax_pair(left_value, left_id, right_value, right_id):
    left_nan = left_value != left_value  # noqa: PLR0124
    right_nan = right_value != right_value  # noqa: PLR0124
    take_right = (
        (right_nan & ~left_nan)
        | (~left_nan & ~right_nan & (right_value > left_value))
        | (
            ((left_value == right_value) | (left_nan & right_nan))
            & (right_id < left_id)
        )
    )
    return (
        tl.where(take_right, right_value, left_value),
        tl.where(take_right, right_id, left_id),
    )


@triton.jit
def _local_max_pack(
    Logits,
    Partials,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    LOCAL_VOCAB: tl.constexpr,
    VOCAB_START: tl.constexpr,
    GLOBAL_VOCAB: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    local_id = tile * BLOCK + tl.arange(0, BLOCK)
    global_id = local_id + VOCAB_START
    valid = (local_id < LOCAL_VOCAB) & (global_id < GLOBAL_VOCAB)
    values = tl.load(
        Logits + row * ROW_STRIDE + local_id * COL_STRIDE,
        mask=valid,
        other=float("-inf"),
    ).to(tl.float32)
    ids = tl.where(valid, global_id, GLOBAL_VOCAB)
    value, index = tl.reduce((values, ids), 0, _argmax_pair)
    tl.store(Partials + (row * TILES + tile) * 2, value)
    tl.store(Partials + (row * TILES + tile) * 2 + 1, index.to(tl.float32))


@triton.jit(do_not_specialize=["TOKENS"])
def _merge_max(
    Partials,
    Output,
    TOKENS,
    TILES: tl.constexpr,
    WORLD: tl.constexpr,
    GLOBAL_VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.arange(0, BLOCK)
    valid = slot < WORLD * TILES
    rank, tile = slot // TILES, slot % TILES
    offset = ((rank * TOKENS + row) * TILES + tile) * 2
    values = tl.load(Partials + offset, mask=valid, other=float("-inf"))
    ids = tl.load(Partials + offset + 1, mask=valid, other=GLOBAL_VOCAB).to(tl.int32)
    _, index = tl.reduce((values, ids), 0, _argmax_pair)
    tl.store(Output + row, index.to(tl.int64))


def from_local_logits(
    logits: torch.Tensor, vocab_start: int, vocab_size: int, *, block: int = 4096
) -> torch.Tensor:
    """Gather compact partial pairs in rank-major order, without a transpose copy."""
    if logits.ndim != 2 or not 0 < vocab_size < 2**24:
        raise ValueError("Require 2D logits and exactly representable FP32 token IDs")
    tokens, local_vocab = logits.shape
    group = get_tp_group()
    tiles = triton.cdiv(local_vocab, block)
    partials = torch.empty(
        (tokens, tiles, 2), device=logits.device, dtype=torch.float32
    )
    output = torch.empty(tokens, device=logits.device, dtype=torch.int64)
    _local_max_pack[(tokens, tiles)](
        logits,
        partials,
        logits.stride(0),
        logits.stride(1),
        local_vocab,
        vocab_start,
        vocab_size,
        tiles,
        block,
        num_warps=4,
    )
    # dim=0 keeps the physical rank-major layout; the merge reads its strides.
    gathered = group.all_gather(partials, dim=0) if group.world_size > 1 else partials
    _merge_max[(tokens,)](
        gathered,
        output,
        tokens,
        tiles,
        group.world_size,
        vocab_size,
        triton.next_power_of_2(tiles * group.world_size),
        num_warps=4,
    )
    return output


def identity_vocab_top_tokens(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """The caller must establish identity D2T at load time, outside graph capture."""
    processor = model.logits_processor
    if processor.scale != 1.0 or processor.soft_cap is not None:
        raise ValueError(
            "This experiment preserves the MiniMax unmodified-logit contract"
        )
    if model.lm_head.tp_size != get_tp_group().world_size:
        raise ValueError("LM head and sampling collective must use the same TP group")
    logits = processor._apply_head(model.lm_head, hidden_states, None)
    return from_local_logits(
        logits,
        model.lm_head.shard_indices.org_vocab_start_index,
        processor.org_vocab_size,
    )
