# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def tl_rand64(seed, offset, includes_zero: tl.constexpr):
    lo, hi, _, _ = tl.randint4x(seed, offset)
    lo = lo.to(tl.uint32, bitcast=True).to(tl.uint64)
    hi = hi.to(tl.uint32, bitcast=True).to(tl.uint64)
    r = (hi << 32) | lo

    # 1 / 2**64
    scale = 5.421010862427522170037e-20
    u = r.to(tl.float64) * scale
    if not includes_zero:
        u = tl.maximum(u, 2.2250738585072014e-308)  # float64 tiny
    return u


@triton.jit
def tl_rand32(seed, offset, includes_zero: tl.constexpr):
    # Generate uniform random in fp64 and then truncate fo fp32
    # to preserve full 64-bit entropy.
    u = tl_rand64(seed, offset, includes_zero=includes_zero)
    return u.to(tl.float32)


@triton.jit
def _gumbel_block_argmax(
    logits,
    token_block,
    token_mask,
    vocab_block,
    vocab_mask,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    APPLY_TEMPERATURE: tl.constexpr,
):
    req_state_idxs = tl.load(
        expanded_idx_mapping_ptr + token_block, mask=token_mask, other=0
    )
    temps = tl.load(temp_ptr + req_state_idxs, mask=token_mask, other=1.0).to(
        tl.float32
    )
    is_greedy = temps == 0.0

    if APPLY_TEMPERATURE:
        # Apply temperature.
        # NOTE(woosuk): Match the behavior of _temperature_kernel.
        # E.g., if the kernel uses tl.div_rn, we should use tl.div_rn here too.
        temps = tl.where(is_greedy, 1.0, temps)
        logits = logits / temps[:, None]

    # Calculate the seeds and offsets for gumbel noise.
    seeds = tl.load(seeds_ptr + req_state_idxs, mask=token_mask, other=0)
    positions = tl.load(pos_ptr + token_block, mask=token_mask, other=0)
    gumbel_seed = tl.randint(seeds, positions)

    # FP32 noise is used because FP64 ops are significantly slower on recent
    # hardware (e.g B300).
    u = tl_rand32(gumbel_seed[:, None], vocab_block[None, :], includes_zero=False)
    gumbel_noise = -tl.log(-tl.log(u))

    # Zero out noise for greedy rows so that argmax sees raw logits.
    gumbel_noise = tl.where(is_greedy[:, None], 0.0, gumbel_noise)

    # Apply gumbel noise.
    logits = tl.where(
        token_mask[:, None] & vocab_mask[None, :],
        logits + gumbel_noise,
        float("-inf"),
    )

    values, idxs = tl.max(logits, axis=1, return_indices=True)
    return values, idxs


@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    processed_logits_ptr,
    processed_logits_stride,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp != 0.0 and APPLY_TEMPERATURE:
        # Apply temperature.
        # NOTE(woosuk): Match the behavior of _temperature_kernel.
        # E.g., if the kernel uses tl.div_rn, we should use tl.div_rn here too.
        logits = logits / temp

    # Store the temperature-applied logits.
    if processed_logits_ptr is not None:
        tl.store(
            processed_logits_ptr + req_state_idx * processed_logits_stride + block,
            logits,
            mask=mask,
        )

    logits = logits.to(tl.float64)
    if temp != 0.0:
        # Calculate the seed for gumbel noise.
        seed = tl.load(seeds_ptr + req_state_idx)
        pos = tl.load(pos_ptr + token_idx)
        gumbel_seed = tl.randint(seed, pos)

        # tl.rand returns fp32, so build a true fp64 uniform from 64 random
        # bits before applying the double-log transform.
        u = tl_rand64(gumbel_seed, block, includes_zero=False)
        gumbel_noise = -tl.log(-tl.log(u))

        # Apply gumbel noise.
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    value, idx = tl.max(logits, axis=0, return_indices=True)
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


@triton.jit
def _fused_mm_gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    hidden_states_ptr,
    hidden_states_stride,
    lm_head_weights_ptr,
    lm_head_weights_stride,
    logits_scale_ptr,
    draft_vocab_offsets_ptr,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    num_tokens,
    hidden_dim,
    vocab_size,
    vocab_start,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
):
    token_block_idx = tl.program_id(0)
    token_block = token_block_idx * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    token_mask = token_block < num_tokens

    vocab_block_idx = tl.program_id(1)
    vocab_block = vocab_block_idx * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    vocab_mask = vocab_block < vocab_size

    # Matrix multiply hidden states x LM head weights.
    logits = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_V), dtype=tl.float32)
    for d_start in range(0, hidden_dim, BLOCK_SIZE_H):
        d_block = d_start + tl.arange(0, BLOCK_SIZE_H)
        d_mask = d_block < hidden_dim
        h_tile = tl.load(
            hidden_states_ptr
            + token_block[:, None] * hidden_states_stride
            + d_block[None, :],
            mask=token_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        w_tile = tl.load(
            lm_head_weights_ptr
            + d_block[:, None]
            + vocab_block[None, :] * lm_head_weights_stride,
            mask=d_mask[:, None] & vocab_mask[None, :],
            other=0.0,
        )
        logits += tl.dot(h_tile, w_tile)

    # De-quantize and scale the logits.
    if logits_scale_ptr is not None:
        logits *= tl.load(logits_scale_ptr)

    # Offset vocab to ensure consistent gumbel noise RNG
    # with the target model. This is necessary for TP.
    vocab_block += vocab_start
    if draft_vocab_offsets_ptr is not None:
        # Re-map from draft vocab to target vocab.
        vocab_block += tl.load(
            draft_vocab_offsets_ptr + vocab_block,
            mask=vocab_mask,
            other=0,
        )

    # Gumbel sample from the logits.
    values, idxs = _gumbel_block_argmax(
        logits,
        token_block,
        token_mask,
        vocab_block,
        vocab_mask,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
    )

    # Convert block-local argmax to global token IDs.
    token_ids = vocab_start + vocab_block_idx * BLOCK_SIZE_V + idxs
    if draft_vocab_offsets_ptr is not None:
        # Re-map from draft token ids to target token ids.
        token_ids += tl.load(
            draft_vocab_offsets_ptr + token_ids,
            mask=token_mask,
            other=0,
        )

    tl.store(
        local_argmax_ptr + token_block * local_argmax_stride + vocab_block_idx,
        token_ids,
        mask=token_mask,
    )
    tl.store(
        local_max_ptr + token_block * local_max_stride + vocab_block_idx,
        values,
        mask=token_mask,
    )


@triton.jit
def _block_argmax_kernel(
    out_ptr,
    max_out_ptr,
    block_max_ptr,
    block_max_row_stride,
    block_max_col_stride,
    block_argmax_ptr,
    block_argmax_row_stride,
    block_argmax_col_stride,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_MAX: tl.constexpr,
):
    token_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks

    max_vals = tl.load(
        block_max_ptr
        + token_idx * block_max_row_stride
        + offsets * block_max_col_stride,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    best_val, best_idx = tl.max(max_vals, axis=0, return_indices=True)
    best_idx = tl.minimum(best_idx, num_blocks - 1)

    sampled = tl.load(
        block_argmax_ptr
        + token_idx * block_argmax_row_stride
        + best_idx * block_argmax_col_stride,
    )
    tl.store(out_ptr + token_idx, sampled.to(tl.int64))

    if OUTPUT_MAX:
        tl.store(max_out_ptr + token_idx, best_val)


def _block_argmax(
    block_max: torch.Tensor,
    block_argmax: torch.Tensor,
    max_out: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens, num_blocks = block_max.shape
    out = block_max.new_empty(num_tokens, dtype=torch.int64)
    BLOCK_SIZE = triton.next_power_of_2(num_blocks)
    _block_argmax_kernel[(num_tokens,)](
        out,
        max_out,
        block_max,
        block_max.stride(0),
        block_max.stride(1),
        block_argmax,
        block_argmax.stride(0),
        block_argmax.stride(1),
        num_blocks,
        BLOCK_SIZE=BLOCK_SIZE,
        OUTPUT_MAX=max_out is not None,
    )
    return out


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    processed_logits_out: torch.Tensor | None = None,  # [num_reqs, vocab_size]
) -> torch.Tensor:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float64)
    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        processed_logits_out,
        processed_logits_out.stride(0) if processed_logits_out is not None else 0,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled


def gumbel_flash_sample(
    hidden_states: torch.Tensor,  # [num_tokens, hidden_size]
    lm_head_weight: torch.Tensor,  # [vocab_size, hidden_size]
    vocab_shard_indices: VocabParallelEmbeddingShardIndices,
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [max_num_reqs]
    apply_temperature: bool,
    logits_scale: torch.Tensor | None = None,
    draft_vocab_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens, hidden_size = hidden_states.shape
    vocab_start = vocab_shard_indices.org_vocab_start_index
    vocab_size = vocab_shard_indices.num_org_elements

    if num_tokens <= 16:
        BLOCK_SIZE_T = 16
    elif num_tokens <= 32:
        BLOCK_SIZE_T = 32
    else:
        BLOCK_SIZE_T = 64
    BLOCK_SIZE_H = 128
    BLOCK_SIZE_V = 128
    num_token_blocks = triton.cdiv(num_tokens, BLOCK_SIZE_T)
    num_vocab_blocks = triton.cdiv(vocab_size, BLOCK_SIZE_V)
    # [num_tokens, num_blocks]
    block_argmax = hidden_states.new_empty(
        num_tokens, num_vocab_blocks, dtype=torch.int64
    )
    # [num_tokens, num_blocks]
    block_max = hidden_states.new_empty(
        num_tokens, num_vocab_blocks, dtype=torch.float32
    )
    _fused_mm_gumbel_sample_kernel[(num_token_blocks, num_vocab_blocks)](
        block_argmax,
        block_argmax.stride(0),
        block_max,
        block_max.stride(0),
        hidden_states,
        hidden_states.stride(0),
        lm_head_weight,
        lm_head_weight.stride(0),
        logits_scale,
        draft_vocab_offsets,
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        num_tokens,
        hidden_size,
        vocab_size,
        vocab_start,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        APPLY_TEMPERATURE=apply_temperature,
        num_warps=8,
        num_stages=6,
    )

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return _block_argmax(block_max, block_argmax)

    # Get local argmax/max.
    local_max = block_max.new_empty(num_tokens, dtype=torch.float32)
    local_argmax = _block_argmax(block_max, block_argmax, max_out=local_max)

    # Get argmax/max across TP ranks.
    local_pair = torch.stack([local_max, local_argmax.float()], dim=-1)
    # [num_tokens, 2 * tp_size]
    gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
    # [num_tokens, tp_size], [num_tokens, tp_size]
    tp_max, tp_argmax = gathered.view(num_tokens, tp_size, 2).unbind(-1)
    return _block_argmax(tp_max, tp_argmax)
