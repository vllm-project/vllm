# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax_2d


# Flash samping performance is optimal for num_tokens <= 64.
MAX_NUM_FLASH_TOKENS = 64


@dataclass
class FlashSamplingConfig:
    lm_head_weight: torch.Tensor
    shard_indices: VocabParallelEmbeddingShardIndices
    logits_scale: torch.Tensor | None = None
    max_num_flash_tokens: int = MAX_NUM_FLASH_TOKENS

@triton.jit
def _lm_head_matmul(
    target_hidden_states_ptr,
    target_hidden_states_stride,
    lm_head_weights_ptr,
    lm_head_weights_stride,
    target_logits_scale_ptr,
    token_idxs,
    token_mask,
    vocab_block,
    vocab_mask,
    hidden_dim,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
):
    acc = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_V), dtype=tl.float32)
    for d_start in range(0, hidden_dim, BLOCK_SIZE_H):
        d_block = d_start + tl.arange(0, BLOCK_SIZE_H)
        d_mask = d_block < hidden_dim
        h_tile = tl.load(
            target_hidden_states_ptr
            + token_idxs[:, None] * target_hidden_states_stride
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
        acc = tl.dot(h_tile, w_tile, acc)

    if target_logits_scale_ptr is not None:
        acc *= tl.load(target_logits_scale_ptr)

    logits = tl.where(
        token_mask[:, None] & vocab_mask[None, :],
        acc,
        float("-inf"),
    )
    return logits


@triton.jit
def _fused_lm_head_sample_kernel(
    # [num_reqs, num_blocks]
    local_argmax_ptr,
    local_argmax_stride,
    # [num_reqs, num_blocks]
    local_max_ptr,
    local_max_stride,
    processed_logits_ptr,
    processed_logits_stride_0,
    processed_logits_stride_1,
    processed_logits_col_ptr,
    # [num_reqs, hidden_dim]
    hidden_states_ptr,
    hidden_states_stride,
    # [local_vocab_size, hidden_dim]
    lm_head_weights_ptr,
    lm_head_weights_stride,
    target_logits_scale_ptr,
    # [num_reqs]
    expanded_idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_reqs]
    pos_ptr,
    num_reqs,
    hidden_dim,
    vocab_size,
    vocab_start,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    USE_GREEDY: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    req_block_idx = tl.program_id(0)
    req_block = req_block_idx * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    req_mask = req_block < num_reqs

    vocab_block_idx = tl.program_id(1)
    vocab_block = vocab_block_idx * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    vocab_mask = vocab_block < vocab_size

    # Compute logits from hidden states and LM head.
    logits = _lm_head_matmul(
        hidden_states_ptr,
        hidden_states_stride,
        lm_head_weights_ptr,
        lm_head_weights_stride,
        target_logits_scale_ptr,
        req_block,
        req_mask,
        vocab_block,
        vocab_mask,
        hidden_dim,
        BLOCK_SIZE_T,
        BLOCK_SIZE_H,
        BLOCK_SIZE_V,
    )

    # Sample from the logits.
    global_vocab_block = vocab_block + vocab_start
    if USE_GREEDY:
        values, idxs = tl.max(logits, axis=1, return_indices=True)
    else:
        values, idxs = gumbel_block_argmax_2d(
            logits,
            req_block,
            req_mask,
            global_vocab_block,
            vocab_mask,
            expanded_idx_mapping_ptr,
            temp_ptr,
            seed_ptr,
            pos_ptr,
            processed_logits_ptr,
            processed_logits_stride_0,
            processed_logits_stride_1,
            processed_logits_col_ptr,
            APPLY_TEMPERATURE=True,
            USE_FP64=USE_FP64,
        )
        values = values.to(tl.float32)
    token_ids = vocab_start + vocab_block_idx * BLOCK_SIZE_V + idxs
    tl.store(
        local_argmax_ptr + req_block * local_argmax_stride + vocab_block_idx,
        token_ids,
        mask=req_mask,
    )
    tl.store(
        local_max_ptr + req_block * local_max_stride + vocab_block_idx,
        values,
        mask=req_mask,
    )


@triton.jit
def _reduce_block_argmax_kernel(
    # [num_reqs]
    sampled_ptr,
    # [num_reqs, num_blocks]
    local_argmax_ptr,
    local_argmax_stride,
    # [num_reqs, num_blocks]
    local_max_ptr,
    local_max_stride,
    num_blocks,
    PADDED_NUM_BLOCKS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block = tl.arange(0, PADDED_NUM_BLOCKS)
    mask = block < num_blocks
    local_max = tl.load(
        local_max_ptr + req_idx * local_max_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    max_block_idx = tl.argmax(local_max, axis=0)
    sampled = tl.load(
        local_argmax_ptr + req_idx * local_argmax_stride + max_block_idx,
    )
    tl.store(sampled_ptr + req_idx, sampled)


def flash_sample(
    # [num_reqs, hidden_size]
    hidden_states: torch.Tensor,
    # [num_reqs]
    positions: torch.Tensor,
    # [num_reqs]
    expanded_idx_mapping: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    # [local_vocab_size, hidden_size]
    lm_head_weight: torch.Tensor,
    vocab_shard_indices: VocabParallelEmbeddingShardIndices,
    target_logits_scale: torch.Tensor | None = None,
    use_greedy: bool = False,
    use_fp64: bool = False,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs, hidden_dim = hidden_states.shape
    vocab_size = vocab_shard_indices.num_org_elements
    vocab_start = vocab_shard_indices.org_vocab_start_index

    if num_reqs <= 16:
        BLOCK_SIZE_T = 16
        BLOCK_SIZE_H = 256
        BLOCK_SIZE_V = 128
    elif num_reqs <= 32:
        BLOCK_SIZE_T = 32
        BLOCK_SIZE_H = 128
        BLOCK_SIZE_V = 256
    else:
        BLOCK_SIZE_T = 64
        BLOCK_SIZE_H = 128
        BLOCK_SIZE_V = 128

    num_req_blocks = triton.cdiv(num_reqs, BLOCK_SIZE_T)
    num_vocab_blocks = triton.cdiv(vocab_size, BLOCK_SIZE_V)

    local_argmax = hidden_states.new_empty(
        num_reqs, num_vocab_blocks, dtype=torch.int64
    )
    local_max = hidden_states.new_empty(
        num_reqs, num_vocab_blocks, dtype=torch.float32
    )
    _fused_lm_head_sample_kernel[(num_req_blocks, num_vocab_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        output_processed_logits,
        output_processed_logits.stride(0) if output_processed_logits is not None else 0,
        output_processed_logits.stride(1) if output_processed_logits is not None else 0,
        output_processed_logits_col,
        hidden_states,
        hidden_states.stride(0),
        lm_head_weight,
        lm_head_weight.stride(0),
        target_logits_scale,
        expanded_idx_mapping,
        temperature,
        seed,
        positions,
        num_reqs,
        hidden_dim,
        vocab_size,
        vocab_start,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        USE_GREEDY=use_greedy,
        USE_FP64=use_fp64,
        num_warps=8,
        num_stages=4,
    )

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size > 1:
        local_pair = torch.stack(
            [local_max, local_argmax.float()], dim=-1
        )
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        gathered = gathered.view(num_reqs, -1, 2)
        local_max = gathered[:, :, 0].contiguous()
        local_argmax = gathered[:, :, 1].to(torch.int64).contiguous()

    num_blocks = local_max.shape[-1]
    padded_num_blocks = triton.next_power_of_2(num_blocks)
    sampled = local_argmax.new_empty(num_reqs, dtype=torch.int64)
    _reduce_block_argmax_kernel[(num_reqs,)](
        sampled,
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        num_blocks,
        PADDED_NUM_BLOCKS=padded_num_blocks,
        num_warps=1,
    )
    num_sampled = hidden_states.new_ones(num_reqs, dtype=torch.int32)
    return sampled.unsqueeze(1), num_sampled
