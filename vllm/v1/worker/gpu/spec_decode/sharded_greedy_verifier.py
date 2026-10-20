# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy speculative verification without gathering full-vocabulary logits.

With a vocab-parallel LM head every TP rank computes logits for its own slice
of the vocabulary, and the native path all-gathers them before rejection
sampling. For a batch in which every request is greedy, verification only
needs the target argmax of each row, so each rank instead reduces its slice to
compact per-block ``(max, index)`` statistics and only those are all-gathered.
A single kernel then finishes the global argmax, acceptance (including
synthetic acceptance), the bonus token and the sampled/rejected counts.

The statistics keep the vocabulary blocks of the native kernels (8192 for the
draft positions, 1024 for the bonus position) at their global boundaries, also
where a block straddles two ranks, so ties resolve to the lowest token id
exactly as they do on the full logits.

Models opt in through ``compute_logits_local``, the same hook batch-sharded
sampling uses: it returns this rank's equal-width, contiguous vocabulary shard,
in TP rank order, with the same processing ``compute_logits`` applies.
"""

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

logger = init_logger(__name__)

# Vocabulary block sizes of the native greedy verification and bonus kernels.
_BLOCK = 8192
_BONUS_BLOCK = 1024
# Token ids travel as fp32 next to the maxima, which is exact below 2**24.
_MAX_VOCAB_SIZE = 1 << 24


@triton.jit
def _local_greedy_stats_kernel(
    logits_ptr,
    logits_stride,
    stats_ptr,
    cu_num_logits_ptr,
    stats_stride,
    local_width,
    vocab_start,
    vocab_size,
    BLOCK: tl.constexpr,
    BONUS_BLOCK: tl.constexpr,
):
    req, step, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row = tl.load(cu_num_logits_ptr + req).to(tl.int64) + step
    end = tl.load(cu_num_logits_ptr + req + 1)
    if row < end:
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        local_offsets = offsets - vocab_start
        in_shard = (local_offsets >= 0) & (local_offsets < local_width)
        valid = in_shard & (offsets < vocab_size)
        values = tl.load(
            logits_ptr + row * logits_stride + local_offsets,
            mask=valid,
            other=float("-inf"),
        ).to(tl.float32)
        if row == end - 1:
            # The last row of a request only ever yields the bonus token.
            bonus_values = values.reshape(BLOCK // BONUS_BLOCK, BONUS_BLOCK)
            bonus_max, bonus_index = tl.max(bonus_values, axis=1, return_indices=True)
            subblock = block * (BLOCK // BONUS_BLOCK) + tl.arange(
                0, BLOCK // BONUS_BLOCK
            )
            mask = subblock < stats_stride
            bonus_dst = (row * stats_stride + subblock) * 2
            tl.store(stats_ptr + bonus_dst, bonus_max, mask=mask)
            tl.store(
                stats_ptr + bonus_dst + 1,
                (subblock * BONUS_BLOCK + bonus_index).to(tl.float32),
                mask=mask,
            )
        else:
            value, index = tl.max(values, axis=0, return_indices=True)
            dst = (row * stats_stride + block) * 2
            tl.store(stats_ptr + dst, value)
            tl.store(stats_ptr + dst + 1, (block * BLOCK + index).to(tl.float32))


@triton.jit
def _global_argmax(
    stats_ptr,
    stats_stride,
    row,
    num_logits,
    num_blocks,
    PADDED_BLOCKS: tl.constexpr,
    TP_SIZE: tl.constexpr,
    PADDED_TP_SIZE: tl.constexpr,
):
    # The gathered statistics are rank-major. Reassemble each vocabulary block
    # from its shards before picking one. NaN logits have no defined argmax;
    # skipping any block with a NaN in some shard keeps the result in range.
    ranks = tl.arange(0, PADDED_TP_SIZE)
    blocks = tl.arange(0, PADDED_BLOCKS)
    mask = (ranks[:, None] < TP_SIZE) & (blocks[None, :] < num_blocks)
    offsets = ((ranks[:, None] * num_logits + row) * stats_stride + blocks[None, :]) * 2
    values = tl.load(stats_ptr + offsets, mask=mask, other=float("-inf"))
    is_nan = values != values  # noqa: PLR0124
    any_nan = tl.sum(is_nan.to(tl.int32), axis=0) > 0
    clean = tl.where(is_nan, float("-inf"), values)
    block_max = tl.max(clean, axis=0)
    # The first rank holding the block maximum owns its lowest token id.
    source_rank = tl.argmax(clean, axis=0)
    scores = tl.where(any_nan, float("-inf"), block_max)
    winning_block = tl.argmax(scores, axis=0)
    indices = tl.load(
        stats_ptr + ((source_rank * num_logits + row) * stats_stride + blocks) * 2 + 1,
        mask=blocks < num_blocks,
        other=0,
    ).to(tl.int64)
    return tl.sum(tl.where(blocks == winning_block, indices, 0), axis=0)


@triton.jit(do_not_specialize=["num_logits"])
def _sharded_greedy_verify_kernel(
    stats_ptr,
    input_ids_ptr,
    positions_ptr,
    logits_indices_ptr,
    expanded_local_pos_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    seeds_ptr,
    rates_ptr,
    sampled_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    num_logits,
    num_blocks,
    num_bonus_blocks,
    OUTPUT_WIDTH: tl.constexpr,
    PADDED_BLOCKS: tl.constexpr,
    PADDED_BONUS_BLOCKS: tl.constexpr,
    SYNTHETIC: tl.constexpr,
    TP_SIZE: tl.constexpr,
    PADDED_TP_SIZE: tl.constexpr,
):
    req = tl.program_id(0)
    state = tl.load(idx_mapping_ptr + req).to(tl.int64)
    start = tl.load(cu_num_logits_ptr + req).to(tl.int64)
    end = tl.load(cu_num_logits_ptr + req + 1)
    num_draft = end - start - 1
    seed = tl.load(seeds_ptr + state)
    prefill_len = tl.load(prefill_len_ptr + state)
    accepted_length = tl.zeros((), tl.int64)
    verifying = True
    for step in range(num_draft):
        if verifying:
            row = start + step
            token_index = tl.load(logits_indices_ptr + row + 1).to(tl.int64)
            draft = tl.load(input_ids_ptr + token_index).to(tl.int64)
            draft_position = tl.load(positions_ptr + token_index)
            local_position = tl.load(expanded_local_pos_ptr + row + 1)
            # Draft slots of a step that starts within the prefill are
            # placeholders and must be rejected (see gather_draft_sampled).
            placeholder = (local_position > 0) & (
                draft_position - local_position < prefill_len
            )
            valid_draft = (draft >= 0) & ~placeholder
            draft = tl.maximum(0, draft)
            target = _global_argmax(
                stats_ptr,
                num_bonus_blocks,
                row,
                num_logits,
                num_blocks,
                PADDED_BLOCKS,
                TP_SIZE,
                PADDED_TP_SIZE,
            )
            if SYNTHETIC:
                pos_index = tl.load(logits_indices_ptr + row).to(tl.int64)
                pos = tl.load(positions_ptr + pos_index)
                uniform = tl_rand32(seed, pos, includes_zero=False)
                accepted = uniform < tl.load(rates_ptr + step)
            else:
                accepted = target == draft
            accepted &= valid_draft
            verifying = accepted
            accepted_length += accepted
            tl.store(
                sampled_ptr + req * OUTPUT_WIDTH + step,
                tl.where(accepted, draft, target),
            )
    if verifying:
        bonus = _global_argmax(
            stats_ptr,
            num_bonus_blocks,
            end - 1,
            num_logits,
            num_bonus_blocks,
            PADDED_BONUS_BLOCKS,
            TP_SIZE,
            PADDED_TP_SIZE,
        )
        tl.store(sampled_ptr + req * OUTPUT_WIDTH + num_draft, bonus)

    # Same accounting as get_num_sampled_and_rejected.
    prefilling = tl.load(seq_lens_ptr + req) < prefill_len
    sampled = tl.where(prefilling, 0, accepted_length + 1)
    rejected = tl.where(prefilling, 0, end - start - sampled)
    tl.store(num_sampled_ptr + req, sampled)
    tl.store(num_rejected_ptr + req, rejected)


def local_greedy_stats(
    logits: torch.Tensor,
    batch: InputBatch,
    vocab_start: int,
    vocab_size: int,
    num_speculative_steps: int,
) -> torch.Tensor:
    """Reduce this rank's vocabulary shard to per-block ``(max, id)`` pairs.

    Returns ``[num_logits, cdiv(vocab_size, 1024), 2]`` fp32. Draft rows fill
    the first ``cdiv(vocab_size, 8192)`` blocks; the last row of each request
    fills all of its 1024-wide bonus blocks. Blocks outside the shard hold
    ``-inf``.
    """
    if logits.ndim != 2 or logits.stride(1) != 1:
        raise ValueError("Expected 2D logits with a contiguous vocabulary dimension")
    if not 0 < vocab_size < _MAX_VOCAB_SIZE:
        raise ValueError(f"vocab_size must be in (0, {_MAX_VOCAB_SIZE})")
    num_logits, local_width = logits.shape
    stats_stride = triton.cdiv(vocab_size, _BONUS_BLOCK)
    stats = logits.new_empty((num_logits, stats_stride, 2), dtype=torch.float32)
    _local_greedy_stats_kernel[
        (batch.num_reqs, num_speculative_steps + 1, triton.cdiv(vocab_size, _BLOCK))
    ](
        logits,
        logits.stride(0),
        stats,
        batch.cu_num_logits,
        stats_stride,
        local_width,
        vocab_start,
        vocab_size,
        BLOCK=_BLOCK,
        BONUS_BLOCK=_BONUS_BLOCK,
    )
    return stats


def verify_gathered_greedy_stats(
    gathered: torch.Tensor,
    batch: InputBatch,
    seeds: torch.Tensor,
    prefill_len: torch.Tensor,
    vocab_size: int,
    num_speculative_steps: int,
    synthetic_conditional_rates: torch.Tensor | None,
    tp_size: int,
) -> SamplerOutput:
    """Greedy rejection sampling from rank-major gathered statistics."""
    num_logits = gathered.shape[0] // tp_size
    sampled = gathered.new_empty(
        (batch.num_reqs, num_speculative_steps + 1), dtype=torch.int64
    )
    num_sampled = gathered.new_empty(batch.num_reqs, dtype=torch.int32)
    num_rejected = torch.empty_like(num_sampled)
    num_blocks = triton.cdiv(vocab_size, _BLOCK)
    num_bonus_blocks = triton.cdiv(vocab_size, _BONUS_BLOCK)
    _sharded_greedy_verify_kernel[(batch.num_reqs,)](
        gathered,
        batch.input_ids,
        batch.positions,
        batch.logits_indices,
        batch.expanded_local_pos,
        batch.cu_num_logits,
        batch.idx_mapping,
        batch.seq_lens,
        prefill_len,
        seeds,
        synthetic_conditional_rates,
        sampled,
        num_sampled,
        num_rejected,
        num_logits,
        num_blocks,
        num_bonus_blocks,
        OUTPUT_WIDTH=num_speculative_steps + 1,
        PADDED_BLOCKS=triton.next_power_of_2(num_blocks),
        PADDED_BONUS_BLOCKS=triton.next_power_of_2(num_bonus_blocks),
        SYNTHETIC=synthetic_conditional_rates is not None,
        TP_SIZE=tp_size,
        PADDED_TP_SIZE=triton.next_power_of_2(tp_size),
        num_warps=4,
    )
    return SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
    )


class ShardedGreedyVerifier:
    """Verify all-greedy batches from compact vocab-parallel statistics.

    ``try_sample`` returns None, before any logits are computed, whenever the
    batch needs anything beyond the target argmax, so the caller falls back to
    the native path.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        rejection_sampler: RejectionSampler,
        vocab_size: int,
    ):
        self.model = model
        self.rejection_sampler = rejection_sampler
        self.vocab_size = vocab_size
        self.tp_group = get_tp_group()

    def can_verify(self, batch: InputBatch, draft_logits: torch.Tensor | None) -> bool:
        rejection_sampler = self.rejection_sampler
        sampler = rejection_sampler.sampler
        if (
            batch.num_reqs == 0
            or batch.num_draft_tokens == 0
            or draft_logits is not None
            or batch.has_structured_output_reqs
            or rejection_sampler.enable_adaptive_verification
            or rejection_sampler.use_block_verification
            or rejection_sampler.watermark_key is not None
            or sampler.compute_nans
            or sampler.return_sampling_mask
            or sampler.trace_replay_state is not None
        ):
            return False
        slots = batch.idx_mapping_np
        return not (
            sampler.needs_logits_processing[slots].any()
            or (sampler.sampling_states.temperature.np[slots] != 0).any()
            or sampler.get_logprobs_dims(slots) is not None
            # The stats grid covers num_speculative_steps + 1 rows per request.
            or np.diff(batch.cu_num_logits_np).max()
            > rejection_sampler.num_speculative_steps + 1
        )

    def try_sample(
        self,
        hidden_states: torch.Tensor,
        batch: InputBatch,
        grammar_output,
        draft_logits: torch.Tensor | None,
    ) -> SamplerOutput | None:
        if grammar_output is not None or not self.can_verify(batch, draft_logits):
            return None
        logits = self.model.compute_logits_local(hidden_states[batch.logits_indices])
        group = self.tp_group
        stats = local_greedy_stats(
            logits,
            batch,
            group.rank_in_group * logits.shape[1],
            self.vocab_size,
            self.rejection_sampler.num_speculative_steps,
        )
        # dim=0 keeps the rank-major layout the verify kernel indexes.
        gathered = group.all_gather(stats, dim=0)
        sampler = self.rejection_sampler.sampler
        return verify_gathered_greedy_stats(
            gathered,
            batch,
            sampler.sampling_states.seeds.gpu,
            sampler.req_states.prefill_len.gpu,
            self.vocab_size,
            self.rejection_sampler.num_speculative_steps,
            self.rejection_sampler.synthetic_conditional_rates,
            group.world_size,
        )


def create_sharded_greedy_verifier(
    model: torch.nn.Module,
    rejection_sampler: RejectionSampler | None,
    vllm_config: VllmConfig,
    vocab_size: int,
) -> ShardedGreedyVerifier | None:
    if not envs.VLLM_SHARDED_GREEDY_VERIFY or rejection_sampler is None:
        return None
    reason = None
    if get_tp_group().world_size == 1:
        reason = "tensor_parallel_size is 1"
    elif not hasattr(model, "compute_logits_local"):
        reason = f"{type(model).__name__} does not implement compute_logits_local"
    elif type(rejection_sampler) is not RejectionSampler or (
        type(rejection_sampler.sampler) is not Sampler
    ):
        reason = "a custom sampler is in use"
    elif not 0 < vocab_size < _MAX_VOCAB_SIZE:
        reason = f"vocab_size {vocab_size} is out of range"
    elif vllm_config.lora_config is not None:
        reason = "LoRA may replace the LM head"
    if reason is not None:
        logger.warning_once("Disabling sharded greedy verification: %s", reason)
        return None
    logger.info_once("Sharded greedy verification enabled.")
    return ShardedGreedyVerifier(model, rejection_sampler, vocab_size)
