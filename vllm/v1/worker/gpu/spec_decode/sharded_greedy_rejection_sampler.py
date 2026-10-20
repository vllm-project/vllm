# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy verification with compact vocabulary-parallel statistics."""

import torch

from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.greedy_rejection_sampler import (
    _greedy_verify_and_count,
)

logger = init_logger(__name__)


@triton.jit
def _sharded_greedy_target_stats(
    logits_ptr,
    logits_stride,
    packed_ptr,
    cu_num_logits_ptr,
    stats_stride,
    LOCAL_WIDTH: tl.constexpr,
    VOCAB_START: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    BONUS_BLOCK: tl.constexpr,
):
    req, step, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row = tl.load(cu_num_logits_ptr + req).to(tl.int64) + step
    end = tl.load(cu_num_logits_ptr + req + 1)
    if row < end:
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        local_offsets = offsets - VOCAB_START
        valid = (
            (local_offsets >= 0)
            & (local_offsets < LOCAL_WIDTH)
            & (offsets < VOCAB_SIZE)
        )
        values = tl.load(
            logits_ptr + row * logits_stride + local_offsets,
            mask=valid,
            other=float("-inf"),
        ).to(tl.float32)
        if row == end - 1:
            bonus_values = values.reshape(BLOCK // BONUS_BLOCK, BONUS_BLOCK)
            bonus_max, bonus_index = tl.max(bonus_values, axis=1, return_indices=True)
            subblock = block * (BLOCK // BONUS_BLOCK) + tl.arange(
                0, BLOCK // BONUS_BLOCK
            )
            mask = subblock < stats_stride
            bonus_dst = (row * stats_stride + subblock) * 2
            tl.store(packed_ptr + bonus_dst, bonus_max, mask=mask)
            tl.store(
                packed_ptr + bonus_dst + 1,
                (subblock * BONUS_BLOCK + bonus_index).to(tl.float32),
                mask=mask,
            )
        else:
            value, index = tl.max(values, axis=0, return_indices=True)
            dst = (row * stats_stride + block) * 2
            tl.store(packed_ptr + dst, value)
            tl.store(packed_ptr + dst + 1, (block * BLOCK + index).to(tl.float32))


def local_greedy_stats(
    logits: torch.Tensor,
    batch: InputBatch,
    vocab_start: int,
    vocab_size: int,
    num_speculative_steps: int,
) -> torch.Tensor:
    """Keep native global block boundaries, including those crossing TP shards."""
    rows, width = logits.shape
    if width < 8192 or not 0 < vocab_size < 2**24:
        raise ValueError("Require a complete first native block and exact FP32 IDs")
    stats_stride = triton.cdiv(vocab_size, 1024)
    packed = torch.empty(
        (rows, stats_stride, 2), device=logits.device, dtype=torch.float32
    )
    _sharded_greedy_target_stats[
        (batch.num_reqs, num_speculative_steps + 1, triton.cdiv(vocab_size, 8192))
    ](
        logits,
        logits.stride(0),
        packed,
        batch.cu_num_logits,
        stats_stride,
        width,
        vocab_start,
        vocab_size,
        8192,
        1024,
    )
    return packed


def verify_gathered_greedy_stats(
    gathered: torch.Tensor,
    batch: InputBatch,
    seeds: torch.Tensor,
    prefill_len: torch.Tensor,
    vocab_size: int,
    num_speculative_steps: int,
    rates: torch.Tensor | None,
    tp_size: int,
) -> SamplerOutput:
    """Fuse TP reduction, acceptance, bonus selection and output count updates."""
    if tp_size not in (2, 4, 8):
        raise ValueError("Packed statistics require TP2, TP4 or TP8")
    rows = gathered.shape[0] // tp_size
    sampled = torch.empty(
        (batch.num_reqs, num_speculative_steps + 1),
        dtype=torch.int64,
        device=gathered.device,
    )
    num_sampled = torch.empty(batch.num_reqs, dtype=torch.int32, device=gathered.device)
    num_rejected = torch.empty_like(num_sampled)
    blocks, bonus_blocks = triton.cdiv(vocab_size, 8192), triton.cdiv(vocab_size, 1024)
    _greedy_verify_and_count[(batch.num_reqs,)](
        gathered,
        None,
        batch.input_ids,
        batch.positions,
        batch.logits_indices,
        batch.expanded_local_pos,
        batch.cu_num_logits,
        batch.idx_mapping,
        batch.seq_lens,
        prefill_len,
        seeds,
        rates,
        sampled,
        num_sampled,
        num_rejected,
        blocks,
        bonus_blocks,
        num_speculative_steps + 1,
        triton.next_power_of_2(blocks),
        triton.next_power_of_2(bonus_blocks),
        rates is not None,
        num_logits=rows,
        TP_SIZE=tp_size,
        num_warps=4,
    )
    return SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
    )


def sharded_greedy_rejection_sample(
    logits: torch.Tensor,
    batch: InputBatch,
    seeds: torch.Tensor,
    prefill_len: torch.Tensor,
    vocab_start: int,
    vocab_size: int,
    num_speculative_steps: int,
    rates: torch.Tensor | None,
) -> SamplerOutput:
    """Exchange statistics rather than full logits, preserving greedy semantics."""
    group = get_tp_group()
    if group.world_size not in (2, 4, 8) or vocab_start != (
        group.rank_in_group * logits.shape[1]
    ):
        raise ValueError("Require equal, contiguous vocabulary shards in TP order")
    packed = local_greedy_stats(
        logits, batch, vocab_start, vocab_size, num_speculative_steps
    )
    gathered = group.all_gather(packed, dim=0)
    return verify_gathered_greedy_stats(
        gathered,
        batch,
        seeds,
        prefill_len,
        vocab_size,
        num_speculative_steps,
        rates,
        group.world_size,
    )


def can_use_sharded_greedy(rejection_sampler, batch: InputBatch) -> bool:
    """Use the nightly no-processor eligibility, retaining every native fallback."""
    from vllm.models.minimax_m3.amd.ops.n03_sampling import can_use_target

    return can_use_target(rejection_sampler, batch, Sampler)


class ShardedGreedyVerifier:
    """Optional sampler fast path for models exposing contiguous local logits."""

    def __init__(
        self, model, rejection_sampler, speculator, vocab_start: int, vocab_size: int
    ):
        self.model = model
        self.rejection_sampler = rejection_sampler
        self.speculator = speculator
        self.vocab_start = vocab_start
        self.vocab_size = vocab_size

    def try_sample(
        self, hidden_states: torch.Tensor, batch: InputBatch, grammar_output
    ) -> SamplerOutput | None:
        """Return None before computing logits when native sampling is required."""
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[1] != 6144
            or hidden_states.dtype != torch.bfloat16
            or grammar_output is not None
            or batch.has_structured_output_reqs
            or self.speculator.draft_logits is not None
            or not can_use_sharded_greedy(self.rejection_sampler, batch)
        ):
            return None
        hidden = hidden_states[batch.logits_indices]
        logits = self.model.logits_processor(
            self.model.lm_head, hidden, skip_gather=True
        )
        sampler = self.rejection_sampler.sampler
        return sharded_greedy_rejection_sample(
            logits,
            batch,
            sampler.sampling_states.seeds.gpu,
            sampler.req_states.prefill_len.gpu,
            self.vocab_start,
            self.vocab_size,
            self.rejection_sampler.num_speculative_steps,
            self.rejection_sampler.synthetic_conditional_rates,
        )


def create_sharded_greedy_verifier(
    model, rejection_sampler, speculator, vllm_config
) -> ShardedGreedyVerifier | None:
    """Use the verified nightly model's stock processor with only gather skipped."""
    from vllm.models.minimax_m3.amd.ops.n03_sampling import target_model

    inner = target_model(model, rejection_sampler, speculator, vllm_config)
    if inner is None:
        return None
    logger.info(
        "N03 target sampling enabled at GPUModelRunner.sample: "
        "stock BF16 head, native-block compact statistics, nightly prefill masking"
    )
    return ShardedGreedyVerifier(
        inner,
        rejection_sampler,
        speculator,
        inner.lm_head.shard_indices.org_vocab_start_index,
        inner.logits_processor.org_vocab_size,
    )
