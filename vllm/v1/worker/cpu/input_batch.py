# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU implementations for Model Runner V2 input-batch kernels."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.utils import random_uuid

if TYPE_CHECKING:
    from vllm.v1.worker.cpu.attn_utils import FastPrefillBatchMetadata


class InputBuffers:
    def __init__(self, max_num_reqs: int, max_num_tokens: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.device = device

        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.is_padding = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
        self.query_start_loc = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)


@dataclass
class InputBatch:
    # batch_idx -> req_id
    req_ids: list[str]
    num_reqs: int
    num_reqs_after_padding: int

    # batch_idx -> req_state_idx
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray
    # Identical to idx_mapping except for spec decoding.
    expanded_idx_mapping: torch.Tensor
    # [total_num_logits] position within request for each logit
    expanded_local_pos: torch.Tensor

    # [num_reqs]
    # batch_idx -> num_scheduled_tokens
    num_scheduled_tokens: np.ndarray
    # number of tokens in the batch
    num_tokens: int
    num_tokens_after_padding: int

    # [num_reqs + 1]
    query_start_loc: torch.Tensor
    query_start_loc_np: np.ndarray
    # [num_reqs]
    seq_lens: torch.Tensor
    # [num_reqs] CPU upper bound on seq_lens (see CommonAttentionMetadata).
    seq_lens_cpu_upper_bound: torch.Tensor
    # [num_reqs]
    num_computed_tokens_np: np.ndarray
    # [num_reqs]
    prefill_len_np: np.ndarray
    # [num_reqs]
    num_computed_prefill_tokens_np: np.ndarray
    # [num_reqs] CPU bool array == (num_computed_prefill_tokens_np < prefill_len_np).
    is_prefilling_np: np.ndarray
    # == np.any(is_prefilling_np)
    has_prefill: bool
    # No prefills, or only prefill rows that run as decodes.
    decode_graph_eligible: bool

    # [num_tokens_after_padding]
    input_ids: torch.Tensor
    # [num_tokens_after_padding]
    positions: torch.Tensor
    # [num_tokens_after_padding]
    is_padding: torch.Tensor

    # [total_num_logits]
    logits_indices: torch.Tensor
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor
    cu_num_logits_np: np.ndarray

    # Whether any requests in batch use structured output.
    has_structured_output_reqs: bool

    # [num_reqs] per-request prompt length, only populated for R-SWA.
    prompt_lens: torch.Tensor | None

    # [num_reqs] only populated for non-dummy pipeline-parallel batches.
    max_seq_len_np: np.ndarray | None = None

    # Longest query the batch may contain.
    max_query_len: int | None = None

    # Arms the KV-sharing fast prefill path for this step.
    fast_prefill: "FastPrefillBatchMetadata | None" = None

    # [num_reqs] prefilling rows that schedule one new prompt token over
    # existing context and so compute exactly like decodes.
    prefill_runs_as_decode_np: np.ndarray | None = None

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        input_buffers: InputBuffers,
        max_query_len: int | None = None,
        is_padding: bool = True,
    ) -> "InputBatch":
        assert 0 < num_reqs <= num_tokens
        device = input_buffers.device

        req_ids = [f"req_{i}_{random_uuid()}" for i in range(num_reqs)]
        idx_mapping_np = np.arange(num_reqs, dtype=np.intp)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int64, device=device)
        expanded_idx_mapping = idx_mapping
        expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=device)

        # Distribute the remainder evenly so that no dummy request exceeds
        # ceil(num_tokens / num_reqs) <= max_model_len tokens.
        base_tokens = num_tokens // num_reqs
        num_extra = num_tokens % num_reqs
        assert max_query_len is None or base_tokens + (num_extra > 0) <= max_query_len
        num_scheduled_tokens = np.full(num_reqs, base_tokens, dtype=np.int32)
        if num_extra > 0:
            num_scheduled_tokens[-num_extra:] += 1
        assert int(num_scheduled_tokens.sum()) == num_tokens

        # seq_len equals to query_len
        input_buffers.seq_lens[: num_reqs - num_extra] = base_tokens
        input_buffers.seq_lens[num_reqs - num_extra : num_reqs] = base_tokens + 1
        # Pad unused rows.
        input_buffers.seq_lens[num_reqs:] = 0
        seq_lens = input_buffers.seq_lens[:num_reqs]

        query_start_loc_np = np.empty(num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
        input_buffers.query_start_loc[:1] = 0
        torch.cumsum(
            seq_lens, dim=0, out=input_buffers.query_start_loc[1 : num_reqs + 1]
        )
        # Pad unused rows.
        input_buffers.query_start_loc[num_reqs + 1 :] = num_tokens
        query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]

        input_ids = input_buffers.input_ids[:num_tokens].zero_()
        positions = input_buffers.positions[:num_tokens].zero_()

        input_buffers.is_padding[:num_tokens].fill_(is_padding)
        is_padding = input_buffers.is_padding[:num_tokens]

        logits_indices = query_start_loc[1:] - 1
        cu_num_logits = torch.arange(num_reqs + 1, device=device, dtype=torch.int32)
        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
        # Copy so dummy context can be added without touching
        # num_scheduled_tokens.
        seq_lens_cpu_upper_bound = torch.from_numpy(num_scheduled_tokens.copy())
        return cls(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
            num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
            has_prefill=False,
            max_seq_len_np=None,
            decode_graph_eligible=True,
            input_ids=input_ids,
            positions=positions,
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=False,
            prompt_lens=None,
            max_query_len=max_query_len,
        )


class BatchReqState(NamedTuple):
    """CPU request state for a scheduled batch, in batch (sorted) order."""

    req_ids: list[str]
    num_scheduled_tokens: np.ndarray  # [num_reqs]
    num_tokens: int
    idx_mapping_np: np.ndarray  # [num_reqs]
    prefill_len_np: np.ndarray  # [num_reqs]
    num_computed_prefill_tokens_np: np.ndarray  # [num_reqs]
    is_prefilling_np: np.ndarray  # [num_reqs]
    max_seq_len_np: np.ndarray | None  # [num_reqs], None in non-pp case
    has_prefill: bool
    prefill_runs_as_decode_np: np.ndarray | None  # [num_reqs]
    decode_graph_eligible: bool


def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    ops.prepare_prefill_inputs_cpu(
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_len,
        num_computed_tokens,
    )


def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    ops.prepare_pos_seq_lens_cpu(
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        pos,
        seq_lens,
    )


def combine_sampled_and_draft_tokens(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_logits: int,
    num_new_sampled_tokens: int = 1,  # excl accepted draft tokens, a.k.a bonus tokens
) -> torch.Tensor:
    assert num_new_sampled_tokens in (0, 1), (
        f"num_new_sampled_tokens must be 0 or 1, got {num_new_sampled_tokens}"
    )
    return ops.combine_sampled_and_draft_tokens_cpu(
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        cu_num_logits,
        num_logits,
        num_new_sampled_tokens,
    )


def get_num_sampled_and_rejected(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = idx_mapping.shape[0]
    num_rejected = torch.empty_like(num_sampled)
    is_chunked_prefilling = seq_lens[:num_reqs] < prefill_len[idx_mapping]
    num_sampled.masked_fill_(is_chunked_prefilling, 0)
    num_rejected.copy_(cu_num_logits[1 : num_reqs + 1] - cu_num_logits[:num_reqs])
    num_rejected.sub_(num_sampled)
    num_rejected.masked_fill_(is_chunked_prefilling, 0)
    return num_sampled, num_rejected


def post_update(
    # [num_reqs] batch_idx -> req_state_idx; negative index means skip.
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [max_num_reqs]
    last_sampled_tokens: torch.Tensor,
    # [max_num_reqs, vocab_size]
    output_bin_counts: torch.Tensor | None,
    # [num_reqs, num_speculative_steps + 1]
    sampled_tokens: torch.Tensor,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor | None,
    # [max_num_reqs, max_model_len]
    all_token_ids: torch.Tensor,
    # [max_num_reqs]
    total_len: torch.Tensor,
) -> None:
    ops.post_update_cpu(
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        total_len,
    )


def post_update_num_computed_tokens(
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
) -> None:
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_computed_tokens.index_add_(0, idx_mapping, query_lens)
