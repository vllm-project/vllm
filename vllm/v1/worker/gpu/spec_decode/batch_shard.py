# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.distributed

from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_tp_group
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput


@dataclass
class BatchShardMetadata:
    """Collective layout for one sampling step.

    Requests (and their logits) are owner-sorted: all requests owned by
    rank 0 first, then rank 1, etc. Within an owner, batch order is
    preserved (stable sort). Every field is derived from replicated CPU
    arrays, so all ranks build identical plans without communication —
    which relies on request slots (idx_mapping) being assigned identically
    on every rank (see finish_requests in the model runner).
    """

    tp_size: int
    # Per-owner-rank logits counts (all-to-all send splits).
    num_logits_per_rank: list[int]
    num_local_logits: int
    num_local_reqs: int
    # The number of request entries contributed by each rank to the
    # gathered result (shorter shards are padded up to this value).
    max_num_reqs_per_rank: int
    # [num_reqs] For batch request i, its source index in the rank-major
    # gathered results. Indexing the gathered tensors with this restores
    # the original batch order.
    gathered_src_indices: torch.Tensor
    # The number of entries occupied by each request in the gathered
    # tensor. Derived from the global cu_num_logits, so each rank agrees
    # ont he gather shapes, including empty shards.
    max_num_logits_per_req: int


def _layout(lengths: list[int], align_elems: int) -> tuple[list[int], int]:
    """Pack offsets for `lengths`, each aligned up to `align_elems`.

    Alignment keeps every unpacked view 16-byte aligned, so Triton's
    tt.divisibility specialization stays constant as the per-step lengths
    change; otherwise the samplers recompile mid-serving.
    """
    offsets = []
    total = 0
    for n in lengths:
        total = cdiv(total, align_elems) * align_elems
        offsets.append(total)
        total += n
    return offsets, total


class _StagingPool:
    """Round-robin pinned host buffers feeding one persistent device buffer.

    The pool depth covers max_concurrency in-flight steps, since a step's
    copy may still be running when the next step's arrays are written. The
    device buffer needs no rotation because its reuse is stream-ordered.
    """

    def __init__(
        self,
        capacity: int,
        dtype: torch.dtype,
        device: torch.device,
        depth: int,
    ):
        pin = device.type == "cuda"
        self._cpu = [
            torch.empty(capacity, dtype=dtype, pin_memory=pin) for _ in range(depth)
        ]
        self._np = [t.numpy() for t in self._cpu]
        self._gpu = torch.empty(capacity, dtype=dtype, device=device)
        self._align = 16 // self._gpu.element_size()
        self._idx = 0

    def pack_and_copy(self, arrays: list[np.ndarray]) -> list[torch.Tensor]:
        """Ship several index arrays to the GPU in a single H2D copy."""
        offsets, total = _layout([a.shape[0] for a in arrays], self._align)
        assert total <= self._gpu.shape[0]

        self._idx = (self._idx + 1) % len(self._cpu)
        buf = self._np[self._idx]
        for a, offset in zip(arrays, offsets):
            buf[offset : offset + a.shape[0]] = a

        packed = self._gpu[:total]
        packed.copy_(self._cpu[self._idx][:total], non_blocking=True)
        return [packed[o : o + a.shape[0]] for a, o in zip(arrays, offsets)]


class BatchSharder:
    """Owner-sorts the sampler inputs across TP ranks."""

    def __init__(
        self,
        max_num_reqs: int,
        max_num_logits_per_req: int,
        device: torch.device,
        max_concurrency: int = 2,
    ):
        tp_group = get_tp_group()
        self.tp_rank = tp_group.rank_in_group
        self.tp_size = tp_group.world_size
        self.max_num_reqs = max_num_reqs
        self.device = device

        max_num_logits = max_num_reqs * max_num_logits_per_req
        self._arange = np.arange(
            max(max_num_logits, max_num_reqs, self.tp_size), dtype=np.int32
        )

        # Worst-case packed layout of the arrays shipped by
        # shard_sampler_inputs (see the pack_and_copy calls there).
        depth = max(2, max_concurrency)
        _, cap64 = _layout(
            [max_num_logits, max_num_reqs, max_num_reqs, max_num_logits], 2
        )
        _, cap32 = _layout(
            [max_num_reqs, max_num_logits, max_num_logits, max_num_reqs + 1], 4
        )
        self._pool64 = _StagingPool(cap64, torch.int64, device, depth)
        self._pool32 = _StagingPool(cap32, torch.int32, device, depth)

    def shard_sampler_inputs(
        self,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[InputBatch, torch.Tensor, GrammarOutput | None, BatchShardMetadata]:
        """Owner-sort the batch and build this rank's local sampler inputs.

        Returns the local sub-batch, the owner-sorted logits_indices (gather
        the sampling hidden states with these so `compute_logits_local` emits
        logits in all-to-all send order), the local grammar output (None if
        this rank owns none of the structured-output requests), and the shard
        metadata (used for all-gathering the sampler outputs).
        """
        tp_rank = self.tp_rank
        tp_size = self.tp_size
        idx_mapping_np = input_batch.idx_mapping_np
        cu_num_logits_np = input_batch.cu_num_logits_np
        num_reqs = idx_mapping_np.shape[0]
        num_logits = int(cu_num_logits_np[-1])
        arange_np = self._arange

        # Deterministically assign requests to ranks, round-robin over slot
        # indices so ownership stays balanced when the slot allocator fills
        # low slots first (partial occupancy).
        req_owner_np = idx_mapping_np % tp_size
        sorted_req_indices_np = np.argsort(req_owner_np, kind="stable")
        num_reqs_per_rank_np = np.bincount(req_owner_np, minlength=tp_size)
        cu_num_reqs_np = np.zeros(tp_size + 1, dtype=np.int64)
        np.cumsum(num_reqs_per_rank_np, out=cu_num_reqs_np[1:])
        req_start = int(cu_num_reqs_np[tp_rank])
        req_end = int(cu_num_reqs_np[tp_rank + 1])
        num_local_reqs = req_end - req_start
        local_req_indices_np = sorted_req_indices_np[req_start:req_end].astype(
            np.int64, copy=False
        )

        # Derive the number of logits owned by each rank, as well as
        # the local rank.
        num_logits_per_req_np = np.diff(cu_num_logits_np)
        sorted_num_logits_per_req_np = num_logits_per_req_np[sorted_req_indices_np]
        sorted_cu_num_logits_np = np.zeros(num_reqs + 1, dtype=np.int64)
        np.cumsum(sorted_num_logits_per_req_np, out=sorted_cu_num_logits_np[1:])
        cu_num_logits_per_rank_np = sorted_cu_num_logits_np[cu_num_reqs_np]
        num_local_logits = int(
            cu_num_logits_per_rank_np[tp_rank + 1] - cu_num_logits_per_rank_np[tp_rank]
        )

        # Derive the logits indices owned by the local rank.
        sorted_logits_starts_np = (
            input_batch.query_start_loc_np[1 : num_reqs + 1].astype(np.int64)
            - num_logits_per_req_np
        )[sorted_req_indices_np]
        sorted_logits_indices_np = arange_np[:num_logits] + np.repeat(
            sorted_logits_starts_np - sorted_cu_num_logits_np[:-1],
            sorted_num_logits_per_req_np,
        )
        local_logits_start = int(cu_num_logits_per_rank_np[tp_rank])
        local_logits_indices_np = sorted_logits_indices_np[
            local_logits_start : local_logits_start + num_local_logits
        ]

        # Derive the mapping of output tensor request indices => gathered
        # tensor request indices.
        # Batch request i's result lives in the padded, gathered results at
        # rank_of(i) * max_num_reqs_per_rank + offset_within_rank(i).
        max_num_reqs_per_rank = int(num_reqs_per_rank_np.max())
        pos_in_sorted_np = np.empty(num_reqs, dtype=np.int64)
        pos_in_sorted_np[sorted_req_indices_np] = arange_np[:num_reqs]
        owner_of_sorted_np = np.repeat(arange_np[:tp_size], num_reqs_per_rank_np)
        src_index_sorted_np = (
            owner_of_sorted_np * max_num_reqs_per_rank
            + arange_np[:num_reqs]
            - cu_num_reqs_np[owner_of_sorted_np]
        )
        gathered_src_indices_np = src_index_sorted_np[pos_in_sorted_np]

        # Calculate the sub-batch CPU arrays needed for sampling.
        local_num_logits_per_req_np = sorted_num_logits_per_req_np[req_start:req_end]
        local_cu_num_logits_np = (
            sorted_cu_num_logits_np[req_start : req_end + 1]
            - sorted_cu_num_logits_np[req_start]
        ).astype(np.int32)
        local_idx_mapping_np = idx_mapping_np[local_req_indices_np]
        local_expanded_idx_mapping_np = np.repeat(
            local_idx_mapping_np, local_num_logits_per_req_np
        )
        local_expanded_local_pos_np = arange_np[:num_local_logits] - np.repeat(
            local_cu_num_logits_np[:-1], local_num_logits_per_req_np
        )

        # Asynchronously copy the sub-batch CPU arrays => GPU tensors.
        (
            sorted_logits_indices,
            gathered_src_indices,
            local_req_indices,
            local_logits_indices,
        ) = self._pool64.pack_and_copy(
            [
                sorted_logits_indices_np,
                gathered_src_indices_np,
                local_req_indices_np,
                local_logits_indices_np,
            ]
        )
        (
            local_idx_mapping,
            local_expanded_idx_mapping,
            local_expanded_local_pos,
            local_cu_num_logits,
        ) = self._pool32.pack_and_copy(
            [
                local_idx_mapping_np,
                local_expanded_idx_mapping_np,
                local_expanded_local_pos_np,
                local_cu_num_logits_np,
            ]
        )
        # Under async scheduling, the exact sequence length values are only
        # tracked via GPU state. They must be gathered here on the GPU using
        # local_req_indices.
        local_seq_lens = input_batch.seq_lens[local_req_indices]

        # Compute the local number of draft tokens.
        num_draft_tokens_per_req = None
        num_draft_tokens = 0
        if input_batch.num_draft_tokens_per_req is not None:
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req[
                local_req_indices_np
            ]
            num_draft_tokens = int(num_draft_tokens_per_req.sum())

        req_ids = input_batch.req_ids
        local_req_ids = [req_ids[i] for i in local_req_indices_np.tolist()]
        local_batch = replace(
            input_batch,
            req_ids=local_req_ids,
            num_reqs=num_local_reqs,
            idx_mapping=local_idx_mapping,
            idx_mapping_np=local_idx_mapping_np,
            expanded_idx_mapping=local_expanded_idx_mapping,
            expanded_local_pos=local_expanded_local_pos,
            seq_lens=local_seq_lens,
            logits_indices=local_logits_indices,
            cu_num_logits=local_cu_num_logits,
            cu_num_logits_np=local_cu_num_logits_np,
            num_draft_tokens=num_draft_tokens,
            num_draft_tokens_per_req=num_draft_tokens_per_req,
        )
        local_grammar_output = None
        if grammar_output is not None:
            local_grammar_output = _shard_grammar_output(
                grammar_output, input_batch, local_req_ids
            )
        metadata = BatchShardMetadata(
            tp_size=tp_size,
            num_logits_per_rank=np.diff(cu_num_logits_per_rank_np).tolist(),
            num_local_logits=num_local_logits,
            num_local_reqs=num_local_reqs,
            max_num_reqs_per_rank=max_num_reqs_per_rank,
            gathered_src_indices=gathered_src_indices,
            max_num_logits_per_req=int(num_logits_per_req_np.max())
            if num_reqs > 0
            else 1,
        )
        return local_batch, sorted_logits_indices, local_grammar_output, metadata


def all_to_all_logits(
    logits_shard: torch.Tensor,
    metadata: BatchShardMetadata,
) -> torch.Tensor:
    recv = logits_shard.new_empty(
        metadata.tp_size * metadata.num_local_logits, logits_shard.shape[-1]
    )
    torch.distributed.all_to_all_single(
        recv,
        logits_shard.contiguous(),
        output_split_sizes=[metadata.num_local_logits] * metadata.tp_size,
        input_split_sizes=metadata.num_logits_per_rank,
        group=get_tp_group().device_group,
    )
    shard_width = recv.shape[-1]
    return (
        recv.view(metadata.tp_size, metadata.num_local_logits, shard_width)
        .permute(1, 0, 2)
        .reshape(metadata.num_local_logits, metadata.tp_size * shard_width)
    )


def _shard_grammar_output(
    grammar_output: GrammarOutput,
    input_batch: InputBatch,
    local_req_ids: list[str],
) -> GrammarOutput | None:
    owned = set(local_req_ids)
    req_id_to_idx = {req_id: i for i, req_id in enumerate(input_batch.req_ids)}
    cu_num_logits_np = input_batch.cu_num_logits_np

    local_ids: list[str] = []
    keep_indices: list[int] = []
    cursor = 0
    for req_id in grammar_output.structured_output_request_ids:
        req_idx = req_id_to_idx[req_id]
        num_req_logits = int(cu_num_logits_np[req_idx + 1] - cu_num_logits_np[req_idx])
        if req_id in owned:
            local_ids.append(req_id)
            keep_indices.extend(range(cursor, cursor + num_req_logits))
        cursor += num_req_logits
    if not local_ids:
        return None
    return GrammarOutput(
        structured_output_request_ids=local_ids,
        grammar_bitmask=grammar_output.grammar_bitmask[keep_indices],
    )


@triton.jit(
    do_not_specialize=["max_num_logits_per_req", "num_src_cols", "packed_stride"]
)
def _pack_sampler_output_kernel(
    packed_ptr,
    packed_stride,
    sampled_token_ids_ptr,
    sampled_token_ids_stride,
    num_sampled_ptr,
    num_rejected_ptr,
    num_nans_ptr,
    local_cu_num_logits_ptr,
    max_num_logits_per_req,
    num_src_cols,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_ptr = packed_ptr + req_idx * packed_stride

    token_ids = tl.load(
        sampled_token_ids_ptr + req_idx * sampled_token_ids_stride + cols,
        mask=cols < num_src_cols,
        other=0,
    )
    tl.store(row_ptr + cols, token_ids, mask=cols < max_num_logits_per_req)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    tl.store(row_ptr + max_num_logits_per_req, num_sampled.to(tl.int64))
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    tl.store(row_ptr + max_num_logits_per_req + 1, num_rejected.to(tl.int64))
    if num_nans_ptr is not None:
        # num_nans is per logits row; reduce it per request. The replicated
        # samplers report per-row counts that downstream zips per request,
        # so a per-request sum is the well-defined equivalent.
        start = tl.load(local_cu_num_logits_ptr + req_idx)
        num_req_logits = tl.load(local_cu_num_logits_ptr + req_idx + 1) - start
        nans = tl.load(num_nans_ptr + start + cols, mask=cols < num_req_logits, other=0)
        tl.store(
            row_ptr + max_num_logits_per_req + 2, tl.sum(nans.to(tl.int64), axis=0)
        )


@triton.jit(do_not_specialize=["max_num_logits_per_req", "gathered_stride"])
def _unpack_gathered_output_kernel(
    gathered_ptr,
    gathered_stride,
    gathered_src_indices_ptr,
    sampled_token_ids_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    num_nans_ptr,
    max_num_logits_per_req,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    src = tl.load(gathered_src_indices_ptr + req_idx)
    row_ptr = gathered_ptr + src * gathered_stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < max_num_logits_per_req

    token_ids = tl.load(row_ptr + cols, mask=mask)
    tl.store(
        sampled_token_ids_ptr + req_idx * max_num_logits_per_req + cols,
        token_ids,
        mask=mask,
    )
    num_sampled = tl.load(row_ptr + max_num_logits_per_req)
    tl.store(num_sampled_ptr + req_idx, num_sampled.to(tl.int32))
    num_rejected = tl.load(row_ptr + max_num_logits_per_req + 1)
    tl.store(num_rejected_ptr + req_idx, num_rejected.to(tl.int32))
    if num_nans_ptr is not None:
        num_nans = tl.load(row_ptr + max_num_logits_per_req + 2)
        tl.store(num_nans_ptr + req_idx, num_nans.to(tl.int32))


@triton.jit(
    do_not_specialize=[
        "max_num_logits_per_req",
        "num_logprob_cols",
        "num_src_cols",
        "send_ids_stride",
        "send_logprobs_stride",
        "src_ids_stride",
        "src_logprobs_stride",
    ]
)
def _pack_logprobs_kernel(
    send_ids_ptr,
    send_ids_stride,
    send_logprobs_ptr,
    send_logprobs_stride,
    src_ids_ptr,
    src_ids_stride,
    src_logprobs_ptr,
    src_logprobs_stride,
    selected_token_ranks_ptr,
    local_cu_num_logits_ptr,
    max_num_logits_per_req,
    num_logprob_cols,
    num_src_cols,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    pos = tl.program_id(1)
    start = tl.load(local_cu_num_logits_ptr + req_idx)
    if pos < tl.load(local_cu_num_logits_ptr + req_idx + 1) - start:
        src = start + pos
        dst = req_idx * max_num_logits_per_req + pos
        cols = tl.arange(0, BLOCK_SIZE)
        # A rank whose sub-batch has no draft tokens runs the regular sampler
        # while others run the rejection sampler, and their column counts can
        # differ (e.g. logprob_token_ids widening); the two masks pad or
        # truncate to the agreed width.
        src_mask = cols < num_src_cols
        dst_mask = cols < num_logprob_cols
        ids = tl.load(src_ids_ptr + src * src_ids_stride + cols, mask=src_mask, other=0)
        logprobs = tl.load(
            src_logprobs_ptr + src * src_logprobs_stride + cols,
            mask=src_mask,
            other=float("-inf"),
        )
        ids_row = send_ids_ptr + dst * send_ids_stride
        tl.store(ids_row + cols, ids.to(tl.int64), mask=dst_mask)
        rank = tl.load(selected_token_ranks_ptr + src)
        tl.store(ids_row + num_logprob_cols, rank.to(tl.int64))
        tl.store(
            send_logprobs_ptr + dst * send_logprobs_stride + cols,
            logprobs.to(tl.float32),
            mask=dst_mask,
        )


@triton.jit(
    do_not_specialize=[
        "max_num_logits_per_req",
        "num_logprob_cols",
        "gathered_ids_stride",
        "gathered_logprobs_stride",
    ]
)
def _unpack_logprobs_kernel(
    gathered_ids_ptr,
    gathered_ids_stride,
    gathered_logprobs_ptr,
    gathered_logprobs_stride,
    gathered_src_indices_ptr,
    cu_num_logits_ptr,
    logprob_token_ids_ptr,
    logprobs_ptr,
    selected_token_ranks_ptr,
    max_num_logits_per_req,
    num_logprob_cols,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    pos = tl.program_id(1)
    start = tl.load(cu_num_logits_ptr + req_idx)
    if pos < tl.load(cu_num_logits_ptr + req_idx + 1) - start:
        src = tl.load(gathered_src_indices_ptr + req_idx) * max_num_logits_per_req + pos
        dst = start + pos
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < num_logprob_cols
        src_ids_row = gathered_ids_ptr + src * gathered_ids_stride
        ids = tl.load(src_ids_row + cols, mask=mask)
        tl.store(logprob_token_ids_ptr + dst * num_logprob_cols + cols, ids, mask=mask)
        rank = tl.load(src_ids_row + num_logprob_cols)
        tl.store(selected_token_ranks_ptr + dst, rank)
        logprobs = tl.load(
            gathered_logprobs_ptr + src * gathered_logprobs_stride + cols, mask=mask
        )
        tl.store(logprobs_ptr + dst * num_logprob_cols + cols, logprobs, mask=mask)


def _gather_logprobs_tensors(
    local_output: SamplerOutput | None,
    metadata: BatchShardMetadata,
    global_batch: InputBatch,
    local_batch: InputBatch,
    logprobs_dims: tuple[int, int],
    device: torch.device,
) -> LogprobsTensors:
    num_logprobs, max_token_ids = logprobs_dims
    num_logprob_cols = 1 + max(num_logprobs, max_token_ids)
    max_num_logits_per_req = metadata.max_num_logits_per_req
    num_send_rows = metadata.max_num_reqs_per_rank * max_num_logits_per_req
    block_size = triton.next_power_of_2(num_logprob_cols)

    # Padded slots are never read back: gathered_src_indices only addresses
    # rows that an owner rank wrote.
    send_ids_ranks = torch.empty(
        num_send_rows, num_logprob_cols + 1, dtype=torch.int64, device=device
    )
    send_logprobs = torch.empty(
        num_send_rows, num_logprob_cols, dtype=torch.float32, device=device
    )
    if local_output is not None and local_output.logprobs_tensors is not None:
        lp = local_output.logprobs_tensors
        _pack_logprobs_kernel[(metadata.num_local_reqs, max_num_logits_per_req)](
            send_ids_ranks,
            send_ids_ranks.stride(0),
            send_logprobs,
            send_logprobs.stride(0),
            lp.logprob_token_ids,
            lp.logprob_token_ids.stride(0),
            lp.logprobs,
            lp.logprobs.stride(0),
            lp.selected_token_ranks,
            local_batch.cu_num_logits,
            max_num_logits_per_req,
            num_logprob_cols,
            lp.logprob_token_ids.shape[1],
            BLOCK_SIZE=block_size,
        )

    gathered_ids_ranks = tensor_model_parallel_all_gather(send_ids_ranks, dim=0)
    gathered_logprobs = tensor_model_parallel_all_gather(send_logprobs, dim=0)

    num_reqs = global_batch.num_reqs
    num_logits = int(global_batch.cu_num_logits_np[-1])
    logprob_token_ids = torch.empty(
        num_logits, num_logprob_cols, dtype=torch.int64, device=device
    )
    logprobs = torch.empty(
        num_logits, num_logprob_cols, dtype=torch.float32, device=device
    )
    selected_token_ranks = torch.empty(num_logits, dtype=torch.int64, device=device)
    _unpack_logprobs_kernel[(num_reqs, max_num_logits_per_req)](
        gathered_ids_ranks,
        gathered_ids_ranks.stride(0),
        gathered_logprobs,
        gathered_logprobs.stride(0),
        metadata.gathered_src_indices,
        global_batch.cu_num_logits,
        logprob_token_ids,
        logprobs,
        selected_token_ranks,
        max_num_logits_per_req,
        num_logprob_cols,
        BLOCK_SIZE=block_size,
    )
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=selected_token_ranks,
        cu_num_generated_tokens=(
            global_batch.cu_num_logits_np.tolist() if num_logits != num_reqs else None
        ),
    )


def gather_sampler_output(
    local_output: SamplerOutput | None,
    metadata: BatchShardMetadata,
    device: torch.device,
    global_batch: InputBatch,
    local_batch: InputBatch,
    gather_num_nans: bool = False,
    logprobs_dims: tuple[int, int] | None = None,
) -> SamplerOutput:
    max_num_logits_per_req = metadata.max_num_logits_per_req
    num_packed_cols = max_num_logits_per_req + 2 + (1 if gather_num_nans else 0)
    block_size = triton.next_power_of_2(max_num_logits_per_req)

    # Pack the sampler output tensors (excluding logprobs) into a single
    # tensor for the all-gather.
    packed = torch.empty(
        metadata.max_num_reqs_per_rank,
        num_packed_cols,
        dtype=torch.int64,
        device=device,
    )
    if local_output is not None:
        assert local_output.num_sampled is not None
        assert local_output.num_rejected is not None
        assert not gather_num_nans or local_output.num_nans is not None
        num_src_cols = min(
            local_output.sampled_token_ids.shape[1], max_num_logits_per_req
        )
        _pack_sampler_output_kernel[(metadata.num_local_reqs,)](
            packed,
            packed.stride(0),
            local_output.sampled_token_ids,
            local_output.sampled_token_ids.stride(0),
            local_output.num_sampled,
            local_output.num_rejected,
            local_output.num_nans if gather_num_nans else None,
            local_batch.cu_num_logits if gather_num_nans else None,
            max_num_logits_per_req,
            num_src_cols,
            BLOCK_SIZE=block_size,
        )

    # All-gather the sampler output tensors (excluding logprobs).
    gathered = tensor_model_parallel_all_gather(packed, dim=0)
    # All-gather the logprobs tensors.
    logprobs_tensors = None
    if logprobs_dims is not None:
        logprobs_tensors = _gather_logprobs_tensors(
            local_output, metadata, global_batch, local_batch, logprobs_dims, device
        )

    # Unpack the gathered tensor into a sampler output object.
    num_reqs = metadata.gathered_src_indices.shape[0]
    sampled_token_ids = torch.empty(
        num_reqs, max_num_logits_per_req, dtype=torch.int64, device=device
    )
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=device)
    num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=device)
    num_nans = (
        torch.empty(num_reqs, dtype=torch.int32, device=device)
        if gather_num_nans
        else None
    )
    _unpack_gathered_output_kernel[(num_reqs,)](
        gathered,
        gathered.stride(0),
        metadata.gathered_src_indices,
        sampled_token_ids,
        num_sampled,
        num_rejected,
        num_nans,
        max_num_logits_per_req,
        BLOCK_SIZE=block_size,
    )
    return SamplerOutput(
        sampled_token_ids=sampled_token_ids,
        logprobs_tensors=logprobs_tensors,
        num_nans=num_nans,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
    )
