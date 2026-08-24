# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing-math tests for batch-sharded sampling.

These simulate every TP rank in-process (no torch.distributed): the local
sub-batch, the all-to-all send/recv layout, and the padded result gather must
compose to a bit-exact round trip of the replicated computation.
"""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample import batch_shard
from vllm.v1.worker.gpu.sample.batch_shard import (
    BatchSharder,
    _shard_grammar_output,
)

DEVICE = "cuda"

# The owner-sorted layout is expanded by a Triton kernel, so the routing math
# can only be exercised on device.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _np(t: torch.Tensor) -> np.ndarray:
    return t.cpu().numpy()


def _make_batch(
    rng: np.random.Generator,
    num_reqs: int,
    max_num_reqs: int,
    max_spec: int,
    slot_pool: np.ndarray | None = None,
) -> InputBatch:
    """Consistent batch with random unique request slots, 1 + k_i logits
    rows per request at the tail of each query segment."""
    if slot_pool is None:
        slot_pool = np.arange(max_num_reqs)
    slots = rng.choice(slot_pool, size=num_reqs, replace=False)
    slots = slots.astype(np.int32)
    k = rng.integers(0, max_spec + 1, size=num_reqs)
    num_logits_per_req = (1 + k).astype(np.int32)
    cu = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(num_logits_per_req, out=cu[1:])

    query_lens = num_logits_per_req + rng.integers(0, 4, size=num_reqs).astype(np.int32)
    qsl = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(query_lens, out=qsl[1:])
    num_tokens = int(qsl[-1])

    logits_indices_np = np.concatenate(
        [
            np.arange(qsl[i + 1] - num_logits_per_req[i], qsl[i + 1])
            for i in range(num_reqs)
        ]
    ).astype(np.int64)
    expanded_idx_np = np.repeat(slots, num_logits_per_req)
    expanded_pos_np = np.concatenate(
        [np.arange(n, dtype=np.int32) for n in num_logits_per_req]
    )
    seq_lens_np = (query_lens + rng.integers(0, 100, size=num_reqs)).astype(np.int32)

    def dev(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a).to(DEVICE)

    return InputBatch(
        req_ids=[f"req-{i}" for i in range(num_reqs)],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=dev(slots),
        idx_mapping_np=slots,
        expanded_idx_mapping=dev(expanded_idx_np),
        expanded_local_pos=dev(expanded_pos_np),
        num_scheduled_tokens=query_lens,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_draft_tokens=int(k.sum()),
        num_draft_tokens_per_req=k.astype(np.int32),
        query_start_loc=dev(qsl),
        query_start_loc_np=qsl,
        seq_lens=dev(seq_lens_np),
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_np.copy()),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
        has_prefill=False,
        max_seq_len_np=None,
        input_ids=torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE),
        positions=torch.arange(num_tokens, dtype=torch.int64, device=DEVICE),
        is_padding=torch.zeros(num_tokens, dtype=torch.bool, device=DEVICE),
        logits_indices=dev(logits_indices_np),
        cu_num_logits=dev(cu),
        cu_num_logits_np=cu,
        has_structured_output_reqs=False,
        prompt_lens=None,
    )


def _shard_all_ranks(batch: InputBatch, max_num_reqs: int, tp_size: int):
    """Simulate every TP rank by faking get_tp_group() per BatchSharder."""
    results = []
    for rank in range(tp_size):
        fake_group = SimpleNamespace(rank_in_group=rank, world_size=tp_size)
        with mock.patch.object(batch_shard, "get_tp_group", return_value=fake_group):
            sharder = BatchSharder(
                max_num_reqs=max_num_reqs,
                max_num_logits_per_req=8,
                device=torch.device(DEVICE),
            )
        local_batch, sorted_logits_indices, _, metadata = sharder.shard_sampler_inputs(
            batch, None
        )
        results.append((local_batch, sorted_logits_indices, metadata))
    return results


def _owned_batch_indices(local_batch: InputBatch) -> np.ndarray:
    return np.array([int(r.split("-")[1]) for r in local_batch.req_ids], dtype=np.int64)


def _owned_rows(cu: np.ndarray, owned: np.ndarray) -> np.ndarray:
    if len(owned) == 0:
        return np.array([], dtype=np.int64)
    return np.concatenate([np.arange(cu[r], cu[r + 1]) for r in owned])


@requires_cuda
@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("seed", [0, 1])
def test_local_batch_partition(tp_size: int, seed: int):
    """Local sub-batches partition the batch by owner, preserve batch order
    within an owner, and carry exactly the owned requests' entries of every
    per-request and per-row array."""
    rng = np.random.default_rng(seed)
    max_num_reqs = 64
    batch = _make_batch(rng, num_reqs=23, max_num_reqs=max_num_reqs, max_spec=3)
    num_logits = int(batch.cu_num_logits_np[-1])
    results = _shard_all_ranks(batch, max_num_reqs, tp_size)
    cu = batch.cu_num_logits_np

    metadata0 = results[0][2]
    assert sum(metadata0.num_logits_per_rank) == num_logits
    # Gather width = max logits rows of any request (bounds num_sampled).
    assert metadata0.max_num_logits_per_req == int(np.diff(cu).max())

    all_owned: list[int] = []
    for rank, (local, _, metadata) in enumerate(results):
        owned = _owned_batch_indices(local)
        assert local.num_reqs == len(owned) == metadata.num_local_reqs
        assert (batch.idx_mapping_np[owned] % tp_size == rank).all()
        # Stable sort: batch order is preserved within an owner.
        assert (np.diff(owned) > 0).all()
        all_owned.extend(owned.tolist())

        rows = _owned_rows(cu, owned)
        assert (
            metadata.num_local_logits
            == len(rows)
            == metadata0.num_logits_per_rank[rank]
        )

        expected_idx = batch.idx_mapping_np[owned]
        assert (local.idx_mapping_np == expected_idx).all()
        assert _np(local.idx_mapping).tolist() == expected_idx.tolist()
        assert (np.diff(local.cu_num_logits_np) == np.diff(cu)[owned]).all()
        assert _np(local.cu_num_logits).tolist() == local.cu_num_logits_np.tolist()
        assert (
            _np(local.expanded_idx_mapping) == _np(batch.expanded_idx_mapping)[rows]
        ).all()
        assert (
            _np(local.expanded_local_pos) == _np(batch.expanded_local_pos)[rows]
        ).all()
        assert (_np(local.logits_indices) == _np(batch.logits_indices)[rows]).all()
        assert (_np(local.seq_lens) == _np(batch.seq_lens)[owned]).all()
        assert batch.num_draft_tokens_per_req is not None
        assert local.num_draft_tokens_per_req is not None
        assert (
            local.num_draft_tokens_per_req == batch.num_draft_tokens_per_req[owned]
        ).all()
        assert local.num_draft_tokens == int(
            batch.num_draft_tokens_per_req[owned].sum()
        )
        # Triton specializes kernels on pointer alignment: every packed-buffer
        # view fed to kernels must stay 16-byte aligned across batch shapes,
        # or serving JITs new kernel variants mid-collective.
        for t in (
            local.idx_mapping,
            local.expanded_idx_mapping,
            local.expanded_local_pos,
            local.logits_indices,
            local.cu_num_logits,
            metadata.gathered_src_indices,
        ):
            assert t.data_ptr() % 16 == 0
    assert sorted(all_owned) == list(range(batch.num_reqs))

    # sorted_logits_indices is identical on all ranks: the owner-sorted logits row
    # positions, i.e. logits_indices permuted rank-major.
    sorted_logits_indices = _np(results[0][1])
    for _, gi, _ in results:
        assert (_np(gi) == sorted_logits_indices).all()
    rows_all = np.concatenate(
        [_owned_rows(cu, _owned_batch_indices(local)) for local, _, _ in results]
    )
    assert (
        sorted_logits_indices.tolist() == _np(batch.logits_indices)[rows_all].tolist()
    )


@requires_cuda
@pytest.mark.parametrize("tp_size", [2, 4])
def test_all_to_all_reassembly_simulated(tp_size: int):
    """Vocab shards computed from owner-sorted hidden states and routed
    through the simulated all-to-all reassemble into exactly the full-vocab
    logits of each rank's owned rows."""
    rng = np.random.default_rng(7)
    max_num_reqs = 32
    batch = _make_batch(rng, num_reqs=11, max_num_reqs=max_num_reqs, max_spec=4)
    vocab = 64 * tp_size
    shard_w = vocab // tp_size
    token_logits = torch.randn(batch.num_tokens, vocab, device=DEVICE)

    results = _shard_all_ranks(batch, max_num_reqs, tp_size)
    sorted_logits_indices = results[0][1]
    num_logits_per_rank = results[0][2].num_logits_per_rank
    bounds = np.zeros(tp_size + 1, dtype=np.int64)
    np.cumsum(num_logits_per_rank, out=bounds[1:])

    # send_r = rank r's vocab shard of the owner-sorted rows, with no further
    # permute; chunk d of send_r goes to rank d.
    sends = [
        token_logits[sorted_logits_indices][:, r * shard_w : (r + 1) * shard_w]
        for r in range(tp_size)
    ]
    for rank, (local, _, metadata) in enumerate(results):
        recv = torch.cat(
            [sends[src][bounds[rank] : bounds[rank + 1]] for src in range(tp_size)]
        )
        # Reassemble source-rank-major chunks into full-vocab rows, mirroring
        # all_to_all_logits (source-rank order is vocab-shard order).
        reassembled = (
            recv.view(tp_size, metadata.num_local_logits, shard_w)
            .permute(1, 0, 2)
            .reshape(metadata.num_local_logits, vocab)
        )
        expected = token_logits[local.logits_indices]
        assert torch.equal(reassembled, expected)


@requires_cuda
def test_shard_no_spec():
    """Batches without draft tokens (one logits row per request) degenerate
    cleanly: rows mirror requests."""
    rng = np.random.default_rng(11)
    tp_size = 4
    max_num_reqs = 32
    batch = _make_batch(rng, num_reqs=10, max_num_reqs=max_num_reqs, max_spec=0)
    results = _shard_all_ranks(batch, max_num_reqs, tp_size)

    assert results[0][2].max_num_logits_per_req == 1
    for local, _, metadata in results:
        assert metadata.num_local_logits == metadata.num_local_reqs == local.num_reqs
        owned = _owned_batch_indices(local)
        assert _np(local.logits_indices).tolist() == (
            _np(batch.logits_indices)[owned].tolist()
        )


@pytest.mark.parametrize("tp_size", [2, 4])
def test_shard_grammar_output(tp_size: int):
    """The grammar filter keeps exactly the owned requests' bitmask rows, in
    grammar order, and the per-rank pieces partition the global bitmask."""
    rng = np.random.default_rng(5)
    max_num_reqs = 32
    num_reqs = 12
    slots = rng.choice(np.arange(max_num_reqs), size=num_reqs, replace=False)
    k = rng.integers(0, 4, size=num_reqs)
    cu = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(1 + k, out=cu[1:])
    owner = slots % tp_size

    # Grammar requests: every other request, in a scrambled order.
    req_ids = [f"req-{i}" for i in range(num_reqs)]
    grammar_idx = list(rng.permutation(np.arange(0, num_reqs, 2)))
    grammar_req_ids = [req_ids[i] for i in grammar_idx]
    num_grammar_rows = sum(int(cu[i + 1] - cu[i]) for i in grammar_idx)
    bitmask = rng.integers(0, 2**31, size=(num_grammar_rows, 4), dtype=np.int32)
    input_batch = SimpleNamespace(req_ids=req_ids, cu_num_logits_np=cu)

    # Global row offset of each grammar request within the bitmask.
    row_offsets = {}
    cursor = 0
    for i in grammar_idx:
        row_offsets[i] = cursor
        cursor += int(cu[i + 1] - cu[i])

    grammar_output = GrammarOutput(
        structured_output_request_ids=grammar_req_ids,
        grammar_bitmask=bitmask,
    )

    kept_total = 0
    for rank in range(tp_size):
        local_req_ids = [req_ids[i] for i in range(num_reqs) if owner[i] == rank]
        local = _shard_grammar_output(grammar_output, input_batch, local_req_ids)
        expected_idx = [i for i in grammar_idx if owner[i] == rank]
        if not expected_idx:
            assert local is None
            continue
        assert local is not None
        assert local.structured_output_request_ids == [req_ids[i] for i in expected_idx]
        expected_rows = np.concatenate(
            [
                np.arange(row_offsets[i], row_offsets[i] + cu[i + 1] - cu[i])
                for i in expected_idx
            ]
        )
        assert (local.grammar_bitmask == bitmask[expected_rows.astype(int)]).all()
        kept_total += local.grammar_bitmask.shape[0]
    assert kept_total == num_grammar_rows


@requires_cuda
@pytest.mark.parametrize("tp_size", [2, 4])
def test_gather_src_indices_round_trip(tp_size: int):
    """Padded per-rank result blocks indexed by gathered_src_indices restore
    batch order, including when some rank owns zero requests."""
    rng = np.random.default_rng(3)
    max_num_reqs = 40
    all_slots = np.arange(max_num_reqs)
    batch = _make_batch(
        rng,
        num_reqs=17,
        max_num_reqs=max_num_reqs,
        max_spec=2,
        slot_pool=all_slots[all_slots % tp_size != 0],  # rank 0 owns nothing
    )
    results = _shard_all_ranks(batch, max_num_reqs, tp_size)
    metadata0 = results[0][2]
    assert metadata0.num_local_reqs == 0

    # Value of request j is j; each rank packs its owned requests' values into
    # a zero-padded (max_num_reqs_per_rank, 1) block, gathered rank-major.
    pad = metadata0.max_num_reqs_per_rank
    blocks = []
    for local, _, _ in results:
        owned = _owned_batch_indices(local)
        block = torch.zeros(pad, 1, dtype=torch.int64, device=DEVICE)
        block[: len(owned), 0] = torch.from_numpy(owned).to(DEVICE)
        blocks.append(block)
    gathered = torch.cat(blocks)

    restored = gathered[metadata0.gathered_src_indices][:, 0]
    assert restored.tolist() == list(range(batch.num_reqs))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tp_size", [2, 4])
def test_gather_sampler_output_kernels(tp_size: int):
    """The fused pack/unpack kernels round-trip per-rank sampler results
    into batch order: token ids zero-filled past the local width, counts
    cast back to int32, identical on every rank."""
    from dataclasses import replace as dc_replace

    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    rng = np.random.default_rng(9)
    max_num_reqs = 32
    batch = _make_batch(rng, num_reqs=13, max_num_reqs=max_num_reqs, max_spec=3)
    results = _shard_all_ranks(batch, max_num_reqs, tp_size)
    device = torch.device("cuda")
    num_reqs = batch.num_reqs
    width = results[0][2].max_num_logits_per_req

    # Distinctive per-rank local outputs. Even ranks mimic the regular
    # sampler (one column, exercising zero-fill); odd ranks mimic the
    # rejection sampler's over-allocation (exercising truncation).
    local_outputs: list[SamplerOutput | None] = []
    expected_ids = torch.zeros(num_reqs, width, dtype=torch.int64)
    expected_sampled = torch.zeros(num_reqs, dtype=torch.int32)
    expected_rejected = torch.zeros(num_reqs, dtype=torch.int32)
    for rank, (local, _, metadata) in enumerate(results):
        if metadata.num_local_reqs == 0:
            local_outputs.append(None)
            continue
        owned = torch.from_numpy(_owned_batch_indices(local))
        src_width = 1 if rank % 2 == 0 else width + 3
        ids = owned[:, None] * 1000 + torch.arange(src_width)[None, :]
        num_sampled = (owned % width + 1).to(torch.int32)
        num_rejected = (owned % 3).to(torch.int32)
        local_outputs.append(
            SamplerOutput(
                sampled_token_ids=ids.to(device),
                logprobs_tensors=None,
                num_nans=None,
                num_sampled=num_sampled.to(device),
                num_rejected=num_rejected.to(device),
            )
        )
        copy_width = min(src_width, width)
        expected_ids[owned, :copy_width] = ids[:, :copy_width]
        expected_sampled[owned] = num_sampled
        expected_rejected[owned] = num_rejected

    # Two passes over a mocked all-gather: capture each rank's packed send
    # block, then replay with the concatenated blocks as the gathered result.
    packed_blocks: dict[int, torch.Tensor] = {}
    gathered_full: list[torch.Tensor] = []

    def run(rank: int) -> SamplerOutput:
        metadata = dc_replace(
            results[rank][2],
            gathered_src_indices=results[rank][2].gathered_src_indices.to(device),
        )

        def fake_all_gather(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
            packed_blocks[rank] = x.clone()
            if gathered_full:
                return gathered_full[0]
            return torch.zeros(
                tp_size * x.shape[0], x.shape[1], dtype=x.dtype, device=x.device
            )

        with mock.patch.object(
            batch_shard, "tensor_model_parallel_all_gather", fake_all_gather
        ):
            return batch_shard.gather_sampler_output(
                local_outputs[rank],
                metadata,
                device,
                global_batch=batch,
                local_batch=results[rank][0],
            )

    for rank in range(tp_size):
        run(rank)
    # Zero the uninitialized padding rows so the comparison is deterministic;
    # the real path never reads them (gathered_src_indices skips padding).
    for rank, (local, _, metadata) in enumerate(results):
        packed_blocks[rank][metadata.num_local_reqs :] = 0
    gathered_full.append(torch.cat([packed_blocks[r] for r in range(tp_size)]))

    for rank in range(tp_size):
        out = run(rank)
        assert out.sampled_token_ids.dtype == torch.int64
        assert out.num_sampled is not None and out.num_sampled.dtype == torch.int32
        assert out.num_rejected is not None
        assert torch.equal(out.sampled_token_ids.cpu(), expected_ids)
        assert torch.equal(out.num_sampled.cpu(), expected_sampled)
        assert torch.equal(out.num_rejected.cpu(), expected_rejected.to(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gather_sampler_output_logprobs_and_nans():
    """num_nans is reduced per request and gathered; LogprobsTensors are
    packed into fixed per-request blocks, width-aligned across ranks (pad
    and truncate paths), and restored to global row order."""
    from dataclasses import replace as dc_replace

    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    tp_size = 2
    rng = np.random.default_rng(21)
    max_num_reqs = 32
    batch = _make_batch(rng, num_reqs=6, max_num_reqs=max_num_reqs, max_spec=3)
    results = _shard_all_ranks(batch, max_num_reqs, tp_size)
    device = torch.device("cuda")
    num_reqs = batch.num_reqs
    cu = batch.cu_num_logits_np
    num_logits = int(cu[-1])
    width = results[0][2].max_num_logits_per_req
    num_cols = 3

    def cuda_fields(b: InputBatch) -> InputBatch:
        return dc_replace(
            b,
            cu_num_logits=b.cu_num_logits.to(device),
            expanded_local_pos=b.expanded_local_pos.to(device),
        )

    global_cuda = cuda_fields(batch)

    local_outputs: list[SamplerOutput | None] = []
    local_batches: list[InputBatch] = []
    expected_ids = torch.zeros(num_logits, num_cols, dtype=torch.int64)
    expected_logprobs = torch.full(
        (num_logits, num_cols), float("-inf"), dtype=torch.float32
    )
    expected_ranks = torch.zeros(num_logits, dtype=torch.int64)
    expected_nans = torch.zeros(num_reqs, dtype=torch.int32)
    for rank, (local, _, metadata) in enumerate(results):
        local_batches.append(cuda_fields(local))
        if metadata.num_local_reqs == 0:
            local_outputs.append(None)
            continue
        owned = _owned_batch_indices(local)
        rows = torch.from_numpy(_owned_rows(cu, owned))
        # Rank 0 sends fewer columns (pad path), rank 1 more (truncate path).
        local_cols = num_cols - 1 if rank == 0 else num_cols + 2
        ids = rows[:, None] * 10 + torch.arange(local_cols)[None, :]
        logprobs = ids.float() * 0.5
        ranks_t = rows + 7
        nans = (rows % 4).to(torch.int32)
        local_outputs.append(
            SamplerOutput(
                sampled_token_ids=torch.zeros(
                    metadata.num_local_reqs, width, dtype=torch.int64, device=device
                ),
                logprobs_tensors=LogprobsTensors(
                    logprob_token_ids=ids.to(device),
                    logprobs=logprobs.to(device),
                    selected_token_ranks=ranks_t.to(device),
                    cu_num_generated_tokens=None,
                ),
                num_nans=nans.to(device),
                num_sampled=torch.ones(
                    metadata.num_local_reqs, dtype=torch.int32, device=device
                ),
                num_rejected=torch.zeros(
                    metadata.num_local_reqs, dtype=torch.int32, device=device
                ),
            )
        )
        copy_cols = min(local_cols, num_cols)
        expected_ids[rows, :copy_cols] = ids[:, :copy_cols]
        expected_logprobs[rows, :copy_cols] = logprobs[:, :copy_cols]
        expected_ranks[rows] = ranks_t
        for j in owned:
            in_req = (rows >= int(cu[j])) & (rows < int(cu[j + 1]))
            expected_nans[j] = int(nans[in_req].sum())

    recorded: dict[int, list[torch.Tensor]] = {}
    gathered_full: list[torch.Tensor] = []

    def run(rank: int) -> SamplerOutput:
        metadata = dc_replace(
            results[rank][2],
            gathered_src_indices=results[rank][2].gathered_src_indices.to(device),
        )
        calls = {"i": 0}

        def fake_all_gather(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
            i = calls["i"]
            calls["i"] += 1
            if gathered_full:
                return gathered_full[i]
            recorded.setdefault(rank, []).append(x.clone())
            return torch.zeros(
                tp_size * x.shape[0], *x.shape[1:], dtype=x.dtype, device=x.device
            )

        with mock.patch.object(
            batch_shard, "tensor_model_parallel_all_gather", fake_all_gather
        ):
            return batch_shard.gather_sampler_output(
                local_outputs[rank],
                metadata,
                device,
                global_batch=global_cuda,
                local_batch=local_batches[rank],
                gather_num_nans=True,
                logprobs_dims=(num_cols - 1, 0),
            )

    for rank in range(tp_size):
        run(rank)
    gathered_full.extend(
        torch.cat([recorded[r][i] for r in range(tp_size)]) for i in range(3)
    )

    for rank in range(tp_size):
        out = run(rank)
        assert out.num_nans is not None
        assert torch.equal(out.num_nans.cpu(), expected_nans)
        lp = out.logprobs_tensors
        assert lp is not None
        assert torch.equal(lp.logprob_token_ids.cpu(), expected_ids)
        assert torch.equal(lp.logprobs.cpu(), expected_logprobs)
        assert torch.equal(lp.selected_token_ranks.cpu(), expected_ranks)
        if num_logits != num_reqs:
            assert lp.cu_num_generated_tokens == cu.tolist()
        else:
            assert lp.cu_num_generated_tokens is None


def test_finish_requests_frees_slots_in_sorted_order():
    """Request slots must be freed in the same order on every TP rank:
    ownership derives from slot indices, and `finished_req_ids` is a set
    whose iteration order is per-process hash-randomized."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    removed: list[str] = []
    fake_runner = SimpleNamespace(_remove_request=removed.append, pooling_runner=None)
    scheduler_output = SimpleNamespace(
        finished_req_ids={"req-b", "req-c", "req-a"},
        preempted_req_ids={"req-e", "req-d"},
    )
    GPUModelRunner.finish_requests(fake_runner, scheduler_output)
    assert removed == sorted(removed)
    assert set(removed) == {"req-a", "req-b", "req-c", "req-d", "req-e"}
