# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.worker.gpu import cp_utils as gpu_cp_utils
from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker


def _copy_to_cpu(value, out=None, device=None):
    tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    if out is not None:
        return out.copy_(tensor)
    return tensor


def test_replicated_decode_piecewise_graph_padding(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)

    segments_by_rank, per_rank_num_tokens = manager._build_batch_layout(
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        num_computed_tokens=np.full(3, 16, dtype=np.int32),
        is_prefilling=np.zeros(3, dtype=np.bool_),
        query_start_loc_np=np.arange(4, dtype=np.int32),
        padded_num_tokens=4,
    )

    assert per_rank_num_tokens == [3, 3]
    request_indices = [
        [segment.global_batch_req_idx for segment in rank] for rank in segments_by_rank
    ]
    assert request_indices == [[0, 1, 2], [0, 1, 2]]
    assert manager._hidden_restore_idx is None
    assert torch.equal(
        manager._padded_gather_idx,
        torch.tensor([0, 1, 2, 0, 0, 1, 2, 0]),
    )
    assert torch.equal(
        manager._gathered_kv_write_mask,
        torch.tensor([True, True, True, False, False, False, False, False]),
    )


def test_input_buffers_are_exposed_for_cudagraph_capture():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=8,
    )

    assert manager.input_buffers is manager._input_buffers
    assert manager.input_buffers.input_ids.shape == (8,)
    assert manager.input_buffers.positions.shape == (8,)
    assert manager.input_buffers.is_padding.shape == (8,)


@pytest.mark.parametrize(
    ("pcp_world_size", "num_scheduled_tokens", "is_prefilling", "expected"),
    [
        (2, [8], [True], 4),
        (2, [7], [True], 4),
        (2, [3], [False], 3),
        (2, [3, 8], [False, True], 7),
        (4, [2, 9], [False, True], 5),
    ],
)
def test_num_tokens_for_dispatch_uses_largest_pcp_rank(
    pcp_world_size, num_scheduled_tokens, is_prefilling, expected
):
    manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=0,
        device=torch.device("cpu"),
    )

    actual = manager.get_num_tokens_for_dispatch(
        np.asarray(num_scheduled_tokens, dtype=np.int32),
        np.asarray(is_prefilling, dtype=np.bool_),
    )

    assert actual == expected


def test_graph_padding_cannot_be_smaller_than_largest_pcp_rank(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)

    with pytest.raises(ValueError, match="smaller than the largest rank-local batch"):
        manager._build_batch_layout(
            num_scheduled_tokens=np.ones(3, dtype=np.int32),
            num_computed_tokens=np.full(3, 16, dtype=np.int32),
            is_prefilling=np.zeros(3, dtype=np.bool_),
            query_start_loc_np=np.arange(4, dtype=np.int32),
            padded_num_tokens=2,
        )


def _rank_rows(
    pcp_rank: int,
    pcp_world_size: int,
    num_scheduled_tokens: np.ndarray,
    num_computed_tokens: np.ndarray,
    is_prefilling: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One PCP+DCP rank's rows as (global request, extent)."""
    manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=pcp_rank,
        device=torch.device("cpu"),
        dcp_world_size=2,
    )
    query_start_loc_np = np.concatenate([[0], np.cumsum(num_scheduled_tokens)]).astype(
        np.int32
    )
    segments = manager._get_rank_segments(
        pcp_rank, num_scheduled_tokens, is_prefilling, query_start_loc_np
    )
    rows = np.array([segment.global_batch_req_idx for segment in segments])
    extents = (num_computed_tokens + num_scheduled_tokens)[rows]
    return rows, extents


@pytest.mark.parametrize(
    ("num_scheduled_tokens", "num_computed_tokens"),
    [
        ([32, 32], [0, 0]),
        ([32, 32, 32, 32], [0, 0, 0, 0]),
        ([32, 32], [100, 0]),  # one continued, one fresh
        ([64, 32, 48], [0, 512, 0]),  # ragged lengths and mixed contexts
        # A replicated prefill BEHIND a split one. Sorting replicated rows first
        # put request 1 ahead of request 0.
        ([64, 3], [0, 0]),
        ([64, 3, 64], [0, 0, 0]),
    ],
)
@pytest.mark.parametrize("pcp_world_size", [2, 4])
def test_published_row_order_is_identical_on_every_pcp_rank(
    num_scheduled_tokens, num_computed_tokens, pcp_world_size
):
    """Every rank must map its rows to the same global requests, in the same order."""
    num_scheduled_tokens = np.array(num_scheduled_tokens, dtype=np.int32)
    num_computed_tokens = np.array(num_computed_tokens, dtype=np.int32)
    is_prefilling = np.ones(len(num_scheduled_tokens), dtype=np.bool_)

    orders = [
        _rank_rows(
            rank,
            pcp_world_size,
            num_scheduled_tokens,
            num_computed_tokens,
            is_prefilling,
        )[0]
        for rank in range(pcp_world_size)
    ]
    for rank, order in enumerate(orders[1:], start=1):
        assert np.array_equal(orders[0], order), (
            f"rank {rank} rows map to {order.tolist()}, rank 0 to {orders[0].tolist()}"
        )
    # Grouped by request and ascending: what the indexer plan indexes by.
    assert np.all(np.diff(orders[0]) >= 0)


def test_published_row_order_puts_every_decode_before_every_prefill():
    """split_decodes_and_prefills takes the FIRST prefilling row as the boundary."""
    # req 0 is a continued prefill, req 1 is a decode.
    req_idx, _ = _rank_rows(
        pcp_rank=0,
        pcp_world_size=2,
        num_scheduled_tokens=np.array([32, 1], dtype=np.int32),
        num_computed_tokens=np.array([100, 20], dtype=np.int32),
        is_prefilling=np.array([True, False], dtype=np.bool_),
    )
    # The decode (request 1) must be row 0.
    assert req_idx[0] == 1


def test_split_prefill_rows_repeat_the_request_and_its_full_extent():
    """Equal adjacent request indices are what let a backend share a KV region."""
    req_idx, extents = _rank_rows(
        pcp_rank=1,
        pcp_world_size=2,
        num_scheduled_tokens=np.array([64, 3], dtype=np.int32),
        num_computed_tokens=np.array([100, 0], dtype=np.int32),
        is_prefilling=np.ones(2, dtype=np.bool_),
    )
    assert req_idx.tolist() == [0, 0, 1]
    assert extents.tolist() == [164, 164, 3]


@pytest.mark.parametrize("query_len", [1, 2, 3, 5, 6, 9])
def test_dcp_replicates_prefills_too_short_to_split(query_len):
    """A prefill that cannot fill 2*pcp chunks is replicated, not split."""
    pcp_world_size = 2
    num_scheduled_tokens = np.array([query_len], dtype=np.int32)
    is_prefilling = np.ones(1, dtype=np.bool_)

    rows_per_rank = []
    for rank in range(pcp_world_size):
        manager = PCPManager(
            pcp_world_size=pcp_world_size,
            pcp_rank=rank,
            device=torch.device("cpu"),
            dcp_world_size=2,
        )
        assert manager.replicated_requests(num_scheduled_tokens, is_prefilling)[0]
        rows = list(
            manager._iter_rank_chunks(rank, num_scheduled_tokens, is_prefilling)
        )
        assert rows == [(0, 0, query_len)]
        rows_per_rank.append(rows)
    assert rows_per_rank[0] == rows_per_rank[1]


@pytest.mark.parametrize("pcp_world_size", [2, 4, 8])
@pytest.mark.parametrize(
    "query_len", [16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 4097]
)
def test_pcp_first_chunk_row_is_never_short(pcp_world_size, query_len):
    num_scheduled_tokens = np.array([query_len], dtype=np.int32)
    is_prefilling = np.ones(1, dtype=np.bool_)

    charged_per_rank = []
    for rank in range(pcp_world_size):
        manager = PCPManager(
            pcp_world_size=pcp_world_size,
            pcp_rank=rank,
            device=torch.device("cpu"),
            dcp_world_size=pcp_world_size,
        )
        chunk_lens = [
            chunk_len
            for _, _, chunk_len in manager._iter_rank_chunks(
                rank, num_scheduled_tokens, is_prefilling
            )
        ]
        assert chunk_lens, f"rank {rank} got no rows for {query_len=}"
        assert chunk_lens[0] == max(chunk_lens), (
            f"rank {rank} emitted a short first chunk for {query_len=}: {chunk_lens}"
        )
        charged_per_rank.append(len(chunk_lens) * chunk_lens[0])

    assert len(set(charged_per_rank)) == 1, (
        f"ranks would chunk differently for {query_len=}: {charged_per_rank}"
    )


def _make_global_decode_batch(
    num_computed_tokens: list[int], buffers: InputBuffers, device: torch.device
) -> InputBatch:
    """A replicate-decode global batch as `prepare_inputs` would build it."""
    num_reqs = len(num_computed_tokens)
    num_tokens = num_reqs
    seq_lens_np = np.asarray(num_computed_tokens, dtype=np.int32) + 1

    base = InputBatch.make_dummy(num_reqs, num_tokens, buffers)
    buffers.seq_lens[:num_reqs] = torch.from_numpy(seq_lens_np).to(device)
    buffers.positions[:num_reqs] = torch.tensor(num_computed_tokens, device=device)
    query_start_loc_np = np.arange(num_reqs + 1, dtype=np.int32)
    buffers.query_start_loc[: num_reqs + 1] = torch.from_numpy(query_start_loc_np).to(
        device
    )

    return replace(
        base,
        req_ids=[f"req_{i}" for i in range(num_reqs)],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=device),
        idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
        num_scheduled_tokens=np.ones(num_reqs, dtype=np.int32),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_draft_tokens=0,
        num_draft_tokens_per_req=np.zeros(num_reqs, dtype=np.int32),
        query_start_loc=buffers.query_start_loc[: num_reqs + 1],
        query_start_loc_np=query_start_loc_np,
        seq_lens=buffers.seq_lens[:num_reqs],
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_np),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=np.asarray(num_computed_tokens, dtype=np.int32),
        prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
        input_ids=buffers.input_ids[:num_tokens],
        positions=buffers.positions[:num_tokens],
        is_padding=buffers.is_padding[:num_tokens],
        prompt_lens=None,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU kernels")
def test_partition_defers_dcp_metadata_to_post_partition_batch():
    """DCP-local lengths must derive from the partitioned batch, not the
    global one: the partition replaces seq_lens, so a pre-partition value is
    stale. partition_batch therefore returns None, and the runtime populates
    the field afterwards from the PCP-owned buffers.
    """
    device = torch.device("cuda:0")
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=device,
        max_num_reqs=4,
        max_num_tokens=8,
        dcp_world_size=2,
        dcp_rank=0,
    )

    global_buffers = InputBuffers(4, 8, device)
    global_batch = _make_global_decode_batch([16, 24], global_buffers, device)
    # A leftover from an earlier DCP batch must not survive the partition.
    global_batch.dcp_local_seq_lens = global_buffers.dcp_local_seq_lens[:2]
    global_batch.dcp_local_seq_lens.fill_(-1)

    local_batch = manager.partition_batch(global_batch, padded_num_tokens=2)

    assert local_batch.dcp_local_seq_lens is None
    assert local_batch.seq_lens.tolist() == [17, 25]

    # What execute_model does next: derive DCP metadata from the final batch
    # on the PCP-owned buffers.
    local_batch.dcp_local_seq_lens = gpu_cp_utils.maybe_prepare_dcp_local_seq_lens(
        manager.input_buffers.dcp_local_seq_lens,
        local_batch.seq_lens,
        local_batch.num_reqs,
        dcp_size=2,
        dcp_rank=0,
        cp_interleave=1,
        num_reqs_padded=local_batch.num_reqs_after_padding,
    )
    expected = get_dcp_local_seq_lens(
        torch.tensor([17, 25], dtype=torch.int32), 2, 0, 1
    )
    assert local_batch.dcp_local_seq_lens is not None
    assert torch.equal(local_batch.dcp_local_seq_lens.cpu(), expected)


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize(
    "queries,prefilling",
    [
        ([9, 5], [True, True]),
        ([9, 1, 17, 1], [True, False, True, False]),
        ([1, 1], [False, False]),
        ([2, 1], [False, False]),
        ([1], [True]),
    ],
)
@pytest.mark.parametrize(
    "consumer", ["sampling", "batch_sharding", "speculation", "prompt_logprobs"]
)
@pytest.mark.parametrize("multicast", [False, True])
def test_sampling_matches_global_rows(
    monkeypatch, world, queries, prefilling, consumer, multicast
):
    """Preserve sampled order and dense prompt rows across ragged/padded PCP batches."""
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)
    q = np.array(queries, dtype=np.int32)
    dense = consumer != "sampling"
    prompt_logprobs_worker = PromptLogprobsWorker(max_num_reqs=len(q))
    if consumer == "prompt_logprobs":
        prompt_logprobs_worker.uses_prompt_logprobs[0] = True
        prompt_logprobs_worker.in_progress_prompt_logprobs["req0"] = []
    prefilling = np.array(prefilling)
    starts = np.concatenate(([0], np.cumsum(q))).astype(np.int32)
    global_hidden = torch.arange(int(starts[-1]) * 3).reshape(-1, 3).float()
    managers, local = [], []
    for rank in range(world):
        manager = PCPManager(world, rank, torch.device("cpu"))
        segments, counts = manager._build_batch_layout(
            q, np.zeros_like(q), prefilling, starts
        )
        hidden = torch.zeros(max(counts), 3)
        for segment in segments[rank]:
            hidden[segment.rank_local_batch_slice] = global_hidden[
                segment.global_batch_slice
            ]
        manager._global_batch = NS(
            # Dense consumers may request multiple logits per request.
            logits_indices=(
                torch.arange(len(global_hidden))
                if dense
                else torch.tensor(starts[1:] - 1, dtype=torch.int64)
            ),
            num_reqs=len(q),
            num_tokens=int(q.sum()),
            num_scheduled_tokens=q,
            has_prefill=bool(np.any(prefilling)),
            idx_mapping_np=np.arange(len(q)),
            num_computed_prefill_tokens_np=np.zeros_like(q),
            prefill_len_np=q,
        )
        managers.append(manager)
        local.append(hidden)
    replicated = not np.any(prefilling)
    full = torch.cat(local)
    packed = (
        None
        if replicated
        else torch.cat(
            [
                hidden[manager._sample_local_row_idx]
                for manager, hidden in zip(managers, local)
            ]
        )
    )
    for manager, hidden in zip(managers, local):
        if multicast and not replicated:
            n = manager._sample_local_row_idx.numel()
            manager._restore_buffers = (
                torch.empty(n, 3),
                torch.empty(n * world, 3),
                torch.empty(2, len(q), 3),
            )

        def publish(tensor, group_name, out, manager=manager, hidden=hidden):
            assert not dense and packed is not None
            torch.testing.assert_close(tensor, hidden[manager._sample_local_row_idx])
            out.copy_(packed)

        monkeypatch.setattr(
            torch.ops.symm_mem, "multimem_all_gather_out", publish, raising=False
        )

        def gather(tensor, dim, manager=manager, hidden=hidden):
            assert not replicated
            assert dim == 0
            if dense:
                torch.testing.assert_close(tensor, hidden)
                return full
            torch.testing.assert_close(tensor, hidden[manager._sample_local_row_idx])
            return packed

        monkeypatch.setattr(
            pcp_manager_module,
            "get_pcp_group",
            lambda: NS(all_gather=gather, cpu_group=NS(group_name="test")),
        )
        runner = NS(
            pcp_manager=manager,
            batch_sharder=NS() if consumer == "batch_sharding" else None,
            speculator=NS() if consumer == "speculation" else None,
            prompt_logprobs_worker=prompt_logprobs_worker,
            req_states=NS(prompt_len=NS(np=q)),
        )
        restored, sampled, batch = pcp_manager_module.maybe_restore_pcp_for_sampling(
            runner, hidden, NS()
        )
        assert batch is manager._global_batch
        torch.testing.assert_close(
            sampled, global_hidden[manager._global_batch.logits_indices]
        )
        if dense:
            torch.testing.assert_close(restored, global_hidden)


def test_restore_without_pcp_preserves_inputs():
    """Non-PCP runners need no restore metadata or prompt-logprob worker."""
    hidden = torch.zeros(2, 3)
    input_batch = NS()
    restored, sampled, batch = pcp_manager_module.maybe_restore_pcp_for_sampling(
        NS(pcp_manager=None), hidden, input_batch
    )
    assert restored is hidden and sampled is None and batch is input_batch


@pytest.mark.parametrize("failed_phase", [1, 2])
def test_restore_allocation_fallback_is_rank_consistent(monkeypatch, failed_phase):
    import torch.distributed._symmetric_memory as symm_mem

    calls = []

    def agree(ready, **kwargs):
        calls.append("agree")
        if calls.count("agree") == failed_phase:
            ready.zero_()

    def rendezvous(*args):
        calls.append("rendezvous")
        return NS(multicast_ptr=1)

    monkeypatch.setattr(PCPManager, "validate_config", lambda *args: None)
    monkeypatch.setattr(
        pcp_manager_module, "get_pcp_group", lambda: NS(cpu_group=NS(size=lambda: 4))
    )
    monkeypatch.setattr(pcp_manager_module.dist, "all_reduce", agree)
    monkeypatch.setattr(symm_mem, "empty", torch.empty)
    monkeypatch.setattr(symm_mem, "rendezvous", rendezvous)
    config = NS(
        parallel_config=NS(prefill_context_parallel_size=4),
        scheduler_config=NS(max_num_seqs=4),
        model_config=NS(dtype=torch.bfloat16, get_hidden_size=lambda: 16),
    )
    assert (
        pcp_manager_module.allocate_pcp_restore_buffers(
            config, torch.device("cpu"), False
        )
        is None
    )
    assert calls == (
        ["agree"] if failed_phase == 1 else ["agree", "rendezvous", "agree"]
    )


def _multicast_restore_worker(rank, port):
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", rank=rank, world_size=4)
    nccl = dist.new_group(backend="nccl")
    pcp_manager_module.get_pcp_group = lambda: NS(cpu_group=dist.group.WORLD)
    PCPManager.validate_config = staticmethod(lambda *args: None)
    for dtype in (torch.bfloat16, torch.float16):
        config = NS(
            parallel_config=NS(prefill_context_parallel_size=4),
            scheduler_config=NS(max_num_seqs=64),
            model_config=NS(dtype=dtype, get_hidden_size=lambda: 32),
        )
        buffers = pcp_manager_module.allocate_pcp_restore_buffers(
            config, torch.device("cuda", rank), False
        )
        assert buffers is not None
        manager = PCPManager(
            4, rank, torch.device("cuda", rank), restore_buffers=buffers
        )
        manager._global_batch = NS(has_prefill=True)
        for rows, skew in ((1, False), (7, False), (64, True)):
            n = rows if skew else (rows + 3) // 4
            ids = torch.arange(rows, device="cuda")
            owners = torch.zeros_like(ids) if skew else ids % 4
            slots = ids if skew else ids // 4
            manager._sample_local_row_idx = torch.arange(n, device="cuda") * 2
            manager._sample_restore_idx = owners * n + slots
            previous = expected_previous = None
            for epoch in range(3):
                hidden = torch.randn(2 * n + 1, 32, dtype=dtype, device="cuda")
                gathered = torch.empty(
                    4 * hidden.shape[0], 32, dtype=dtype, device="cuda"
                )
                dist.all_gather_into_tensor(gathered, hidden, group=nccl)
                expected = gathered[owners * hidden.shape[0] + slots * 2]
                if rank == 0 and epoch == 1:
                    time.sleep(0.01)
                output = manager.restore_sample_hidden_states(hidden)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                if previous is not None:
                    torch.testing.assert_close(
                        previous, expected_previous, rtol=0, atol=0
                    )
                previous, expected_previous = output, expected.clone()
        manager.release_restore_buffers()
        manager.release_restore_buffers()
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4, reason="requires four CUDA GPUs"
)
def test_multicast_restore_buffers_match_dense_oracle():
    import torch.multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    mp.spawn(_multicast_restore_worker, args=(get_open_port(),), nprocs=4, join=True)


def test_prompt_logprob_worker_exposes_dense_hidden_requirement() -> None:
    worker = PromptLogprobsWorker(max_num_reqs=4)
    worker.uses_prompt_logprobs[:2] = True
    worker.in_progress_prompt_logprobs["req0"] = []
    worker.in_progress_prompt_logprobs["req1"] = []
    input_batch = NS(
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 8], dtype=np.int32),
        prefill_len_np=np.array([8, 8], dtype=np.int32),
    )
    prompt_lens = np.array([8, 8, 0, 0], dtype=np.int32)

    assert worker.needs_prompt_hidden_states(input_batch, prompt_lens)

    input_batch.num_computed_prefill_tokens_np[:] = 8
    assert not worker.needs_prompt_hidden_states(input_batch, prompt_lens)

    input_batch.num_computed_prefill_tokens_np[:] = 0
    input_batch.prefill_len_np[0] = 8
    prompt_lens[0] = 4
    worker.uses_prompt_logprobs[1] = False
    assert not worker.needs_prompt_hidden_states(input_batch, prompt_lens)


def test_prompt_logprob_worker_skips_mask_without_active_requests() -> None:
    worker = PromptLogprobsWorker(max_num_reqs=4)

    assert not worker.needs_prompt_hidden_states(
        NS(),
        np.empty(0, dtype=np.int32),
    )
