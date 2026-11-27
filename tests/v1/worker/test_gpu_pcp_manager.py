# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.spec_decode.utils import fake_speculative_config
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.v1.attention.backends.utils import PAD_SLOT_ID, get_dcp_local_seq_lens
from vllm.v1.attention.ops import pcp as attention_pcp
from vllm.v1.worker.gpu import cp_utils as gpu_cp_utils
from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers, set_dummy_context
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.pcp_manager import (
    PCPManager,
    apply_prefill_as_decode_mask,
    has_requests_to_run_as_prefill,
)
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.dspark.config import DSparkSpeculatorConfig


def _copy_to_cpu(value, device=None, dtype=None, out=None):
    tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    if out is not None:
        return out.copy_(tensor)
    return tensor


def _make_config(cudagraph_mode: CUDAGraphMode):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
            dcp_comm_backend="ag_rs",
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(),
        ),
        lora_config=None,
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
    )


def _make_capture_manager(block_table: torch.Tensor):
    block_tables = SimpleNamespace(
        input_block_tables=(block_table,),
        num_kv_cache_groups=1,
        block_sizes=(2,),
        cp_interleave=1,
    )
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=False,
        max_num_reqs=4,
        max_num_tokens=8,
        block_tables=block_tables,
    )
    return manager, block_tables


@pytest.mark.parametrize(
    "cudagraph_mode",
    [CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL_AND_PIECEWISE],
)
def test_validate_config_accepts_decode_only_full_graphs(cudagraph_mode):
    PCPManager.validate_config(_make_config(cudagraph_mode), supports_mm_inputs=False)


def test_validate_config_rejects_full_graph_for_prefills():
    with pytest.raises(NotImplementedError, match="decode-only routines"):
        PCPManager.validate_config(
            _make_config(CUDAGraphMode.FULL), supports_mm_inputs=False
        )


@pytest.mark.parametrize("draft_uses_mla", [False, True])
@pytest.mark.parametrize("dcp_size", [1, 2, 4])
def test_validate_config_dcp_for_dspark(draft_uses_mla, dcp_size):
    config = _make_config(CUDAGraphMode.NONE)
    config.parallel_config.prefill_context_parallel_size = 4
    config.parallel_config.decode_context_parallel_size = dcp_size
    config.model_config.hf_text_config.index_topk = 1
    config.speculative_config = fake_speculative_config(
        DSparkSpeculatorConfig,
        draft_decode_context_parallel_size=dcp_size if draft_uses_mla else 1,
    )
    if draft_uses_mla and dcp_size == 2:
        with pytest.raises(NotImplementedError, match="requires DCP=1 or DCP=PCP"):
            PCPManager.validate_config(config, supports_mm_inputs=False)
        return
    PCPManager.validate_config(config, supports_mm_inputs=False)


def test_sharded_decode_piecewise_graph_padding(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_tensor_h2d", _copy_to_cpu)

    segments_by_rank, per_rank_num_tokens = manager._build_batch_layout(
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        num_computed_tokens=np.full(3, 16, dtype=np.int32),
        is_prefilling=np.zeros(3, dtype=np.bool_),
        query_start_loc_np=np.arange(4, dtype=np.int32),
        padded_num_tokens=4,
    )

    assert per_rank_num_tokens == [2, 1]
    request_indices = [
        [segment.global_batch_req_idx for segment in rank] for rank in segments_by_rank
    ]
    assert request_indices == [[0, 2], [1]]
    assert torch.equal(manager._hidden_restore_idx, torch.tensor([0, 4, 1]))
    assert torch.equal(
        manager._padded_gather_idx,
        torch.tensor([0, 2, 0, 0, 1, 0, 0, 0]),
    )
    assert torch.equal(
        manager._gathered_kv_write_mask,
        torch.tensor([True, True, False, False, True, False, False, False]),
    )


def test_input_buffers_are_exposed_for_cudagraph_capture():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=False,
        max_num_reqs=4,
        max_num_tokens=8,
    )

    assert manager.input_buffers is manager._input_buffers
    assert manager.input_buffers.input_ids.shape == (8,)
    assert manager.input_buffers.positions.shape == (8,)
    assert manager.input_buffers.is_padding.shape == (8,)


def test_sync_drafts_writes_logits_to_request_slots(monkeypatch):
    """Rank 0's draft logits land in each batch request's persistent slot."""
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=1,
        device=torch.device("cpu"),
        shard_decode_requests=False,
        max_num_reqs=4,
        max_num_tokens=8,
    )
    root_tokens = torch.tensor([[11, 12, 13]])
    root_logits = torch.arange(24, dtype=torch.float32).view(1, 3, 8)
    broadcasts = iter([root_tokens, root_logits])
    group = SimpleNamespace(
        broadcast=lambda tensor, src: tensor.copy_(next(broadcasts))
    )
    monkeypatch.setattr(pcp_manager_module, "get_pcp_group", lambda: group)
    draft_tokens = torch.full((1, 3), -1)
    draft_logits = DraftLogitsCache(
        4, 3, 8, dtype=torch.float32, fill=-1.0, device="cpu"
    )

    manager.sync_drafts(draft_tokens, torch.tensor([2]), draft_logits)

    torch.testing.assert_close(draft_tokens, root_tokens)
    torch.testing.assert_close(draft_logits.values[2], root_logits[0])
    assert (draft_logits.values[[0, 1, 3]] == -1).all()


@pytest.mark.parametrize(
    ("pcp_world_size", "num_scheduled_tokens", "is_prefilling", "expected"),
    [
        (2, [8], [True], 4),
        (2, [7], [True], 4),
        (2, [3], [False], 3),
        (2, [3, 8], [False, True], 7),
        (4, [2, 9], [False, True], 4),
    ],
)
def test_num_tokens_for_dispatch_uses_largest_pcp_rank(
    pcp_world_size, num_scheduled_tokens, is_prefilling, expected
):
    manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
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
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_tensor_h2d", _copy_to_cpu)

    with pytest.raises(ValueError, match="smaller than the largest rank-local batch"):
        manager._build_batch_layout(
            num_scheduled_tokens=np.ones(3, dtype=np.int32),
            num_computed_tokens=np.full(3, 16, dtype=np.int32),
            is_prefilling=np.zeros(3, dtype=np.bool_),
            query_start_loc_np=np.arange(4, dtype=np.int32),
            padded_num_tokens=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU kernels")
@pytest.mark.parametrize(
    ("cg_mode", "num_reqs", "expected_tokens", "expected_reqs"),
    [
        (CUDAGraphMode.NONE, 4, 2, 2),
        (CUDAGraphMode.PIECEWISE, None, 8, 2),
        (CUDAGraphMode.FULL, 4, 8, 4),
    ],
)
def test_partition_padding_is_derived_from_batch_descriptor(
    cg_mode, num_reqs, expected_tokens, expected_reqs
):
    """partition_batch derives graph padding from the batch descriptor: no
    padding without graphs, token padding only for PIECEWISE, and request
    padding on top for FULL."""
    device = torch.device("cuda:0")
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=device,
        shard_decode_requests=False,
        max_num_reqs=4,
        max_num_tokens=8,
    )
    global_batch = _make_global_decode_batch(
        [16, 24], InputBuffers(4, 8, device), device
    )

    local_batch = manager.partition_batch(
        global_batch,
        BatchExecutionDescriptor(
            cg_mode=cg_mode,
            num_tokens=8,
            num_reqs=num_reqs,
        ),
    )

    assert local_batch.num_tokens_after_padding == expected_tokens
    assert local_batch.num_reqs_after_padding == expected_reqs


def test_dummy_draft_does_not_reuse_previous_graph_batch():
    manager, _ = _make_capture_manager(torch.ones((4, 2), dtype=torch.int32))
    dummy_batch = InputBatch.make_dummy(1, 4, manager.input_buffers)
    manager.draft_prefill_batch = replace(dummy_batch)

    manager.prepare_draft_prefill(dummy_batch, dummy_batch.input_ids)

    draft_inputs, _ = manager.get_draft_input_buffers(
        manager.input_buffers, dummy_batch.num_tokens_after_padding
    )
    assert draft_inputs is manager.input_buffers


def test_capture_uses_pcp_persistent_inputs():
    manager, _ = _make_capture_manager(torch.ones((4, 2), dtype=torch.int32))

    dummy_batch = InputBatch.make_dummy(
        num_reqs=4,
        num_tokens=4,
        input_buffers=InputBuffers(4, 8, torch.device("cpu")),
        max_query_len=1,
    )
    dummy_batch.input_ids.copy_(torch.arange(4))
    dummy_batch.positions.fill_(7)
    input_batch = manager.prepare_inputs_for_capture_or_profile(dummy_batch)

    assert input_batch is not dummy_batch
    assert input_batch.req_ids == dummy_batch.req_ids
    for name in ("input_ids", "positions", "is_padding", "query_start_loc", "seq_lens"):
        actual = getattr(input_batch, name)
        torch.testing.assert_close(actual, getattr(dummy_batch, name))
        assert actual.data_ptr() == getattr(manager.input_buffers, name).data_ptr()


def test_block_table_rebind_updates_interleave():
    manager, block_tables = _make_capture_manager(torch.ones((4, 2), dtype=torch.int32))
    final_block_tables = SimpleNamespace(**vars(block_tables))
    final_block_tables.cp_interleave = 64

    manager.set_block_tables(final_block_tables)

    assert manager._block_tables is final_block_tables
    assert manager.cp_interleave == 64


def test_dummy_context_updates_pcp_local_block_tables():
    global_block_table = torch.full((4, 4), -1, dtype=torch.int32)
    manager, block_tables = _make_capture_manager(global_block_table)
    input_batch = InputBatch.make_dummy(
        num_reqs=2,
        num_tokens=2,
        input_buffers=manager.input_buffers,
        max_query_len=1,
    )
    input_batch = manager.prepare_inputs_for_capture_or_profile(input_batch)
    local_block_tables = manager.get_dummy_block_tables(input_batch.num_reqs)

    set_dummy_context(
        input_batch,
        block_tables,
        context_len=3,
        num_kv_blocks=16,
        max_model_len=16,
        input_block_tables=local_block_tables,
    )

    torch.testing.assert_close(
        local_block_tables[0][:2, :2],
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
    )
    assert torch.all(global_block_table == -1)


def _rank_request_ids(
    manager: PCPManager,
    rank: int,
    req_ids: list[str],
    *,
    is_prefilling: np.ndarray | None = None,
) -> list[str]:
    num_reqs = len(req_ids)
    if is_prefilling is None:
        is_prefilling = np.zeros(num_reqs, dtype=np.bool_)
    segments = manager._get_rank_segments(
        rank=rank,
        num_scheduled_tokens=np.ones(num_reqs, dtype=np.int32),
        is_prefilling=is_prefilling,
        query_start_loc_np=np.arange(num_reqs + 1, dtype=np.int32),
    )
    return [req_ids[segment.global_batch_req_idx] for segment in segments]


def test_pcp_only_decode_requests_are_round_robin_balanced_each_step():
    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    req_ids = [f"request-{idx}" for idx in range(18)]

    owners: dict[str, int] = {}
    for rank in range(manager.pcp_world_size):
        for req_id in _rank_request_ids(manager, rank, req_ids):
            assert req_id not in owners
            owners[req_id] = rank

    assert owners == {
        req_id: index % manager.pcp_world_size for index, req_id in enumerate(req_ids)
    }
    counts = [list(owners.values()).count(rank) for rank in range(4)]
    assert max(counts) - min(counts) == 1

    reordered_req_ids = req_ids[::2] + req_ids[1::2]
    reordered_owners = {
        req_id: rank
        for rank in range(manager.pcp_world_size)
        for req_id in _rank_request_ids(manager, rank, reordered_req_ids)
    }
    assert reordered_owners == {
        req_id: index % manager.pcp_world_size
        for index, req_id in enumerate(reordered_req_ids)
    }
    assert reordered_owners != owners


def test_decode_requests_remain_replicated_when_dcp_is_enabled():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=False,
        dcp_world_size=2,
    )
    req_ids = ["request-a", "request-b", "request-c"]

    assert _rank_request_ids(manager, 0, req_ids) == req_ids
    assert _rank_request_ids(manager, 1, req_ids) == req_ids


def test_decode_sharding_allows_ranks_with_no_owned_request():
    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    req_ids = ["request-a"]

    assert _rank_request_ids(manager, 0, req_ids) == req_ids
    assert _rank_request_ids(manager, 1, req_ids) == []
    assert _rank_request_ids(manager, 2, req_ids) == []
    assert _rank_request_ids(manager, 3, req_ids) == []


def test_prefill_partitioning_is_preserved_with_sharded_decode():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    segments_by_rank = [
        manager._get_rank_segments(
            rank=rank,
            num_scheduled_tokens=np.array([8, 1, 0, 1], dtype=np.int32),
            is_prefilling=np.array([True, False, False, False]),
            query_start_loc_np=np.array([0, 8, 9, 9, 10], dtype=np.int32),
        )
        for rank in range(manager.pcp_world_size)
    ]

    prefill_tokens = sorted(
        token_idx
        for segments in segments_by_rank
        for segment in segments
        if segment.global_batch_req_idx == 0
        for token_idx in range(
            segment.global_batch_slice.start, segment.global_batch_slice.stop
        )
    )
    decode_owners = {
        segment.global_batch_req_idx: rank
        for rank, segments in enumerate(segments_by_rank)
        for segment in segments
        if segment.global_batch_req_idx in (1, 3)
    }

    assert prefill_tokens == list(range(8))
    assert decode_owners == {1: 0, 3: 1}


def test_sharded_decode_layout_selects_owner_kv_for_replication(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_tensor_h2d", _copy_to_cpu)
    manager._build_batch_layout(
        num_scheduled_tokens=np.array([1, 1, 1], dtype=np.int32),
        num_computed_tokens=np.array([16, 16, 16], dtype=np.int32),
        is_prefilling=np.array([False, False, False]),
        query_start_loc_np=np.array([0, 1, 2, 3], dtype=np.int32),
    )

    gathered_slot_mapping = manager._convert_to_gathered_slot_mappings(
        torch.tensor([[123, 456, 789]], dtype=torch.int64)
    )
    assert torch.equal(
        gathered_slot_mapping,
        torch.tensor([[123, 789, 456, PAD_SLOT_ID]], dtype=torch.int64),
    )
    assert torch.equal(manager._hidden_restore_idx, torch.tensor([0, 2, 1]))

    class FakePCPGroup:
        world_size = 2

        def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
            assert dim == 0
            assert tensor.shape == (2, 1)
            return torch.tensor([[11.0], [33.0], [22.0], [0.0]])

    monkeypatch.setattr(attention_pcp, "get_pcp_group", FakePCPGroup)
    (gathered_kv,), cache_slot_mapping = attention_pcp._gather_prefill_cache_inputs(
        (torch.tensor([[11.0], [33.0]]),),
        gathered_slot_mapping[0],
        num_decode_tokens=0,
        shard_decode_requests=True,
    )

    assert torch.equal(gathered_kv, torch.tensor([[11.0], [33.0], [22.0], [0.0]]))
    assert torch.equal(cache_slot_mapping, torch.tensor([123, 789, 456, PAD_SLOT_ID]))


def test_replicated_draft_cache_inputs_keep_unexpanded_slot_mapping(monkeypatch):
    def unexpected_gather():
        pytest.fail("replicated draft decode must not gather PCP cache inputs")

    monkeypatch.setattr(attention_pcp, "get_pcp_group", unexpected_gather)
    kv = torch.tensor([[11.0], [22.0], [0.0]])
    slot_mapping = torch.tensor([123, 456])

    (cache_kv,), cache_slot_mapping = attention_pcp._gather_prefill_cache_inputs(
        (kv,), slot_mapping, num_decode_tokens=2, shard_decode_requests=True
    )

    assert torch.equal(cache_kv, kv[:2])
    assert torch.equal(cache_slot_mapping, slot_mapping)


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
        shard_decode_requests=False,
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
            shard_decode_requests=False,
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
            shard_decode_requests=False,
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
        shard_decode_requests=False,
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

    local_batch = manager.partition_batch(
        global_batch,
        BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=4,
            num_reqs=4,
        ),
    )

    assert local_batch.dcp_local_seq_lens is None
    assert local_batch.seq_lens.tolist() == [17, 25, 0, 0]
    assert local_batch.pcp is not None
    assert local_batch.pcp.dcp_local_seq_lens_cpu_upper_bound is not None
    expected_dcp_upper_bound = get_dcp_local_seq_lens(
        torch.tensor([17, 25], dtype=torch.int32), 2, 0, 1
    )
    assert torch.equal(
        local_batch.pcp.dcp_local_seq_lens_cpu_upper_bound,
        torch.cat([expected_dcp_upper_bound, torch.zeros(2, dtype=torch.int32)]),
    )

    # What execute_model does next: derive DCP metadata from the final batch
    # on the PCP-owned buffers.
    local_batch.dcp_local_seq_lens = gpu_cp_utils.prepare_dcp_local_seq_lens(
        manager.input_buffers.dcp_local_seq_lens,
        local_batch.seq_lens,
        local_batch.num_reqs,
        dcp_size=2,
        dcp_rank=0,
        cp_interleave=1,
        num_reqs_padded=local_batch.num_reqs_after_padding,
    )
    expected = get_dcp_local_seq_lens(
        torch.tensor([17, 25, 0, 0], dtype=torch.int32), 2, 0, 1
    )
    assert local_batch.dcp_local_seq_lens is not None
    assert torch.equal(local_batch.dcp_local_seq_lens.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU kernels")
def test_moe_padding_matches_pcp_all_gather(monkeypatch):
    """The PCP batch carries the all-gather of every rank's local padding mask,
    the rows the MoE routes, even when ranks own different numbers of tokens."""
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    device = torch.device("cuda:0")
    local_masks, batch_masks = [], []
    for pcp_rank in range(2):
        manager = PCPManager(
            pcp_world_size=2,
            pcp_rank=pcp_rank,
            device=device,
            shard_decode_requests=True,
            max_num_reqs=4,
            max_num_tokens=8,
            moe_gathers_tokens=True,
        )
        global_batch = _make_global_decode_batch(
            [16, 24, 32], InputBuffers(4, 8, device), device
        )
        local_batch = manager.partition_batch(
            global_batch,
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=4, num_reqs=None
            ),
        )
        num_tokens = local_batch.num_tokens_after_padding
        local_masks.append(manager.input_buffers.is_padding[:num_tokens].cpu())
        batch_masks.append(local_batch.is_padding.clone())

    # Sharded decode puts two requests on rank 0 and one on rank 1.
    assert [mask.logical_not().sum().item() for mask in local_masks] == [2, 1]
    for batch_mask in batch_masks:
        assert torch.equal(batch_mask.cpu(), torch.cat(local_masks))
        num_tokens = batch_mask.numel()
        hidden_states = torch.randn(num_tokens, 4, device=device)
        router_logits = torch.randn(num_tokens, 8, device=device)
        with set_forward_context(None, VllmConfig(), is_padding=batch_mask):
            _, topk_ids, _ = fused_topk(hidden_states, router_logits, 2, False)
        assert (topk_ids[~batch_mask] >= 0).all()
        assert (topk_ids[batch_mask] == -1).all()


@pytest.mark.parametrize(
    ("pcp_world_size", "dcp_world_size", "expected"),
    [(1, 1, False), (2, 1, True), (2, 2, False)],
)
def test_parallel_config_manages_decode_sharding(
    pcp_world_size: int, dcp_world_size: int, expected: bool
):
    parallel_config = ParallelConfig(
        prefill_context_parallel_size=pcp_world_size,
        decode_context_parallel_size=dcp_world_size,
        distributed_executor_backend="ray",
    )

    assert parallel_config.pcp_shard_decode_requests is expected


def test_apply_prefill_as_decode_mask_validates_shape():
    flags = np.array([True, False, True], dtype=np.bool_)
    assert np.array_equal(
        apply_prefill_as_decode_mask(flags, None),
        flags,
    )
    assert np.array_equal(
        apply_prefill_as_decode_mask(
            flags, np.array([True, False, False], dtype=np.bool_)
        ),
        np.array([False, False, True], dtype=np.bool_),
    )
    with pytest.raises(ValueError, match="does not match"):
        apply_prefill_as_decode_mask(flags, np.array([True, False], dtype=np.bool_))


def test_num_tokens_for_dispatch_accounts_for_prefill_running_as_decode():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
    )
    num_scheduled_tokens = np.array([4, 4], dtype=np.int32)
    is_prefilling = np.array([False, True], dtype=np.bool_)
    # Without override: decode (4 tokens on rank 0) + split prefill
    # (2 tokens per rank) -> max 6.
    assert manager.get_num_tokens_for_dispatch(num_scheduled_tokens, is_prefilling) == 6
    # With override: 2 sharded decodes (rank 0 gets req 0, rank 1 gets req 1) -> max 4.
    prefill_runs_as_decode = np.array([False, True], dtype=np.bool_)
    assert (
        manager.get_num_tokens_for_dispatch(
            num_scheduled_tokens, is_prefilling, prefill_runs_as_decode
        )
        == 4
    )


def test_partition_batch_runs_cached_single_token_prefill_as_sharded_decode(
    monkeypatch,
):
    device = torch.device("cpu")
    monkeypatch.setattr(pcp_manager_module, "async_tensor_h2d", _copy_to_cpu)

    # Global batch:
    # req 0: ordinary decode (computed=16, prefill_len=16, scheduled=1)
    # req 1: cached 1-token prompt tail running as decode
    #        (computed=127, prefill_len=128, scheduled=1)
    # req 2: multi-token prefill (computed=0, prefill_len=8, scheduled=8)
    global_buffers = InputBuffers(4, 16, device)
    base = InputBatch.make_dummy(3, 10, global_buffers)
    query_start_loc_np = np.array([0, 1, 2, 10], dtype=np.int32)
    global_buffers.query_start_loc[:4] = torch.from_numpy(query_start_loc_np)
    global_buffers.input_ids[:10] = torch.arange(100, 110, dtype=torch.int32)
    positions_np = np.array([16, 127, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    global_buffers.positions[:10] = torch.from_numpy(positions_np)
    seq_lens_np = np.array([17, 128, 8], dtype=np.int32)
    global_buffers.seq_lens[:3] = torch.from_numpy(seq_lens_np)

    global_batch = replace(
        base,
        req_ids=["d0", "tail1", "p2"],
        num_reqs=3,
        num_reqs_after_padding=3,
        idx_mapping=torch.arange(3, dtype=torch.int64, device=device),
        idx_mapping_np=np.arange(3, dtype=np.int64),
        num_scheduled_tokens=np.array([1, 1, 8], dtype=np.int32),
        num_tokens=10,
        num_tokens_after_padding=10,
        query_start_loc=global_buffers.query_start_loc[:4],
        query_start_loc_np=query_start_loc_np,
        seq_lens=global_buffers.seq_lens[:3],
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_np),
        num_computed_tokens_np=np.array([16, 127, 0], dtype=np.int32),
        prefill_len_np=np.array([16, 128, 8], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([16, 127, 0], dtype=np.int32),
        is_prefilling_np=np.array([False, True, True], dtype=np.bool_),
        has_prefill=True,
        prefill_runs_as_decode_np=np.array([False, True, False], dtype=np.bool_),
        decode_graph_eligible=False,
        input_ids=global_buffers.input_ids[:10],
        positions=global_buffers.positions[:10],
        is_padding=global_buffers.is_padding[:10],
    )

    local_batches = []
    for pcp_rank in range(2):
        manager = PCPManager(
            pcp_world_size=2,
            pcp_rank=pcp_rank,
            device=device,
            shard_decode_requests=True,
            max_num_reqs=4,
            max_num_tokens=16,
        )
        local_batch = manager.partition_batch(
            global_batch,
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=5, num_reqs=None
            ),
        )
        local_batches.append((manager, local_batch))

    # Rank 0 owns decode d0 + two chunks of p2 (offsets 0..2 and 6..8).
    # Rank 1 owns converted decode tail1 + two chunks of p2 (offsets 2..4 and 4..6).
    _, batch_r0 = local_batches[0]
    _, batch_r1 = local_batches[1]
    assert batch_r0.req_ids == ["d0", "p2", "p2"]
    assert batch_r1.req_ids == ["tail1", "p2", "p2"]
    assert batch_r0.is_prefilling_np.tolist() == [False, True, True]
    assert batch_r0.prefill_runs_as_decode_np is None
    assert batch_r1.is_prefilling_np.tolist() == [False, True, True]
    assert batch_r1.prefill_runs_as_decode_np is None
    # Global batch lifecycle flags remain untouched for sampling / prompt logprobs.
    assert global_batch.is_prefilling_np.tolist() == [False, True, True]
    assert global_batch.has_prefill is True


def test_pcp_full_decode_graph_allows_short_prefill_running_as_decode(monkeypatch):
    device = torch.device("cpu")
    monkeypatch.setattr(pcp_manager_module, "async_tensor_h2d", _copy_to_cpu)

    # Single request in global batch: cached 1-token prompt tail running as decode.
    # Rank 0 owns it; Rank 1 gets a 0-token placeholder pointing to req 0.
    global_buffers = InputBuffers(4, 8, device)
    global_batch = _make_global_decode_batch([127], global_buffers, device)
    global_batch = replace(
        global_batch,
        prefill_len_np=np.array([128], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([127], dtype=np.int32),
        is_prefilling_np=np.array([True], dtype=np.bool_),
        has_prefill=True,
        prefill_runs_as_decode_np=np.array([True], dtype=np.bool_),
        decode_graph_eligible=True,
    )
    assert not has_requests_to_run_as_prefill(global_batch)

    seen_is_prefilling: list[torch.Tensor] = []
    seen_max_query_len: list[int] = []

    def _fake_build_attn_metadata(**kwargs):
        seen_is_prefilling.append(kwargs["is_prefilling"])
        seen_max_query_len.append(kwargs["max_query_len"])
        return {}

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_states.default.build_attn_metadata",
        _fake_build_attn_metadata,
    )
    model_state = object.__new__(DefaultModelState)
    model_state.model_config = SimpleNamespace(max_model_len=2048)
    model_state.mm_prefix_ranges = None

    for pcp_rank in range(2):
        manager = PCPManager(
            pcp_world_size=2,
            pcp_rank=pcp_rank,
            device=device,
            shard_decode_requests=True,
            max_num_reqs=4,
            max_num_tokens=8,
        )
        local_batch = manager.partition_batch(
            global_batch,
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=2,
                num_reqs=2,
                uniform_token_count=1,
            ),
        )
        assert local_batch.num_reqs_after_padding == 2
        assert local_batch.num_tokens_after_padding == 2
        assert local_batch.has_prefill is False
        assert local_batch.decode_graph_eligible is True
        assert local_batch.max_query_len == 1
        assert not has_requests_to_run_as_prefill(local_batch)
        assert local_batch.prefill_runs_as_decode_np is None
        assert local_batch.is_prefilling_np.tolist() == [False, False]
        if pcp_rank == 0:
            # Rank 0 owns the converted request (padded to 2 reqs).
            assert local_batch.num_tokens == 1
        else:
            # Rank 1 is empty (0 tokens): placeholder must not appear as prefilling.
            assert local_batch.num_tokens == 0

        model_state.prepare_attn(
            local_batch,
            CUDAGraphMode.FULL,
            block_tables=(),
            slot_mappings=torch.zeros((1, 4), dtype=torch.int64),
            attn_groups=[],
            kv_cache_config=SimpleNamespace(),
        )

    assert [t.tolist() for t in seen_is_prefilling] == [
        [False, False],
        [False, False],
    ]
    assert seen_max_query_len == [1, 1]


def test_pcp_sharded_decode_keeps_global_padded_reqs_at_least_num_reqs(monkeypatch):
    """When PCP shards a 4-request uniform decode batch across 2 ranks,
    batch_desc.num_reqs is the rank-local request count (2). Global input
    preparation before partition_batch must still allocate for all 4 global
    requests before partition_batch slices each rank's 2 local requests.
    """
    device = torch.device("cpu")
    monkeypatch.setattr(pcp_manager_module, "async_tensor_h2d", _copy_to_cpu)

    global_buffers = InputBuffers(4, 8, device)
    global_batch = _make_global_decode_batch([64, 64, 127, 127], global_buffers, device)
    global_batch = replace(
        global_batch,
        prefill_len_np=np.array([64, 64, 128, 128], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([64, 64, 127, 127], dtype=np.int32),
        is_prefilling_np=np.array([False, False, True, True], dtype=np.bool_),
        has_prefill=True,
        prefill_runs_as_decode_np=np.array([False, False, True, True], dtype=np.bool_),
        decode_graph_eligible=True,
    )
    batch_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=2,
        num_reqs=2,
        uniform_token_count=1,
    )
    num_reqs_padded = max(
        batch_desc.num_reqs or global_batch.num_reqs, global_batch.num_reqs
    )
    assert num_reqs_padded == 4

    for pcp_rank in range(2):
        manager = PCPManager(
            pcp_world_size=2,
            pcp_rank=pcp_rank,
            device=device,
            shard_decode_requests=True,
            max_num_reqs=4,
            max_num_tokens=8,
        )
        local_batch = manager.partition_batch(global_batch, batch_desc)
        assert local_batch.num_reqs == 2
        assert local_batch.num_reqs_after_padding == 2
        assert local_batch.decode_graph_eligible is True


def test_pcp_full_decode_graph_mamba_hybrid_handles_padded_local_is_prefilling(
    monkeypatch,
):
    """When 3 global decode/short-prefill requests are sharded across 2 PCP ranks
    and replay a 2-request FULL decode graph, rank 1 has num_local_reqs=1 and
    num_reqs_after_padding=2. MambaHybridModelState.prepare_attn must slice
    local_batch.is_prefilling_np to local_batch.num_reqs before copying into its
    padded is_prefilling buffer.
    """
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    device = torch.device("cpu")
    monkeypatch.setattr(pcp_manager_module, "async_tensor_h2d", _copy_to_cpu)

    global_buffers = InputBuffers(4, 8, device)
    global_batch = _make_global_decode_batch([64, 127, 64], global_buffers, device)
    global_batch = replace(
        global_batch,
        prefill_len_np=np.array([64, 128, 64], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([64, 127, 64], dtype=np.int32),
        is_prefilling_np=np.array([False, True, False], dtype=np.bool_),
        has_prefill=True,
        prefill_runs_as_decode_np=np.array([False, True, False], dtype=np.bool_),
        decode_graph_eligible=True,
    )

    seen_mamba_is_prefilling: list[torch.Tensor] = []

    def _fake_build_attn_metadata(**kwargs):
        seen_mamba_is_prefilling.append(
            kwargs["model_specific_attn_metadata"].is_prefilling
        )
        return {}

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_states.mamba_hybrid.build_attn_metadata",
        _fake_build_attn_metadata,
    )
    model_state = object.__new__(MambaHybridModelState)
    model_state.model_config = SimpleNamespace(max_model_len=2048)
    model_state.vllm_config = SimpleNamespace(num_speculative_tokens=1)
    model_state.num_accepted_tokens_gpu = torch.ones(4, dtype=torch.int32)
    model_state.prefix_caching = False
    model_state._use_replayssm_trackers = False
    model_state.recoverssm = None

    for pcp_rank in range(2):
        manager = PCPManager(
            pcp_world_size=2,
            pcp_rank=pcp_rank,
            device=device,
            shard_decode_requests=True,
            max_num_reqs=4,
            max_num_tokens=8,
        )
        local_batch = manager.partition_batch(
            global_batch,
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=2,
                num_reqs=2,
                uniform_token_count=1,
            ),
        )
        model_state.prepare_attn(
            local_batch,
            CUDAGraphMode.FULL,
            block_tables=(),
            slot_mappings=torch.zeros((1, 4), dtype=torch.int64),
            attn_groups=[],
            kv_cache_config=SimpleNamespace(),
        )

    assert [t.tolist() for t in seen_mamba_is_prefilling] == [
        [False, False],
        [False, False],
    ]


def test_pcp_sharded_decode_execute_model_scales_reqs_for_dispatch(
    monkeypatch,
):
    """Under PCP decode sharding, execute_model dispatches target CUDA graphs
    with the sharded per-rank request count via get_num_reqs_for_dispatch.
    """
    from unittest.mock import Mock

    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu import model_runner as model_runner_module
    from vllm.v1.worker.gpu.dp_utils import DPSyncState
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
    )
    assert manager.get_num_reqs_for_dispatch(4, decode_graph_eligible=True) == 2
    assert manager.get_num_reqs_for_dispatch(3, decode_graph_eligible=True) == 2
    assert manager.get_num_reqs_for_dispatch(4, decode_graph_eligible=False) == 4
    runner = Mock(
        lora_config=None,
        aux_output_connector=None,
        encoder_cache=None,
        pcp_manager=manager,
        is_encoder_decoder=False,
        max_num_reqs=8,
        decode_query_len=1,
        dp_size=2,
        dp_rank=0,
        parallel_config=SimpleNamespace(),
        ubatch_runner=None,
        cudagraph_manager=object(),
        observability_config=SimpleNamespace(cudagraph_metrics=False),
    )
    runner.gather_batch_req_state.return_value = (
        SimpleNamespace(
            num_tokens=4,
            num_scheduled_tokens=np.array([1, 1, 1, 1], dtype=np.int32),
            is_prefilling_np=np.array([False, False, True, True], dtype=np.bool_),
            prefill_runs_as_decode_np=np.array(
                [False, False, True, True], dtype=np.bool_
            ),
            has_prefill=True,
            decode_graph_eligible=True,
        ),
        1,
    )
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {f"r{i}": 1 for i in range(4)}
    scheduler_output.total_num_scheduled_tokens = 4

    seen_dispatch_args = {}

    def _fake_dispatch(
        cudagraph_manager, num_reqs, num_toks, uniform_tok_count, *a, **kw
    ):
        seen_dispatch_args["num_reqs"] = num_reqs
        seen_dispatch_args["num_reqs_for_dispatch"] = kw.get("num_reqs_for_dispatch")
        seen_dispatch_args["num_toks"] = num_toks
        seen_dispatch_args["uniform_tok_count"] = uniform_tok_count
        return (
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=2,
                num_reqs=kw.get("num_reqs_for_dispatch", num_reqs),
                uniform_token_count=1,
            ),
            DPSyncState(
                num_tokens_across_dp=torch.tensor([2, 2], dtype=torch.int32),
                uniform_token_count=1,
                max_query_len=1,
                eager=False,
                num_reqs=num_reqs,
            ),
        )

    monkeypatch.setattr(model_runner_module, "dispatch_cg_and_sync_dp", _fake_dispatch)
    runner.prepare_inputs.side_effect = RuntimeError("stop after dispatch")

    with pytest.raises(RuntimeError, match="stop after dispatch"):
        GPUModelRunner.execute_model(runner, scheduler_output)

    assert seen_dispatch_args == {
        "num_reqs": 4,
        "num_reqs_for_dispatch": 2,
        "num_toks": 2,
        "uniform_tok_count": 1,
    }

    from vllm.v1.worker.gpu import dp_utils as dp_utils_module

    monkeypatch.setattr(dp_utils_module, "should_skip_dp_coordination", lambda: True)
    seen_cg_reqs: list[int] = []

    class _FakeCudaGraphManager:
        def dispatch(
            self, num_reqs, num_tokens, uniform_token_count, **kwargs
        ) -> BatchExecutionDescriptor:
            seen_cg_reqs.append(num_reqs)
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                uniform_token_count=uniform_token_count,
            )

    batch_desc, dp_sync = dp_utils_module.dispatch_cg_and_sync_dp(
        _FakeCudaGraphManager(),  # type: ignore[arg-type]
        num_reqs=4,
        num_tokens=2,
        uniform_token_count=1,
        dp_size=2,
        dp_rank=0,
        max_query_len=1,
        num_reqs_for_dispatch=2,
    )
    assert seen_cg_reqs == [2, 2]
    assert batch_desc.num_reqs == 2
    assert dp_sync is not None
    assert dp_sync.num_reqs == 4
