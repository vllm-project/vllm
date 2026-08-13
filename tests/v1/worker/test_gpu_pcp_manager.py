# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.model_executor.layers.attention import pcp as attention_pcp
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.pcp_manager import PCPManager


def _copy_to_cpu(value, out=None, device=None):
    tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    if out is not None:
        return out.copy_(tensor)
    return tensor


def test_sharded_decode_piecewise_graph_padding(monkeypatch):
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
            padded_num_tokens=1,
        )


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
        num_computed_tokens=np.full(num_reqs, 16, dtype=np.int32),
        is_prefilling=is_prefilling,
        query_start_loc_np=np.arange(num_reqs + 1, dtype=np.int32),
    )
    return [req_ids[segment.global_batch_req_idx] for segment in segments]


def test_pcp_only_decode_requests_are_round_robin_balanced_each_step():
    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
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
        dcp_world_size=2,
    )
    req_ids = ["request-a", "request-b", "request-c"]

    assert _rank_request_ids(manager, 0, req_ids) == req_ids
    assert _rank_request_ids(manager, 1, req_ids) == req_ids


def test_prefill_partitioning_is_preserved_with_sharded_decode():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        dcp_world_size=1,
    )
    segments_by_rank = [
        manager._get_rank_segments(
            rank=rank,
            num_scheduled_tokens=np.array([8, 1, 0, 1], dtype=np.int32),
            num_computed_tokens=np.array([0, 16, 16, 16], dtype=np.int32),
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
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)
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
