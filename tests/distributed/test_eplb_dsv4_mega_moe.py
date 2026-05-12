# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.deepseek_v4.nvidia.model import (
    _map_mega_moe_logical_to_physical_and_record_load,
)


def test_mega_moe_eplb_maps_logical_to_physical_and_records_load():
    topk_ids = torch.tensor(
        [
            [2, 1],
            [2, -1],
            [3, 2],
        ],
        dtype=torch.int32,
    )
    logical_to_physical_map = torch.tensor(
        [
            [0, -1, -1],
            [1, -1, -1],
            [4, 7, -1],
            [5, 6, 8],
        ],
        dtype=torch.long,
    )
    logical_replica_count = torch.tensor([1, 1, 2, 3], dtype=torch.long)
    expert_load_view = torch.zeros(9, dtype=torch.int32)

    mapped_ids = _map_mega_moe_logical_to_physical_and_record_load(
        topk_ids,
        expert_load_view=expert_load_view,
        logical_to_physical_map=logical_to_physical_map,
        logical_replica_count=logical_replica_count,
    )

    expected_ids = torch.tensor(
        [
            [4, 1],
            [7, -1],
            [8, 4],
        ],
        dtype=topk_ids.dtype,
    )
    expected_load = torch.tensor([0, 1, 0, 0, 2, 0, 0, 1, 1], dtype=torch.int32)
    assert torch.equal(mapped_ids, expected_ids)
    assert torch.equal(expert_load_view, expected_load)


def test_mega_moe_eplb_hot_expert_balances_across_replicas():
    topk_ids = torch.full((8, 4), -1, dtype=torch.int64)
    topk_ids[:, 0] = 0
    logical_to_physical_map = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    logical_replica_count = torch.tensor([4], dtype=torch.long)
    expert_load_view = torch.zeros(4, dtype=torch.int32)

    mapped_ids = _map_mega_moe_logical_to_physical_and_record_load(
        topk_ids,
        expert_load_view=expert_load_view,
        logical_to_physical_map=logical_to_physical_map,
        logical_replica_count=logical_replica_count,
    )

    # Hashing by token avoids slot-0 routes collapsing to replica 0.
    expected_ids = torch.tensor(
        [
            [0, -1, -1, -1],
            [1, -1, -1, -1],
            [2, -1, -1, -1],
            [3, -1, -1, -1],
            [0, -1, -1, -1],
            [1, -1, -1, -1],
            [2, -1, -1, -1],
            [3, -1, -1, -1],
        ],
        dtype=topk_ids.dtype,
    )
    expected_load = torch.tensor([2, 2, 2, 2], dtype=torch.int32)
    assert torch.equal(mapped_ids, expected_ids)
    assert torch.equal(expert_load_view, expected_load)
