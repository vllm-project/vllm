# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu.pcp_manager import (
    PCPManager,
    _get_global_prefill_max_seq_len,
    _validate_owner_history_axis,
)


class _BlockTablesShape:
    num_kv_cache_groups = 2


def _make_owner_slot_manager(max_num_tokens: int = 8) -> PCPManager:
    return PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
        max_num_tokens=max_num_tokens,
        block_tables=_BlockTablesShape(),  # type: ignore[arg-type]
        dcp_world_size=4,
        dcp_rank=0,
        cp_interleave=64,
        owner_history_enabled=True,
    )


def _old_gather_and_compact(
    block_numbers: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
    cp_size: int,
    cp_interleave: int,
) -> torch.Tensor:
    """Reference the removed DCP mask -> all-gather -> compact path."""
    block_offsets = positions % (block_size * cp_size)
    owner_rank = block_offsets // cp_interleave % cp_size
    rounds = block_offsets // (cp_interleave * cp_size)
    local_offsets = rounds * cp_interleave + block_offsets % cp_interleave
    owner_local_slots = block_numbers * block_size + local_offsets
    rank_local_rows = torch.stack(
        [
            torch.where(
                owner_rank == rank,
                owner_local_slots,
                torch.full_like(owner_local_slots, -1),
            )
            for rank in range(cp_size)
        ]
    )
    valid = rank_local_rows >= 0
    compacted_owner_rank = valid.to(torch.int64).argmax(dim=0)
    compacted_local_slot = rank_local_rows.gather(
        0, compacted_owner_rank.unsqueeze(0)
    ).squeeze(0)
    return torch.stack((compacted_owner_rank, compacted_local_slot), dim=-1)


def test_direct_owner_math_matches_removed_dcp_compaction() -> None:
    block_size = 64
    cp_size = 4
    cp_interleave = 64
    positions = torch.tensor(
        [0, 63, 64, 127, 128, 191, 192, 255, 256, 319, 511],
        dtype=torch.int64,
    )
    block_numbers = torch.tensor(
        [5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7],
        dtype=torch.int64,
    )

    block_offsets = positions % (block_size * cp_size)
    owner_rank = block_offsets // cp_interleave % cp_size
    rounds = block_offsets // (cp_interleave * cp_size)
    local_offsets = rounds * cp_interleave + block_offsets % cp_interleave
    direct = torch.stack(
        (owner_rank, block_numbers * block_size + local_offsets), dim=-1
    )
    old = _old_gather_and_compact(
        block_numbers, positions, block_size, cp_size, cp_interleave
    )
    torch.testing.assert_close(direct, old)


def test_owner_slots_expand_to_pcp_rank_major_layout_and_pad(monkeypatch) -> None:
    manager = _make_owner_slot_manager()

    def fail_if_collective_is_requested():
        pytest.fail("owner slot preparation must not request a DCP collective")

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.pcp_manager.get_dcp_group",
        fail_if_collective_is_requested,
    )
    manager._padded_gather_idx = torch.tensor(
        [0, 2, 1, 3, 0, 0, 0, 0], dtype=torch.int64
    )
    manager._gathered_kv_write_mask = torch.tensor(
        [True, True, True, True, False, False, False, False]
    )
    global_owner_slots = torch.tensor(
        [
            [[0, 10], [1, 11], [2, 12], [3, 13]],
            [[3, 20], [2, 21], [1, 22], [0, 23]],
        ],
        dtype=torch.int64,
    )

    gathered = manager._convert_to_gathered_owner_slot_mappings(global_owner_slots)
    expected = torch.tensor(
        [
            [
                [0, 10],
                [2, 12],
                [1, 11],
                [3, 13],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
            ],
            [
                [3, 20],
                [1, 22],
                [2, 21],
                [0, 23],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
            ],
        ],
        dtype=torch.int64,
    )
    torch.testing.assert_close(gathered, expected)
    local_slots = torch.zeros((2, 8), dtype=torch.int64)
    assert manager.get_direct_owner_slot_mappings(local_slots).data_ptr() == (
        gathered.data_ptr()
    )


def test_dummy_owner_slots_keep_forward_shape_and_padding(monkeypatch) -> None:
    manager = _make_owner_slot_manager(max_num_tokens=6)

    def fail_if_collective_is_requested():
        pytest.fail("dummy owner slots must not request a DCP collective")

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.pcp_manager.get_dcp_group",
        fail_if_collective_is_requested,
    )
    local_slots = manager.get_dummy_slot_mappings(num_tokens=3)
    owner_slots = manager.get_direct_owner_slot_mappings(local_slots)

    assert local_slots.shape == (2, 12)
    assert owner_slots.shape == (2, 12, 2)
    assert torch.all(local_slots == -1)
    assert torch.all(owner_slots == -1)


def test_owner_history_axis_accepts_pcp4_dcp4() -> None:
    _validate_owner_history_axis(4, 4)


@pytest.mark.parametrize(
    "pcp_size,dcp_size",
    [(2, 1), (4, 1), (2, 2), (8, 8), (4, 2)],
)
def test_owner_history_axis_rejects_unsupported_sizes(
    pcp_size: int, dcp_size: int
) -> None:
    with pytest.raises(NotImplementedError, match="PCP=4 and DCP=4"):
        _validate_owner_history_axis(pcp_size, dcp_size)


def test_owner_sharded_property_fails_closed_for_pcp2_dcp2() -> None:
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        dcp_world_size=2,
        owner_history_enabled=True,
    )
    with pytest.raises(NotImplementedError, match="PCP=4 and DCP=4"):
        _ = manager.owner_sharded_history_enabled


def test_global_prefill_max_is_scheduler_global_and_ignores_decodes() -> None:
    assert (
        _get_global_prefill_max_seq_len(
            torch.tensor([2048, 8192, 1536], dtype=torch.int32),
            is_prefilling=torch.tensor([True, False, True]).numpy(),
            num_reqs=3,
        )
        == 2048
    )
    assert (
        _get_global_prefill_max_seq_len(
            torch.tensor([8192], dtype=torch.int32),
            is_prefilling=torch.tensor([False]).numpy(),
            num_reqs=1,
        )
        == 0
    )
