# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.v1.worker.gpu.pcp_manager import PCPManager


def _manager(rank: int, size: int = 2, dcp_size: int = 1) -> PCPManager:
    return PCPManager(
        pcp_world_size=size,
        pcp_rank=rank,
        device=torch.device("cpu"),
        dcp_world_size=dcp_size,
    )


def test_dual_chunk_swap_even_partition() -> None:
    num_scheduled = np.array([8], dtype=np.int32)
    num_computed = np.array([100], dtype=np.int32)
    is_prefilling = np.array([True], dtype=np.bool_)

    assert _manager(rank=0)._get_rank_segments(
        0, num_scheduled, num_computed, is_prefilling
    ) == [(0, 100, 2), (0, 106, 2)]
    assert _manager(rank=1)._get_rank_segments(
        1, num_scheduled, num_computed, is_prefilling
    ) == [(0, 102, 2), (0, 104, 2)]


def test_virtual_request_padding_uses_pcp_multiple() -> None:
    manager = _manager(rank=0, size=4)

    assert manager._pad_num_reqs(1) == 4
    assert manager._pad_num_reqs(4) == 4
    assert manager._pad_num_reqs(5) == 8


def test_dual_chunk_swap_uneven_partition() -> None:
    num_scheduled = np.array([10], dtype=np.int32)
    num_computed = np.array([0], dtype=np.int32)
    is_prefilling = np.array([True], dtype=np.bool_)

    assert _manager(rank=0)._get_rank_segments(
        0, num_scheduled, num_computed, is_prefilling
    ) == [(0, 8, 2), (0, 0, 3)]
    assert _manager(rank=1)._get_rank_segments(
        1, num_scheduled, num_computed, is_prefilling
    ) == [(0, 3, 3), (0, 6, 2)]


def test_short_prefill_allows_empty_rank_segments() -> None:
    num_scheduled = np.array([1], dtype=np.int32)
    num_computed = np.array([0], dtype=np.int32)
    is_prefilling = np.array([True], dtype=np.bool_)

    assert _manager(rank=0)._get_rank_segments(
        0, num_scheduled, num_computed, is_prefilling
    ) == [(0, 0, 1)]
    assert (
        _manager(rank=1)._get_rank_segments(
            1, num_scheduled, num_computed, is_prefilling
        )
        == []
    )


def test_decode_rows_are_duplicated_before_restore() -> None:
    num_scheduled = np.array([1, 8], dtype=np.int32)
    num_computed = np.array([8, 0], dtype=np.int32)
    is_prefilling = np.array([False, True], dtype=np.bool_)

    assert _manager(rank=0)._get_rank_segments(
        0, num_scheduled, num_computed, is_prefilling
    ) == [(0, 8, 1), (1, 6, 2), (1, 0, 2)]
    assert _manager(rank=1)._get_rank_segments(
        1, num_scheduled, num_computed, is_prefilling
    ) == [(0, 8, 1), (1, 2, 2), (1, 4, 2)]


def test_hidden_restore_indices_reconstruct_physical_order() -> None:
    manager = _manager(rank=0)
    num_scheduled = np.array([8], dtype=np.int32)
    num_computed = np.array([0], dtype=np.int32)
    is_prefilling = np.array([True], dtype=np.bool_)
    query_start_loc = np.array([0, 8], dtype=np.int32)

    manager._build_hidden_restore_idx(
        num_scheduled,
        num_computed,
        is_prefilling,
        query_start_loc,
    )

    assert manager._per_rank_num_tokens == [4, 4]
    assert manager._hidden_restore_idx is not None
    assert manager._hidden_restore_idx.tolist() == [2, 3, 4, 5, 6, 7, 0, 1]


def test_hidden_restore_indices_drop_duplicate_decode_rows() -> None:
    manager = _manager(rank=0)
    num_scheduled = np.array([1, 8], dtype=np.int32)
    num_computed = np.array([8, 0], dtype=np.int32)
    is_prefilling = np.array([False, True], dtype=np.bool_)
    query_start_loc = np.array([0, 1, 9], dtype=np.int32)

    manager._build_hidden_restore_idx(
        num_scheduled,
        num_computed,
        is_prefilling,
        query_start_loc,
    )

    assert manager._per_rank_num_tokens == [5, 5]
    assert manager._hidden_restore_idx is not None
    assert manager._hidden_restore_idx.tolist() == [0, 3, 4, 6, 7, 8, 9, 1, 2]


def test_dcp_uses_virtual_token_mapping() -> None:
    manager = _manager(rank=1, dcp_size=2)
    num_scheduled = np.array([1, 8], dtype=np.int32)
    num_computed = np.array([8, 0], dtype=np.int32)
    is_prefilling = np.array([False, True], dtype=np.bool_)
    query_start_loc = np.array([0, 1, 9], dtype=np.int32)

    manager._build_hidden_restore_idx(
        num_scheduled,
        num_computed,
        is_prefilling,
        query_start_loc,
    )

    assert not manager.dcp_passthrough
    assert manager._local_virtual_to_physical_idx is not None
    assert manager._local_virtual_to_physical_idx.tolist() == [0, 3, 4, 5, 6]


def test_mla_cache_gather_exchanges_single_latent_payload() -> None:
    manager = _manager(rank=0)
    gathered_payloads: list[torch.Tensor] = []

    def fake_allgather_and_restore_tokens(tensor: torch.Tensor) -> torch.Tensor:
        gathered_payloads.append(tensor.clone())
        return torch.flip(tensor, dims=(0,))

    manager._allgather_and_restore_tokens = fake_allgather_and_restore_tokens  # type: ignore[method-assign]
    manager.set_physical_slot_mappings(
        {
            "layer": torch.tensor([40, 41, 42, 43], dtype=torch.int64),
        }
    )

    kv_c_normed = torch.arange(12, dtype=torch.float32).view(4, 3)
    k_pe = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
    slot_mapping = torch.arange(4, dtype=torch.int64)

    restored_kv_c, restored_k_pe, restored_slots = (
        manager.gather_and_restore_mla_latent_cache_inputs(
            kv_c_normed,
            k_pe,
            slot_mapping,
            "layer",
        )
    )

    assert len(gathered_payloads) == 1
    assert gathered_payloads[0].shape == (4, 5)
    torch.testing.assert_close(gathered_payloads[0][:, :3], kv_c_normed)
    torch.testing.assert_close(gathered_payloads[0][:, 3:], k_pe.view(4, 2))
    torch.testing.assert_close(restored_kv_c, torch.flip(kv_c_normed, dims=(0,)))
    torch.testing.assert_close(restored_k_pe, torch.flip(k_pe, dims=(0,)))
    torch.testing.assert_close(restored_slots, torch.tensor([40, 41, 42, 43]))


def test_indexer_cache_gather_exchanges_only_latent_k() -> None:
    manager = _manager(rank=0)
    gathered_payloads: list[torch.Tensor] = []

    def fake_allgather_and_restore_tokens(tensor: torch.Tensor) -> torch.Tensor:
        gathered_payloads.append(tensor.clone())
        return torch.flip(tensor, dims=(0,))

    manager._allgather_and_restore_tokens = fake_allgather_and_restore_tokens  # type: ignore[method-assign]
    manager.set_physical_slot_mappings(
        {
            "indexer.k_cache": torch.tensor([80, 81, 82, 83], dtype=torch.int64),
        }
    )

    k = torch.arange(16, dtype=torch.float32).view(4, 4)
    slot_mapping = torch.arange(4, dtype=torch.int64)

    restored_k, restored_slots = manager.gather_and_restore_indexer_cache_inputs(
        k,
        slot_mapping,
        "indexer.k_cache",
    )

    assert len(gathered_payloads) == 1
    torch.testing.assert_close(gathered_payloads[0], k)
    torch.testing.assert_close(restored_k, torch.flip(k, dims=(0,)))
    torch.testing.assert_close(restored_slots, torch.tensor([80, 81, 82, 83]))
