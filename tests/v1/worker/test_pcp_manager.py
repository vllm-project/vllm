# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import torch

from vllm.model_executor.layers.attention import pcp as attention_pcp
from vllm.model_executor.layers.attention.pcp import (
    maybe_gather_indexer_k,
    maybe_gather_mla_latent_cache_inputs,
)
from vllm.v1.worker.gpu import pcp_manager
from vllm.v1.worker.gpu.pcp_manager import PCPManager, RankSegment


def _manager(rank: int = 0, size: int = 2) -> PCPManager:
    return PCPManager(
        pcp_world_size=size,
        pcp_rank=rank,
        device=torch.device("cpu"),
    )


def _patch_fake_pcp_group(monkeypatch) -> None:
    class FakePCPGroup:
        world_size = 2

        def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
            assert dim == 0
            return torch.cat((tensor, tensor + 1000), dim=0)

    monkeypatch.setattr(pcp_manager, "get_pcp_group", lambda: FakePCPGroup())
    monkeypatch.setattr(attention_pcp, "get_pcp_group", lambda: FakePCPGroup())


def test_mixed_batch_layout_restores_decodes_once() -> None:
    manager = _manager()
    segments_by_rank, per_rank_num_tokens = manager._build_batch_layout(
        num_scheduled_tokens=np.array([1, 8], dtype=np.int32),
        is_prefilling=np.array([False, True], dtype=np.bool_),
        query_start_loc_np=np.array([0, 1, 9], dtype=np.int32),
    )

    assert segments_by_rank == [
        [
            RankSegment(0, slice(0, 1), slice(0, 1)),
            RankSegment(1, slice(1, 3), slice(1, 3)),
            RankSegment(1, slice(7, 9), slice(3, 5)),
        ],
        [
            RankSegment(0, slice(0, 1), slice(0, 1)),
            RankSegment(1, slice(3, 5), slice(1, 3)),
            RankSegment(1, slice(5, 7), slice(3, 5)),
        ],
    ]
    assert per_rank_num_tokens == [5, 5]
    assert manager._hidden_restore_idx is not None
    assert manager._hidden_restore_idx.tolist() == [
        0,
        1,
        2,
        6,
        7,
        8,
        9,
        3,
        4,
    ]
    assert manager._expanded_is_padding is not None
    assert manager._expanded_is_padding.tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]


def test_restore_hidden_states_handles_uneven_rank_tokens(monkeypatch) -> None:
    _patch_fake_pcp_group(monkeypatch)
    manager = _manager()
    manager._build_batch_layout(
        num_scheduled_tokens=np.array([5], dtype=np.int32),
        is_prefilling=np.array([True], dtype=np.bool_),
        query_start_loc_np=np.array([0, 5], dtype=np.int32),
    )

    restored = manager.restore_hidden_states(torch.tensor([10, 11, 12]))

    torch.testing.assert_close(restored, torch.tensor([10, 11, 1010, 1011, 1012]))


def test_pcp_cache_gather_keeps_decodes_local(monkeypatch) -> None:
    _patch_fake_pcp_group(monkeypatch)
    kv_c = torch.arange(12, dtype=torch.float32).view(4, 3)
    k_pe = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
    indexer_k = torch.arange(16, dtype=torch.float32).view(4, 4)
    slot_mapping = torch.tensor([10, 11, 12, 13, -1, 21, 22, 23])

    gathered_kv_c, gathered_k_pe, mla_slots = maybe_gather_mla_latent_cache_inputs(
        SimpleNamespace(num_decode_tokens=1), kv_c, k_pe, slot_mapping, True
    )
    gathered_indexer_k, indexer_slots = maybe_gather_indexer_k(
        indexer_k, slot_mapping, 1, True
    )

    torch.testing.assert_close(
        gathered_kv_c,
        torch.cat((kv_c[:1], kv_c[1:], kv_c[1:] + 1000)),
    )
    torch.testing.assert_close(
        gathered_k_pe,
        torch.cat((k_pe[:1], k_pe[1:], k_pe[1:] + 1000)),
    )
    torch.testing.assert_close(
        gathered_indexer_k,
        torch.cat((indexer_k[:1], indexer_k[1:], indexer_k[1:] + 1000)),
    )
    expected_slots = torch.tensor([10, 11, 12, 13, 21, 22, 23])
    torch.testing.assert_close(mla_slots, expected_slots)
    torch.testing.assert_close(indexer_slots, expected_slots)


def test_cache_slot_mapping_covers_the_gathered_layout() -> None:
    manager = _manager()
    manager._padded_gather_idx = torch.tensor([0, 1, 2, 0], dtype=torch.int64)
    manager._expanded_is_padding = torch.tensor([False, False, False, True])

    slot_mapping = manager._convert_to_gathered_slot_mappings(
        torch.tensor([[110, 111, 120]], dtype=torch.int64)
    )

    torch.testing.assert_close(
        slot_mapping,
        torch.tensor([[110, 111, 120, -1]], dtype=torch.int64),
    )
