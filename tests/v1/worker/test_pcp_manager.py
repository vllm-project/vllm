# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.model_executor.layers.attention import pcp as attention_pcp
from vllm.model_executor.layers.attention.pcp import (
    maybe_gather_indexer_k,
    maybe_gather_mla_latent_cache_inputs,
)
from vllm.v1.worker.gpu import pcp_manager
from vllm.v1.worker.gpu.pcp_manager import PCPBatchLayout, PCPManager


def _manager(rank: int = 0, size: int = 2) -> PCPManager:
    return PCPManager(
        pcp_world_size=size,
        pcp_rank=rank,
        device=torch.device("cpu"),
    )


def _patch_fake_pcp_group(monkeypatch) -> None:
    class FakePCPGroup:
        def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
            assert dim == 0
            return torch.cat((tensor, tensor + 1000), dim=0)

    monkeypatch.setattr(pcp_manager, "get_pcp_group", lambda: FakePCPGroup())
    monkeypatch.setattr(attention_pcp, "get_pcp_group", lambda: FakePCPGroup())


def test_mixed_batch_layout_restores_rank_zero_decodes() -> None:
    manager = _manager()
    layout = manager._build_batch_layout(
        num_scheduled_tokens=np.array([1, 8], dtype=np.int32),
        num_computed_tokens=np.array([8, 0], dtype=np.int32),
        is_prefilling=np.array([False, True], dtype=np.bool_),
        query_start_loc_np=np.array([0, 1, 9], dtype=np.int32),
    )

    assert layout.rank_segments == [
        [(0, 8, 1), (1, 6, 2), (1, 0, 2)],
        [(0, 8, 1), (1, 2, 2), (1, 4, 2)],
    ]
    assert layout.per_rank_num_tokens == [5, 5]
    assert layout.hidden_restore_idx.tolist() == [0, 3, 4, 6, 7, 8, 9, 1, 2]


def test_restore_hidden_states_handles_uneven_rank_tokens(monkeypatch) -> None:
    _patch_fake_pcp_group(monkeypatch)
    manager = _manager()
    manager._batch_layout = manager._build_batch_layout(
        num_scheduled_tokens=np.array([5], dtype=np.int32),
        num_computed_tokens=np.array([0], dtype=np.int32),
        is_prefilling=np.array([True], dtype=np.bool_),
        query_start_loc_np=np.array([0, 5], dtype=np.int32),
    )

    restored = manager.restore_hidden_states(torch.tensor([10, 11, 12]))

    torch.testing.assert_close(restored, torch.tensor([11, 12, 1010, 1011, 10]))


def test_pcp_cache_gather_exchanges_latent_payloads(monkeypatch) -> None:
    _patch_fake_pcp_group(monkeypatch)
    kv_c = torch.arange(12, dtype=torch.float32).view(4, 3)
    k_pe = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
    indexer_k = torch.arange(16, dtype=torch.float32).view(4, 4)
    slot_mapping = torch.arange(8, dtype=torch.int64)

    gathered_kv_c, gathered_k_pe, mla_slots = maybe_gather_mla_latent_cache_inputs(
        object(), kv_c, k_pe, slot_mapping, True
    )
    gathered_indexer_k, indexer_slots = maybe_gather_indexer_k(
        indexer_k, slot_mapping, True
    )

    torch.testing.assert_close(gathered_kv_c, torch.cat((kv_c, kv_c + 1000)))
    torch.testing.assert_close(gathered_k_pe, torch.cat((k_pe, k_pe + 1000)))
    torch.testing.assert_close(
        gathered_indexer_k,
        torch.cat((indexer_k, indexer_k + 1000)),
    )
    torch.testing.assert_close(mla_slots, slot_mapping)
    torch.testing.assert_close(indexer_slots, slot_mapping)


def test_cache_slot_mapping_covers_the_gathered_layout() -> None:
    manager = _manager()
    manager._batch_layout = PCPBatchLayout(
        hidden_restore_idx=torch.empty(0, dtype=torch.int64),
        cache_write_idx=torch.tensor([0, 1, 2, 0], dtype=torch.int64),
        cache_is_padding=torch.tensor([False, False, False, True]),
        per_rank_num_tokens=[2, 1],
        rank_segments=[[(0, 10, 2)], [(1, 20, 1)]],
    )

    slot_mapping = manager._expand_slot_mappings(
        torch.tensor([[110, 111, 120]], dtype=torch.int64)
    )

    torch.testing.assert_close(
        slot_mapping,
        torch.tensor([[110, 111, 120, -1]], dtype=torch.int64),
    )
