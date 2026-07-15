# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.attention import pcp as attention_pcp
from vllm.model_executor.layers.attention.pcp import (
    maybe_gather_indexer_k,
    maybe_gather_mla_latent_cache_inputs,
)
from vllm.v1.worker.gpu import pcp_manager
from vllm.v1.worker.gpu.pcp_manager import (
    PCPBatchLayout,
    PCPManager,
    allgather_tokens,
)


def _manager(rank: int, size: int = 2, dcp_size: int = 1) -> PCPManager:
    return PCPManager(
        pcp_world_size=size,
        pcp_rank=rank,
        device=torch.device("cpu"),
        dcp_world_size=dcp_size,
    )


def _patch_fake_pcp_group(
    monkeypatch,
    gathered_payloads: list[tuple[torch.Tensor, ...]],
) -> None:
    class FakePCPGroup:
        rank_in_group = 0
        world_size = 2

        def all_gather(
            self,
            tensor: torch.Tensor,
            dim: int,
        ) -> torch.Tensor:
            assert dim == 0
            gathered_payloads.append((tensor.clone(),))
            return torch.cat((tensor, tensor + 1000), dim=0)

    monkeypatch.setattr(pcp_manager, "get_pcp_group", lambda: FakePCPGroup())
    monkeypatch.setattr(attention_pcp, "get_pcp_group", lambda: FakePCPGroup())


def _pcp_config(
    *,
    pcp_size: int = 2,
    hf_text_config: SimpleNamespace | None = None,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            hf_text_config=hf_text_config or SimpleNamespace(),
            is_encoder_decoder=False,
            is_multimodal_model=False,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
        lora_config=None,
        speculative_config=None,
    )


def test_sparse_mla_pcp_rejects_cuda_graphs() -> None:
    config = _pcp_config(
        hf_text_config=SimpleNamespace(index_topk=2048),
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
    )

    with pytest.raises(NotImplementedError, match="sparse MLA PCP"):
        PCPManager.validate_config(config, supports_mm_inputs=False)


def test_dual_chunk_swap_starts_after_cached_prefix() -> None:
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


def test_virtual_token_padding_mask_marks_collective_padding() -> None:
    is_padding = torch.ones(8, dtype=torch.bool)

    visible_mask = PCPManager._set_padding_mask(
        is_padding,
        num_tokens=5,
        num_tokens_after_padding=7,
    )

    torch.testing.assert_close(
        visible_mask,
        torch.tensor([False, False, False, False, False, True, True]),
    )
    assert bool(is_padding[7])


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

    layout = manager._build_batch_layout(
        num_scheduled,
        num_computed,
        is_prefilling,
        query_start_loc,
    )

    assert layout.per_rank_num_tokens == [4, 4]
    assert layout.hidden_restore_idx.tolist() == [2, 3, 4, 5, 6, 7, 0, 1]


def test_hidden_restore_indices_drop_duplicate_decode_rows() -> None:
    manager = _manager(rank=0)
    num_scheduled = np.array([1, 8], dtype=np.int32)
    num_computed = np.array([8, 0], dtype=np.int32)
    is_prefilling = np.array([False, True], dtype=np.bool_)
    query_start_loc = np.array([0, 1, 9], dtype=np.int32)

    layout = manager._build_batch_layout(
        num_scheduled,
        num_computed,
        is_prefilling,
        query_start_loc,
    )

    assert layout.per_rank_num_tokens == [5, 5]
    assert layout.hidden_restore_idx.tolist() == [0, 3, 4, 6, 7, 8, 9, 1, 2]


def test_hidden_restore_uses_compacted_padded_allgather(monkeypatch) -> None:
    gathered_payloads: list[tuple[torch.Tensor, ...]] = []
    _patch_fake_pcp_group(monkeypatch, gathered_payloads)
    manager = _manager(rank=0)
    num_scheduled = np.array([5], dtype=np.int32)
    num_computed = np.array([0], dtype=np.int32)
    is_prefilling = np.array([True], dtype=np.bool_)
    query_start_loc = np.array([0, 5], dtype=np.int32)

    layout = manager._build_batch_layout(
        num_scheduled,
        num_computed,
        is_prefilling,
        query_start_loc,
    )
    manager._batch_layout = layout
    restored = manager.restore_hidden_states(torch.tensor([10, 11, 12]))

    assert layout.per_rank_num_tokens == [3, 2]
    assert layout.hidden_restore_idx.tolist() == [1, 2, 3, 4, 0]
    assert len(gathered_payloads) == 1
    torch.testing.assert_close(restored, torch.tensor([11, 12, 1010, 1011, 10]))


def test_mla_cache_gather_exchanges_grouped_latent_payloads(monkeypatch) -> None:
    gathered_payloads: list[tuple[torch.Tensor, ...]] = []
    _patch_fake_pcp_group(monkeypatch, gathered_payloads)

    kv_c_normed = torch.arange(12, dtype=torch.float32).view(4, 3)
    k_pe = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
    slot_mapping = torch.arange(40, 48, dtype=torch.int64)

    restored_kv_c, restored_k_pe, restored_slots = maybe_gather_mla_latent_cache_inputs(
        object(),
        kv_c_normed,
        k_pe,
        slot_mapping,
        True,
    )

    assert len(gathered_payloads) == 2
    torch.testing.assert_close(gathered_payloads[0][0], kv_c_normed)
    torch.testing.assert_close(gathered_payloads[1][0], k_pe.view(4, 2))
    torch.testing.assert_close(
        restored_kv_c,
        torch.cat((kv_c_normed, kv_c_normed + 1000), dim=0),
    )
    torch.testing.assert_close(
        restored_k_pe,
        torch.cat((k_pe, k_pe + 1000), dim=0),
    )
    torch.testing.assert_close(
        restored_slots,
        slot_mapping,
    )


def test_mla_cache_gather_gate_uses_pcp_flag() -> None:
    kv_c_normed = torch.arange(4, dtype=torch.float32)
    k_pe = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
    slot_mapping = torch.arange(4, dtype=torch.int64)

    restored_kv, restored_k_pe, restored_slots = maybe_gather_mla_latent_cache_inputs(
        None,
        kv_c_normed,
        k_pe,
        slot_mapping,
        False,
    )

    assert restored_kv is kv_c_normed
    assert restored_k_pe is k_pe
    assert restored_slots is slot_mapping


def test_indexer_cache_gather_exchanges_only_latent_k(monkeypatch) -> None:
    gathered_payloads: list[tuple[torch.Tensor, ...]] = []
    _patch_fake_pcp_group(monkeypatch, gathered_payloads)

    k = torch.arange(16, dtype=torch.float32).view(4, 4)
    slot_mapping = torch.arange(80, 88, dtype=torch.int64)

    restored_k, restored_slots = maybe_gather_indexer_k(k, slot_mapping, True)

    assert len(gathered_payloads) == 1
    torch.testing.assert_close(gathered_payloads[0][0], k)
    torch.testing.assert_close(restored_k, torch.cat((k, k + 1000), dim=0))
    torch.testing.assert_close(restored_slots, slot_mapping)


def test_slot_mapping_is_not_gathered_with_cache_payload(monkeypatch) -> None:
    gathered_payloads: list[tuple[torch.Tensor, ...]] = []
    _patch_fake_pcp_group(monkeypatch, gathered_payloads)

    kv_c_normed = torch.arange(12, dtype=torch.float32).view(4, 3)
    k_pe = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
    k = torch.arange(16, dtype=torch.float32).view(4, 4)
    slot_mapping = torch.arange(8, dtype=torch.int64)

    _, _, mla_slots = maybe_gather_mla_latent_cache_inputs(
        object(), kv_c_normed, k_pe, slot_mapping, True
    )
    _, indexer_slots = maybe_gather_indexer_k(k, slot_mapping, True)

    assert len(gathered_payloads) == 3
    torch.testing.assert_close(gathered_payloads[0][0], kv_c_normed)
    torch.testing.assert_close(gathered_payloads[1][0], k_pe.view(4, 2))
    torch.testing.assert_close(gathered_payloads[2][0], k)
    torch.testing.assert_close(mla_slots, slot_mapping)
    torch.testing.assert_close(indexer_slots, slot_mapping)


def test_cache_slot_mapping_is_computed_for_gathered_layout() -> None:
    manager = _manager(rank=0, size=2)
    manager._physical_batch = SimpleNamespace(
        idx_mapping_np=np.array([7, 8], dtype=np.int32),
    )
    manager._batch_layout = PCPBatchLayout(
        hidden_restore_idx=torch.empty(0, dtype=torch.int64),
        per_rank_num_tokens=[2, 1],
        rank_segments=[[(0, 10, 2)], [(1, 20, 1)]],
    )

    def fake_compute_slot_mappings(
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert out is not None
        assert num_tokens_padded == 4
        torch.testing.assert_close(
            idx_mapping,
            torch.tensor([7, 8, 0], dtype=torch.int32),
        )
        torch.testing.assert_close(
            query_start_loc,
            torch.tensor([0, 2, 3, 4], dtype=torch.int32),
        )
        torch.testing.assert_close(positions, torch.tensor([10, 11, 20, 0]))
        out[0].copy_(positions.to(out.dtype) + 100)
        return out

    manager._compute_slot_mappings = fake_compute_slot_mappings

    local_slot_mapping = torch.empty(1, 2, dtype=torch.int64)
    cache_slot_mapping = manager._compute_cache_slot_mappings(local_slot_mapping)

    torch.testing.assert_close(
        cache_slot_mapping,
        torch.tensor([[110, 111, 120, -1]], dtype=torch.int64),
    )


def test_dummy_cache_slot_mapping_matches_pcp_gathered_tokens() -> None:
    manager = _manager(rank=0, size=2)
    local_slot_mapping = torch.full((1, 3), -1, dtype=torch.int64)

    cache_slot_mapping = manager.dummy_cache_slot_mappings(local_slot_mapping)

    torch.testing.assert_close(
        cache_slot_mapping,
        torch.full((1, 6), -1, dtype=torch.int64),
    )


def test_padded_allgather_compacts_uneven_rank_tokens(monkeypatch) -> None:
    gathered_payloads: list[tuple[torch.Tensor, ...]] = []
    _patch_fake_pcp_group(monkeypatch, gathered_payloads)

    tensor = torch.tensor([0, 1, -1], dtype=torch.int64)

    gathered_tensor = allgather_tokens(tensor, [2, 3])

    assert len(gathered_payloads) == 1
    torch.testing.assert_close(gathered_payloads[0][0], tensor)
    torch.testing.assert_close(
        gathered_tensor,
        torch.tensor([0, 1, 1000, 1001, 999], dtype=torch.int64),
    )
