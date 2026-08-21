# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.pcp_manager as pcp_module
from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


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
    assert torch.equal(manager._hidden_restore_idx, torch.tensor([0, 1, 2]))
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
def test_partition_reuses_gpu_cursor_for_replicated_spec_decode():
    device = torch.device("cuda")
    global_buffers = InputBuffers(max_num_reqs=1, max_num_tokens=4, device=device)
    global_batch = InputBatch.make_dummy(
        num_reqs=1,
        num_tokens=4,
        input_buffers=global_buffers,
    )

    # Model an async step after rejection: the CPU scheduler cursor is still
    # optimistic, while the GPU cursor used to build positions/seq_lens has
    # already rolled back to the accepted prefix.
    global_batch.num_draft_tokens = 3
    global_batch.num_draft_tokens_per_req = np.array([3], dtype=np.int32)
    global_batch.num_computed_tokens_np[:] = 20
    global_batch.prefill_len_np[:] = 8
    global_batch.num_computed_prefill_tokens_np[:] = 8
    global_batch.positions.copy_(torch.arange(10, 14, device=device))
    global_batch.seq_lens.fill_(14)

    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=device,
        req_states=SimpleNamespace(),
        max_num_reqs=1,
        max_num_tokens=4,
    )
    local_batch = manager.partition_batch(global_batch)

    # q_len > 1 remains one replicated decode row. Its actual device metadata
    # follows the corrected GPU cursor, not the stale CPU upper bound.
    assert local_batch.num_reqs == 1
    assert local_batch.num_scheduled_tokens.tolist() == [4]
    torch.testing.assert_close(
        local_batch.positions,
        torch.arange(10, 14, device=device),
    )
    torch.testing.assert_close(
        local_batch.seq_lens,
        torch.tensor([14], dtype=torch.int32, device=device),
    )
    assert local_batch.num_computed_tokens_np.tolist() == [20]


def test_maybe_prepare_replicated_pcp_attn_uses_global_gpu_metadata(monkeypatch):
    speculator = Mock(spec=DraftModelSpeculator)
    speculator.block_tables = Mock()
    speculator.kv_cache_config = object()
    speculator._build_draft_attn_metadata = Mock(return_value=object())
    slot_mappings_tensor = torch.arange(5).reshape(1, 5)
    speculator.block_tables.compute_slot_mappings.return_value = slot_mappings_tensor
    monkeypatch.setattr(
        pcp_module,
        "build_slot_mappings_by_layer",
        lambda slot_mappings, kv_cache_config: {"attention_layer": slot_mappings},
    )

    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=2,
        num_tokens_after_padding=5,
        idx_mapping=torch.tensor([3, 7], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        query_start_loc_np=np.array([0, 2, 5], dtype=np.int32),
        positions=torch.arange(5),
        seq_lens=torch.tensor([9, 11], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([20, 22], dtype=torch.int32),
    )

    attn_metadata, slot_mappings = pcp_module._maybe_prepare_replicated_pcp_attn(
        Mock(spec=PCPManager),
        speculator,
        input_batch,
        None,
        None,
    )

    assert attn_metadata is speculator._build_draft_attn_metadata.return_value
    assert set(slot_mappings) == {"attention_layer"}
    speculator.block_tables.gather_block_tables.assert_called_once_with(
        input_batch.idx_mapping,
        num_reqs_padded=2,
    )
    speculator.block_tables.compute_slot_mappings.assert_called_once_with(
        input_batch.idx_mapping,
        input_batch.query_start_loc,
        input_batch.positions,
        num_tokens_padded=5,
    )
    speculator._build_draft_attn_metadata.assert_called_once_with(
        num_reqs=2,
        num_reqs_padded=2,
        num_tokens_padded=5,
        seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
        step=0,
        query_start_loc_np=input_batch.query_start_loc_np,
        query_start_loc_gpu=input_batch.query_start_loc,
        seq_lens=input_batch.seq_lens,
    )


def test_maybe_prepare_replicated_pcp_attn_preserves_inputs_when_disabled():
    speculator = Mock(spec=DraftModelSpeculator)
    speculator.block_tables = Mock()
    attn_metadata = object()
    slot_mappings = {"attention_layer": torch.arange(2)}

    actual_metadata, actual_slot_mappings = (
        pcp_module._maybe_prepare_replicated_pcp_attn(
            None,
            speculator,
            SimpleNamespace(),
            attn_metadata,
            slot_mappings,
        )
    )

    assert actual_metadata is attn_metadata
    assert actual_slot_mappings is slot_mappings
    speculator.block_tables.gather_block_tables.assert_not_called()


def test_maybe_prepare_replicated_pcp_attn_preserves_inputs_when_skipping_attn():
    speculator = Mock(spec=DraftModelSpeculator)
    speculator.block_tables = Mock()
    attn_metadata = object()
    slot_mappings = {"attention_layer": torch.arange(2)}

    actual_metadata, actual_slot_mappings = (
        pcp_module._maybe_prepare_replicated_pcp_attn(
            Mock(spec=PCPManager),
            speculator,
            SimpleNamespace(),
            attn_metadata,
            slot_mappings,
            skip_attn=True,
        )
    )

    assert actual_metadata is attn_metadata
    assert actual_slot_mappings is slot_mappings
    speculator.block_tables.gather_block_tables.assert_not_called()
