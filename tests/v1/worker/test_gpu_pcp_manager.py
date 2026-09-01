# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.pcp_manager import PCPManager


def _copy_to_cpu(value, out=None, device=None):
    tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    if out is not None:
        return out.copy_(tensor)
    return tensor


def _sparse_mla_config(cudagraph_mode: CUDAGraphMode):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(index_topk=2048),
        ),
        lora_config=None,
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
    )


def test_sparse_mla_piecewise_cudagraph_requires_direct_kv():
    config = _sparse_mla_config(CUDAGraphMode.PIECEWISE)

    with pytest.raises(NotImplementedError, match="require direct KV"):
        pcp_manager_module._validate_sparse_mla_cudagraph(
            config,
            use_direct_kv=False,
        )

    pcp_manager_module._validate_sparse_mla_cudagraph(
        config,
        use_direct_kv=True,
    )


def test_sparse_mla_direct_kv_rejects_full_cudagraph():
    config = _sparse_mla_config(CUDAGraphMode.FULL)

    with pytest.raises(NotImplementedError, match="PIECEWISE"):
        PCPManager.validate_config(
            config,
            supports_mm_inputs=False,
        )


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


@pytest.mark.parametrize(
    ("pcp_rank", "num_local_tokens", "expected"),
    [
        (0, 2, [[4, 5], [104, 105]]),
        (1, 3, [[9, 10, 11], [109, 110, 111]]),
    ],
)
def test_direct_kv_builds_rank_local_slot_mappings(
    pcp_rank: int,
    num_local_tokens: int,
    expected: list[list[int]],
):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=pcp_rank,
        device=torch.device("cpu"),
        use_local_kv_slot_mappings=True,
    )
    manager._padded_gather_idx = torch.tensor([0, 1, 0, 2, 3, 4])
    manager._gathered_kv_write_mask = torch.tensor(
        [True, True, False, True, True, True]
    )

    global_slot_mappings = torch.tensor(
        [
            [4, 5, 9, 10, 11],
            [104, 105, 109, 110, 111],
        ]
    )

    slot_mappings = manager._convert_slot_mappings(
        global_slot_mappings,
        num_local_tokens,
    )

    assert slot_mappings.tolist() == expected


def test_default_kv_builds_gathered_slot_mappings():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    manager._padded_gather_idx = torch.tensor([0, 1, 0, 2, 3, 4])
    manager._gathered_kv_write_mask = torch.tensor(
        [True, True, False, True, True, True]
    )

    slot_mappings = manager._convert_slot_mappings(
        torch.tensor([[4, 5, 9, 10, 11]]),
        num_local_tokens=2,
    )

    assert slot_mappings.tolist() == [[4, 5, pcp_manager_module.PAD_SLOT_ID, 9, 10, 11]]


def test_direct_kv_preserves_rank_zero_decode_write_ownership():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=1,
        device=torch.device("cpu"),
        use_local_kv_slot_mappings=True,
    )
    manager._padded_gather_idx = torch.tensor([0, 1, 2, 0, 1, 2])
    manager._gathered_kv_write_mask = torch.tensor(
        [True, True, True, False, False, False]
    )

    slot_mappings = manager._convert_slot_mappings(
        torch.tensor([[4, 5, 6]]),
        num_local_tokens=3,
    )

    assert slot_mappings.tolist() == [
        [pcp_manager_module.PAD_SLOT_ID] * 3,
    ]


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
