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


def test_local_is_padding_reuses_global_batch_buffer():
    global_is_padding = torch.zeros(8, dtype=torch.bool)
    global_batch = SimpleNamespace(is_padding=global_is_padding)

    local_is_padding = PCPManager._get_local_is_padding(global_batch, 4)

    assert local_is_padding.data_ptr() == global_is_padding.data_ptr()
    local_is_padding[3] = True
    assert global_is_padding.tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]


def test_local_is_padding_rejects_short_global_batch_buffer():
    global_batch = SimpleNamespace(is_padding=torch.zeros(3, dtype=torch.bool))

    with pytest.raises(RuntimeError, match="3 < 4"):
        PCPManager._get_local_is_padding(global_batch, 4)


def test_global_cudagraph_padding_is_disabled_for_none():
    input_batch = SimpleNamespace(num_tokens=3, num_tokens_after_padding=4)

    assert (
        PCPManager._get_cudagraph_padded_num_tokens(input_batch, CUDAGraphMode.NONE)
        is None
    )


@pytest.mark.parametrize(
    "cudagraph_mode",
    [
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_graph_modes_use_global_cudagraph_padding(cudagraph_mode):
    input_batch = SimpleNamespace(num_tokens=3, num_tokens_after_padding=4)
    assert PCPManager._get_cudagraph_padded_num_tokens(input_batch, cudagraph_mode) == 4

    input_batch.num_tokens_after_padding = input_batch.num_tokens
    assert (
        PCPManager._get_cudagraph_padded_num_tokens(input_batch, cudagraph_mode)
        == input_batch.num_tokens
    )
