# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import set_dummy_context
from vllm.v1.worker.gpu.pcp_manager import PCPManager


def _copy_to_cpu(value, out=None, device=None):
    tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    if out is not None:
        return out.copy_(tensor)
    return tensor


def _make_config(cudagraph_mode: CUDAGraphMode):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(),
        ),
        lora_config=None,
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
    )


def _make_capture_manager(block_table: torch.Tensor):
    block_tables = SimpleNamespace(
        input_block_tables=(block_table,),
        num_kv_cache_groups=1,
        kernel_block_sizes=(2,),
        blocks_per_kv_block=(1,),
    )
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=8,
        block_tables=block_tables,
    )
    return manager, block_tables


@pytest.mark.parametrize(
    "cudagraph_mode",
    [CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL_AND_PIECEWISE],
)
def test_validate_config_accepts_decode_only_full_graphs(cudagraph_mode):
    PCPManager.validate_config(_make_config(cudagraph_mode), supports_mm_inputs=False)


def test_validate_config_rejects_full_graph_for_prefills():
    with pytest.raises(NotImplementedError, match="decode-only routines"):
        PCPManager.validate_config(
            _make_config(CUDAGraphMode.FULL), supports_mm_inputs=False
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


@pytest.mark.parametrize(
    ("cg_mode", "num_reqs", "expected_tokens", "expected_reqs"),
    [
        (CUDAGraphMode.NONE, 4, None, None),
        (CUDAGraphMode.PIECEWISE, None, 8, None),
        (CUDAGraphMode.FULL, 4, 8, 4),
    ],
)
def test_partition_padding_is_derived_from_batch_descriptor(
    cg_mode, num_reqs, expected_tokens, expected_reqs
):
    manager = MagicMock()
    input_batch = MagicMock()
    manager.partition_batch.return_value = input_batch
    batch_desc = BatchExecutionDescriptor(
        cg_mode=cg_mode,
        num_tokens=8,
        num_reqs=num_reqs,
    )

    result = pcp_manager_module.maybe_partition_pcp_batch(
        manager,
        input_batch,
        batch_desc,
    )

    assert result is input_batch
    manager.partition_batch.assert_called_once_with(
        input_batch,
        padded_num_tokens=expected_tokens,
        padded_num_reqs=expected_reqs,
    )


def test_capture_uses_pcp_persistent_inputs():
    manager, _ = _make_capture_manager(torch.ones((4, 2), dtype=torch.int32))

    input_batch, _, _ = manager.prepare_inputs_to_capture(
        num_reqs=4,
        num_tokens=4,
        max_query_len=1,
    )

    assert (
        input_batch.input_ids.data_ptr() == manager.input_buffers.input_ids.data_ptr()
    )
    assert (
        input_batch.positions.data_ptr() == manager.input_buffers.positions.data_ptr()
    )
    assert (
        input_batch.is_padding.data_ptr() == manager.input_buffers.is_padding.data_ptr()
    )


def test_dummy_context_updates_pcp_local_block_tables():
    global_block_table = torch.full((4, 4), -1, dtype=torch.int32)
    manager, block_tables = _make_capture_manager(global_block_table)
    input_batch, local_block_tables, _ = manager.prepare_inputs_to_capture(
        num_reqs=2,
        num_tokens=2,
        max_query_len=1,
    )

    set_dummy_context(
        input_batch,
        block_tables,
        context_len=3,
        num_kv_blocks=16,
        max_model_len=16,
        input_block_tables=local_block_tables,
    )

    torch.testing.assert_close(
        local_block_tables[0][:2, :2],
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
    )
    assert torch.all(global_block_table == -1)
