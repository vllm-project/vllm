# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import set_dummy_context
from vllm.v1.worker.gpu.pcp_manager import PCPManager

pytestmark = pytest.mark.cpu_test


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


@pytest.mark.parametrize(
    "cudagraph_mode",
    [
        CUDAGraphMode.NONE,
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_validate_config_accepts_decode_only_full_graphs(cudagraph_mode):
    PCPManager.validate_config(_make_config(cudagraph_mode), supports_mm_inputs=False)


def test_validate_config_rejects_full_graph_for_prefills():
    with pytest.raises(NotImplementedError, match="decode-only routines"):
        PCPManager.validate_config(
            _make_config(CUDAGraphMode.FULL), supports_mm_inputs=False
        )


def test_full_graph_preserves_request_padding():
    input_batch = SimpleNamespace(
        has_prefill=False,
    )

    assert PCPManager._resolve_num_reqs_after_padding(input_batch, 4, 2) == 4
    assert PCPManager._resolve_num_reqs_after_padding(input_batch, None, 2) == 2


def test_full_graph_rejects_prefill_batch():
    input_batch = SimpleNamespace(
        has_prefill=True,
    )

    with pytest.raises(RuntimeError, match="decode-only"):
        PCPManager._resolve_num_reqs_after_padding(input_batch, 4, 2)


def test_full_graph_rejects_insufficient_request_capacity():
    input_batch = SimpleNamespace(
        has_prefill=False,
    )

    with pytest.raises(AssertionError, match="request capacity"):
        PCPManager._resolve_num_reqs_after_padding(input_batch, 1, 2)


def test_full_capture_uses_pcp_persistent_buffers(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=8,
    )
    manager._local_block_tables = (torch.ones((8, 2), dtype=torch.int32),)
    expanded_slot_mappings = torch.full((1, 8), -1, dtype=torch.int64)
    monkeypatch.setattr(
        manager,
        "get_dummy_slot_mappings",
        lambda num_tokens: expanded_slot_mappings[:, : num_tokens * 2],
    )

    input_batch, block_tables, slot_mappings = manager.prepare_inputs_to_capture(
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
    assert torch.count_nonzero(block_tables[0]) == 0
    assert slot_mappings.shape == (1, 8)


def test_dummy_context_updates_pcp_local_block_tables(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=8,
    )
    manager._local_block_tables = (torch.ones((8, 4), dtype=torch.int32),)
    monkeypatch.setattr(
        manager,
        "get_dummy_slot_mappings",
        lambda num_tokens: torch.full((1, num_tokens * 2), -1, dtype=torch.int64),
    )
    input_batch, local_block_tables, _ = manager.prepare_inputs_to_capture(
        num_reqs=2,
        num_tokens=2,
        max_query_len=1,
    )
    global_block_table = torch.full((4, 4), -1, dtype=torch.int32)
    block_tables = SimpleNamespace(
        input_block_tables=(global_block_table,),
        kernel_block_sizes=(2,),
        blocks_per_kv_block=(1,),
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
