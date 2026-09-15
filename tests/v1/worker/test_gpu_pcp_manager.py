# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.worker.gpu import cp_utils as gpu_cp_utils
from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers, set_dummy_context
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
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
            dcp_comm_backend="ag_rs",
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

    dummy_batch = InputBatch.make_dummy(
        num_reqs=4,
        num_tokens=4,
        input_buffers=InputBuffers(4, 8, torch.device("cpu")),
        max_query_len=1,
    )
    dummy_batch.input_ids.copy_(torch.arange(4))
    dummy_batch.positions.fill_(7)
    input_batch = manager.prepare_inputs_to_capture(dummy_batch)

    assert input_batch is not dummy_batch
    assert input_batch.req_ids == dummy_batch.req_ids
    for name in ("input_ids", "positions", "is_padding", "query_start_loc", "seq_lens"):
        actual = getattr(input_batch, name)
        torch.testing.assert_close(actual, getattr(dummy_batch, name))
        assert actual.data_ptr() == getattr(manager.input_buffers, name).data_ptr()


def test_dummy_context_updates_pcp_local_block_tables():
    global_block_table = torch.full((4, 4), -1, dtype=torch.int32)
    manager, block_tables = _make_capture_manager(global_block_table)
    input_batch = InputBatch.make_dummy(
        num_reqs=2,
        num_tokens=2,
        input_buffers=manager.input_buffers,
        max_query_len=1,
    )
    input_batch = manager.prepare_inputs_to_capture(input_batch)
    local_block_tables = manager.get_dummy_block_tables(input_batch.num_reqs)

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


def _rank_rows(
    pcp_rank: int,
    pcp_world_size: int,
    num_scheduled_tokens: np.ndarray,
    num_computed_tokens: np.ndarray,
    is_prefilling: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One PCP+DCP rank's rows as (global request, extent)."""
    manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=pcp_rank,
        device=torch.device("cpu"),
        dcp_world_size=2,
    )
    query_start_loc_np = np.concatenate([[0], np.cumsum(num_scheduled_tokens)]).astype(
        np.int32
    )
    segments = manager._get_rank_segments(
        pcp_rank, num_scheduled_tokens, is_prefilling, query_start_loc_np
    )
    rows = np.array([segment.global_batch_req_idx for segment in segments])
    extents = (num_computed_tokens + num_scheduled_tokens)[rows]
    return rows, extents


@pytest.mark.parametrize(
    ("num_scheduled_tokens", "num_computed_tokens"),
    [
        ([32, 32], [0, 0]),
        ([32, 32, 32, 32], [0, 0, 0, 0]),
        ([32, 32], [100, 0]),  # one continued, one fresh
        ([64, 32, 48], [0, 512, 0]),  # ragged lengths and mixed contexts
        # A replicated prefill BEHIND a split one. Sorting replicated rows first
        # put request 1 ahead of request 0.
        ([64, 3], [0, 0]),
        ([64, 3, 64], [0, 0, 0]),
    ],
)
@pytest.mark.parametrize("pcp_world_size", [2, 4])
def test_published_row_order_is_identical_on_every_pcp_rank(
    num_scheduled_tokens, num_computed_tokens, pcp_world_size
):
    """Every rank must map its rows to the same global requests, in the same order."""
    num_scheduled_tokens = np.array(num_scheduled_tokens, dtype=np.int32)
    num_computed_tokens = np.array(num_computed_tokens, dtype=np.int32)
    is_prefilling = np.ones(len(num_scheduled_tokens), dtype=np.bool_)

    orders = [
        _rank_rows(
            rank,
            pcp_world_size,
            num_scheduled_tokens,
            num_computed_tokens,
            is_prefilling,
        )[0]
        for rank in range(pcp_world_size)
    ]
    for rank, order in enumerate(orders[1:], start=1):
        assert np.array_equal(orders[0], order), (
            f"rank {rank} rows map to {order.tolist()}, rank 0 to {orders[0].tolist()}"
        )
    # Grouped by request and ascending: what the indexer plan indexes by.
    assert np.all(np.diff(orders[0]) >= 0)


def test_published_row_order_puts_every_decode_before_every_prefill():
    """split_decodes_and_prefills takes the FIRST prefilling row as the boundary."""
    # req 0 is a continued prefill, req 1 is a decode.
    req_idx, _ = _rank_rows(
        pcp_rank=0,
        pcp_world_size=2,
        num_scheduled_tokens=np.array([32, 1], dtype=np.int32),
        num_computed_tokens=np.array([100, 20], dtype=np.int32),
        is_prefilling=np.array([True, False], dtype=np.bool_),
    )
    # The decode (request 1) must be row 0.
    assert req_idx[0] == 1


def test_split_prefill_rows_repeat_the_request_and_its_full_extent():
    """Equal adjacent request indices are what let a backend share a KV region."""
    req_idx, extents = _rank_rows(
        pcp_rank=1,
        pcp_world_size=2,
        num_scheduled_tokens=np.array([64, 3], dtype=np.int32),
        num_computed_tokens=np.array([100, 0], dtype=np.int32),
        is_prefilling=np.ones(2, dtype=np.bool_),
    )
    assert req_idx.tolist() == [0, 0, 1]
    assert extents.tolist() == [164, 164, 3]


@pytest.mark.parametrize("query_len", [1, 2, 3, 5, 6, 9])
def test_dcp_replicates_prefills_too_short_to_split(query_len):
    """A prefill that cannot fill 2*pcp chunks is replicated, not split."""
    pcp_world_size = 2
    num_scheduled_tokens = np.array([query_len], dtype=np.int32)
    is_prefilling = np.ones(1, dtype=np.bool_)

    rows_per_rank = []
    for rank in range(pcp_world_size):
        manager = PCPManager(
            pcp_world_size=pcp_world_size,
            pcp_rank=rank,
            device=torch.device("cpu"),
            dcp_world_size=2,
        )
        assert manager.replicated_requests(num_scheduled_tokens, is_prefilling)[0]
        rows = list(
            manager._iter_rank_chunks(rank, num_scheduled_tokens, is_prefilling)
        )
        assert rows == [(0, 0, query_len)]
        rows_per_rank.append(rows)
    assert rows_per_rank[0] == rows_per_rank[1]


@pytest.mark.parametrize("pcp_world_size", [2, 4, 8])
@pytest.mark.parametrize(
    "query_len", [16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 4097]
)
def test_pcp_first_chunk_row_is_never_short(pcp_world_size, query_len):
    num_scheduled_tokens = np.array([query_len], dtype=np.int32)
    is_prefilling = np.ones(1, dtype=np.bool_)

    charged_per_rank = []
    for rank in range(pcp_world_size):
        manager = PCPManager(
            pcp_world_size=pcp_world_size,
            pcp_rank=rank,
            device=torch.device("cpu"),
            dcp_world_size=pcp_world_size,
        )
        chunk_lens = [
            chunk_len
            for _, _, chunk_len in manager._iter_rank_chunks(
                rank, num_scheduled_tokens, is_prefilling
            )
        ]
        assert chunk_lens, f"rank {rank} got no rows for {query_len=}"
        assert chunk_lens[0] == max(chunk_lens), (
            f"rank {rank} emitted a short first chunk for {query_len=}: {chunk_lens}"
        )
        charged_per_rank.append(len(chunk_lens) * chunk_lens[0])

    assert len(set(charged_per_rank)) == 1, (
        f"ranks would chunk differently for {query_len=}: {charged_per_rank}"
    )


def _make_global_decode_batch(
    num_computed_tokens: list[int], buffers: InputBuffers, device: torch.device
) -> InputBatch:
    """A replicate-decode global batch as `prepare_inputs` would build it."""
    num_reqs = len(num_computed_tokens)
    num_tokens = num_reqs
    seq_lens_np = np.asarray(num_computed_tokens, dtype=np.int32) + 1

    base = InputBatch.make_dummy(num_reqs, num_tokens, buffers)
    buffers.seq_lens[:num_reqs] = torch.from_numpy(seq_lens_np).to(device)
    buffers.positions[:num_reqs] = torch.tensor(num_computed_tokens, device=device)
    query_start_loc_np = np.arange(num_reqs + 1, dtype=np.int32)
    buffers.query_start_loc[: num_reqs + 1] = torch.from_numpy(query_start_loc_np).to(
        device
    )

    return replace(
        base,
        req_ids=[f"req_{i}" for i in range(num_reqs)],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=device),
        idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
        num_scheduled_tokens=np.ones(num_reqs, dtype=np.int32),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_draft_tokens=0,
        num_draft_tokens_per_req=np.zeros(num_reqs, dtype=np.int32),
        query_start_loc=buffers.query_start_loc[: num_reqs + 1],
        query_start_loc_np=query_start_loc_np,
        seq_lens=buffers.seq_lens[:num_reqs],
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_np),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=np.asarray(num_computed_tokens, dtype=np.int32),
        prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
        input_ids=buffers.input_ids[:num_tokens],
        positions=buffers.positions[:num_tokens],
        is_padding=buffers.is_padding[:num_tokens],
        prompt_lens=None,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU kernels")
def test_partition_defers_dcp_metadata_to_post_partition_batch():
    """DCP-local lengths must derive from the partitioned batch, not the
    global one: the partition replaces seq_lens, so a pre-partition value is
    stale. partition_batch therefore returns None, and the runtime populates
    the field afterwards from the PCP-owned buffers.
    """
    device = torch.device("cuda:0")
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=device,
        max_num_reqs=4,
        max_num_tokens=8,
        dcp_world_size=2,
        dcp_rank=0,
    )

    global_buffers = InputBuffers(4, 8, device)
    global_batch = _make_global_decode_batch([16, 24], global_buffers, device)
    # A leftover from an earlier DCP batch must not survive the partition.
    global_batch.dcp_local_seq_lens = global_buffers.dcp_local_seq_lens[:2]
    global_batch.dcp_local_seq_lens.fill_(-1)

    local_batch = manager.partition_batch(
        global_batch,
        padded_num_tokens=4,
        padded_num_reqs=4,
    )

    assert local_batch.dcp_local_seq_lens is None
    assert local_batch.seq_lens.tolist() == [17, 25, 0, 0]
    assert local_batch.dcp_local_seq_lens_cpu_upper_bound is not None
    expected_dcp_upper_bound = get_dcp_local_seq_lens(
        torch.tensor([17, 25], dtype=torch.int32), 2, 0, 1
    )
    assert torch.equal(
        local_batch.dcp_local_seq_lens_cpu_upper_bound,
        torch.cat([expected_dcp_upper_bound, torch.zeros(2, dtype=torch.int32)]),
    )

    # What execute_model does next: derive DCP metadata from the final batch
    # on the PCP-owned buffers.
    local_batch.dcp_local_seq_lens = gpu_cp_utils.maybe_prepare_dcp_local_seq_lens(
        manager.input_buffers.dcp_local_seq_lens,
        local_batch.seq_lens,
        local_batch.num_reqs,
        dcp_size=2,
        dcp_rank=0,
        cp_interleave=1,
        num_reqs_padded=local_batch.num_reqs_after_padding,
    )
    expected = get_dcp_local_seq_lens(
        torch.tensor([17, 25, 0, 0], dtype=torch.int32), 2, 0, 1
    )
    assert local_batch.dcp_local_seq_lens is not None
    assert torch.equal(local_batch.dcp_local_seq_lens.cpu(), expected)
