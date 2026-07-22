# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the opt-in MiniMax M3 FlashInfer sparse decode adapter."""

import inspect

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(
        "MiniMax M3 FlashInfer sparse decode requires CUDA.",
        allow_module_level=True,
    )

from vllm.models.minimax_m3.common.ops.sparse_attn import minimax_m3_sparse_attn_decode
from vllm.models.minimax_m3.nvidia import flashinfer_sparse_decode as flashinfer_sparse

NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 128
TOPK = 16
SM_SCALE = HEAD_DIM**-0.5


def _build_decode_inputs(
    seq_lens_list: tuple[int, ...], kv_layout: str
) -> tuple[torch.Tensor, ...]:
    batch = len(seq_lens_list)
    pages_per_req = [
        (seq_len + PAGE_SIZE - 1) // PAGE_SIZE for seq_len in seq_lens_list
    ]
    max_pages = max(pages_per_req)
    num_pages = sum(pages_per_req)

    physical_page_ids = torch.randperm(num_pages, device="cuda", dtype=torch.int32)
    block_table_storage = torch.zeros(
        (batch, max_pages * 2), device="cuda", dtype=torch.int32
    )
    block_table = block_table_storage[:, ::2]
    page_offset = 0
    for request, request_pages in enumerate(pages_per_req):
        block_table[request, :request_pages] = physical_page_ids[
            page_offset : page_offset + request_pages
        ]
        page_offset += request_pages

    seq_lens_storage = torch.full((batch * 2,), -1, device="cuda", dtype=torch.int32)
    seq_lens = seq_lens_storage[::2]
    seq_lens.copy_(torch.tensor(seq_lens_list, device="cuda", dtype=torch.int32))

    topk_storage = torch.full(
        (batch, NUM_KV_HEADS, TOPK),
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    topk = topk_storage.transpose(0, 1)
    for request, request_pages in enumerate(pages_per_req):
        if request_pages == 0:
            continue
        tail_page = request_pages - 1
        older_pages = torch.randperm(tail_page, device="cuda", dtype=torch.int32)[
            : TOPK - 1
        ]
        selected = torch.cat(
            (
                older_pages,
                torch.tensor([tail_page], device="cuda", dtype=torch.int32),
            )
        )
        selected = selected[torch.randperm(selected.numel(), device="cuda")]
        topk[:, request, : selected.numel()] = selected

    physical_shape = (
        (num_pages, NUM_KV_HEADS, PAGE_SIZE, 2 * HEAD_DIM)
        if kv_layout == "HND"
        else (num_pages, PAGE_SIZE, NUM_KV_HEADS, 2 * HEAD_DIM)
    )
    physical_kv = (
        torch.randn(physical_shape, device="cuda", dtype=torch.bfloat16)
        .mul_(0.5)
        .to(torch.float8_e4m3fn)
    )
    kv_cache = physical_kv if kv_layout == "HND" else physical_kv.permute(0, 2, 1, 3)
    query = torch.randn(
        (batch, NUM_Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16
    )
    return query, kv_cache, topk, block_table, seq_lens


def _launch_metadata_adapter(
    topk: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_physical_pages: int,
    k_scale: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    batch = seq_lens.shape[0]
    physical_pages = torch.empty(
        (NUM_KV_HEADS, batch, TOPK), device="cuda", dtype=torch.int32
    )
    sparse_seq_lens = torch.empty(
        (NUM_KV_HEADS, batch), device="cuda", dtype=torch.int32
    )
    metadata_ok = torch.empty_like(sparse_seq_lens)
    metadata_status = torch.ones(1, device="cuda", dtype=torch.int32)
    bmm1_scale_log2 = torch.empty(1, device="cuda", dtype=torch.float32)
    flashinfer_sparse._prepare_sparse_metadata_kernel[(NUM_KV_HEADS * batch,)](
        topk,
        block_table,
        seq_lens,
        k_scale,
        physical_pages,
        sparse_seq_lens,
        metadata_ok,
        metadata_status,
        bmm1_scale_log2,
        batch,
        block_table.shape[1],
        num_physical_pages,
        topk.stride(0),
        topk.stride(1),
        topk.stride(2),
        block_table.stride(0),
        block_table.stride(1),
        seq_lens.stride(0),
        physical_pages.stride(0),
        physical_pages.stride(1),
        physical_pages.stride(2),
        sparse_seq_lens.stride(0),
        sparse_seq_lens.stride(1),
        TOPK_SIZE=TOPK,
        PAGE_SIZE_VALUE=PAGE_SIZE,
        HAS_KV_SCALE=True,
        SM_SCALE_LOG2=SM_SCALE * torch.log2(torch.tensor(torch.e)).item(),
    )
    torch.accelerator.synchronize()
    return (
        physical_pages,
        sparse_seq_lens,
        metadata_ok,
        metadata_status,
        bmm1_scale_log2,
    )


def test_metadata_adapter_honors_strides_deduplicates_and_handles_padding():
    block_table_storage = torch.full((3, 8), -1, device="cuda", dtype=torch.int32)
    block_table = block_table_storage[:, ::2]
    block_table.copy_(
        torch.tensor(
            [[7, 5, 11, 0], [13, 17, 19, 0], [0, 0, 0, 0]],
            device="cuda",
            dtype=torch.int32,
        )
    )
    seq_lens_storage = torch.full((6,), -1, device="cuda", dtype=torch.int32)
    seq_lens = seq_lens_storage[::2]
    seq_lens.copy_(torch.tensor([257, 130, 0], device="cuda", dtype=torch.int32))

    topk_storage = torch.full(
        (3, NUM_KV_HEADS, TOPK), -1, device="cuda", dtype=torch.int32
    )
    topk = topk_storage.transpose(0, 1)
    topk[:, 0, :4] = torch.tensor([2, 0, 2, 1], device="cuda", dtype=torch.int32)
    topk[:, 1, :2] = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    k_scale = torch.tensor([0.75], device="cuda", dtype=torch.float32)

    physical, sparse_lens, metadata_ok, status, bmm1_scale_log2 = (
        _launch_metadata_adapter(
            topk,
            block_table,
            seq_lens,
            num_physical_pages=20,
            k_scale=k_scale,
        )
    )

    expected_physical = torch.zeros_like(physical)
    expected_physical[:, 0, :3] = torch.tensor(
        [7, 5, 11], device="cuda", dtype=torch.int32
    )
    expected_physical[:, 1, :2] = torch.tensor(
        [13, 17], device="cuda", dtype=torch.int32
    )
    expected_lens = torch.tensor(
        [[257, 130, 0]] * NUM_KV_HEADS, device="cuda", dtype=torch.int32
    )
    assert block_table.stride(1) == 2
    assert seq_lens.stride(0) == 2
    assert not topk.is_contiguous()
    assert torch.equal(physical, expected_physical)
    assert torch.equal(sparse_lens, expected_lens)
    assert torch.equal(metadata_ok, torch.ones_like(metadata_ok))
    assert status.item() == 1
    torch.testing.assert_close(
        bmm1_scale_log2,
        k_scale * SM_SCALE * torch.log2(torch.tensor(torch.e, device="cuda")),
    )


def test_buffer_pool_uses_geometric_capacity_and_retains_graph_allocations():
    pool = flashinfer_sparse._DeviceBufferPool(
        torch.device("cuda"), torch.empty(1024, device="cuda", dtype=torch.uint8)
    )

    def counter_size(batch: int, num_heads: int, sm_count: int) -> int:
        assert num_heads == NUM_Q_HEADS
        assert sm_count > 0
        return batch * 17

    batch_three = pool.get(3, counter_size)
    assert pool.current is not None
    assert pool.current.capacity == 4
    first_storage = batch_three.physical_pages.untyped_storage().data_ptr()
    assert batch_three.physical_pages.shape == (NUM_KV_HEADS, 3, TOPK)
    assert batch_three.physical_pages.is_contiguous()
    assert batch_three.counter.numel() == 4 * 17

    batch_four = pool.get(4, counter_size)
    assert batch_four.physical_pages.untyped_storage().data_ptr() == first_storage
    assert pool.retired == []

    batch_five = pool.get(5, counter_size)
    assert pool.current.capacity == 8
    assert len(pool.retired) == 1
    assert pool.retired[0].capacity == 4
    assert batch_five.physical_pages.untyped_storage().data_ptr() != first_storage
    assert batch_five.counter.numel() == 8 * 17


def test_static_guard_rejects_non_cuda_before_querying_capability(
    monkeypatch: pytest.MonkeyPatch,
):
    query = torch.empty((1, NUM_Q_HEADS, HEAD_DIM), dtype=torch.bfloat16)
    output = torch.empty_like(query)
    kv_cache = torch.empty(
        (1, NUM_KV_HEADS, PAGE_SIZE, 2 * HEAD_DIM), dtype=torch.float8_e4m3fn
    )
    topk = torch.zeros((NUM_KV_HEADS, 1, TOPK), dtype=torch.int32)
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    seq_lens = torch.ones(1, dtype=torch.int32)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: pytest.fail("capability queried for a CPU tensor"),
    )

    reason = flashinfer_sparse._static_fallback_reason(
        query,
        kv_cache,
        topk,
        block_table,
        seq_lens,
        output,
        NUM_KV_HEADS,
        PAGE_SIZE,
        TOPK,
        1,
        None,
        None,
    )
    assert reason == "query is not on CUDA"


def test_unavailable_api_falls_back_before_allocating(monkeypatch: pytest.MonkeyPatch):
    query, kv_cache, topk, block_table, seq_lens = _build_decode_inputs((129,), "HND")
    output = torch.empty_like(query)
    monkeypatch.setenv(
        flashinfer_sparse._BACKEND_ENV, flashinfer_sparse._FLASHINFER_BACKEND
    )
    monkeypatch.setattr(
        flashinfer_sparse, "_static_fallback_reason", lambda *args: None
    )
    monkeypatch.setattr(flashinfer_sparse, "_load_api", lambda: None)
    runner = flashinfer_sparse.FlashInferSparseDecodeRunner()

    used = runner.try_decode(
        query,
        kv_cache,
        topk,
        block_table,
        seq_lens,
        output,
        num_kv_heads=NUM_KV_HEADS,
        scale=SM_SCALE,
        block_size=PAGE_SIZE,
        topk_blocks=TOPK,
        decode_query_len=1,
    )
    assert not used
    assert runner._buffer_pools_by_device == {}


def test_adapter_passes_padding_scales_and_reuses_buffers(
    monkeypatch: pytest.MonkeyPatch,
):
    query, kv_cache, topk, block_table, seq_lens = _build_decode_inputs(
        (257, 130, 0), "NHD"
    )
    output = torch.empty_like(query)
    k_scale = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    v_scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    calls: list[dict] = []

    def run(**kwargs) -> None:
        calls.append(
            {
                "block_tables": kwargs["block_tables"].clone(),
                "seq_lens": kwargs["seq_lens"].clone(),
                "bmm1_scale_log2": kwargs["bmm1_scale_log2"].clone(),
                "bmm2_scale": kwargs["bmm2_scale"],
                "physical_ptr": kwargs["block_tables"].untyped_storage().data_ptr(),
                "counter_ptr": kwargs["multi_ctas_kv_counter_buffer"].data_ptr(),
            }
        )
        assert kwargs["enable_block_sparse_attention"]
        kwargs["out"].copy_(kwargs["query"])

    def counter_size(batch: int, num_heads: int, sm_count: int) -> int:
        return batch * num_heads + sm_count

    pool = flashinfer_sparse._DeviceBufferPool(
        query.device, torch.empty(4096, device="cuda", dtype=torch.uint8)
    )
    runner = flashinfer_sparse.FlashInferSparseDecodeRunner()
    runner._buffer_pools_by_device[query.device] = pool
    monkeypatch.setenv(
        flashinfer_sparse._BACKEND_ENV, flashinfer_sparse._FLASHINFER_BACKEND
    )
    monkeypatch.setattr(
        flashinfer_sparse, "_static_fallback_reason", lambda *args: None
    )
    monkeypatch.setattr(flashinfer_sparse, "_load_api", lambda: (run, counter_size))

    for _ in range(2):
        assert runner.try_decode(
            query,
            kv_cache,
            topk,
            block_table,
            seq_lens,
            output,
            num_kv_heads=NUM_KV_HEADS,
            scale=SM_SCALE,
            block_size=PAGE_SIZE,
            topk_blocks=TOPK,
            decode_query_len=1,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    torch.accelerator.synchronize()

    assert len(calls) == 2
    assert calls[0]["seq_lens"][:, -1].eq(0).all()
    assert calls[0]["bmm2_scale"] is v_scale
    torch.testing.assert_close(
        calls[0]["bmm1_scale_log2"],
        k_scale * SM_SCALE * torch.log2(torch.tensor(torch.e, device="cuda")),
    )
    assert calls[0]["physical_ptr"] == calls[1]["physical_ptr"]
    assert calls[0]["counter_ptr"] == calls[1]["counter_ptr"]
    assert pool.current is not None and pool.current.capacity == 4
    assert pool.retired == []
    torch.testing.assert_close(output, query)


def test_adapter_is_cudagraph_replayable(monkeypatch: pytest.MonkeyPatch):
    query, kv_cache, topk, block_table, seq_lens = _build_decode_inputs((257,), "HND")
    output = torch.empty_like(query)

    def run(**kwargs) -> None:
        kwargs["out"].copy_(kwargs["query"])

    def counter_size(batch: int, num_heads: int, sm_count: int) -> int:
        return batch * num_heads + sm_count

    pool = flashinfer_sparse._DeviceBufferPool(
        query.device, torch.empty(4096, device="cuda", dtype=torch.uint8)
    )
    runner = flashinfer_sparse.FlashInferSparseDecodeRunner()
    runner._buffer_pools_by_device[query.device] = pool
    monkeypatch.setenv(
        flashinfer_sparse._BACKEND_ENV, flashinfer_sparse._FLASHINFER_BACKEND
    )
    monkeypatch.setattr(
        flashinfer_sparse, "_static_fallback_reason", lambda *args: None
    )
    monkeypatch.setattr(flashinfer_sparse, "_load_api", lambda: (run, counter_size))

    for _ in range(2):
        assert runner.try_decode(
            query,
            kv_cache,
            topk,
            block_table,
            seq_lens,
            output,
            num_kv_heads=NUM_KV_HEADS,
            scale=SM_SCALE,
            block_size=PAGE_SIZE,
            topk_blocks=TOPK,
            decode_query_len=1,
        )
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert runner.try_decode(
            query,
            kv_cache,
            topk,
            block_table,
            seq_lens,
            output,
            num_kv_heads=NUM_KV_HEADS,
            scale=SM_SCALE,
            block_size=PAGE_SIZE,
            topk_blocks=TOPK,
            decode_query_len=1,
        )

    query.fill_(0.25)
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(output, query)


def _require_real_flashinfer_sparse_decode() -> None:
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("FlashInfer TRTLLM-GEN sparse decode requires SM100 or SM103")
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        from flashinfer.utils import (
            get_trtllm_gen_multi_ctas_kv_counter_bytes as _counter_size,
        )
    except Exception as error:
        pytest.skip(f"FlashInfer sparse decode API is unavailable: {error}")
    parameters = inspect.signature(trtllm_batch_decode_with_kv_cache).parameters
    assert callable(_counter_size)
    missing = flashinfer_sparse._REQUIRED_DECODE_PARAMETERS.difference(parameters)
    if missing:
        pytest.skip(f"FlashInfer sparse decode API lacks {sorted(missing)}")


@pytest.mark.parametrize(
    ("kv_layout", "batch", "seq_len"),
    [("NHD", 32, 1000), ("HND", 1, 8191)],
)
def test_real_flashinfer_sparse_decode_matches_triton_and_reuses_buffers(
    kv_layout: str,
    batch: int,
    seq_len: int,
    monkeypatch: pytest.MonkeyPatch,
):
    _require_real_flashinfer_sparse_decode()
    torch.manual_seed(0)
    query, kv_cache, topk, block_table, seq_lens = _build_decode_inputs(
        (seq_len,) * batch, kv_layout
    )
    # The Triton oracle's page-table kernel assumes contiguous inner metadata.
    block_table = block_table.contiguous()
    seq_lens = seq_lens.contiguous()
    k_scale = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    v_scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    expected = torch.empty_like(query)
    actual = torch.empty_like(query)
    minimax_m3_sparse_attn_decode(
        query,
        kv_cache,
        topk,
        block_table,
        seq_lens,
        NUM_KV_HEADS,
        SM_SCALE,
        expected,
        1,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    monkeypatch.setenv(
        flashinfer_sparse._BACKEND_ENV, flashinfer_sparse._FLASHINFER_BACKEND
    )
    monkeypatch.setattr(flashinfer_sparse, "_api", None)
    monkeypatch.setattr(flashinfer_sparse, "_api_checked", False)
    runner = flashinfer_sparse.FlashInferSparseDecodeRunner()
    for _ in range(2):
        assert runner.try_decode(
            query,
            kv_cache,
            topk,
            block_table,
            seq_lens,
            actual,
            num_kv_heads=NUM_KV_HEADS,
            scale=SM_SCALE,
            block_size=PAGE_SIZE,
            topk_blocks=TOPK,
            decode_query_len=1,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    torch.accelerator.synchronize()

    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    assert len(runner._buffer_pools_by_device) == 1
    pool = next(iter(runner._buffer_pools_by_device.values()))
    assert pool.current is not None
    assert pool.current.capacity == 1 << (batch - 1).bit_length()
    assert pool.retired == []
