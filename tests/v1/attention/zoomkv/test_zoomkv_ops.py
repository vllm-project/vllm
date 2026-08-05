# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the GPU-only ZoomKV ops and block_summary lifecycle."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.attention.ops.zoomkv.kernels as zoomkv_kernels
from vllm.config.attention import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.zoomkv_attn import (
    ZoomKVAttentionBackend,
    ZoomKVAttentionImpl,
    _graph_chunk_bucket,
    _needs_summary_update,
    _should_use_mixed_sparse_decode,
    _should_use_sparse_decode,
)
from vllm.v1.attention.ops.zoomkv.kernels import (
    float_topk_3d_varlen,
    get_quest_ops,
    quest_score_reference,
)
from vllm.v1.attention.ops.zoomkv.kivi_rerank import partial_chunk_kivi_qk_ref
from vllm.v1.attention.ops.zoomkv.paged import (
    assemble_sparse_context_indices,
    gather_kv_by_logical_indices,
    sparse_decode_attention,
)
from vllm.v1.attention.ops.zoomkv.quant_pack import pack_block_kcache_4bit
from vllm.v1.attention.ops.zoomkv.quest import QuestTorchOps
from vllm.v1.attention.ops.zoomkv.retrieval_metadata_triton import (
    build_actual_num_chunks,
    build_stage_budgets,
)
from vllm.v1.attention.ops.zoomkv.retriever import ZoomKVRetriever, ZoomKVRuntimeConfig
from vllm.v1.attention.ops.zoomkv.state import (
    ZoomKVBlockSummary,
    clear_block_summaries,
    copy_block_summaries_for_block_pairs,
    get_or_create_block_summary,
    invalidate_block_summaries_for_blocks,
)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
def test_fused_actual_num_chunks_metadata(input_dtype):
    seq_lens_cpu = torch.tensor([0, 63, 64, 319, 320, 321, 131055])
    seq_lens = seq_lens_cpu.to(device="cuda", dtype=input_dtype)
    out = torch.empty(seq_lens.numel(), device="cuda", dtype=torch.int32)

    result = build_actual_num_chunks(
        seq_lens,
        out,
        sink_size=64,
        local_size=256,
        block_size=16,
        start_block=4,
        max_chunks=8192,
    )
    expected = (
        (torch.maximum(seq_lens_cpu, torch.tensor(320)) - 256) // 16 - 4
    ).clamp_(0, 8192)
    assert result.dtype == torch.int32
    assert torch.equal(result.cpu(), expected.to(torch.int32))
    host = ZoomKVRetriever._actual_num_chunks_host(
        seq_lens_cpu,
        sink_size=64,
        local_size=256,
        block_size=16,
        start_block=4,
        max_chunks=8192,
    )
    assert host == expected.to(torch.int32).tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stage_budgets_follow_actual_request_widths():
    actual = torch.tensor([0, 23, 100, 8192], device="cuda", dtype=torch.int32)
    outputs = [
        torch.empty((4, 2), device="cuda", dtype=torch.int32) for _ in range(6)
    ]
    build_stage_budgets(
        actual,
        *outputs,
        factor=16,
        large_ratio=0.5,
        small_ratio=0.3,
        dense_ratio=0.4,
        max_large=256,
        max_small=1024,
        dense_topk=8,
        sparse_topk=4,
        final_topk=100,
    )
    parent_lengths, large_ks, sub_lengths, small_ks, dense_ks, final_ks = [
        out[:, 0].cpu().tolist() for out in outputs
    ]
    assert parent_lengths == [0, 2, 7, 512]
    assert large_ks == [0, 1, 4, 256]
    assert sub_lengths == [0, 16, 64, 4096]
    assert small_ks == [0, 5, 20, 1024]
    assert dense_ks == [0, 2, 8, 409]
    assert final_ks == [0, 28, 100, 100]
    for out in outputs:
        assert torch.equal(out[:, 0], out[:, 1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ragged_topk_uses_per_row_length_and_k():
    scores = torch.tensor(
        [
            [[9.0, 1.0, 8.0, 7.0, 100.0, 99.0]],
            [[0.0, 5.0, 4.0, 3.0, 2.0, 1.0]],
        ],
        device="cuda",
    )
    lengths = torch.tensor([[4], [6]], device="cuda", dtype=torch.int32)
    ks = torch.tensor([[2], [3]], device="cuda", dtype=torch.int32)
    positions = float_topk_3d_varlen(scores, lengths, ks, 4, strict=True)
    assert set(positions[0, 0, :2].cpu().tolist()) == {0, 2}
    assert positions[0, 0, 2:].cpu().tolist() == [-1, -1]
    assert set(positions[1, 0, :3].cpu().tolist()) == {1, 2, 3}
    assert positions[1, 0, 3:].cpu().tolist() == [-1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batch_meta_cache_reused_across_layers():
    """Metadata is seq_lens geometry only; 2nd call must reuse the GPU buffer."""
    if not zoomkv_kernels.direct_physical_retrieval_available():
        pytest.skip("direct physical ZoomKV extension unavailable")

    ZoomKVRetriever.clear_batch_meta_cache()
    device = torch.device("cuda")
    cfg = ZoomKVRuntimeConfig(full_attention_threshold=512)
    r0 = ZoomKVRetriever(cfg)
    r1 = ZoomKVRetriever(cfg)
    block_size, hkv, d = 16, 2, 128
    num_blocks, batch, n_chunks = 256, 1, 64
    summary = get_or_create_block_summary(
        "test_meta_cache",
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=hkv,
        head_dim=d,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(num_blocks, block_size, hkv, d, device=device, dtype=torch.bfloat16)
    physical_ids = torch.arange(n_chunks, device=device, dtype=torch.int32)
    summary.update_blocks_from_key_cache(key, physical_ids)
    block_table = torch.full((batch, 128), -1, device=device, dtype=torch.int32)
    block_table[0, :n_chunks] = physical_ids
    # pad local/sink slots so available_chunks stays positive
    block_table[0, n_chunks : n_chunks + 20] = torch.arange(
        n_chunks, n_chunks + 20, device=device, dtype=torch.int32
    )
    seq_len = cfg.sink_size + n_chunks * block_size + cfg.local_size
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    seq_lens_host = seq_lens.cpu()
    raw_q = torch.randn(batch, hkv, d, device=device, dtype=torch.bfloat16)

    out0 = r0.retrieve_topk_tokens_batch_result(
        raw_q,
        summary,
        block_table,
        seq_lens,
        summaries_guaranteed_valid=True,
        seq_lens_host=seq_lens_host,
    )
    cached = ZoomKVRetriever._batch_meta_cache
    assert cached is not None
    ptr0 = cached.actual_num_chunks.data_ptr()
    out1 = r1.retrieve_topk_tokens_batch_result(
        raw_q,
        summary,
        block_table,
        seq_lens,
        summaries_guaranteed_valid=True,
        seq_lens_host=seq_lens_host,
    )
    assert ZoomKVRetriever._batch_meta_cache is not None
    assert ZoomKVRetriever._batch_meta_cache.actual_num_chunks.data_ptr() == ptr0
    assert out0.topk.shape == out1.topk.shape
    ZoomKVRetriever.clear_batch_meta_cache()
    clear_block_summaries()


def test_quest_torch_vs_reference():
    device = _device()
    bs, kv, n, d = 1, 2, 8, 256
    q = torch.randn(bs, kv, d, device=device, dtype=torch.bfloat16)
    cmin = torch.randn(bs, kv, n, d, device=device, dtype=torch.bfloat16)
    cmax = cmin + 1
    ref = quest_score_reference(q, cmin, cmax)
    out = torch.empty(bs, kv, n, device=device, dtype=torch.float32)
    QuestTorchOps().quest_chunk_score(q, cmin, cmax, out, n, None)
    assert torch.allclose(ref, out, atol=1e-2, rtol=1e-2)


def test_quest_ops_dispatch():
    ops = get_quest_ops(prefer_triton=True, strict=False)
    device = _device()
    q = torch.randn(1, 2, 128, device=device, dtype=torch.bfloat16)
    cmin = torch.randn(1, 2, 4, 128, device=device, dtype=torch.bfloat16)
    cmax = cmin + 0.5
    out = torch.empty(1, 2, 4, device=device, dtype=torch.float32)
    ops.quest_chunk_score(q, cmin, cmax, out, 4, None)
    assert torch.isfinite(out).all()


def test_kivi_rerank_uses_compact_per_chunk_output():
    device = _device()
    group_size = 16
    head_dim = 8
    chunk_ids = torch.tensor([[[0, 1]]], device=device, dtype=torch.int64)
    dense_mask = torch.tensor([[[True, False]]], device=device)
    packed = torch.randint(
        -(2**31),
        2**31 - 1,
        (1, 1, 2, head_dim // 8, group_size),
        device=device,
        dtype=torch.int32,
    )
    chunk_min = torch.randn(1, 1, 2, head_dim, device=device, dtype=torch.bfloat16)
    chunk_max = chunk_min + 1
    raw_q = torch.randn(1, 1, head_dim, device=device, dtype=torch.bfloat16)

    scores, indices = partial_chunk_kivi_qk_ref(
        chunk_ids,
        dense_mask,
        packed,
        chunk_min,
        chunk_max,
        raw_q,
        group_size=group_size,
        dense_topk=8,
        sparse_topk=4,
    )

    assert scores.shape == indices.shape == (1, 1, 16)
    assert torch.all(scores[..., :8] > -1.0e30)
    assert torch.all(scores[..., 8:12] > -1.0e30)
    assert torch.all(scores[..., 12:] < -1.0e30)


def test_quest_ops_prefers_complete_cuda_extension(monkeypatch):
    class FakeQuestCuda:
        def quest_chunk_score(self):
            raise AssertionError("not called")

        def quest_sub_chunk_score(self):
            raise AssertionError("not called")

        def quest_map_back(self):
            raise AssertionError("not called")

    fake = FakeQuestCuda()
    monkeypatch.setattr(zoomkv_kernels, "try_load_zoomkv_c", lambda: fake)

    ops = zoomkv_kernels.get_quest_ops(prefer_triton=True, strict=False)

    assert ops.__class__.__name__ == "_CudaQuestOps"
    assert ops._mod is fake


def test_quest_cuda_extension_matches_torch_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    mod = zoomkv_kernels.try_load_zoomkv_c()
    if not zoomkv_kernels._has_cuda_quest(mod):
        pytest.skip("vllm._zoomkv_C Quest kernels are not built")

    device = torch.device("cuda")
    q = torch.randn(1, 2, 128, device=device, dtype=torch.bfloat16)
    cmin = torch.randn(1, 2, 8, 128, device=device, dtype=torch.bfloat16)
    cmax = cmin + torch.rand(1, 2, 8, 128, device=device, dtype=torch.bfloat16)
    valid = torch.ones(1, 2, 8, device=device, dtype=torch.bool)
    valid[..., -1] = False

    cuda_ops = zoomkv_kernels.get_quest_ops(prefer_triton=False, strict=True)
    torch_ops = QuestTorchOps()
    cuda_scores = torch.empty(1, 2, 8, device=device, dtype=torch.float32)
    torch_scores = torch.empty_like(cuda_scores)
    cuda_ops.quest_chunk_score(q, cmin, cmax, cuda_scores, 8, valid)
    torch_ops.quest_chunk_score(q, cmin, cmax, torch_scores, 8, valid)
    assert torch.allclose(cuda_scores, torch_scores, atol=1e-2, rtol=1e-2)

    large_idx = torch.tensor([[[0, 2, 3], [1, 0, 2]]], device=device)
    cuda_sub = torch.empty(1, 2, 6, device=device, dtype=torch.float32)
    torch_sub = torch.empty_like(cuda_sub)
    cuda_ops.quest_sub_chunk_score(q, cmin, cmax, large_idx, cuda_sub, 3, 2)
    torch_ops.quest_sub_chunk_score(q, cmin, cmax, large_idx, torch_sub, 3, 2)
    assert torch.allclose(cuda_sub, torch_sub, atol=1e-2, rtol=1e-2)

    sub_pos = torch.tensor([[[0, 1, 4, 5], [2, 3, 0, 1]]], device=device)
    cuda_idx = torch.empty(1, 2, 4, device=device, dtype=torch.int64)
    torch_idx = torch.empty_like(cuda_idx)
    cuda_ops.quest_map_back(large_idx, sub_pos, cuda_idx, 2, 8)
    torch_ops.quest_map_back(large_idx, sub_pos, torch_idx, 2, 8)
    assert torch.equal(cuda_idx, torch_idx)


def test_block_summary_invalidate_and_cow():
    device = _device()
    clear_block_summaries()
    sc = get_or_create_block_summary(
        "test-layer",
        num_blocks=8,
        num_kv_heads=2,
        head_dim=128,
        block_size=16,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(8, 16, 2, 128, device=device, dtype=torch.bfloat16)
    sc.update_blocks_from_key_cache(key, torch.tensor([1, 2], device=device))
    assert bool(sc.valid[1]) and bool(sc.valid[2])
    invalidate_block_summaries_for_blocks([1])
    assert not bool(sc.valid[1])
    sc.copy_blocks([(2, 3)])
    assert bool(sc.valid[3])
    copy_block_summaries_for_block_pairs([(3, 4)])
    assert bool(sc.valid[4])

    sc.valid[:4] = True
    copy_block_summaries_for_block_pairs([(0, 1)], allocation_num_blocks=2)
    assert bool(sc.valid[4:8].all())
    invalidate_block_summaries_for_blocks([1], allocation_num_blocks=2)
    assert not bool(sc.valid[4:8].any())
    clear_block_summaries()


def test_pack_block_roundtrip_shapes():
    device = _device()
    K = torch.randn(2, 16, 256, device=device, dtype=torch.bfloat16)
    packed, cmin, cmax, cent = pack_block_kcache_4bit(K)
    assert packed.shape == (2, 32, 16)
    assert cmin.shape == (2, 256)


def test_retriever_dense_gate_and_range():
    cfg = ZoomKVRuntimeConfig(
        full_attention_threshold=2000, sink_size=64, local_size=256
    )
    r = ZoomKVRetriever(cfg)
    # Decode always uses sparse unless dense_fallback is set.
    assert not r.should_use_dense(100)
    assert not r.should_use_dense(5000)
    assert ZoomKVRetriever(
        ZoomKVRuntimeConfig(dense_fallback=True)
    ).should_use_dense(5000)
    assert (
        r.retrieval_block_range(64 + 256, 16)[0]
        == r.retrieval_block_range(64 + 256, 16)[1]
    )
    s, e = r.retrieval_block_range(4096, 16)
    assert e > s


def test_retriever_pads_when_candidates_are_fewer_than_final_topk():
    device = _device()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.v1.attention.ops.zoomkv.kernels import (
        direct_physical_retrieval_available,
    )

    if not direct_physical_retrieval_available():
        pytest.skip("direct physical ZoomKV extension unavailable")
    cfg = ZoomKVRuntimeConfig(
        sink_size=0,
        local_size=0,
        final_topk=20,
        full_attention_threshold=0,
        quest_large_chunk=16,
    )
    retriever = ZoomKVRetriever(cfg)
    retriever.quest = QuestTorchOps()
    block_summary = ZoomKVBlockSummary(1, 2, 128, 16, device)
    key = torch.randn(1, 16, 2, 128, device=device, dtype=torch.bfloat16)
    block_summary.update_blocks_from_key_cache(key, torch.tensor([0], device=device))
    raw_q = torch.randn(1, 2, 128, device=device, dtype=torch.bfloat16)

    selected = retriever.retrieve_topk_tokens(
        raw_q,
        block_summary,
        torch.tensor([0], device=device),
        seq_len=16,
    )

    assert selected.shape == (1, 2, 20)
    assert torch.equal(selected[..., 16:], torch.full_like(selected[..., 16:], -1))


def test_zoomkv_offload_spec_merge():
    from vllm.v1.kv_cache_interface import KVQuantMode, ZoomKVOffloadSpec

    specs = [
        ZoomKVOffloadSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            kv_quant_mode=KVQuantMode.NONE,
            sink_size=64,
            local_size=256,
        )
        for _ in range(3)
    ]
    merged = ZoomKVOffloadSpec.merge(specs)
    assert isinstance(merged, ZoomKVOffloadSpec)
    assert merged.sink_size == 64
    assert merged.local_size == 256
    assert merged.block_size == 16


def test_assemble_context_into_preallocated_buffer():
    device = _device()
    kv, tk = 2, 8
    topk = torch.arange(tk, device=device).view(1, -1).expand(kv, -1)
    out = torch.full((kv, 64 + 256 + tk), -1, dtype=torch.int64, device=device)
    idx, valid = assemble_sparse_context_indices(1024, topk, 64, 256, device, out=out)
    assert idx is out or idx.shape == out.shape
    assert valid.shape[0] == kv
    assert bool(valid[:, :64].all())
    assert bool((idx[:, -tk:] == topk).all())


def test_zoomkv_config_defaults_match_runtime_defaults():
    attn = AttentionConfig(backend=AttentionBackendEnum.ZOOMKV)
    runtime = ZoomKVRuntimeConfig()

    assert attn.zoomkv_quest_large_ratio == runtime.quest_large_ratio == 0.5
    assert attn.zoomkv_quest_small_ratio == runtime.quest_small_ratio == 0.3
    assert attn.zoomkv_dense_ratio == runtime.dense_ratio
    assert attn.zoomkv_dense_topk == runtime.dense_topk == 8
    assert attn.zoomkv_sparse_topk == runtime.sparse_topk == 4
    assert attn.zoomkv_quest_chunk == runtime.quest_chunk == 16
    assert attn.zoomkv_enable_offload == runtime.enable_offload is False


@pytest.mark.parametrize(
    ("max_model_len", "expected"),
    [(17_408, 1_072), (132_000, 8_240)],
)
def test_zoomkv_graph_bucket_uses_max_model_len(max_model_len, expected):
    cfg = ZoomKVRuntimeConfig(max_model_len=max_model_len)
    assert _graph_chunk_bucket(cfg, block_size=16) == expected


def test_zoomkv_config_rejects_offload_with_dense_fallback():
    with pytest.raises(ValueError, match="enable_offload"):
        AttentionConfig(
            backend=AttentionBackendEnum.ZOOMKV,
            zoomkv_enable_offload=True,
            zoomkv_dense_fallback=True,
        )


def test_zoomkv_config_rejects_misaligned_chunks():
    with pytest.raises(ValueError, match="zoomkv_quest_chunk must be 16"):
        AttentionConfig(
            backend=AttentionBackendEnum.ZOOMKV,
            zoomkv_quest_chunk=8,
        )


def test_kv_cpu_pool_roundtrip():
    if not torch.cuda.is_available():
        return
    from vllm.v1.attention.ops.zoomkv.offload import (
        ZoomKVCpuKeyPool,
        set_cpu_key_pool,
    )
    from vllm.v1.attention.ops.zoomkv.paged import gather_kv_hybrid

    device = torch.device("cuda")
    clear_block_summaries()
    set_cpu_key_pool(None)
    pool = ZoomKVCpuKeyPool(
        num_slots=8,
        num_kv_heads=2,
        head_dim=128,
        block_size=16,
        dtype=torch.bfloat16,
        device=device,
        layer_names=["layer0"],
        strict=False,
    )
    set_cpu_key_pool(pool)
    key = torch.randn(4, 16, 2, 128, device=device, dtype=torch.bfloat16)
    value = torch.randn(4, 16, 2, 128, device=device, dtype=torch.bfloat16)
    sc = ZoomKVBlockSummary(4, 2, 128, 16, device)
    sc.update_blocks_from_key_cache(key, torch.tensor([1], device=device))
    original_k = key[1].clone()
    original_v = value[1].clone()

    # GPU-only -> warm: CPU copy exists, GPU page intact.
    assert (
        pool.offload_blocks_bulk(
            "layer0", key, value, sc, torch.tensor([1], device=device)
        )
        == 1
    )
    torch.accelerator.synchronize()
    mask = pool.offloaded_mask.get("layer0")
    assert mask is None or not bool(mask[1])
    assert torch.equal(key[1], original_k)
    slot = pool.lookup_slot("layer0", 1)
    assert torch.allclose(pool.key["layer0"][slot], original_k.cpu())
    assert torch.allclose(pool.value["layer0"][slot], original_v.cpu())
    # Re-offloading an already-mapped block is a no-op.
    assert (
        pool.offload_blocks_bulk(
            "layer0", key, value, sc, torch.tensor([1], device=device)
        )
        == 0
    )

    # warm -> cold: GPU pages zeroed, no PCIe traffic.
    assert pool.mark_cold("layer0", key, value, [0, 1, 2]) == 1
    torch.accelerator.synchronize()
    assert bool(pool.offloaded_mask["layer0"][1])
    assert torch.equal(key[1], torch.zeros_like(key[1]))
    assert torch.equal(value[1], torch.zeros_like(value[1]))
    assert pool.has_cold_blocks("layer0")

    # Hybrid gather reads cold tokens straight from the pinned CPU pool.
    bt = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
    lids = torch.arange(16, 32, device=device)
    gk, gv = gather_kv_hybrid(key, value, bt, lids, 16, pool, "layer0", 1, 2)
    assert gk.shape[0] == 2
    assert torch.allclose(gk[0, 0].cpu(), original_k[0, 0].cpu(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(gv[0, 0].cpu(), original_v[0, 0].cpu(), atol=1e-2, rtol=1e-2)

    # cold -> warm: dense readers get the full-precision page back.
    assert pool.restore_blocks("layer0", key, value, [1, 3]) == 1
    torch.accelerator.synchronize()
    assert not bool(pool.offloaded_mask["layer0"][1])
    assert torch.allclose(key[1], original_k)
    assert torch.allclose(value[1], original_v)
    assert not pool.has_cold_blocks("layer0")

    # warm -> cold again is free (no new D2H) because content is immutable.
    d2h_before = pool.metrics.d2h_events
    assert pool.mark_cold("layer0", key, value, [1]) == 1
    assert pool.metrics.d2h_events == d2h_before
    torch.accelerator.synchronize()
    assert torch.equal(key[1], torch.zeros_like(key[1]))

    pool.free_gpu_blocks("layer0", [1])
    assert pool.lookup_slot("layer0", 1) is None
    assert not pool.has_cold_blocks("layer0")
    set_cpu_key_pool(None)
    clear_block_summaries()


def test_prepare_retrieval_query_gqa_mean():
    from vllm.v1.attention.ops.zoomkv.retriever import prepare_retrieval_query

    device = _device()
    q = torch.zeros(1, 4, 8, device=device, dtype=torch.bfloat16)
    q[0, 0] = 2.0
    q[0, 1] = 4.0
    q[0, 2] = 6.0
    q[0, 3] = 8.0
    out = prepare_retrieval_query(q, num_kv_heads=2)
    assert out.shape == (1, 2, 8)
    assert out.is_contiguous()
    assert torch.allclose(out[0, 0], torch.tensor(3.0, device=device, dtype=torch.bfloat16))
    assert torch.allclose(out[0, 1], torch.tensor(7.0, device=device, dtype=torch.bfloat16))


def test_sparse_decode_gate():
    impl = ZoomKVAttentionImpl.__new__(ZoomKVAttentionImpl)
    impl._retriever = None
    metadata = SimpleNamespace(
        max_query_len=1,
        num_decodes=1,
        num_prefills=0,
        num_reqs=1,
        seq_lens_cpu=torch.tensor([100]),  # short seq still sparse
        seq_lens=torch.tensor([100]),
    )
    cfg = ZoomKVRuntimeConfig(full_attention_threshold=512)

    assert impl._should_sparse_decode(metadata, cfg)
    metadata.num_prefills = 1
    # Pure-batch use_sparse stays False for mixed; mixed uses use_mixed_sparse.
    assert not impl._should_sparse_decode(metadata, cfg)
    assert _should_use_mixed_sparse_decode(
        cfg=cfg, num_decodes=1, num_prefills=1
    )
    metadata.num_prefills = 0
    metadata.max_query_len = 2
    assert not impl._should_sparse_decode(metadata, cfg)
    metadata.max_query_len = 1
    assert not impl._should_sparse_decode(
        metadata, ZoomKVRuntimeConfig(dense_fallback=True)
    )
    assert not _should_use_mixed_sparse_decode(
        cfg=ZoomKVRuntimeConfig(enable_offload=True),
        num_decodes=1,
        num_prefills=1,
    )


def test_forward_owns_kv_update_and_preserves_cache_sharing():
    impl = ZoomKVAttentionImpl.__new__(ZoomKVAttentionImpl)
    impl.do_kv_cache_update = Mock()
    impl._should_sparse_decode = Mock(return_value=False)
    output = torch.empty(1, 2, 4)
    impl._dense_flash_forward = Mock(return_value=output)

    slot_mapping = torch.tensor([3], dtype=torch.int64)
    metadata = SimpleNamespace(
        zoomkv=ZoomKVRuntimeConfig(),
        slot_mapping=slot_mapping,
        use_sparse=False,
        use_mixed_sparse=False,
        num_decodes=0,
        num_prefills=1,
    )
    layer = SimpleNamespace(kv_sharing_target_layer_name=None)
    query = torch.empty(1, 2, 4)
    key = torch.empty(1, 1, 4)
    value = torch.empty_like(key)
    kv_cache = torch.empty(1)

    result = impl.forward(
        layer,
        query,
        key,
        value,
        kv_cache,
        metadata,
        output,
    )
    assert result is output
    impl.do_kv_cache_update.assert_called_once()
    call_args = impl.do_kv_cache_update.call_args
    assert call_args.args == (layer, key, value, kv_cache, slot_mapping)
    impl._dense_flash_forward.assert_called_once()

    impl.do_kv_cache_update.reset_mock()
    layer.kv_sharing_target_layer_name = "model.layers.0.self_attn.attn"
    impl.forward(layer, query, key, value, kv_cache, metadata, output)
    impl.do_kv_cache_update.assert_not_called()


def test_assemble_and_sparse_attn():
    device = _device()
    if device.type != "cuda":
        return
    hq, hkv, d = 4, 2, 128
    q = torch.randn(1, hq, d, device=device, dtype=torch.bfloat16)
    topk = torch.arange(16, device=device).view(1, -1).expand(hkv, -1)
    idx, valid = assemble_sparse_context_indices(512, topk, 64, 256, device)
    k = torch.randn(hkv, idx.shape[1], d, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = sparse_decode_attention(q, k, v, 0.1, valid)
    assert out.shape == (1, hq, d)


def test_sparse_decode_attention_masks_invalid_slots():
    # A masked run must equal attention over only the valid slots: padding /
    # invalid context tokens must not take any softmax weight.
    device = _device()
    if device.type != "cuda":
        return
    hq, hkv, d, n = 4, 2, 128, 32
    q = torch.randn(1, hq, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(hkv, n, d, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    valid = torch.ones(hkv, n, dtype=torch.bool, device=device)
    valid[:, n // 2 :] = False
    masked = sparse_decode_attention(q, k, v, 0.1, valid)
    ref = sparse_decode_attention(
        q, k[:, : n // 2].contiguous(), v[:, : n // 2].contiguous(), 0.1, None
    )
    assert masked.shape == (1, hq, d)
    assert torch.allclose(masked.float(), ref.float(), atol=1e-2, rtol=1e-2)


def test_gather_kv_physical():
    device = _device()
    key = torch.randn(4, 16, 2, 64, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    bt = torch.tensor([0, 1, 2, 3], device=device)
    lids = torch.arange(0, 16, device=device)
    gk, gv = gather_kv_by_logical_indices(key, value, bt, lids, 16)
    assert gk.shape[0] == 2
    assert torch.allclose(gk[0, 0], key[0, 0, 0])


def test_decode_block_summary_triton_matches_reference():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    key = torch.randn(8, 16, 2, 256, device=device, dtype=torch.bfloat16)
    fused = ZoomKVBlockSummary(8, 2, 256, 16, device)
    reference = ZoomKVBlockSummary(8, 2, 256, 16, device)
    fused.update_completed_slots(key, torch.tensor([47], device=device))
    reference.update_blocks_from_key_cache(key, torch.tensor([2], device=device))
    torch.accelerator.synchronize()
    assert torch.equal(fused.valid, reference.valid)
    assert torch.equal(fused.chunk_min[2], reference.chunk_min[2])
    assert torch.equal(fused.chunk_max[2], reference.chunk_max[2])
    assert torch.equal(fused.centroid[2], reference.centroid[2])
    assert torch.equal(fused.packed[2], reference.packed[2])


def test_gather_and_assemble_batch_matches_serial():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.v1.attention.ops.zoomkv.paged import (
        assemble_sparse_context_indices_batch,
        gather_kv_by_logical_indices_batch,
        sparse_decode_attention_batch,
    )

    device = torch.device("cuda")
    block_size, hkv, d = 16, 2, 64
    num_blocks, batch = 32, 4
    key = torch.randn(num_blocks, block_size, hkv, d, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    # Distinct physical pages per request.
    block_table = torch.arange(
        batch * 8, device=device, dtype=torch.int32
    ).view(batch, 8)
    seq_lens = torch.tensor([96, 112, 128, 144], device=device, dtype=torch.int32)
    topk = torch.stack(
        [
            torch.arange(16, 16 + 8, device=device).view(1, -1).expand(hkv, -1)
            + i * 8
            for i in range(batch)
        ],
        dim=0,
    ).to(torch.int64)

    idx_b, valid_b = assemble_sparse_context_indices_batch(
        seq_lens, topk, sink_size=16, local_size=32
    )
    for i in range(batch):
        idx_i, valid_i = assemble_sparse_context_indices(
            int(seq_lens[i]),
            topk[i],
            sink_size=16,
            local_size=32,
            device=device,
        )
        assert torch.equal(idx_b[i, :, : idx_i.shape[1]], idx_i)
        assert torch.equal(valid_b[i, :, : valid_i.shape[1]], valid_i)

    gk_b, gv_b = gather_kv_by_logical_indices_batch(
        key, value, block_table, idx_b, block_size
    )
    for i in range(batch):
        gk_i, gv_i = gather_kv_by_logical_indices(
            key, value, block_table[i], idx_b[i], block_size
        )
        assert torch.allclose(gk_b[i], gk_i)
        assert torch.allclose(gv_b[i], gv_i)

    q = torch.randn(batch, 4, d, device=device, dtype=torch.bfloat16)
    out_b = sparse_decode_attention_batch(q, gk_b, gv_b, 0.1, valid_mask=None)
    assert out_b.shape == (batch, 4, d)
    out_buf = torch.empty_like(out_b)
    out_reused = sparse_decode_attention_batch(
        q, gk_b, gv_b, 0.1, valid_mask=None, out=out_buf
    )
    assert out_reused is out_buf
    assert torch.allclose(
        out_reused.float(), out_b.float(), atol=2e-2, rtol=2e-2
    )
    for i in range(batch):
        out_i = sparse_decode_attention(q[i : i + 1], gk_b[i], gv_b[i], 0.1, None)
        assert torch.allclose(out_b[i].float(), out_i[0].float(), atol=2e-2, rtol=2e-2)


def test_retrieve_topk_batch_matches_serial():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    clear_block_summaries()
    block_size, hkv, d = 16, 2, 128
    num_blocks = 512
    key = torch.randn(num_blocks, block_size, hkv, d, device=device, dtype=torch.bfloat16)

    cfg = ZoomKVRuntimeConfig(
        sink_size=64,
        local_size=256,
        final_topk=32,
        quest_chunk=16,
        quest_large_chunk=256,
        full_attention_threshold=500,
    )
    retriever = ZoomKVRetriever(cfg)
    batch = 4
    # Keep seq_lens long enough for hierarchical Quest, but small enough that
    # disjoint block tables fit in the summary pool.
    seq_lens = [1280, 1280, 1280, 1280]
    max_blocks = max((s + block_size - 1) // block_size for s in seq_lens)
    needed = sum((s + block_size - 1) // block_size for s in seq_lens)
    assert needed <= num_blocks
    block_table = torch.full(
        (batch, max_blocks), -1, dtype=torch.int32, device=device
    )
    cursor = 0
    for i, seq_len in enumerate(seq_lens):
        n_b = (seq_len + block_size - 1) // block_size
        block_table[i, :n_b] = torch.arange(
            cursor, cursor + n_b, device=device, dtype=torch.int32
        )
        cursor += n_b

    used = block_table[block_table >= 0].unique()
    summary = ZoomKVBlockSummary(
        num_blocks, hkv, d, block_size, device, dtype=torch.bfloat16
    )
    summary.update_blocks_from_key_cache(key, used)

    raw_q = torch.randn(batch, hkv, d, device=device, dtype=torch.bfloat16)
    batch_topk = retriever.retrieve_topk_tokens_batch(
        raw_q, summary, block_table, torch.tensor(seq_lens, device=device)
    )
    assert batch_topk.shape == (batch, hkv, cfg.final_topk)

    if not zoomkv_kernels.direct_physical_retrieval_available():
        pytest.skip("direct physical ZoomKV extension unavailable")

    uniform_bucket = retriever._chunk_bucket(
        max_blocks
        - cfg.sink_size // block_size
        - (cfg.local_size + block_size - 1) // block_size
    )
    for i, seq_len in enumerate(seq_lens):
        start_b, end_b = retriever.retrieval_block_range(seq_len, block_size)
        actual = end_b - start_b
        phys = torch.full(
            (1, uniform_bucket),
            -1,
            dtype=torch.int32,
            device=device,
        )
        phys[:, :actual].copy_(block_table[i : i + 1, start_b:end_b])
        serial = retriever._retrieve_topk_physical(
            raw_q[i : i + 1],
            summary,
            phys,
            uniform_bucket,
            start_b * block_size,
            actual_num_chunks=torch.tensor(
                [actual], dtype=torch.int32, device=device
            ),
        )
        a = set(batch_topk[i].reshape(-1).tolist()) - {-1}
        b = set(serial[0].reshape(-1).tolist()) - {-1}
        assert a == b, f"req {i} topk set mismatch"

    # Unequal retrieval widths keep exact per-request budgets but still use
    # direct physical retrieval rather than materializing summaries.
    mixed_lens = [1024, 1280, 1536, 1152]
    mixed_max = max((s + block_size - 1) // block_size for s in mixed_lens)
    mixed_bt = torch.full(
        (batch, mixed_max), -1, dtype=torch.int32, device=device
    )
    cursor = 0
    for i, seq_len in enumerate(mixed_lens):
        n_b = (seq_len + block_size - 1) // block_size
        mixed_bt[i, :n_b] = torch.arange(
            cursor, cursor + n_b, device=device, dtype=torch.int32
        )
        cursor += n_b
    used2 = mixed_bt[mixed_bt >= 0].unique()
    summary.update_blocks_from_key_cache(key, used2)
    topk_out = torch.full(
        (batch, hkv, cfg.final_topk),
        -777,
        dtype=torch.int64,
        device=device,
    )
    mixed_result = retriever.retrieve_topk_tokens_batch_result(
        raw_q,
        summary,
        mixed_bt,
        torch.tensor(mixed_lens, device=device),
        summaries_guaranteed_valid=True,
        topk_out=topk_out,
    )
    mixed_topk = mixed_result.topk
    assert mixed_topk is topk_out
    assert mixed_result.used_direct_physical is True
    assert mixed_result.context_fully_valid is True
    graph_result = retriever.retrieve_topk_tokens_batch_result(
        raw_q,
        summary,
        mixed_bt,
        torch.tensor(mixed_lens, device=device),
        summaries_guaranteed_valid=True,
        chunk_bucket=retriever._chunk_bucket(
            mixed_bt.shape[1]
            - cfg.sink_size // block_size
            - (cfg.local_size + block_size - 1) // block_size
        ),
        seq_lens_host=torch.tensor(mixed_lens),
        use_cudagraph=True,
    )
    assert graph_result.used_direct_physical is True
    assert graph_result.context_fully_valid is True
    for i in range(batch):
        eager = set(mixed_topk[i].reshape(-1).tolist()) - {-1}
        graphed = set(graph_result.topk[i].reshape(-1).tolist()) - {-1}
        assert graphed == eager
    start_b = cfg.sink_size // block_size
    bucket = retriever._chunk_bucket(mixed_bt.shape[1] - start_b)
    padded_ids = torch.full(
        (1, bucket), -1, dtype=torch.int32, device=device
    )
    for i, seq_len in enumerate(mixed_lens):
        start_b, end_b = retriever.retrieval_block_range(seq_len, block_size)
        actual = end_b - start_b
        padded_ids.fill_(-1)
        padded_ids[:, :actual].copy_(mixed_bt[i : i + 1, start_b:end_b])
        serial = retriever._retrieve_topk_physical(
            raw_q[i : i + 1],
            summary,
            padded_ids,
            bucket,
            start_b * block_size,
            actual_num_chunks=torch.tensor(
                [actual], dtype=torch.int32, device=device
            ),
        )
        a = set(mixed_topk[i].reshape(-1).tolist()) - {-1}
        b = set(serial[0].reshape(-1).tolist()) - {-1}
        assert a == b, f"mixed req {i} topk set mismatch"
        assert all(
            start_b * block_size <= token < end_b * block_size
            for token in a
        )

    # Reuse the same bucket with shorter actual widths. Padding must be
    # overwritten rather than leaking scores/indices from the previous call.
    shorter_lens = [768, 896, 1024, 832]
    stale_check = retriever.retrieve_topk_tokens_batch_result(
        raw_q,
        summary,
        mixed_bt,
        torch.tensor(shorter_lens, device=device),
        summaries_guaranteed_valid=True,
        topk_out=topk_out,
    ).topk
    for i, seq_len in enumerate(shorter_lens):
        start_b, end_b = retriever.retrieval_block_range(seq_len, block_size)
        valid = stale_check[i][stale_check[i] >= 0]
        assert bool(
            ((valid >= start_b * block_size) & (valid < end_b * block_size)).all()
        )


def test_direct_physical_retrieval_matches_materialized():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not zoomkv_kernels.direct_physical_retrieval_available():
        pytest.skip("direct physical ZoomKV extension unavailable")

    device = torch.device("cuda")
    block_size, hkv, d = 16, 2, 128
    num_blocks, batch, n_chunks = 256, 2, 64
    cfg = ZoomKVRuntimeConfig(
        sink_size=64,
        local_size=256,
        final_topk=32,
        quest_chunk=16,
        quest_large_chunk=256,
        full_attention_threshold=500,
    )
    retriever = ZoomKVRetriever(cfg)
    summary = ZoomKVBlockSummary(
        num_blocks, hkv, d, block_size, device, dtype=torch.bfloat16
    )
    key = torch.randn(
        num_blocks,
        block_size,
        hkv,
        d,
        device=device,
        dtype=torch.bfloat16,
    )
    # Non-monotonic IDs catch accidental logical indexing into global pools.
    physical_ids = torch.stack(
        (
            torch.randperm(num_blocks, device=device)[:n_chunks],
            torch.randperm(num_blocks, device=device)[:n_chunks],
        )
    ).to(torch.int32)
    summary.update_blocks_from_key_cache(
        key, physical_ids.reshape(-1).unique()
    )
    raw_q = torch.randn(
        batch, hkv, d, device=device, dtype=torch.bfloat16
    )
    token_offset = cfg.sink_size

    direct = retriever._retrieve_topk_physical(
        raw_q, summary, physical_ids, n_chunks, token_offset
    )
    packed, cmin, cmax, centroid, valid = (
        summary.gather_batch_block_summaries(
            physical_ids.to(torch.int64),
            chunk_valid=None,
            assume_valid_ids=True,
        )
    )
    materialized = retriever.retrieve_topk_from_block_summaries(
        raw_q,
        packed,
        cmin,
        cmax,
        centroid,
        valid,
        seq_len=token_offset + n_chunks * block_size + cfg.local_size,
        block_size=block_size,
        start_b=token_offset // block_size,
    )
    assert direct.shape == materialized.shape
    for b in range(batch):
        for h in range(hkv):
            assert set(direct[b, h].tolist()) == set(
                materialized[b, h].tolist()
            )
    assert ((direct == -1) | (direct >= token_offset)).all()


def test_batched_sparse_backend_matches_serial_path():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from types import SimpleNamespace

    device = torch.device("cuda")
    clear_block_summaries()
    torch.manual_seed(0)

    block_size, hq, hkv, d = 16, 4, 2, 128
    cfg = ZoomKVRuntimeConfig(
        sink_size=32,
        local_size=64,
        final_topk=16,
        quest_chunk=16,
        quest_large_chunk=256,
        full_attention_threshold=200,
        dense_fallback=False,
        enable_offload=False,
    )
    batch = 2
    seq_lens = [768, 768]
    max_blocks = max((s + block_size - 1) // block_size for s in seq_lens)
    num_blocks = sum((s + block_size - 1) // block_size for s in seq_lens) + 4

    key_cache = torch.randn(
        num_blocks, block_size, hkv, d, device=device, dtype=torch.bfloat16
    )
    value_cache = torch.randn_like(key_cache)
    kv_cache = torch.stack([key_cache, value_cache], dim=1)
    # Match the device object the backend passes (e.g. cuda:0), otherwise
    # get_or_create_block_summary recreates an empty summary.
    cache_device = kv_cache.device

    block_table = torch.full(
        (batch, max_blocks), -1, dtype=torch.int32, device=cache_device
    )
    cursor = 0
    for i, seq_len in enumerate(seq_lens):
        n_b = (seq_len + block_size - 1) // block_size
        block_table[i, :n_b] = torch.arange(
            cursor, cursor + n_b, device=cache_device, dtype=torch.int32
        )
        cursor += n_b

    summary = get_or_create_block_summary(
        layer_name="test_batched_layer",
        num_blocks=num_blocks,
        num_kv_heads=hkv,
        head_dim=d,
        block_size=block_size,
        device=cache_device,
        dtype=torch.bfloat16,
        blocks_per_parent=max(1, cfg.quest_large_chunk // cfg.quest_chunk),
    )
    for b in range(cursor):
        summary.update_blocks_from_key_cache(
            key_cache, torch.tensor([b], device=cache_device)
        )

    query = torch.randn(batch, hq, d, device=cache_device, dtype=torch.bfloat16)
    seq_lens_t = torch.tensor(seq_lens, device=cache_device, dtype=torch.int32)
    seq_lens_cpu = seq_lens_t.cpu()
    q_start = torch.arange(batch + 1, dtype=torch.int32)
    # Sparse paths only consume ZoomKV-specific fields; avoid depending on the
    # full TritonAttentionMetadata constructor surface.
    metadata = SimpleNamespace(
        num_actual_tokens=batch,
        max_query_len=1,
        query_start_loc=q_start.to(cache_device),
        max_seq_len=max(seq_lens),
        seq_lens=seq_lens_t,
        block_table=block_table,
        slot_mapping=torch.zeros(batch, dtype=torch.int64, device=cache_device),
        num_reqs=batch,
        num_decodes=batch,
        num_prefills=0,
        num_decode_tokens=batch,
        zoomkv=cfg,
        query_start_loc_cpu=q_start,
        seq_lens_cpu=seq_lens_cpu,
        topk_indices_buffer=torch.full(
            (batch, hkv, cfg.final_topk), -1, dtype=torch.int64, device=cache_device
        ),
        context_indices_buffer=torch.full(
            (batch, hkv, cfg.sink_size + cfg.local_size + cfg.final_topk),
            -1,
            dtype=torch.int64,
            device=cache_device,
        ),
    )

    impl = ZoomKVAttentionImpl.__new__(ZoomKVAttentionImpl)
    impl.num_heads = hq
    impl.head_size = d
    impl.scale = d**-0.5
    impl.num_kv_heads = hkv
    impl.kv_cache_dtype = "auto"
    impl.block_size = block_size
    impl._retriever = None
    impl._layer_name = "test_batched_layer"

    out_batched = torch.empty_like(query)
    out_serial = torch.empty_like(query)
    layer = SimpleNamespace(layer_name="test_batched_layer")

    impl._sparse_decode_forward_batched(
        layer, query, kv_cache, metadata, out_batched, cfg
    )
    impl._sparse_decode_forward(layer, query, kv_cache, metadata, out_serial, cfg)
    assert torch.allclose(
        out_batched.float(), out_serial.float(), atol=5e-2, rtol=5e-2
    )


def test_gather_kv_from_topk_batch_matches_assemble_gather():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.v1.attention.ops.zoomkv.paged import (
        assemble_sparse_context_indices_batch,
        gather_kv_by_logical_indices_batch,
        gather_kv_from_topk_batch,
    )

    device = torch.device("cuda")
    block_size, hkv, d = 16, 2, 64
    sink, local, tk = 16, 32, 8
    batch = 4
    seq_lens = torch.tensor([512, 528, 544, 560], device=device, dtype=torch.int32)
    max_blocks = int((int(seq_lens.max()) + block_size - 1) // block_size)
    num_blocks = batch * max_blocks
    key = torch.randn(
        num_blocks, block_size, hkv, d, device=device, dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    block_table = torch.arange(
        batch * max_blocks, device=device, dtype=torch.int32
    ).view(batch, max_blocks)
    # Long enough that sink+local are fully filled (fully-valid width).
    topk = torch.stack(
        [
            torch.arange(64, 64 + tk, device=device).view(1, -1).expand(hkv, -1)
            + i * 8
            for i in range(batch)
        ],
        dim=0,
    ).to(torch.int64)

    gk_f, gv_f = gather_kv_from_topk_batch(
        key, value, block_table, seq_lens, topk, block_size, sink, local
    )
    idx, valid = assemble_sparse_context_indices_batch(seq_lens, topk, sink, local)
    assert bool(valid.all())
    gk_r, gv_r = gather_kv_by_logical_indices_batch(
        key, value, block_table, idx, block_size
    )
    assert torch.equal(gk_f, gk_r)
    assert torch.equal(gv_f, gv_r)
    gk_packed, gv_packed = gather_kv_from_topk_batch(
        key,
        value,
        block_table,
        seq_lens,
        topk,
        block_size,
        sink,
        local,
        output_bthd=True,
    )
    assert gk_packed.is_contiguous()
    assert gv_packed.is_contiguous()
    assert torch.equal(gk_packed, gk_r.permute(0, 2, 1, 3))
    assert torch.equal(gv_packed, gv_r.permute(0, 2, 1, 3))


def test_gather_kv_from_topk_batch_handles_short_seq_padding():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.v1.attention.ops.zoomkv.paged import (
        assemble_sparse_context_indices_batch,
        gather_kv_by_logical_indices_batch,
        gather_kv_from_topk_batch,
    )

    device = torch.device("cuda")
    block_size, hkv, d = 16, 2, 64
    num_blocks, batch = 32, 2
    sink, local, tk = 16, 32, 8
    key = torch.randn(
        num_blocks, block_size, hkv, d, device=device, dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    block_table = torch.arange(
        batch * 8, device=device, dtype=torch.int32
    ).view(batch, 8)
    # Variable / short lengths exercise invalid padding slots.
    seq_lens = torch.tensor([40, 96], device=device, dtype=torch.int32)
    topk = torch.full((batch, hkv, tk), -1, dtype=torch.int64, device=device)
    topk[0, :, :3] = torch.arange(16, 19, device=device)
    topk[1, :, :] = torch.arange(16, 16 + tk, device=device)

    gk_f, gv_f = gather_kv_from_topk_batch(
        key, value, block_table, seq_lens, topk, block_size, sink, local
    )
    idx, _ = assemble_sparse_context_indices_batch(seq_lens, topk, sink, local)
    gk_r, gv_r = gather_kv_by_logical_indices_batch(
        key, value, block_table, idx, block_size
    )
    assert torch.equal(gk_f, gk_r)
    assert torch.equal(gv_f, gv_r)


def test_direct_physical_bf16_d128_specialized_kernels_match_reference():
    """Parent/sub/density BF16-D128 specialized kernels vs torch reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.v1.attention.ops.zoomkv.kernels import (
        density_score_physical,
        direct_physical_retrieval_available,
        quest_chunk_score_physical,
        quest_parent_score_physical,
        quest_sub_score_physical,
    )

    if not direct_physical_retrieval_available():
        pytest.skip("_zoomkv_C direct physical kernels unavailable")

    device = torch.device("cuda")
    batch, hkv, d = 2, 2, 128
    n_chunks, factor = 64, 16
    num_blocks = 128
    torch.manual_seed(0)
    q = torch.randn(batch, hkv, d, device=device, dtype=torch.bfloat16)
    physical_ids = torch.arange(n_chunks, device=device, dtype=torch.int32).view(
        1, -1
    ).expand(batch, -1).contiguous()
    # Offset second request's physical ids.
    physical_ids = physical_ids + torch.arange(
        batch, device=device, dtype=torch.int32
    ).view(batch, 1) * n_chunks
    physical_ids = physical_ids.clamp(max=num_blocks - 1).contiguous()

    gmin = torch.randn(num_blocks, hkv, d, device=device, dtype=torch.bfloat16)
    gmax = gmin + torch.rand_like(gmin).abs()
    gcent = torch.randn(num_blocks, hkv, d, device=device, dtype=torch.bfloat16)
    gvalid = torch.ones(num_blocks, dtype=torch.bool, device=device)
    gvalid[::7] = False

    n_parent = n_chunks // factor
    parent_scores = torch.empty(batch, hkv, n_parent, device=device, dtype=torch.float32)
    quest_parent_score_physical(
        q, physical_ids, gmin, gmax, gvalid, parent_scores, n_chunks, factor
    )
    # Reference: gather children and reduce.
    ref_parent = torch.full_like(parent_scores, float("-inf"))
    for b in range(batch):
        for h in range(hkv):
            for p in range(n_parent):
                pmin = torch.full((d,), float("inf"), device=device)
                pmax = torch.full((d,), float("-inf"), device=device)
                any_v = False
                for c in range(factor):
                    child = p * factor + c
                    pid = int(physical_ids[b, child])
                    if pid < 0 or not bool(gvalid[pid]):
                        continue
                    pmin = torch.minimum(pmin, gmin[pid, h].float())
                    pmax = torch.maximum(pmax, gmax[pid, h].float())
                    any_v = True
                if any_v:
                    qv = q[b, h].float()
                    ref_parent[b, h, p] = torch.maximum(qv * pmin, qv * pmax).sum()
    assert torch.allclose(parent_scores, ref_parent, atol=2e-2, rtol=2e-2)

    n_selected = 8
    large_ids = torch.randint(
        0, n_parent, (batch, hkv, n_selected), device=device, dtype=torch.int64
    )
    sub_scores = torch.empty(
        batch, hkv, n_selected * factor, device=device, dtype=torch.float32
    )
    quest_sub_score_physical(
        q,
        physical_ids,
        gmin,
        gmax,
        gvalid,
        large_ids,
        sub_scores,
        n_selected,
        factor,
        n_chunks,
    )
    ref_sub = torch.full_like(sub_scores, float("-inf"))
    for b in range(batch):
        for h in range(hkv):
            for s in range(n_selected):
                parent = int(large_ids[b, h, s])
                for c in range(factor):
                    child = parent * factor + c
                    pos = s * factor + c
                    if child < 0 or child >= n_chunks:
                        continue
                    pid = int(physical_ids[b, child])
                    if pid < 0 or not bool(gvalid[pid]):
                        continue
                    qv = q[b, h].float()
                    mn = gmin[pid, h].float()
                    mx = gmax[pid, h].float()
                    ref_sub[b, h, pos] = torch.maximum(qv * mn, qv * mx).sum()
    assert torch.allclose(sub_scores, ref_sub, atol=2e-2, rtol=2e-2)

    chunk_ids = torch.randint(
        0, n_chunks, (batch, hkv, 24), device=device, dtype=torch.int64
    )
    density = torch.empty(batch, hkv, 24, device=device, dtype=torch.float32)
    density_score_physical(
        chunk_ids, physical_ids, gcent, gvalid, q, density, n_chunks
    )
    ref_den = torch.full_like(density, float("-inf"))
    for b in range(batch):
        for h in range(hkv):
            for s in range(24):
                chunk = int(chunk_ids[b, h, s])
                if chunk < 0 or chunk >= n_chunks:
                    continue
                pid = int(physical_ids[b, chunk])
                if pid < 0 or not bool(gvalid[pid]):
                    continue
                ref_den[b, h, s] = (
                    gcent[pid, h].float() * q[b, h].float()
                ).sum()
    assert torch.allclose(density, ref_den, atol=2e-2, rtol=2e-2)

    # Bucket launches must overwrite every padded slot. Seed scratch with
    # finite values to catch stale data surviving a shorter request.
    actual = torch.tensor([n_chunks, 23], dtype=torch.int32, device=device)
    flat_scores = torch.full(
        (batch, hkv, n_chunks), 123.0, device=device, dtype=torch.float32
    )
    quest_chunk_score_physical(
        q,
        physical_ids,
        gmin,
        gmax,
        gvalid,
        flat_scores,
        n_chunks,
        actual,
    )
    assert bool(torch.isneginf(flat_scores[1, :, 23:]).all())

    parent_scores.fill_(123.0)
    quest_parent_score_physical(
        q,
        physical_ids,
        gmin,
        gmax,
        gvalid,
        parent_scores,
        n_chunks,
        factor,
        actual,
    )
    # Parent 1 contains actual chunks 16..22. Later parents are outside the
    # per-row Top-K scan length, so the producer deliberately leaves that
    # contiguous tail untouched instead of spending work writing sentinels.
    assert bool((parent_scores[1, :, 2:] == 123.0).all())

    chunk_ids[1, :, :] = torch.arange(
        24, device=device, dtype=torch.int64
    )
    density.fill_(123.0)
    density_score_physical(
        chunk_ids,
        physical_ids,
        gcent,
        gvalid,
        q,
        density,
        n_chunks,
        actual,
    )
    assert bool(torch.isneginf(density[1, :, 23:]).all())


def test_retrieve_marks_fully_valid_direct_path():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.v1.attention.ops.zoomkv.kernels import (
        direct_physical_retrieval_available,
    )

    if not direct_physical_retrieval_available():
        pytest.skip("_zoomkv_C direct physical kernels unavailable")

    device = torch.device("cuda")
    clear_block_summaries()
    block_size, hkv, d = 16, 2, 128
    num_blocks = 512
    key = torch.randn(
        num_blocks, block_size, hkv, d, device=device, dtype=torch.bfloat16
    )
    cfg = ZoomKVRuntimeConfig(
        sink_size=64,
        local_size=256,
        final_topk=32,
        quest_chunk=16,
        quest_large_chunk=256,
        full_attention_threshold=500,
    )
    retriever = ZoomKVRetriever(cfg)
    seq_lens = [1280, 1280]
    batch = len(seq_lens)
    max_blocks = max((s + block_size - 1) // block_size for s in seq_lens)
    block_table = torch.full(
        (batch, max_blocks), -1, dtype=torch.int32, device=device
    )
    cursor = 0
    for i, seq_len in enumerate(seq_lens):
        n_b = (seq_len + block_size - 1) // block_size
        block_table[i, :n_b] = torch.arange(
            cursor, cursor + n_b, device=device, dtype=torch.int32
        )
        cursor += n_b
    used = block_table[block_table >= 0].unique()
    summary = ZoomKVBlockSummary(
        num_blocks, hkv, d, block_size, device, dtype=torch.bfloat16
    )
    summary.update_blocks_from_key_cache(key, used)
    raw_q = torch.randn(batch, hkv, d, device=device, dtype=torch.bfloat16)
    result = retriever.retrieve_topk_tokens_batch_result(
        raw_q,
        summary,
        block_table,
        torch.tensor(seq_lens, device=device),
        summaries_guaranteed_valid=True,
    )
    topk = result.topk
    assert topk.shape == (batch, hkv, cfg.final_topk)
    assert result.used_direct_physical is True
    assert result.context_fully_valid is True
    assert bool((topk >= 0).all())

    # Without the lifecycle guarantee, direct retrieval may still execute but
    # must explicitly force the backend onto its safe masked fallback.
    summary.valid.index_fill_(0, used.to(torch.int64), False)
    invalid = retriever.retrieve_topk_tokens_batch_result(
        raw_q,
        summary,
        block_table,
        torch.tensor(seq_lens, device=device),
        summaries_guaranteed_valid=False,
    )
    assert invalid.used_direct_physical is True
    assert invalid.context_fully_valid is False
    assert bool((invalid.topk < 0).all())


def test_need_summary_update_only_on_block_boundary():
    """Host metadata should request summary updates only when a block completes."""
    common = dict(
        max_query_len=1,
        num_decodes=3,
        num_reqs=3,
        block_size=16,
    )
    assert not _needs_summary_update(
        num_prefills=0,
        seq_lens_cpu=torch.tensor([1000, 1001, 1002]),
        **common,
    )
    assert _needs_summary_update(
        num_prefills=0,
        seq_lens_cpu=torch.tensor([1008, 1001, 1024]),
        **common,
    )
    assert _needs_summary_update(
        num_prefills=1,
        seq_lens_cpu=torch.tensor([1000, 1001, 1002]),
        **common,
    )


def test_use_sparse_gate_matches_builder_helper():
    """Sparse routing should be computable once per step from host metadata."""
    cfg = ZoomKVRuntimeConfig(full_attention_threshold=512)
    assert _should_use_sparse_decode(
        cfg=cfg,
        max_query_len=1,
        num_decodes=1,
        num_prefills=0,
        num_reqs=1,
        seq_lens_cpu=torch.tensor([100]),
    )
    # Pure-batch flag is False for mixed; mixed uses the separate helper.
    assert not _should_use_sparse_decode(
        cfg=cfg,
        max_query_len=1,
        num_decodes=1,
        num_prefills=1,
        num_reqs=1,
        seq_lens_cpu=torch.tensor([4096]),
    )
    assert _should_use_mixed_sparse_decode(
        cfg=cfg, num_decodes=2, num_prefills=1
    )
    assert not _should_use_sparse_decode(
        cfg=ZoomKVRuntimeConfig(dense_fallback=True),
        max_query_len=1,
        num_decodes=1,
        num_prefills=0,
        num_reqs=1,
        seq_lens_cpu=torch.tensor([4096]),
    )


def test_mixed_forward_routes_sparse_prefix_and_dense_suffix():
    """Mixed GPU-only batches must split decode sparse / prefill dense."""
    impl = ZoomKVAttentionImpl.__new__(ZoomKVAttentionImpl)
    calls: list[tuple[str, dict]] = []

    def update(layer, key, value, kv_cache, slot_mapping, **kwargs):
        calls.append(("update", {}))

    def sparse_batched(layer, query, kv_cache, attn_metadata, output, cfg, **kwargs):
        calls.append(("sparse", dict(kwargs)))
        return output

    def dense_forward(
        layer,
        query,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
        **kwargs,
    ):
        calls.append(("dense", dict(kwargs)))
        return output

    impl.do_kv_cache_update = update
    impl._sparse_decode_forward_batched = sparse_batched
    impl._dense_flash_forward = dense_forward

    tensor = torch.zeros((3, 1, 4))
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
        zoomkv=ZoomKVRuntimeConfig(enable_offload=False),
        use_sparse=False,
        use_mixed_sparse=True,
        num_decodes=2,
        num_prefills=1,
        num_decode_tokens=2,
        num_reqs=3,
        max_query_len=4,
        block_table=None,
        seq_lens=None,
        graph_chunk_bucket=None,
    )
    layer = SimpleNamespace(kv_sharing_target_layer_name=None)
    impl.forward(layer, tensor, tensor, tensor, tensor, metadata, tensor.clone())
    assert [name for name, _ in calls] == ["update", "sparse", "dense"]
    assert calls[1][1] == {"num_decode_reqs": 2, "num_decode_tokens": 2}
    assert calls[2][1]["req_start"] == 2
    assert calls[2][1]["tok_start"] == 2


def test_dense_prefill_slice_rebases_query_start_loc(monkeypatch):
    """Prefill suffix FA must rebase cu_seqlens and use local max_seq_len."""
    captured: dict = {}

    def fake_fa(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "vllm.v1.attention.backends.zoomkv_attn.flash_attn_varlen_func",
        fake_fa,
    )
    monkeypatch.setattr(
        "vllm.v1.attention.backends.zoomkv_attn.is_quantized_kv_cache",
        lambda *_a, **_k: False,
    )

    impl = ZoomKVAttentionImpl.__new__(ZoomKVAttentionImpl)
    impl.num_kv_heads = 1
    impl.scale = 1.0
    impl.kv_cache_dtype = "auto"
    impl._fa = SimpleNamespace(
        vllm_flash_attn_version=3,
        sliding_window=None,
        supports_quant_query_input=False,
        alibi_slopes=None,
        logits_soft_cap=0.0,
        sinks=None,
    )
    impl._split_kv_cache = lambda kv: (kv[:, 0], kv[:, 1])

    # 2 decode tokens + 4 prefill tokens.
    query = torch.zeros(6, 1, 8)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros(8, 2, 16, 1, 8)
    qsl = torch.tensor([0, 1, 2, 6], dtype=torch.int32)
    seq_lens = torch.tensor([4096, 4096, 128], dtype=torch.int32)
    metadata = SimpleNamespace(
        use_cascade=False,
        num_reqs=3,
        num_actual_tokens=6,
        query_start_loc=qsl.clone(),
        query_start_loc_cpu=qsl.clone(),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        block_table=torch.zeros(3, 8, dtype=torch.int32),
        max_query_len=4,
        max_seq_len=4096,
        causal=True,
        scheduler_metadata=object(),
        max_num_splits=2,
    )
    layer = SimpleNamespace(_q_scale=torch.ones(1), _k_scale=torch.ones(1), _v_scale=torch.ones(1))
    impl._dense_flash_forward(
        layer, query, kv_cache, metadata, output, req_start=2, tok_start=2
    )
    assert captured["max_seqlen_q"] == 4
    assert captured["max_seqlen_k"] == 128
    assert captured["scheduler_metadata"] is None
    assert torch.equal(captured["cu_seqlens_q"], torch.tensor([0, 4], dtype=torch.int32))
    assert captured["q"].shape[0] == 4
    assert captured["out"].shape[0] == 4


def test_forward_owns_kv_cache_update_before_attention():
    """The unified attention op must update cache before either read path."""
    assert ZoomKVAttentionBackend.forward_includes_kv_cache_update

    impl = ZoomKVAttentionImpl.__new__(ZoomKVAttentionImpl)
    calls: list[str] = []

    def update(layer, key, value, kv_cache, slot_mapping, **kwargs):
        assert torch.equal(slot_mapping, torch.tensor([3], dtype=torch.int32))
        calls.append("update")

    def dense_forward(*args, **kwargs):
        assert calls == ["update"]
        calls.append("attention")
        return args[4]

    impl.do_kv_cache_update = update
    impl._should_sparse_decode = lambda metadata, cfg: False
    impl._dense_flash_forward = dense_forward

    tensor = torch.zeros((1, 1, 4))
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([3], dtype=torch.int32),
        zoomkv=ZoomKVRuntimeConfig(enable_offload=False),
        use_sparse=False,
        use_mixed_sparse=False,
        num_decodes=0,
        num_prefills=1,
    )
    layer = SimpleNamespace(kv_sharing_target_layer_name=None)
    result = impl.forward(
        layer,
        tensor,
        tensor,
        tensor,
        tensor,
        metadata,
        tensor.clone(),
    )

    assert calls == ["update", "attention"]
    assert result.shape == tensor.shape


def test_parent_summary_invalidate_and_cow():
    device = _device()
    clear_block_summaries()
    sc = get_or_create_block_summary(
        "parent-layer",
        num_blocks=32,
        num_kv_heads=2,
        head_dim=128,
        block_size=16,
        device=device,
        dtype=torch.bfloat16,
    )
    sc.parent_valid[10] = True
    sc.parent_first_child[10] = 3
    sc.parent_min[10].fill_(1.0)
    invalidate_block_summaries_for_blocks([10])
    assert not bool(sc.parent_valid[10])
    assert int(sc.parent_first_child[10]) == -1
    assert float(sc.parent_min[10, 0, 0]) == 0.0

    sc.parent_valid[11] = True
    sc.parent_min[11].fill_(2.0)
    sc.copy_blocks([(11, 12)])
    assert bool(sc.parent_valid[12])
    assert float(sc.parent_min[12, 0, 0]) == 2.0
    copy_block_summaries_for_block_pairs([(12, 13)])
    assert bool(sc.parent_valid[13])
    clear_block_summaries()


def test_parent_finalize_matches_build_parent_minmax():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.v1.attention.ops.zoomkv.block_summary_triton import (
        finalize_parent_summaries,
    )

    device = torch.device("cuda")
    hkv, d, factor = 2, 128, 16
    start_b = 4
    n_chunks = 32
    num_blocks = 64
    sc = ZoomKVBlockSummary(num_blocks, hkv, d, 16, device)
    key = torch.randn(num_blocks, 16, hkv, d, device=device, dtype=torch.bfloat16)
    phys = torch.arange(start_b, start_b + n_chunks, device=device, dtype=torch.int32)
    sc.update_blocks_from_key_cache(key, phys)
    block_table = torch.full(
        (1, start_b + n_chunks), -1, device=device, dtype=torch.int32
    )
    block_table[0, start_b : start_b + n_chunks] = phys
    seq_len = (start_b + n_chunks) * 16
    finalize_parent_summaries(
        block_table,
        sc,
        start_block=start_b,
        seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
        scan_all=True,
    )
    torch.cuda.synchronize()

    packed, cmin, cmax, centroid, valid = sc.gather_request_block_summaries(phys)
    ref_min, ref_max, ref_valid = sc.build_parent_minmax(phys, cmin, cmax, valid)
    n_parent = n_chunks // factor
    for p in range(n_parent):
        anchor = int(phys[p * factor + factor - 1])
        assert bool(sc.parent_valid[anchor])
        assert int(sc.parent_first_child[anchor]) == int(phys[p * factor])
        assert torch.allclose(sc.parent_min[anchor], ref_min[0, :, p], atol=1e-3)
        assert torch.allclose(sc.parent_max[anchor], ref_max[0, :, p], atol=1e-3)


def test_parent_precomputed_scoring_matches_inline():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.v1.attention.ops.zoomkv.kernels import (
        direct_physical_retrieval_available,
        quest_parent_score_physical,
    )

    if not direct_physical_retrieval_available():
        pytest.skip("_zoomkv_C direct physical kernels unavailable")

    device = torch.device("cuda")
    batch, hkv, d = 1, 2, 128
    factor, n_chunks = 16, 64
    num_blocks = 128
    start_b = 4
    torch.manual_seed(7)
    q = torch.randn(batch, hkv, d, device=device, dtype=torch.bfloat16)
    phys = torch.tensor(
        [[12, 5, 20, 33, 7, 44, 19, 2, 55, 61, 8, 14, 27, 39, 48, 50] * 4],
        device=device,
        dtype=torch.int32,
    )
    sc = ZoomKVBlockSummary(num_blocks, hkv, d, 16, device)
    key = torch.randn(num_blocks, 16, hkv, d, device=device, dtype=torch.bfloat16)
    sc.update_blocks_from_key_cache(key, phys.reshape(-1).unique())
    block_table = torch.full(
        (1, start_b + n_chunks), -1, device=device, dtype=torch.int32
    )
    block_table[0, start_b : start_b + n_chunks] = phys[0]
    seq_len = (start_b + n_chunks) * 16
    sc.update_completed_slots(
        key,
        torch.tensor([(start_b + n_chunks - 1) * 16 + 15], device=device),
        block_table=block_table,
        start_block=start_b,
        seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
        scan_all_parents=True,
    )
    torch.cuda.synchronize()

    n_parent = n_chunks // factor
    scores_pre = torch.empty(batch, hkv, n_parent, device=device, dtype=torch.float32)
    quest_parent_score_physical(
        q,
        block_table[:, start_b : start_b + n_chunks],
        sc.chunk_min,
        sc.chunk_max,
        sc.valid,
        scores_pre,
        n_chunks,
        factor,
        parent_min=sc.parent_min,
        parent_max=sc.parent_max,
        parent_valid=sc.parent_valid,
        parent_first_child=sc.parent_first_child,
    )
    scores_inline = torch.empty_like(scores_pre)
    sc.parent_valid.zero_()
    quest_parent_score_physical(
        q,
        block_table[:, start_b : start_b + n_chunks],
        sc.chunk_min,
        sc.chunk_max,
        sc.valid,
        scores_inline,
        n_chunks,
        factor,
    )
    assert torch.allclose(scores_pre, scores_inline, atol=2e-2, rtol=2e-2)


def test_parent_stale_first_child_falls_back_to_inline():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.v1.attention.ops.zoomkv.kernels import (
        direct_physical_retrieval_available,
        quest_parent_score_physical,
    )

    if not direct_physical_retrieval_available():
        pytest.skip("_zoomkv_C direct physical kernels unavailable")

    device = torch.device("cuda")
    batch, hkv, d = 1, 2, 128
    factor, n_chunks = 16, 32
    num_blocks = 64
    phys = torch.arange(n_chunks, device=device, dtype=torch.int32).view(1, -1)
    sc = ZoomKVBlockSummary(num_blocks, hkv, d, 16, device)
    key = torch.randn(num_blocks, 16, hkv, d, device=device, dtype=torch.bfloat16)
    sc.update_blocks_from_key_cache(key, phys.reshape(-1))
    anchor = int(phys[0, factor - 1])
    sc.parent_valid[anchor] = True
    sc.parent_first_child[anchor] = 999  # stale
    sc.parent_min[anchor].fill_(0.0)
    sc.parent_max[anchor].fill_(0.0)

    q = torch.randn(batch, hkv, d, device=device, dtype=torch.bfloat16)
    n_parent = n_chunks // factor
    scores = torch.empty(batch, hkv, n_parent, device=device, dtype=torch.float32)
    quest_parent_score_physical(
        q,
        phys,
        sc.chunk_min,
        sc.chunk_max,
        sc.valid,
        scores,
        n_chunks,
        factor,
        parent_min=sc.parent_min,
        parent_max=sc.parent_max,
        parent_valid=sc.parent_valid,
        parent_first_child=sc.parent_first_child,
    )
    ref = torch.empty_like(scores)
    quest_parent_score_physical(
        q,
        phys,
        sc.chunk_min,
        sc.chunk_max,
        sc.valid,
        ref,
        n_chunks,
        factor,
    )
    assert torch.allclose(scores, ref, atol=2e-2, rtol=2e-2)
