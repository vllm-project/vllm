# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the GPU-only ZoomKV ops and block_summary lifecycle."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.attention.ops.zoomkv.kernels as zoomkv_kernels
from vllm.config.attention import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.zoomkv_attn import ZoomKVAttentionImpl
from vllm.v1.attention.ops.zoomkv.kernels import get_quest_ops, quest_score_reference
from vllm.v1.attention.ops.zoomkv.paged import (
    assemble_sparse_context_indices,
    gather_kv_by_logical_indices,
    sparse_decode_attention,
)
from vllm.v1.attention.ops.zoomkv.quant_pack import pack_block_kcache_4bit
from vllm.v1.attention.ops.zoomkv.quest import QuestTorchOps
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
    assert r.should_use_dense(100)
    assert not r.should_use_dense(5000)
    assert (
        r.retrieval_block_range(64 + 256, 16)[0]
        == r.retrieval_block_range(64 + 256, 16)[1]
    )
    s, e = r.retrieval_block_range(4096, 16)
    assert e > s


def test_retriever_pads_when_candidates_are_fewer_than_final_topk():
    device = _device()
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

    assert attn.zoomkv_quest_large_ratio == runtime.quest_large_ratio
    assert attn.zoomkv_quest_small_ratio == runtime.quest_small_ratio
    assert attn.zoomkv_dense_ratio == runtime.dense_ratio
    assert attn.zoomkv_quest_chunk == runtime.quest_chunk == 16
    assert attn.zoomkv_enable_offload == runtime.enable_offload is False
    # GQA group-mean retrieval query, matching the original ZoomKV
    # implementation; the max-head variant measured ~0.11 lower Top-K recall.
    assert attn.zoomkv_per_query_head == runtime.per_query_head is False


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


def test_prepare_retrieval_query_picks_strongest_head():
    from vllm.v1.attention.ops.zoomkv.retriever import prepare_retrieval_query

    device = _device()
    q = torch.zeros(1, 4, 8, device=device, dtype=torch.bfloat16)
    q[0, 1] = 3.0
    q[0, 3] = 1.0
    out = prepare_retrieval_query(q, num_kv_heads=2, per_query_head=True)
    assert out.shape == (1, 2, 8)
    assert torch.allclose(out[0, 0], q[0, 1])
    assert torch.allclose(out[0, 1], q[0, 3])


def test_sparse_decode_gate():
    impl = ZoomKVAttentionImpl.__new__(ZoomKVAttentionImpl)
    impl._retriever = None
    metadata = SimpleNamespace(
        max_query_len=1,
        num_decodes=1,
        num_prefills=0,
        num_reqs=1,
        seq_lens_cpu=torch.tensor([4096]),
        seq_lens=torch.tensor([4096]),
    )
    cfg = ZoomKVRuntimeConfig(full_attention_threshold=512)

    assert impl._should_sparse_decode(metadata, cfg)
    metadata.num_prefills = 1
    assert not impl._should_sparse_decode(metadata, cfg)
    metadata.num_prefills = 0
    metadata.max_query_len = 2
    assert not impl._should_sparse_decode(metadata, cfg)
    metadata.max_query_len = 1
    assert not impl._should_sparse_decode(
        metadata, ZoomKVRuntimeConfig(dense_fallback=True)
    )


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
