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
from vllm.v1.attention.backends.zoomkv_attn import (
    ZoomKVAttentionImpl,
    _needs_summary_update,
)
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

    for i, seq_len in enumerate(seq_lens):
        start_b, end_b = retriever.retrieval_block_range(seq_len, block_size)
        phys = block_table[i, start_b:end_b].to(torch.int64)
        serial = retriever.retrieve_topk_tokens(
            raw_q[i : i + 1], summary, phys, seq_len
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
    mixed_result = retriever.retrieve_topk_tokens_batch_result(
        raw_q,
        summary,
        mixed_bt,
        torch.tensor(mixed_lens, device=device),
        summaries_guaranteed_valid=True,
    )
    mixed_topk = mixed_result.topk
    assert mixed_result.used_direct_physical is True
    assert mixed_result.context_fully_valid is True
    for i, seq_len in enumerate(mixed_lens):
        start_b, end_b = retriever.retrieval_block_range(seq_len, block_size)
        phys = mixed_bt[i, start_b:end_b].to(torch.int64)
        serial = retriever.retrieve_topk_tokens(
            raw_q[i : i + 1], summary, phys, seq_len
        )
        a = set(mixed_topk[i].reshape(-1).tolist()) - {-1}
        b = set(serial[0].reshape(-1).tolist()) - {-1}
        assert a == b, f"mixed req {i} topk set mismatch"


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
