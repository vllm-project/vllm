# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 MXFP4 caches: gfx950 K/V and index geometry, the gfx942
index-only cache and its kernels, and dtype dispatch."""

import importlib.util

import pytest
import torch

from vllm.config.cache import CacheConfig
from vllm.models.minimax_m3.amd.ops.index_norm_rope_gfx942 import (
    index_norm_rope_gfx942,
)
from vllm.models.minimax_m3.amd.ops.mxfp4_gfx942 import (
    MANAGER_BLOCK_BYTES,
    PAGE_DATA_BYTES,
    PAGE_SCALE_BYTES,
    expand_block_table_page64,
    expand_block_table_page64_into,
    gfx942_mxfp4_cache_strides,
    gfx942_mxfp4_cache_views,
    store_k_gfx942,
)
from vllm.models.minimax_m3.amd.ops.mxfp4_pages import (
    PAGE_BYTES,
    PAGE_TOKENS,
    mxfp4_page_views,
)
from vllm.models.minimax_m3.amd.sparse_attention_mxfp4 import (
    MiniMaxM3SparseRocmMxfp4Backend,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    select_main_backend_and_impl_cls,
)
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    MLAAttentionSpec,
    SparseCacheRole,
    get_kv_quant_mode,
)


def test_page_is_data_then_scale():
    assert PAGE_BYTES == PAGE_TOKENS * 68 == 8704
    pages = torch.arange(3 * PAGE_BYTES, dtype=torch.int64).remainder(251)
    pages = pages.to(torch.uint8).view(3, PAGE_BYTES)
    data, scale = mxfp4_page_views(pages)
    assert data.shape == (3, 128, 64) and scale.shape == (3, 128, 4)
    # Token t's data is bytes [64 t, 64 t + 64), its scales bytes
    # [8192 + 4 t, 8192 + 4 t + 4) of the page.
    assert torch.equal(data[2, 5], pages[2, 5 * 64 : 6 * 64])
    assert torch.equal(scale[2, 5], pages[2, 8192 + 20 : 8192 + 24])


def test_page_views_keep_head_and_page_strides():
    """A K/V pool [pages, 2 heads, 8704] is viewed per side without a copy."""
    pool = torch.zeros(4, 2, PAGE_BYTES, dtype=torch.uint8)
    data, scale = mxfp4_page_views(pool[:, 1:])
    assert data.shape == (4, 1, 128, 64)
    assert data.stride() == (2 * PAGE_BYTES, PAGE_BYTES, 64, 1)
    assert data.data_ptr() == pool[:, 1:].data_ptr()
    assert scale.data_ptr() == pool[:, 1:].data_ptr() + 128 * 64
    with pytest.raises(ValueError):
        mxfp4_page_views(pool[:, :, :-1])


@pytest.mark.parametrize("num_kv_heads", [1, 4])
def test_main_spec_page_holds_k_and_v_pages_per_head(num_kv_heads):
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=num_kv_heads,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.MXFP4,
    )
    spec = MiniMaxM3SparseRocmMxfp4Backend.customize_spec(spec)
    assert spec.num_heads == 2 * num_kv_heads
    assert spec.page_size_bytes == 2 * num_kv_heads * PAGE_BYTES
    # Idempotent: a spec that already carries its packing is left alone.
    assert MiniMaxM3SparseRocmMxfp4Backend.customize_spec(spec) == spec


def test_main_spec_of_other_modes_is_untouched():
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )
    assert MiniMaxM3SparseRocmMxfp4Backend.customize_spec(spec) is spec


def test_index_spec_page_is_one_head_of_68_byte_tokens():
    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128 // 2 + 128 // 32,
        dtype=torch.uint8,
        cache_role=SparseCacheRole.INDEXER,
    )
    assert spec.page_size_bytes == PAGE_BYTES


def test_gfx942_page64_table():
    table = torch.tensor([[7, 2], [0, 11]], dtype=torch.int32)
    expected = torch.tensor([[14, 15, 4, 5], [0, 1, 22, 23]], dtype=torch.int32)
    assert torch.equal(expand_block_table_page64(table), expected)


def test_gfx942_page64_table_into_overwrites_previous_step():
    buf = torch.full((4, 12), 999, dtype=torch.int32)
    expand_block_table_page64_into(torch.arange(12, dtype=torch.int32).view(2, 6), buf)
    second = torch.tensor([[40, 0, 3, 41, 2, 5]], dtype=torch.int32)
    got = expand_block_table_page64_into(second, buf)
    assert got.data_ptr() == buf.data_ptr()
    assert torch.equal(got, expand_block_table_page64(second))
    with pytest.raises(ValueError):
        expand_block_table_page64_into(torch.zeros((5, 6), dtype=torch.int32), buf)


def test_gfx942_in_place_strides_match_dense_pages():
    """The strided operands address exactly the bytes of the dense pages."""
    cache = torch.arange(3 * MANAGER_BLOCK_BYTES, dtype=torch.int64)
    cache = cache.remainder(251).to(torch.uint8).view(3, MANAGER_BLOCK_BYTES)
    k, k_scale, kw = gfx942_mxfp4_cache_strides(cache)
    data, scales = gfx942_mxfp4_cache_views(cache)
    assert data.shape == (6, 1, 4, 64, 16) and scales.shape == (6, 1, 4, 64)
    assert k.data_ptr() == k_scale.data_ptr() == cache.data_ptr()
    flat = cache.flatten()
    for page in range(6):
        d0 = page * kw["kv_page_stride"]
        s0 = page * kw["kv_scale_page_stride"] + kw["kv_scale_offset"]
        assert torch.equal(flat[d0 : d0 + PAGE_DATA_BYTES], data[page].flatten())
        assert torch.equal(flat[s0 : s0 + PAGE_SCALE_BYTES], scales[page].flatten())


requires_gpu_aiter = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("aiter") is None,
    reason="needs a GPU and aiter",
)


@requires_gpu_aiter
def test_gfx942_store_k_round_trips_through_dense_pages():
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    torch.manual_seed(0)
    k = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros((3, MANAGER_BLOCK_BYTES), dtype=torch.uint8, device="cuda")
    slots = torch.tensor([0, 63, 64, 130, 383], device="cuda")
    store_k_gfx942(k, cache, slots)
    data, scales = gfx942_mxfp4_cache_views(cache)
    ref_q, ref_s = dynamic_mxfp4_quant(k)
    ref_s = ref_s.view(torch.uint8).view(5, 4)
    for t, slot in enumerate(slots.tolist()):
        page, off = divmod(slot, 64)
        assert torch.equal(data[page, 0, :, off].flatten(), ref_q[t])
        scale = scales[page, 0].view(4, 16, 4)[:, off % 16, off // 16]
        assert torch.equal(scale, ref_s[t])


@requires_gpu_aiter
@pytest.mark.parametrize("tokens", [1, 33])
def test_gfx942_index_norm_rope_matches_eager(default_vllm_config, tokens):
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.models.minimax_m3.amd.model import MiniMAXGemmaRMSNorm

    torch.manual_seed(0)
    rope = get_rope(
        128,
        max_position=1 << 20,
        rope_parameters={"rope_theta": 1e7, "partial_rotary_factor": 0.5},
    ).cuda()
    qn = MiniMAXGemmaRMSNorm(128).cuda().to(torch.bfloat16)
    kn = MiniMAXGemmaRMSNorm(128).cuda().to(torch.bfloat16)
    with torch.no_grad():
        qn.weight.uniform_(-0.5, 0.5)
        kn.weight.uniform_(-0.5, 0.5)
    q_start = 14 * 128
    qkv = torch.randn(tokens, 16 * 128, device="cuda", dtype=torch.bfloat16) * 4
    pos = torch.randint(0, 1 << 20, (tokens,), device="cuda")

    ref_q, ref_k = rope.forward_native(
        pos,
        qn(qkv[:, q_start : q_start + 128].view(tokens, 1, 128)),
        kn(qkv[:, q_start + 128 : q_start + 256]),
    )
    q, k = index_norm_rope_gfx942(
        qkv,
        pos,
        qn.weight,
        kn.weight,
        rope.cos_sin_cache,
        q_start=q_start,
        num_heads=1,
        head_dim=128,
        rotary_dim=64,
        eps=qn.variance_epsilon,
    )
    assert torch.equal(q, ref_q.reshape(tokens, 128))
    assert torch.equal(k, ref_k.reshape(tokens, 128))


@requires_gpu_aiter
@pytest.mark.parametrize("heads,query_len", [(1, 4), (2, 4), (4, 4), (1, 1)])
def test_gfx942_pack_decode_rows(heads, query_len):
    """Slot h*query_len + r of request b's group is head h of row r: its MXFP4,
    its e8m0 bytes, output row h*rows + row and causal end; dead slots are
    never stored."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.models.minimax_m3.amd.ops.mxfp4_gfx942 import pack_decode_rows_gfx942

    torch.manual_seed(0)
    batch = 3
    rows = batch * query_len
    iq = (torch.randn(rows, heads, 128, device="cuda") * 3).to(torch.bfloat16)
    seq_lens = torch.tensor([7, 300, 4097], dtype=torch.int32, device="cuda")
    q = torch.full((batch + 1, 16, 64), 0xAB, dtype=torch.uint8, device="cuda")
    qs = torch.zeros((batch + 1, 1, 4, 16, 4), dtype=torch.uint8, device="cuda")
    meta = torch.zeros(3 * 16 * (batch + 1), dtype=torch.int32, device="cuda")
    m = pack_decode_rows_gfx942(iq, seq_lens, q, qs, meta, query_len=query_len)
    assert m.shape == (3, 16 * batch)
    for b in range(batch):
        for s in range(16):
            slot = 16 * b + s
            if s >= heads * query_len:
                assert m[0, slot] == -1 and m[2, slot] == 0
                continue
            h, r = divmod(s, query_len)
            row = b * query_len + r
            ref_q, ref_s = dynamic_mxfp4_quant(iq[row, h][None].contiguous())
            assert torch.equal(q[b, s], ref_q[0])
            assert torch.equal(qs[b, 0, :, s, 0], ref_s.view(torch.uint8)[0])
            assert m[0, slot] == h * rows + row
            assert m[1, slot] == 0
            assert m[2, slot] == int(seq_lens[b]) - query_len + r + 1


@requires_gpu_aiter
@pytest.mark.parametrize("heads,query_len", [(1, 4), (2, 4), (1, 1)])
def test_gfx942_pack_decode_rows_fp8(heads, query_len):
    """FP8-Q decode packing: slot h*query_len + r holds e4m3fnuz
    q * 224 / max|q| and the fp32 scale max|q| / 224 (correctly rounded), with
    the same slot metadata as the MXFP4 packing."""
    from vllm.models.minimax_m3.amd.ops.mxfp4_gfx942 import (
        FP8Q_ROW_MAX,
        pack_decode_rows_fp8_gfx942,
        pack_decode_rows_gfx942,
    )

    torch.manual_seed(0)
    batch = 3
    rows = batch * query_len
    iq = (torch.randn(rows, heads, 128, device="cuda") * 3).to(torch.bfloat16)
    iq[0, 0] = 0  # an all-zero row packs to zeros with scale 0
    seq_lens = torch.tensor([7, 300, 4097], dtype=torch.int32, device="cuda")
    q = torch.full((batch + 1, 16, 128), 0xAB, dtype=torch.uint8, device="cuda")
    qs = torch.zeros((batch + 1, 16), dtype=torch.float32, device="cuda")
    meta = torch.zeros(3 * 16 * (batch + 1), dtype=torch.int32, device="cuda")
    m = pack_decode_rows_fp8_gfx942(iq, seq_lens, q, qs, meta, query_len=query_len)
    meta4 = torch.zeros_like(meta)
    m4 = pack_decode_rows_gfx942(
        iq,
        seq_lens,
        torch.zeros((batch, 16, 64), dtype=torch.uint8, device="cuda"),
        torch.zeros((batch, 1, 4, 16, 4), dtype=torch.uint8, device="cuda"),
        meta4,
        query_len=query_len,
    )
    assert torch.equal(m, m4)
    for b in range(batch):
        for s in range(heads * query_len):
            h, r = divmod(s, query_len)
            x = iq[b * query_len + r, h].float()
            amax = x.abs().max()
            if amax == 0:
                assert (q[b, s] == 0).all() and qs[b, s] == 0
                continue
            inv = (FP8Q_ROW_MAX / amax.double()).float()
            ref = (x * inv).to(torch.float8_e4m3fnuz).view(torch.uint8)
            assert torch.equal(q[b, s], ref)
            assert qs[b, s] == (amax.double() / FP8Q_ROW_MAX).float()


@requires_gpu_aiter
@pytest.mark.parametrize("q_fp8", [False, True])
def test_gfx942_decode_scores_match_reference(monkeypatch, q_fp8):
    """The row-packed decode scorer (FP8 Q against the MXFP4 K cache, or the
    MXFP4-Q fallback of an aiter without ``q_fp8``) reproduces an fp32
    reference on the dequantized operands."""
    import types

    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.models.minimax_m3.amd import indexer_mxfp4 as M
    from vllm.models.minimax_m3.amd.ops.mxfp4_gfx942 import FP8Q_ROW_MAX

    if q_fp8 and not M._aiter_rowpack_has_fp8q():
        pytest.skip("aiter row-packed scorer lacks q_fp8")
    if not q_fp8:
        monkeypatch.setattr(M, "_aiter_rowpack_has_fp8q", lambda: False)
    fp4 = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )

    def deq(x):
        p, s = dynamic_mxfp4_quant(x.contiguous())
        p = p.view(torch.uint8)
        v = torch.stack([fp4[(p & 15).long()], fp4[(p >> 4).long()]], -1)
        sc = torch.exp2(s.view(torch.uint8).float() - 127)
        return (v.flatten(-2).view(x.shape[0], -1, 32) * sc[..., None]).flatten(1)

    torch.manual_seed(0)
    batch, ql, heads, ctx, msl = 2, 4, 2, 1000, 2048
    blocks = msl // 128
    bt = torch.randperm(batch * blocks, device="cuda").view(batch, blocks)
    bt = bt.to(torch.int32)
    cache = torch.zeros(
        (batch * blocks, MANAGER_BLOCK_BYTES), dtype=torch.uint8, device="cuda"
    )
    k = torch.randn(batch, ctx, 128, device="cuda").to(torch.bfloat16)
    for b in range(batch):
        slots = bt[b].long()[:, None] * 128 + torch.arange(128, device="cuda")
        store_k_gfx942(k[b], cache, slots.flatten()[:ctx])
    seq_lens = torch.tensor([ctx, 517], dtype=torch.int32, device="cuda")
    iq = torch.randn(batch * ql, heads, 128, device="cuda").to(torch.bfloat16)
    fake = types.SimpleNamespace(
        max_model_len=msl,
        sparse_block_size=128,
        _page64_table=None,
        _max_decode_reqs=batch,
        device=torch.device("cuda"),
        _gfx942_buffers=None,
        _gfx942_index_heads=heads,
        max_decode_query_len=ql,
    )
    d = types.SimpleNamespace(
        seq_lens=seq_lens, block_table=bt, max_seq_len=msl, decode_query_len=ql
    )
    plan = M.MiniMaxM3IndexerMxfp4MetadataBuilder._gfx942_decode_plan(fake, d)
    score = M.mxfp4_gfx942_index_scores_decode(
        iq,
        cache.view(-1, PAGE_BYTES),
        seq_lens,
        plan,
        max_seq_len=msl,
        query_len=ql,
        scale=128**-0.5,
        init_blocks=0,
        local_blocks=0,
    )
    for b in range(batch):
        kd = deq(k[b]).double()
        for r in range(ql):
            kv_len = int(seq_lens[b]) - ql + r + 1
            nb = (kv_len + 127) // 128
            for h in range(heads):
                x = iq[b * ql + r, h].float()
                if q_fp8:
                    amax = x.abs().max()
                    inv = (FP8Q_ROW_MAX / amax.double()).float()
                    qd = (x * inv).to(torch.float8_e4m3fnuz).double() * (
                        (amax.double() / FP8Q_ROW_MAX).float().double()
                    )
                else:
                    qd = deq(x[None].to(torch.bfloat16))[0].double()
                s = (kd[:kv_len] @ qd).float() * 128**-0.5
                s = torch.nn.functional.pad(s, (0, nb * 128 - kv_len), value=-1e30)
                ref = s.view(nb, 128).amax(-1)
                got = score[h, b * ql + r, :nb]
                torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


@requires_gpu_aiter
def test_gfx942_decode_page_max_reduce_and_pin():
    """64-token page maxima fold into 128-token block scores over each row's
    causal blocks only; a half-filled last block ignores its unwritten page."""
    from vllm.models.minimax_m3.amd.ops.mxfp4_gfx942 import (
        decode_page_max_to_block_scores_gfx942,
    )

    torch.manual_seed(0)
    pages = torch.randn(1, 2, 8, device="cuda")
    # Row 1 (kv_len 257) owns pages 0..4; page 5 is unwritten garbage.
    pages[0, 1, 5] = 1e9
    seq_lens = torch.tensor([257], dtype=torch.int32, device="cuda")
    out = torch.full((1, 2, 4), -7.0, device="cuda")
    decode_page_max_to_block_scores_gfx942(
        pages,
        seq_lens,
        out,
        query_len=2,
        init_blocks=1,
        local_blocks=1,
        sentinel_init=1e30,
        sentinel_local=1e29,
    )
    p = pages.cpu()[0]
    # row 0: kv_len 256 -> blocks 0, 1
    assert out[0, 0, 0] == 1e30 and out[0, 0, 1] == 1e29
    assert (out[0, 0, 2:] == -7.0).all()
    # row 1: kv_len 257 -> blocks 0, 1, 2; block 2 is page 4 only
    assert out[0, 1, 0] == 1e30
    assert out[0, 1, 1] == torch.maximum(p[1, 2], p[1, 3]).item()
    assert out[0, 1, 2] == 1e29
    assert out[0, 1, 3] == -7.0


def test_gfx942_requires_scheduled_in_place_rowpack_scorer(monkeypatch):
    """An AITER whose row-packed scorer cannot read the paged cache in place or
    take a caller schedule is rejected at selection, not mid-step."""
    import sys
    import types

    import vllm.models.minimax_m3.amd.indexer_mxfp4 as indexer

    def full(
        q,
        *a,
        kv_page_stride=None,
        kv_scale_page_stride=None,
        kv_scale_offset=0,
        cta_info=None,
        total_ctas=None,
        **k,
    ):
        pass

    def unscheduled(
        q,
        *a,
        kv_page_stride=None,
        kv_scale_page_stride=None,
        kv_scale_offset=0,
        **k,
    ):
        pass

    def dense_only(q, *a, **k):
        pass

    def helper(*a, **k):
        pass

    fake = types.ModuleType("aiter.ops.flydsl")
    kernels = types.ModuleType("aiter.ops.flydsl.kernels.mqa_logits")
    kernels.pa_mqa_logits_fp4_prefill_gfx942 = types.ModuleType("prefill")
    kernels.pa_mqa_logits_fp4_prefill_gfx942.compute_varqlen_windows = helper
    kernels.pa_mqa_logits_fp4_prefill_gfx942.compute_prefill_schedule = helper
    rowpack = types.ModuleType("rowpack")
    rowpack.flydsl_pa_mqa_page_max_fp4_rowpack_prefill_gfx942 = full
    rowpack.rowpack_prefill_inputs_gfx942 = helper
    kernels.pa_mqa_page_max_fp4_rowpack_prefill_gfx942 = rowpack
    msa = types.ModuleType("aiter.ops.msa_attention")
    msa.pa_sparse_block_topk = helper
    for name in ("aiter", "aiter.ops"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["aiter.ops"].flydsl = fake
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", fake)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.kernels", types.ModuleType("k"))
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.kernels.mqa_logits", kernels)
    monkeypatch.setitem(sys.modules, "aiter.ops.msa_attention", msa)
    monkeypatch.setattr(indexer.rocm_aiter_ops, "is_enabled", lambda: True)

    def reason():
        return indexer.rocm_mxfp4_aiter_unavailable_reason("gfx942")

    assert reason() is None
    for scorer in (unscheduled, dense_only):
        rowpack.flydsl_pa_mqa_page_max_fp4_rowpack_prefill_gfx942 = scorer
        assert reason() == indexer.AITER_GFX942_REQUIREMENT
    rowpack.flydsl_pa_mqa_page_max_fp4_rowpack_prefill_gfx942 = full
    del kernels.pa_mqa_logits_fp4_prefill_gfx942.compute_prefill_schedule
    assert reason() == indexer.AITER_GFX942_REQUIREMENT


def test_gfx942_selection_gate_is_strict(monkeypatch):
    import vllm.models.minimax_m3.amd.indexer_mxfp4 as indexer

    monkeypatch.setattr(indexer.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(indexer, "_mxfp4_arch", lambda: "gfx942")
    monkeypatch.setattr(
        indexer, "rocm_mxfp4_aiter_unavailable_reason", lambda arch=None: None
    )
    good = dict(
        topk_blocks=16,
        sparse_block_size=128,
        num_index_heads=1,
        index_head_dim=128,
        score_type="max",
        max_model_len=8192 * 128,
    )
    reason = indexer.rocm_mxfp4_indexer_unsupported_reason
    assert reason(**good) is None
    assert reason(**(good | {"num_index_heads": 2})) is None
    assert "8192" in reason(**(good | {"max_model_len": 8192 * 128 + 1}))
    assert "num_index_heads" in reason(**(good | {"num_index_heads": 3}))


def test_gfx942_index_only_mxfp4_requires_fp8_main_cache(monkeypatch):
    from vllm.models.minimax_m3.amd import model

    monkeypatch.setattr(model, "on_gfx942", lambda: True)
    assert model.use_gfx942_mxfp4_indexer("fp8", "mxfp4")
    assert model.use_gfx942_mxfp4_indexer("fp8_e4m3", "mxfp4")
    assert not model.use_gfx942_mxfp4_indexer("auto", "mxfp4")
    assert not model.use_gfx942_mxfp4_indexer("fp8", "bf16")
    # MXFP4 main K/V on gfx942 needs the MXFP4 indexer (its attend reads the
    # page table the indexer emits), so this pairing is enabled, not rejected.
    assert model.use_gfx942_mxfp4_indexer("mxfp4", "mxfp4")


def test_kv_cache_dtype_name():
    assert CacheConfig(cache_dtype="mxfp4").cache_dtype == "mxfp4"
    assert STR_DTYPE_TO_TORCH_DTYPE["mxfp4"] == torch.uint8
    mode = get_kv_quant_mode("mxfp4")
    assert mode == KVQuantMode.MXFP4 and mode.is_mxfp4
    # A duplicate enum value would make MXFP4 an alias of another mode.
    assert KVQuantMode(mode.value).name == "MXFP4"
    assert not mode.is_nvfp4 and not get_kv_quant_mode("nvfp4").is_mxfp4
    assert get_kv_quant_mode("auto") == KVQuantMode.NONE


def test_mxfp4_kv_selects_the_rocm_attend():
    from vllm.models.minimax_m3.amd.sparse_attention_mxfp4 import (
        MiniMaxM3SparseRocmMxfp4Impl,
    )
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        assert select_main_backend_and_impl_cls(
            topk_blocks=16, kv_cache_dtype="mxfp4", num_kv_heads=1
        ) == (MiniMaxM3SparseRocmMxfp4Backend, MiniMaxM3SparseRocmMxfp4Impl)
    else:
        with pytest.raises(ValueError, match="only on ROCm gfx950"):
            select_main_backend_and_impl_cls(
                topk_blocks=16, kv_cache_dtype="mxfp4", num_kv_heads=1
            )


def test_mxfp4_decode_work_map_is_graph_stable(monkeypatch):
    """A captured decode graph keeps the work map's address and row count it saw
    at capture, so every step must refill one buffer and use one block bound."""
    import sys
    import types

    from vllm.models.minimax_m3.amd import indexer_mxfp4 as mod
    from vllm.models.minimax_m3.amd.indexer_msa import (
        MiniMaxM3IndexerMSAMetadata,
        MiniMaxM3IndexerMSAMetadataBuilder,
    )
    from vllm.models.minimax_m3.common.indexer import MiniMaxM3IndexerDecodeMetadata

    calls = []

    def work_map(seq_lens, max_block, query_len, heads, out=None):
        calls.append((out, max_block))
        return out

    fake = types.ModuleType("aiter.ops.flydsl")
    fake.minimax_m3_index_score_work_map = work_map
    fake.minimax_m3_index_score_work_map_capacity = lambda *a: 64
    for name in ("aiter", "aiter.ops"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", fake)

    def fake_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        n = common_attn_metadata  # the max_seq_len of this step
        t = torch.zeros(1, dtype=torch.int32)
        decode = MiniMaxM3IndexerDecodeMetadata(
            seq_lens=t,
            block_table=t,
            max_seq_len=n,
            decode_query_len=1,
            max_decode_query_len=1,
        )
        return MiniMaxM3IndexerMSAMetadata(
            seq_lens=t,
            max_seq_len=n,
            slot_mapping=t,
            num_actual_tokens=1,
            num_decodes=1,
            num_decode_tokens=1,
            num_prefills=0,
            num_prefill_tokens=0,
            decode=decode,
        )

    monkeypatch.setattr(MiniMaxM3IndexerMSAMetadataBuilder, "build", fake_build)
    builder = object.__new__(mod.MiniMaxM3IndexerMxfp4MetadataBuilder)
    builder.num_index_heads = 1
    builder.max_model_len = 4096
    builder._max_decode_reqs = 32
    builder.max_decode_query_len = 1
    builder.sparse_block_size = 128
    builder.device = torch.device("cpu")
    builder._work_map_buf = None
    builder.is_gfx942 = False

    first = builder.build(0, 4096)  # the capture step: max_seq_len == max_model_len
    second = builder.build(0, 300)  # a later step with a short real context
    assert len(calls) == 2
    assert calls[0][0] is calls[1][0] is builder._work_map_buf
    assert calls[0][1] == calls[1][1] == 4096 // 128
    assert first.decode.max_seq_len == second.decode.max_seq_len == 4096


def _decode_scores(M, iq, cache, seq_lens, plan, msl, query_len):
    return M.mxfp4_gfx942_index_scores_decode(
        iq,
        cache.view(-1, PAGE_BYTES),
        seq_lens,
        plan,
        max_seq_len=msl,
        query_len=query_len,
        scale=128**-0.5,
        init_blocks=1,
        local_blocks=2,
    ).clone()


@requires_gpu_aiter
@pytest.mark.parametrize("heads,query_len", [(2, 4), (1, 1), (2, 3)])
def test_gfx942_fused_decode_matches_unfused(monkeypatch, heads, query_len):
    """The one-launch decode (block scores from the scorer's epilogue, Q pack
    in its prologue) and its separate-pack variant (a Q the prologue cannot
    read) reproduce the 3-launch FP8-Q chain of an aiter without the
    block-score scorer bit for bit on every causal block, init/local pins
    included."""
    import types

    from vllm.models.minimax_m3.amd import indexer_mxfp4 as M

    if not (M._aiter_rowpack_has_fp8q() and M._aiter_rowpack_has_block_out()):
        pytest.skip("aiter row-packed scorer lacks the fused decode variant")
    torch.manual_seed(0)
    batch, msl = 3, 4096
    blocks = msl // 128
    bt = torch.randperm(batch * blocks, device="cuda").view(batch, blocks)
    bt = bt.to(torch.int32)
    cache = torch.zeros(
        (batch * blocks, MANAGER_BLOCK_BYTES), dtype=torch.uint8, device="cuda"
    )
    for b in range(batch):
        k = torch.randn(msl, 128, device="cuda").to(torch.bfloat16)
        slots = bt[b].long()[:, None] * 128 + torch.arange(128, device="cuda")
        store_k_gfx942(k, cache, slots.flatten())
    seq_lens = torch.tensor([4000, 517, 129], dtype=torch.int32, device="cuda")
    iq = torch.randn(batch * query_len, heads, 128, device="cuda").to(torch.bfloat16)
    # Odd head stride: the scorer's prologue cannot read it, so Q is packed by
    # a separate launch.
    iq_odd = torch.zeros(batch * query_len, heads, 129, device="cuda")
    iq_odd = iq_odd.to(torch.bfloat16)[..., :128]
    iq_odd.copy_(iq)
    fake = types.SimpleNamespace(
        max_model_len=msl,
        sparse_block_size=128,
        _page64_table=None,
        _max_decode_reqs=batch,
        device=torch.device("cuda"),
        _gfx942_buffers=None,
        _gfx942_index_heads=heads,
        max_decode_query_len=query_len,
    )
    d = types.SimpleNamespace(
        seq_lens=seq_lens, block_table=bt, max_seq_len=msl, decode_query_len=query_len
    )
    plan = M.MiniMaxM3IndexerMxfp4MetadataBuilder._gfx942_decode_plan(fake, d)
    out = {}
    for mode, q in ((0, iq), (1, iq_odd), (2, iq)):
        with monkeypatch.context() as mp:
            if mode == 0:
                mp.setattr(M, "_aiter_rowpack_has_block_out", lambda: False)
            out[mode] = _decode_scores(M, q, cache, seq_lens, plan, msl, query_len)
    for b in range(batch):
        for r in range(query_len):
            nb = (int(seq_lens[b]) - query_len + r + 1 + 127) // 128
            row = b * query_len + r
            for fuse in (1, 2):
                assert torch.equal(
                    out[fuse][:, row, :nb].view(torch.int32),
                    out[0][:, row, :nb].view(torch.int32),
                ), (fuse, b, r)


@requires_gpu_aiter
def test_gfx942_rowpack_call_cache_does_not_grow_or_pin_tensors(monkeypatch):
    """The resolved-launch cache is keyed on the specialization, not on data
    pointers: decode steps with fresh activations / outputs at the same shapes
    reuse one entry and leave no tensor alive (keying on data_ptr and holding
    the bound args leaked device memory per step until launches failed with
    hipErrorLaunchOutOfResources mid-serve)."""
    import gc
    import types
    import weakref

    from vllm.models.minimax_m3.amd import indexer_mxfp4 as M

    if not (M._aiter_rowpack_has_fp8q() and M._aiter_rowpack_has_block_out()):
        pytest.skip("aiter row-packed scorer lacks the fused decode variant")
    torch.manual_seed(0)
    batch, ql, heads, msl = 2, 4, 2, 2048
    blocks = msl // 128
    bt = torch.arange(batch * blocks, dtype=torch.int32, device="cuda").view(batch, -1)
    cache = torch.zeros(
        (batch * blocks, MANAGER_BLOCK_BYTES), dtype=torch.uint8, device="cuda"
    )
    seq_lens = torch.tensor([2000, 700], dtype=torch.int32, device="cuda")
    fake = types.SimpleNamespace(
        max_model_len=msl,
        sparse_block_size=128,
        _page64_table=None,
        _max_decode_reqs=batch,
        device=torch.device("cuda"),
        _gfx942_buffers=None,
        _gfx942_index_heads=heads,
        max_decode_query_len=ql,
    )
    d = types.SimpleNamespace(
        seq_lens=seq_lens, block_table=bt, max_seq_len=msl, decode_query_len=ql
    )
    plan = M.MiniMaxM3IndexerMxfp4MetadataBuilder._gfx942_decode_plan(fake, d)
    for fuse in (0, 2):
        if fuse == 0:
            # The 3-launch chain of an aiter without the block-score scorer.
            monkeypatch.setattr(M, "_aiter_rowpack_has_block_out", lambda: False)
        else:
            monkeypatch.undo()
        refs = []
        for step in range(6):
            iq = torch.randn(batch * ql, heads, 128, device="cuda").to(torch.bfloat16)
            score = M.mxfp4_gfx942_index_scores_decode(
                iq,
                cache.view(-1, PAGE_BYTES),
                seq_lens,
                plan,
                max_seq_len=msl,
                query_len=ql,
                scale=128**-0.5,
                init_blocks=0,
                local_blocks=1,
            )
            refs += [weakref.ref(iq), weakref.ref(score)]
            if step == 1:
                entries = len(M._ROWPACK_CALLS)
            del iq, score
        torch.cuda.synchronize()
        gc.collect()
        assert len(M._ROWPACK_CALLS) == entries, fuse
        assert all(r() is None for r in refs), fuse


def _random_decode_topk_case(gen, heads, kv_heads, query_len, batch, msl):
    """Score rows with ties, NaN, +-0, -inf, AITER's init/local sentinels and
    garbage past each row's causal end; ragged and zero-length requests."""
    from vllm.models.minimax_m3.amd.indexer_msa import _score_width

    max_blk = -(-msl // 128)
    lens = torch.randint(0, msl + 1, (batch,), generator=gen)
    lens[0] = msl
    if batch > 1:
        lens[1] = int(torch.randint(0, 16 * 128, (1,), generator=gen))  # < 16 blocks
    if batch > 2:
        lens[2] = 0 if query_len == 1 else query_len - 1  # causal_len 0 rows
    rows = batch * query_len
    width = _score_width(msl, 128)
    if bool(torch.randint(0, 2, (1,), generator=gen)):
        score = torch.randint(-3, 4, (heads, rows, width), generator=gen).float()
    else:
        score = torch.randn(heads, rows, width, generator=gen)
    flat = score.view(-1)
    idx = torch.randint(0, flat.numel(), (64,), generator=gen)
    specials = torch.tensor(
        [float("nan"), -0.0, 0.0, float("-inf"), float("inf"), 1e30, 1e29]
    )
    flat[idx] = specials[torch.randint(0, len(specials), (64,), generator=gen)]
    for r in range(rows):
        b, t = divmod(r, query_len)
        nb = -(-max(int(lens[b]) - query_len + t + 1, 0) // 128)
        if nb:
            score[:, r, 0] = 1e30  # init block
            score[:, r, nb - 1] = 1e29  # local block
    bt = torch.randint(0, 1 << 20, (batch, max_blk + 3), generator=gen)
    return (
        score.cuda(),
        bt.to(torch.int32).cuda(),
        lens.to(torch.int32).cuda(),
        rows,
    )


@requires_gpu_aiter
@pytest.mark.parametrize(
    "heads,kv_heads,query_len,batch,msl",
    [
        (1, 1, 1, 1, 32768),
        (1, 1, 4, 3, 16384),
        (2, 2, 1, 5, 524288),
        (2, 2, 4, 4, 131072),
        (1, 2, 2, 7, 2048),
        (2, 4, 3, 2, 65536),
        (1, 1, 1, 32, 524288),
        (2, 2, 4, 16, 32768),
    ],
)
def test_gfx942_triton_decode_topk_matches_aiter(
    heads, kv_heads, query_len, batch, msl
):
    """The Triton decode selector selects exactly what AITER's
    pa_sparse_block_topk selects -- same blocks in the same rank order (ties
    to the larger block, NaN as -inf, -0 below +0), -1 padding on short rows --
    and emits an identical sparse block table and context lengths, at every
    launch geometry the selector can take."""
    from aiter.ops.msa_attention import pa_sparse_block_topk

    from vllm.models.minimax_m3.amd.ops.mxfp4_decode_topk import mxfp4_decode_topk

    gen = torch.Generator().manual_seed(heads * 1000 + query_len * 100 + batch + msl)
    for case in range(4):
        score, bt, lens, rows = _random_decode_topk_case(
            gen, heads, kv_heads, query_len, batch, msl
        )
        args = dict(
            max_seq_len=msl,
            block_size=128,
            query_len=query_len,
            num_kv_heads=kv_heads,
            pages_per_block=8,
        )

        def bufs(rows=rows):
            return (
                torch.full((heads, rows, 16), -7, dtype=torch.int32, device="cuda"),
                torch.full(
                    (rows * kv_heads, 128), -7, dtype=torch.int32, device="cuda"
                ),
                torch.full((rows * kv_heads,), -7, dtype=torch.int32, device="cuda"),
            )

        ref = bufs()
        pa_sparse_block_topk(
            score, ref[0], bt, lens, sparse_bt=ref[1], sparse_ctx=ref[2], **args
        )
        counter = torch.zeros(heads, rows, dtype=torch.int32, device="cuda")
        # default geometry, a forced multi-chunk split and a single chunk
        for kw in ({}, {"block_k": 128, "max_chunks": 4}, {"max_chunks": 1}):
            got = bufs()
            mxfp4_decode_topk(
                score,
                got[0],
                bt,
                lens,
                sparse_bt=got[1],
                sparse_ctx=got[2],
                counter=counter,
                **args,
                **kw,
            )
            for name, a, b in zip(("topk_idx", "sparse_bt", "sparse_ctx"), ref, got):
                assert torch.equal(a, b), (case, kw, name)
            assert not counter.any(), (case, kw)


def test_gfx942_decode_topk_backend_auto():
    """The decode selector is picked from launch-time shapes only."""
    from vllm.models.minimax_m3.amd import indexer_mxfp4 as M

    assert M.gfx942_decode_topk_backend(1, 1, 524288, 128) == "triton"
    assert M.gfx942_decode_topk_backend(32, 2, 524288, 128) == "triton"
    assert M.gfx942_decode_topk_backend(128, 1, 524288, 128) == "aiter"
