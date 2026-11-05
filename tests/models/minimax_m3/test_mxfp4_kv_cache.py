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
    decode_block_scores_gfx942,
    expand_block_table_page64,
    expand_block_table_page64_into,
    gfx942_mxfp4_cache_strides,
    gfx942_mxfp4_cache_views,
    quantize_q_gfx942,
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
def test_gfx942_q_quant_matches_dynamic_mxfp4_quant():
    """Slot 0 of the tile is the row's MXFP4, scales in the preshuffled order."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    torch.manual_seed(0)
    iq = (torch.randn(9, 2, 128, device="cuda") * 3).to(torch.bfloat16)
    iq[0, 0, :32] = 0
    for head in range(2):
        q, s, w = quantize_q_gfx942(iq, head)
        ref_q, ref_s = dynamic_mxfp4_quant(iq[:, head].contiguous())
        ref_s = ref_s.view(torch.uint8).view(9, 4)
        assert torch.equal(q[:, 0], ref_q) and not q[:, 1:].any()
        assert torch.equal(s[:, 0, :, 0, 0], ref_s)
        s[:, 0, :, 0, 0] = 0
        assert not s.any()
        assert torch.equal(w[:, 0], torch.ones_like(w[:, 0])) and not w[:, 1:].any()


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
def test_gfx942_decode_block_scores_reduce_and_pin():
    """Two MTP rows of one request see 256 and 257 tokens: the local block is
    block 1 for the first and block 2 for the second."""
    torch.manual_seed(0)
    logits = torch.randn(2, 400, device="cuda")
    seq_lens = torch.tensor([257], dtype=torch.int32, device="cuda")
    out = torch.empty((2, 4), device="cuda")
    decode_block_scores_gfx942(
        logits,
        seq_lens,
        out,
        query_len=2,
        init_blocks=1,
        local_blocks=1,
        sentinel_init=1e30,
        sentinel_local=1e29,
    )
    padded = torch.nn.functional.pad(logits.cpu(), (0, 112), value=float("-inf"))
    expect = padded.view(2, 4, 128).amax(-1)
    expect[:, 0] = 1e30
    expect[0, 1] = 1e29
    expect[1, 2] = 1e29
    assert torch.equal(out.cpu(), expect)


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


def test_gfx942_requires_signed_in_place_scorers(monkeypatch):
    """A ReLU-only or dense-only AITER is rejected at selection, not mid-step."""
    import sys
    import types

    import vllm.models.minimax_m3.amd.indexer_mxfp4 as indexer

    def full(
        q,
        *a,
        relu=True,
        kv_page_stride=None,
        kv_scale_page_stride=None,
        kv_scale_offset=0,
        **k,
    ):
        pass

    def relu_only(
        q,
        *a,
        kv_page_stride=None,
        kv_scale_page_stride=None,
        kv_scale_offset=0,
        **k,
    ):
        pass

    def dense_only(q, *a, relu=True, **k):
        pass

    def helper(*a, **k):
        pass

    fake = types.ModuleType("aiter.ops.flydsl")
    fake.recommended_gfx942_fp4_longctx_config = helper
    for name in indexer.GFX942_SCORERS:
        setattr(fake, name, full)
    kernels = types.ModuleType("aiter.ops.flydsl.kernels.mqa_logits")
    kernels.pa_mqa_logits_fp4_prefill_gfx942 = types.ModuleType("prefill")
    kernels.pa_mqa_logits_fp4_prefill_gfx942.compute_varqlen_windows = helper
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
    for scorer in (relu_only, dense_only):
        monkeypatch.setattr(fake, indexer.GFX942_SCORERS[1], scorer)
        assert reason() == indexer.AITER_GFX942_REQUIREMENT
    monkeypatch.setattr(fake, indexer.GFX942_SCORERS[1], full)
    rowpack.flydsl_pa_mqa_page_max_fp4_rowpack_prefill_gfx942 = dense_only
    assert reason() == indexer.AITER_GFX942_REQUIREMENT
    del kernels.pa_mqa_page_max_fp4_rowpack_prefill_gfx942
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
    assert not model.use_gfx942_mxfp4_indexer("mxfp4", "mxfp4")


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
