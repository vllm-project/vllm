# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_K31_BEAT_CTXKV's one-launch context K/V store (kimi_k3/amd/ops/
ctx_kv_store, through specialized_decode/levers/ctxkv) against the stock
DFlashQwen3Model.precompute_and_store_context_kv, at the release Kimi-K3.1
DSpark draft's TP8 shapes (5 layers, 2 KV heads, head dim 64).

Both run the stock K/V projection; the stock path's copy, grouped K RMSNorm,
RoPE and TRITON_ATTN cache writes (TritonAttentionImpl.do_kv_cache_update) are
the reference. V must match exactly; K may differ in its last bf16 bit where the
variance's summation order (or an FMA) rounds the other way: at most 1 in 100
K elements, by at most one bf16 ulp. Untouched cache rows (negative slots) must
stay as they were."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="the launch is for ROCm"
)

L, NKV, HD, HIDDEN, BS, NBLK, MAX_POS = 5, 2, 64, 512, 16, 64, 4096


class _Fake:
    pass


def _model(device, neox: bool, cs_dtype: torch.dtype, seed: int):
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    for name in (
        "_project_context_kv_token_major",
        "_project_context_kv",
        "_normalize_context_states",
        "_normalize_context_k",
    ):
        setattr(_Fake, name, getattr(DFlashQwen3Model, name))
    g = torch.Generator(device=device).manual_seed(seed)
    m = _Fake()
    m._hidden_norm_weight = (
        1 + 0.1 * torch.randn(HIDDEN, device=device, generator=g)
    ).bfloat16()
    m._fused_kv_weight = (
        torch.randn(L * 2 * NKV * HD, HIDDEN, device=device, generator=g) / 20
    ).bfloat16()
    m._fused_kv_bias = None
    m._context_kv_proj = None
    m._k_norm_weights = (
        1 + 0.2 * torch.randn(L, HD, device=device, generator=g)
    ).bfloat16()
    inv = 1.0 / (1e6 ** (torch.arange(0, HD, 2, device=device).float() / HD))
    freqs = torch.arange(MAX_POS, device=device).float()[:, None] * inv[None, :]
    m._rope_cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], -1).to(cs_dtype)
    m._rope_head_size, m._rope_is_neox = HD, neox
    m._num_attn_layers, m._kv_size, m._head_dim = L, NKV * HD, HD
    m._num_kv_heads, m._rms_norm_eps = NKV, 1e-6
    m.layers = [SimpleNamespace(self_attn=SimpleNamespace()) for _ in range(L)]
    # One allocation for every layer's [blocks, kv heads, block size, 2 * hd]
    # cache, as the KV cache manager's.
    pool = torch.randn(L, NBLK, NKV, BS, 2 * HD, device=device, generator=g)
    pool = pool.bfloat16()
    m._attn_layers = []
    for i in range(L):
        impl = object.__new__(TritonAttentionImpl)
        impl.attn_type = AttentionType.DECODER
        impl.kv_cache_dtype = "auto"
        impl._is_per_token_head_quant = False
        impl.head_size = HD
        impl._set_kv_ptr_range = True
        one = torch.ones((), device=device)
        m._attn_layers.append(
            SimpleNamespace(impl=impl, kv_cache=pool[i], _k_scale=one, _v_scale=one)
        )
    return m, pool


def _stock():
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    fn = DFlashQwen3Model.precompute_and_store_context_kv
    return getattr(fn, "__wrapped__", fn)


@pytest.mark.parametrize("num_ctx", [1, 8, 37])
@pytest.mark.parametrize("neox", [True, False])
@pytest.mark.parametrize("cs_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("per_layer", [False, True])
def test_k31_ctx_kv_store(monkeypatch, num_ctx, neox, cs_dtype, per_layer):
    from vllm.models.kimi_k3.amd.specialized_decode import levers
    from vllm.models.kimi_k3.amd.specialized_decode.levers import ctxkv

    monkeypatch.setenv("VLLM_K31_BEAT_CTXKV", "1")
    monkeypatch.setattr(levers, "_active", True)
    device = torch.device("cuda")
    ref, ref_pool = _model(device, neox, cs_dtype, seed=1)
    new, new_pool = _model(device, neox, cs_dtype, seed=1)
    before = new_pool.clone()
    g = torch.Generator(device=device).manual_seed(2)
    states = torch.randn(num_ctx, HIDDEN, device=device, generator=g).bfloat16()
    pos = torch.randint(0, MAX_POS, (num_ctx,), device=device, generator=g)

    def slots():
        s = torch.randperm(NBLK * BS, device=device, generator=g)[:num_ctx]
        s[::3] = -1  # padding rows
        return s

    slot = [slots() for _ in range(L)] if per_layer else slots()

    _stock()(ref, states, pos, slot)
    ctxkv._wrap_precompute(_stock())(new, states, pos, slot)
    assert getattr(new, ctxkv._REASON) is None

    v_ref, v_new = ref_pool[..., HD:], new_pool[..., HD:]
    torch.testing.assert_close(v_new, v_ref, rtol=0, atol=0)
    k_ref, k_new = ref_pool[..., :HD].float(), new_pool[..., :HD].float()
    diff = (k_new != k_ref).sum().item()
    written = (before[..., :HD] != k_ref).any(-1).sum().item() * HD
    assert diff <= max(1, written // 100), (diff, written)
    ulp = k_ref.abs().clamp_min(1e-30) * 2.0**-7
    assert ((k_new - k_ref).abs() <= ulp).all()
    # Negative slots wrote nothing (rows the reference left alone stay as they
    # were in both).
    untouched = (ref_pool == before).all(-1)
    assert (new_pool == before).all(-1)[untouched].all()


def test_k31_ctx_kv_store_why_not():
    from vllm.models.kimi_k3.amd.ops import ctx_kv_store
    from vllm.models.kimi_k3.amd.specialized_decode.levers import ctxkv

    device = torch.device("cuda")
    m, _ = _model(device, True, torch.float32, seed=0)
    assert ctxkv.model_why_not(m) is None
    caches = [a.kv_cache for a in m._attn_layers]
    assert ctx_kv_store.layout_why_not(caches, NKV, HD) is None
    assert "dtype" in ctx_kv_store.layout_why_not(
        [c.view(torch.int16) for c in caches], NKV, HD
    )
    assert "layers" in ctx_kv_store.layout_why_not(caches * 2, NKV, HD)
    m._attn_layers[2].impl.kv_cache_dtype = "fp8"
    assert "fp8" in ctxkv.model_why_not(m)
    m._attn_layers[2].impl.kv_cache_dtype = "auto"
    m.layers[0].self_attn.v_scale = 1.0
    assert ctxkv.model_why_not(m) == "V scale"
