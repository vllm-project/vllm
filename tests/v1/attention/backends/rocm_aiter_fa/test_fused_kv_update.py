# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_FA's fused KV-cache update hooks against a float32 reference.

``do_rope_and_kv_cache_update`` (RoPE + cache write) and
``do_qk_norm_rope_kvcache_update`` (per-head RMSNorm + RoPE + cache write)
replace the separate ops the compile fusion passes match. Each must produce
the rotated q/k and the cache bytes the unfused path would.
"""

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_fa.fa_harness import (
    DEVICE,
    FaCase,
)
from tests.v1.attention.backends.rocm_aiter_fa.test_kv_cache import _read_back
from tests.v1.attention.model_shapes import MUSE_SPARK_FULL, QWEN3_8_27B, QWEN3_235B
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import is_quantized_kv_cache

NUM_TOKENS = 37
EPS = 1e-6


def _rope_ref(x, positions, cos_sin_cache, is_neox):
    """x: [T, H, D] float32; rotates the first rotary_dim channels."""
    rot = cos_sin_cache.shape[-1]
    cos, sin = cos_sin_cache[positions].float().chunk(2, dim=-1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    xr, xp = x[..., :rot], x[..., rot:]
    if is_neox:
        x1, x2 = xr.chunk(2, dim=-1)
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    else:
        x1, x2 = xr[..., 0::2], xr[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        out = out.flatten(-2)
    return torch.cat([out, xp], dim=-1)


def _rms_ref(x, weight):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight.float()


def _cache_value(x, kv_cache_dtype, scale):
    if not is_quantized_kv_cache(kv_cache_dtype):
        return x
    fp8 = current_platform.fp8_dtype()
    return (x / scale).to(fp8).float() * scale


def _setup(shape, kv_cache_dtype, monkeypatch):
    monkeypatch.setattr(rocm_aiter_ops, "_SHUFFLE_KV_CACHE_ENABLED", False)
    monkeypatch.setattr(rocm_aiter_ops, "_AITER_ENABLED", True)
    case = FaCase(shape, kv_cache_dtype=kv_cache_dtype).setup(16)
    slots = torch.randperm(case.num_blocks * case.block_size, device=DEVICE)
    return case, slots[:NUM_TOKENS].contiguous()


@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize(
    "shape", [MUSE_SPARK_FULL, QWEN3_235B, QWEN3_8_27B], ids=lambda s: s.id
)
def test_rope_and_kv_cache_update(shape, kv_cache_dtype, is_neox, monkeypatch):
    case, slots = _setup(shape, kv_cache_dtype, monkeypatch)
    impl = case.layer.impl
    assert impl.fused_rope_kvcache_supported()
    h, kv, d = shape.num_heads, shape.num_kv_heads, shape.head_size
    with set_current_vllm_config(case.config):
        rope = get_rope(d, 4096, is_neox_style=is_neox, dtype=torch.float32)
    rope = rope.to(DEVICE)
    positions = torch.randint(0, 4096, (NUM_TOKENS,), device=DEVICE)
    q = torch.randn(NUM_TOKENS, h, d, dtype=case.dtype, device=DEVICE)
    k = torch.randn(NUM_TOKENS, kv, d, dtype=case.dtype, device=DEVICE)
    v = torch.randn_like(k)
    q_ref = _rope_ref(q.float(), positions, rope.cos_sin_cache, is_neox)
    k_ref = _rope_ref(k.float(), positions, rope.cos_sin_cache, is_neox)
    v_ref = v.float()

    impl.do_rope_and_kv_cache_update(
        case.layer,
        q,
        k,
        v,
        positions,
        rope.cos_sin_cache,
        is_neox,
        case.kv_cache,
        slots,
    )
    torch.testing.assert_close(q.float(), q_ref, atol=2e-2, rtol=2e-2)
    k_back, v_back = _read_back(case, slots)
    tol = (
        {"atol": 1.5e-1, "rtol": 1.5e-1}
        if is_quantized_kv_cache(kv_cache_dtype)
        else {"atol": 2e-2, "rtol": 2e-2}
    )
    torch.testing.assert_close(
        k_back.float(),
        _cache_value(k_ref, kv_cache_dtype, case.layer._k_scale_float),
        **tol,
    )
    torch.testing.assert_close(
        v_back.float(),
        _cache_value(v_ref, kv_cache_dtype, case.layer._v_scale_float),
        **tol,
    )


@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize(
    "shape", [MUSE_SPARK_FULL, QWEN3_235B, QWEN3_8_27B], ids=lambda s: s.id
)
def test_qk_norm_rope_kv_cache_update(shape, kv_cache_dtype, is_neox, monkeypatch):
    case, slots = _setup(shape, kv_cache_dtype, monkeypatch)
    impl = case.layer.impl
    assert impl.fused_qk_norm_rope_kvcache_supported()
    h, kv, d = shape.num_heads, shape.num_kv_heads, shape.head_size
    with set_current_vllm_config(case.config):
        # The fused kernel reads the cos/sin cache in the activation dtype,
        # as the compile fusion pass passes it.
        rope = get_rope(d, 4096, is_neox_style=is_neox, dtype=case.dtype)
    rope = rope.to(DEVICE)
    positions = torch.randint(0, 4096, (NUM_TOKENS,), device=DEVICE)
    qkv = torch.randn(NUM_TOKENS, (h + 2 * kv) * d, dtype=case.dtype, device=DEVICE)
    q_weight = torch.rand(d, dtype=case.dtype, device=DEVICE) + 0.5
    k_weight = torch.rand(d, dtype=case.dtype, device=DEVICE) + 0.5
    q, k, v = qkv.float().split([h * d, kv * d, kv * d], dim=-1)
    q_ref = _rope_ref(
        _rms_ref(q.view(-1, h, d), q_weight), positions, rope.cos_sin_cache, is_neox
    )
    k_ref = _rope_ref(
        _rms_ref(k.view(-1, kv, d), k_weight), positions, rope.cos_sin_cache, is_neox
    )
    v_ref = v.view(-1, kv, d)

    q_out = torch.empty(NUM_TOKENS, h, d, dtype=case.dtype, device=DEVICE)
    k_out = torch.empty(NUM_TOKENS, kv, d, dtype=case.dtype, device=DEVICE)
    impl.do_qk_norm_rope_kvcache_update(
        case.layer,
        qkv,
        q_out,
        k_out,
        positions,
        q_weight,
        k_weight,
        EPS,
        rope.cos_sin_cache,
        is_neox,
        case.kv_cache,
        slots,
    )
    torch.testing.assert_close(q_out.float(), q_ref, atol=3e-2, rtol=3e-2)
    quantized = is_quantized_kv_cache(kv_cache_dtype)
    k_tol = (
        {"atol": 2e-1, "rtol": 1.5e-1} if quantized else {"atol": 3e-2, "rtol": 3e-2}
    )
    torch.testing.assert_close(k_out.float(), k_ref, **k_tol)
    k_back, v_back = _read_back(case, slots)
    torch.testing.assert_close(
        k_back.float(),
        _cache_value(k_ref, kv_cache_dtype, case.layer._k_scale_float),
        **k_tol,
    )
    torch.testing.assert_close(
        v_back.float(),
        _cache_value(v_ref, kv_cache_dtype, case.layer._v_scale_float),
        **k_tol,
    )


def test_fused_rope_is_disabled_under_shuffle(monkeypatch):
    """Shuffle writes go through the backend's own update path."""
    case, _ = _setup(QWEN3_235B, "auto", monkeypatch)
    monkeypatch.setattr(rocm_aiter_ops, "_SHUFFLE_KV_CACHE_ENABLED", True)
    assert not case.layer.impl.fused_rope_kvcache_supported()
    assert case.layer.impl.fused_qk_norm_rope_kvcache_supported()
