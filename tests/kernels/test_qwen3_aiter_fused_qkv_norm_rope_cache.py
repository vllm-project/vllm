# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_qkv_norm_rope_cache as fused_mod
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform

NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 128
ROTARY_DIM = 64
RMS_NORM_EPS = 1e-6
MAX_POSITION_EMBEDDINGS = 4096
ROPE_THETA = 5_000_000.0


def _make_fake_context(kv_cache: torch.Tensor, slot_mapping: torch.Tensor):
    def _split_kv_cache(kv_cache: torch.Tensor):
        return kv_cache.unbind(dim=1)

    attn_layer = SimpleNamespace(
        impl=SimpleNamespace(_split_kv_cache=_split_kv_cache),
        kv_cache_dtype="auto",
        kv_cache=kv_cache,
        _k_scale=torch.tensor(1.0, device=kv_cache.device),
        _v_scale=torch.tensor(1.0, device=kv_cache.device),
    )
    return None, attn_layer, kv_cache, slot_mapping


def _ref_gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + RMS_NORM_EPS)
    return (x * (1.0 + weight.float())).to(orig_dtype)


def _ref_neox_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    half = ROTARY_DIM // 2
    cos = cos_sin_cache[positions, :half].float()[:, None, :]
    sin = cos_sin_cache[positions, half:ROTARY_DIM].float()[:, None, :]

    x_rot = x[..., :ROTARY_DIM].float()
    x_pass = x[..., ROTARY_DIM:]
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return torch.cat((rotated.to(x.dtype), x_pass), dim=-1)


def _ref_qwen3_qkv_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
):
    num_tokens = qkv.shape[0]
    q_size = NUM_Q_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    q_gate, k, v = qkv.split([q_size * 2, kv_size, kv_size], dim=-1)

    q_gate = q_gate.view(num_tokens, NUM_Q_HEADS, 2 * HEAD_DIM)
    q = q_gate[..., :HEAD_DIM]
    gate = q_gate[..., HEAD_DIM:].reshape(num_tokens, q_size)
    k = k.view(num_tokens, NUM_KV_HEADS, HEAD_DIM)
    v = v.view(num_tokens, NUM_KV_HEADS, HEAD_DIM)

    q = _ref_neox_rope(
        _ref_gemma_rmsnorm(q, q_weight), positions, cos_sin_cache
    ).reshape(num_tokens, q_size)
    k = _ref_neox_rope(
        _ref_gemma_rmsnorm(k, k_weight), positions, cos_sin_cache
    ).reshape(num_tokens, kv_size)
    return q, k, v.reshape(num_tokens, kv_size), gate


def _install_fake_aiter_kernel(monkeypatch: pytest.MonkeyPatch, fake_kernel):
    module_names = [
        "aiter",
        "aiter.ops",
        "aiter.ops.triton",
        "aiter.ops.triton.rope",
        "aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cache",
    ]
    for name in module_names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules[
        "aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cache"
    ].fused_qkv_split_qk_norm_rope_cache = fake_kernel


def test_qwen3_aiter_wrapper_passes_raw_gemma_weights(monkeypatch):
    num_tokens = 3
    q_size = NUM_Q_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    qkv = torch.randn(num_tokens, q_size * 2 + kv_size * 2, dtype=torch.bfloat16)
    q_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16) * 0.1
    k_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16) * 0.1
    kv_cache = torch.empty(1, 2, num_tokens, NUM_KV_HEADS, HEAD_DIM)
    slot_mapping = torch.arange(num_tokens, dtype=torch.long)
    captured = {}

    def fake_kernel(qkv, q_weight_arg, k_weight_arg, *args, **kwargs):
        del args, kwargs
        captured["q_weight"] = q_weight_arg
        captured["k_weight"] = k_weight_arg
        return (
            torch.empty(num_tokens, NUM_Q_HEADS, HEAD_DIM, dtype=qkv.dtype),
            torch.empty(num_tokens, NUM_Q_HEADS, HEAD_DIM, dtype=qkv.dtype),
            torch.empty(num_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=qkv.dtype),
            torch.empty(num_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=qkv.dtype),
        )

    _install_fake_aiter_kernel(monkeypatch, fake_kernel)
    monkeypatch.setattr(
        fused_mod,
        "get_attention_context",
        lambda layer_name: _make_fake_context(kv_cache, slot_mapping),
    )

    fused_mod.qwen3_aiter_fused_qkv_norm_rope_cache_impl(
        qkv,
        torch.arange(num_tokens, dtype=torch.long),
        q_weight,
        k_weight,
        torch.empty(MAX_POSITION_EMBEDDINGS, ROTARY_DIM),
        "layers.0.self_attn.attn",
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        ROTARY_DIM,
        RMS_NORM_EPS,
        True,
    )

    assert captured["q_weight"] is q_weight
    assert captured["k_weight"] is k_weight


def _aiter_fused_qkv_op_available() -> bool:
    fused_mod.is_aiter_fused_qkv_norm_rope_cache_available.cache_clear()
    return fused_mod.is_aiter_fused_qkv_norm_rope_cache_available()


@pytest.mark.skipif(
    not current_platform.is_rocm() or not _aiter_fused_qkv_op_available(),
    reason="Qwen3 AITER fused QKV/RoPE/cache test requires ROCm and AITER",
)
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
@pytest.mark.parametrize("block_size", [16, 64])
@torch.inference_mode()
def test_qwen3_aiter_fused_qkv_norm_rope_cache_matches_reference(
    monkeypatch,
    default_vllm_config,
    num_tokens: int,
    block_size: int,
):
    del default_vllm_config
    assert rocm_aiter_ops.is_enabled()

    device = torch.device("cuda", torch.accelerator.current_device_index())
    dtype = torch.bfloat16
    torch.manual_seed(0)

    rope = RotaryEmbedding(
        head_size=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        base=ROPE_THETA,
        is_neox_style=True,
        dtype=dtype,
    ).to(device)

    q_size = NUM_Q_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    qkv = torch.randn(
        num_tokens, q_size * 2 + kv_size * 2, dtype=dtype, device=device
    )
    qkv_ref = qkv.clone()
    q_weight = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    k_weight = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    positions = torch.randint(
        0, MAX_POSITION_EMBEDDINGS, (num_tokens,), dtype=torch.long, device=device
    )

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    kv_cache = torch.zeros(
        num_blocks, 2, block_size, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device
    )
    slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.long, device=device
    )[:num_tokens]
    monkeypatch.setattr(
        fused_mod,
        "get_attention_context",
        lambda layer_name: _make_fake_context(kv_cache, slot_mapping),
    )

    kv_cache_dummy, q, k, v, gate = (
        fused_mod.qwen3_aiter_fused_qkv_norm_rope_cache_impl(
            qkv,
            positions,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            "layers.0.self_attn.attn",
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            ROTARY_DIM,
            RMS_NORM_EPS,
            True,
        )
    )
    q_ref, k_ref, v_ref, gate_ref = _ref_qwen3_qkv_norm_rope(
        qkv_ref, positions, q_weight, k_weight, rope.cos_sin_cache
    )

    # The AITER fused kernel has a different bf16 materialization boundary from
    # the PyTorch reference; AITER's own op test uses this tolerance.
    atol, rtol = 2e-2, 2e-2
    torch.testing.assert_close(q, q_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(k, k_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(v, v_ref, atol=0, rtol=0)
    torch.testing.assert_close(gate, gate_ref, atol=0, rtol=0)
    assert kv_cache_dummy.numel() == 0

    key_cache, value_cache = kv_cache.unbind(dim=1)
    flat_key_cache = key_cache.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    flat_value_cache = value_cache.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    torch.testing.assert_close(
        flat_key_cache[slot_mapping],
        k_ref.view(num_tokens, NUM_KV_HEADS, HEAD_DIM),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        flat_value_cache[slot_mapping],
        v_ref.view(num_tokens, NUM_KV_HEADS, HEAD_DIM),
        atol=0,
        rtol=0,
    )
