# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm import _custom_ops as ops
from vllm.model_executor.layers.mla import _mla_rope_kvcache_fusion_enabled
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
    RotaryEmbedding,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
KV_LORA_RANK = 512
BLOCK_SIZE = 64
NUM_BLOCKS = 64


def _make_rope(is_neox: bool, dtype: torch.dtype, deepseek: bool):
    if deepseek:
        rope = DeepseekScalingRotaryEmbedding(
            QK_ROPE_HEAD_DIM,
            QK_ROPE_HEAD_DIM,
            max_position_embeddings=4096,
            base=10000,
            is_neox_style=is_neox,
            scaling_factor=40,
            dtype=torch.float32,
            extrapolation_factor=1,
            attn_factor=1,
            beta_fast=32,
            beta_slow=1,
            mscale=1.0,
            mscale_all_dim=1.0,
        )
    else:
        rope = RotaryEmbedding(
            QK_ROPE_HEAD_DIM,
            QK_ROPE_HEAD_DIM,
            4096,
            10000,
            is_neox,
            torch.float32,
        )
    return rope.to(dtype=dtype, device=torch.get_default_device())


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA fused kernel")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("is_neox", [False, True])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("num_tokens", [7, 42])
@pytest.mark.parametrize("num_heads", [16, 128])
@pytest.mark.parametrize("use_deepseek_rope", [False, True])
@torch.inference_mode()
def test_fused_matches_unfused_pipeline(
    default_vllm_config,
    dtype: torch.dtype,
    is_neox: bool,
    kv_cache_dtype: str,
    num_tokens: int,
    num_heads: int,
    use_deepseek_rope: bool,
) -> None:
    """Fused kernel vs the unfused custom-op pipeline, on the exact strided
    views the model seam produces (not contiguous tensors)."""
    set_random_seed(0)
    torch.set_default_device("cuda")
    rope = _make_rope(is_neox, dtype, use_deepseek_rope)

    positions = torch.randint(0, 4096, (num_tokens,))

    # Model-seam layouts:
    #   q: (T, H, nope+rope); q_pe is the strided trailing slice.
    #   kv_lora: (T, kv_lora_rank + rope); k_pe is the strided split view.
    q = torch.randn(
        num_tokens, num_heads, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM, dtype=dtype
    )
    kv_lora = torch.randn(num_tokens, KV_LORA_RANK + QK_ROPE_HEAD_DIM, dtype=dtype)
    kv_c, k_pe = kv_lora.split([KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1)
    kv_c = kv_c.contiguous()  # kv_a_layernorm output is contiguous in the model

    q_fused = q.clone()
    k_pe_fused = kv_lora.clone().split([KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1)[1]
    assert (
        q_fused[..., QK_NOPE_HEAD_DIM:].stride()
        != q_fused[..., QK_NOPE_HEAD_DIM:].contiguous().stride()
    )  # truly strided

    slot_mapping = torch.tensor(
        random.sample(range(NUM_BLOCKS * BLOCK_SIZE), num_tokens), dtype=torch.long
    )
    entry_size = KV_LORA_RANK + QK_ROPE_HEAD_DIM
    cache_torch_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    kv_cache_fused = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, entry_size, dtype=cache_torch_dtype
    )
    kv_cache_unfused = torch.zeros_like(kv_cache_fused)
    scale = torch.tensor([0.1], dtype=torch.float32)

    # --- Unfused reference: the exact production pipeline of the non-fused
    # path (ops.rotary_embedding in-place, then ops.concat_and_cache_mla).
    q_unfused = q.clone()
    k_pe_unfused = kv_lora.clone().split([KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1)[1]
    q_pe_unfused = q_unfused[..., QK_NOPE_HEAD_DIM:]
    ops.rotary_embedding(
        positions,
        q_pe_unfused,
        k_pe_unfused,
        QK_ROPE_HEAD_DIM,
        rope.cos_sin_cache,
        rope.is_neox_style,
    )
    ops.concat_and_cache_mla(
        kv_c,
        k_pe_unfused,
        kv_cache_unfused,
        slot_mapping,
        kv_cache_dtype,
        scale,
    )

    # --- Fused kernel on the strided seam views.
    ops.concat_and_cache_mla_rope_fused(
        positions,
        q_fused[..., QK_NOPE_HEAD_DIM:],
        k_pe_fused,
        kv_c,
        rope.cos_sin_cache,
        rope.is_neox_style,
        slot_mapping,
        kv_cache_fused,
        kv_cache_dtype,
        scale,
    )

    # The fused kernel and ops.rotary_embedding round differently in their
    # internal compute (intermediate precision / multiply ordering), so the
    # outputs of the two implementations can differ by ~1 ulp each. For
    # bf16 (8 mantissa bits) that is up to ~2^-7 relative and, near zero,
    # an absolute spacing of ~0.016 at |x|~2 -- the pytorch-derived
    # defaults (atol=1e-3) assume a same-algorithm comparison and are too
    # tight for that. fp16/fp32 pass with the defaults.
    if dtype == torch.bfloat16:
        atol, rtol = 2e-2, 2**-7
    else:
        atol, rtol = get_default_atol(q_fused), get_default_rtol(q_fused)
    torch.testing.assert_close(q_fused, q_unfused, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_pe_fused, k_pe_unfused, atol=atol, rtol=rtol)
    if kv_cache_dtype == "fp8":
        # Dequantize both caches; rounding differences upstream of the fp8
        # quantization can flip one quantization bucket (1 fp8e4m3 ulp).
        deq_fused = torch.empty_like(kv_cache_fused, dtype=torch.float16)
        deq_unfused = torch.empty_like(kv_cache_unfused, dtype=torch.float16)
        ops.convert_fp8(
            deq_fused,
            kv_cache_fused.contiguous(),
            scale.item(),
            kv_dtype=kv_cache_dtype,
        )
        ops.convert_fp8(
            deq_unfused,
            kv_cache_unfused.contiguous(),
            scale.item(),
            kv_dtype=kv_cache_dtype,
        )
        torch.testing.assert_close(deq_fused, deq_unfused, atol=0.05, rtol=0.15)
    else:
        torch.testing.assert_close(
            kv_cache_fused, kv_cache_unfused, atol=atol, rtol=rtol
        )
    # The nope part of q must be untouched (bit-exact).
    torch.testing.assert_close(
        q_fused[..., :QK_NOPE_HEAD_DIM],
        q[..., :QK_NOPE_HEAD_DIM],
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.cpu_test
def test_mla_rope_kvcache_fusion_enabled(default_vllm_config):
    rope = _make_rope(is_neox=False, dtype=torch.bfloat16, deepseek=False)

    def probe(**overrides):
        kwargs = dict(
            rotary_emb=rope,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            kv_cache_dtype="auto",
            is_sparse=False,
            model_dtype=torch.bfloat16,
            fuse_rope_kvcache_cat_mla=True,
            calculate_kv_scales=False,
        )
        kwargs.update(overrides)
        return _mla_rope_kvcache_fusion_enabled(**kwargs)

    if not current_platform.is_cuda_alike():
        assert probe() is False
        return

    assert probe() is True
    deepseek_rope = _make_rope(is_neox=False, dtype=torch.bfloat16, deepseek=True)
    assert probe(rotary_emb=deepseek_rope) is True
    fp16_rope = _make_rope(is_neox=False, dtype=torch.float16, deepseek=False)
    assert probe(rotary_emb=fp16_rope, model_dtype=torch.float16) is True

    # Flag off.
    assert probe(fuse_rope_kvcache_cat_mla=False) is False

    # Rope subclasses with different math are rejected (exact-type allowlist).
    class _CustomRope(RotaryEmbedding):
        pass

    custom = _CustomRope(
        QK_ROPE_HEAD_DIM, QK_ROPE_HEAD_DIM, 4096, 10000, False, torch.float32
    ).to(dtype=torch.bfloat16)
    assert probe(rotary_emb=custom) is False
    assert probe(rotary_emb=None) is False
    # Partial-width rotation.
    assert probe(qk_rope_head_dim=QK_ROPE_HEAD_DIM * 2) is False
    partial_rope = RotaryEmbedding(
        QK_ROPE_HEAD_DIM * 2,
        QK_ROPE_HEAD_DIM,
        4096,
        10000,
        False,
        torch.float32,
    ).to(dtype=torch.bfloat16)
    assert probe(rotary_emb=partial_rope) is False
    # cos_sin_cache dtype mismatch with activations.
    fp32_rope = _make_rope(is_neox=False, dtype=torch.float32, deepseek=False)
    assert probe(rotary_emb=fp32_rope) is False
    assert probe(model_dtype=torch.half) is False
    # Unsupported cache layouts.
    assert probe(kv_cache_dtype="fp8_ds_mla") is False
    assert probe(is_sparse=True) is False
    # Dynamic KV-scale calibration runs after the fused write would have
    # quantized with the stale scale.
    assert probe(calculate_kv_scales=True) is False


@pytest.mark.cpu_test
def test_mla_attention_forward_skips_kv_update_when_dep_provided():
    """When the producer already wrote the KV cache (dep provided), the
    direct path must not call impl.do_kv_cache_update again."""
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    def run(dep):
        self = MagicMock()
        self.calculate_kv_scales = False
        self.use_direct_call = True
        self.layer_name = "layer.0"
        self.kv_cache_dtype = "auto"
        q = torch.randn(2, 4, 8)
        ctx = MagicMock()
        ctx.attn_metadata = MagicMock()
        ctx.slot_mapping = {"layer.0": torch.zeros(2, dtype=torch.long)}
        with patch(
            "vllm.model_executor.layers.attention.mla_attention.get_forward_context",
            return_value=ctx,
        ):
            MLAAttention.forward(
                self,
                q,
                torch.randn(2, 16),
                torch.randn(2, 1, 8),
                output_shape=(2, 32),
                kv_cache_dummy_dep=dep,
            )
        return self.impl.do_kv_cache_update

    assert run(dep=torch.empty(0)).call_count == 0
    assert run(dep=None).call_count == 1
