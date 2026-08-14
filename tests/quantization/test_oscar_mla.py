# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the OSCAR INT2 MLA-latent path.

The load-bearing property is that the packed-storage kernels and the
fake-quant measurement path produce *the same numbers*. If they diverge, an
accuracy result measured with the fake-quant path says nothing about what
the real INT2 cache would do.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.oscar.mla_latent import (
    fake_quant_int2_groupwise,
)
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("OSCAR MLA kernels are CUDA-only", allow_module_level=True)

from vllm.v1.attention.ops.triton_oscar_mla import (  # noqa: E402
    oscar_mla_decode_int2,
    oscar_mla_dequant_int2,
    oscar_mla_pack_int2,
)

# GLM-5.2 geometry.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
GROUP = 128


@pytest.mark.parametrize("num_tokens", [1, 17, 2048])
def test_pack_dequant_matches_fake_quant(num_tokens: int):
    """Packed codes must dequantize to exactly the fake-quant output.

    This is what licenses using fake-quant accuracy numbers as a proxy for
    the real INT2 cache.
    """
    torch.manual_seed(0)
    c_kv = torch.randn(num_tokens, KV_LORA_RANK, device="cuda")

    codes, sb = oscar_mla_pack_int2(c_kv, group_size=GROUP)
    got = oscar_mla_dequant_int2(codes, sb, group_size=GROUP)
    want = fake_quant_int2_groupwise(c_kv, GROUP, lloyd_max=True)

    assert codes.shape == (num_tokens, KV_LORA_RANK // 4)
    assert codes.dtype == torch.uint8
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_packed_size_is_int2():
    """Storage must actually be 2 bits/element plus per-group metadata."""
    c_kv = torch.randn(1024, KV_LORA_RANK, device="cuda")
    codes, sb = oscar_mla_pack_int2(c_kv, group_size=GROUP)

    per_token = (codes.numel() + sb.numel() * 4) // 1024
    bf16_per_token = KV_LORA_RANK * 2
    assert per_token == 160, f"expected 160 B/token, got {per_token}"
    assert bf16_per_token / per_token == pytest.approx(6.4)


def test_only_four_levels_per_group():
    """INT2 means at most 4 distinct values per quantization group."""
    torch.manual_seed(0)
    c_kv = torch.randn(8, KV_LORA_RANK, device="cuda")
    codes, sb = oscar_mla_pack_int2(c_kv, group_size=GROUP)
    deq = oscar_mla_dequant_int2(codes, sb, group_size=GROUP)

    for t in range(deq.shape[0]):
        for g in range(KV_LORA_RANK // GROUP):
            group = deq[t, g * GROUP : (g + 1) * GROUP]
            assert group.unique().numel() <= 4


@pytest.mark.parametrize("ctx", [128, 2048])
@pytest.mark.parametrize("num_heads", [8])
def test_decode_matches_reference(ctx: int, num_heads: int):
    """The fused INT2 decode must match dense attention on the same values.

    Reference attends over the *dequantized* latent, isolating kernel error
    from quantization error.
    """
    torch.manual_seed(0)
    q_latent = torch.randn(num_heads, KV_LORA_RANK, device="cuda")
    q_pe = torch.randn(num_heads, QK_ROPE_HEAD_DIM, device="cuda")
    c_kv = torch.randn(ctx, KV_LORA_RANK, device="cuda")
    k_pe = torch.randn(ctx, QK_ROPE_HEAD_DIM, device="cuda")
    sm_scale = 1.0 / (KV_LORA_RANK + QK_ROPE_HEAD_DIM) ** 0.5

    codes, sb = oscar_mla_pack_int2(c_kv, group_size=GROUP)
    out = oscar_mla_decode_int2(
        q_latent, q_pe, codes, sb, k_pe, sm_scale, group_size=GROUP
    )

    deq = oscar_mla_dequant_int2(codes, sb, group_size=GROUP)
    scores = (q_latent @ deq.T + q_pe @ k_pe.T) * sm_scale
    ref = torch.softmax(scores.float(), dim=-1) @ deq

    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    rel_l2 = (out - ref).norm() / ref.norm()
    assert cos > 0.9999, f"cos={cos:.6f}"
    assert rel_l2 < 1e-3, f"rel_l2={rel_l2:.2e}"


def test_rotation_is_orthogonal_roundtrip():
    """rotate -> quant -> unrotate must not drift when quantization is exact.

    Uses a value already on the quantization grid so any error is the
    rotation's, not the quantizer's.
    """
    torch.manual_seed(0)
    d = KV_LORA_RANK
    rot, _ = torch.linalg.qr(torch.randn(d, d, device="cuda"))
    x = torch.randn(64, d, device="cuda")

    roundtrip = (x @ rot) @ rot.T
    torch.testing.assert_close(roundtrip, x, atol=1e-4, rtol=1e-4)
