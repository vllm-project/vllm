# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3's fp8 decode must hand a bf16-query backend (TRITON_MLA, the DSpark draft)
the fp8 query dequantized by `q_scale`, bit for bit as an fp32 dequant would."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
from vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache import (
    fused_mla_decode_q_concat_kv_cache_insert,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Kimi-K3 fused MLA requires a GPU"
)


@pytest.mark.parametrize("q_scale", [1.0, 0.75, 0.05, 37.5])
def test_fp8_decode_dequantizes_query_for_bf16_backend(q_scale: float) -> None:
    torch.manual_seed(0)
    num_tokens, num_heads, kv_lora_rank, rope_dim = 13, 8, 512, 64

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    ql_nope = randn(num_tokens, num_heads, kv_lora_rank)
    q_pe = randn(num_tokens, num_heads, rope_dim)
    kv_c = randn(num_tokens, kv_lora_rank)
    k_pe = randn(num_tokens, rope_dim)
    slot_mapping = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    q_scale_t = torch.tensor(q_scale, dtype=torch.float32, device="cuda")
    # The attributes `_decode_concat_cache` reads, with the scales as
    # `BaseKVCacheMethod.process_weights_after_loading` leaves them.
    layer = SimpleNamespace(
        kv_cache_dtype="fp8",
        kv_cache=torch.zeros(
            (1, 16, kv_lora_rank + rope_dim), dtype=torch.uint8, device="cuda"
        ),
        impl=SimpleNamespace(supports_quant_query_input=False),
        _q_scale=q_scale_t,
        _q_scale_float=q_scale,
        _q_scale_inv=q_scale_t.reciprocal().reshape(1),
        _k_scale_inv=torch.ones(1, dtype=torch.float32, device="cuda"),
    )

    mqa_q = MultiHeadLatentAttention._decode_concat_cache(
        layer, ql_nope, q_pe, kv_c, k_pe, None, None, slot_mapping
    )

    q_fp8 = fused_mla_decode_q_concat_kv_cache_insert(
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        torch.zeros_like(layer.kv_cache),
        slot_mapping,
        q_scale_inv=layer._q_scale_inv,
        cache_scale_inv=layer._k_scale_inv,
    )
    expected = (q_fp8.float() * q_scale).to(torch.bfloat16)
    assert mqa_q.dtype == torch.bfloat16
    torch.testing.assert_close(mqa_q, expected, atol=0, rtol=0)
