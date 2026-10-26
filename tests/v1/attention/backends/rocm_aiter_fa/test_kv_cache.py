# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache write and read paths of ROCM_AITER_FA.

``do_kv_cache_update`` must store exactly the (quantized) K/V at each slot for
every layout and cache dtype, skip ``-1`` padding slots, and the attention
read paths must see the same values. The fused RoPE+cache hooks are covered by
tests/compile/passes/test_rope_kvcache_fusion.py and
test_rocm_aiter_qk_norm_rope_kvcache_fusion.py.
"""

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_fa.fa_harness import (
    DEVICE,
    FP8_K_SCALE,
    FP8_V_SCALE,
    FaCase,
    blocks_for,
)
from tests.v1.attention.model_shapes import MUSE_SPARK_FULL, QWEN3_8_27B, QWEN3_235B
from vllm.platforms import current_platform
from vllm.v1.attention.backends.rocm_aiter_fa import (
    AiterFlashAttentionBackend,
    cp_mha_gather_cache,
)
from vllm.v1.kv_cache_interface import KVCacheLayout, is_quantized_kv_cache

KV_DTYPES = ["auto", "bfloat16", "fp8", "fp8_e4m3"]


def _read_back(case: FaCase, slots: torch.Tensor):
    """K/V at ``slots``, dequantized, through the backend's own gather kernel
    (the only reader that understands the shuffle layout)."""
    impl = case.layer.impl
    key_cache, value_cache = impl._split_kv_cache(case.kv_cache)
    quantized = is_quantized_kv_cache(case.kv_cache_dtype)
    if quantized:
        key_cache = key_cache.view(current_platform.fp8_dtype())
        value_cache = value_cache.view(current_platform.fp8_dtype())
    n = slots.numel()
    kv, d = case.shape.num_kv_heads, case.shape.head_size
    key = torch.empty(n, kv, d, dtype=case.dtype, device=DEVICE)
    value = torch.empty_like(key)
    # One "request" per slot: a 1-token sequence whose only block is the
    # slot's block, starting at the slot's offset.
    bs = case.block_size
    from vllm._aiter_ops import rocm_aiter_ops

    cp_mha_gather_cache(
        key_cache=key_cache,
        value_cache=value_cache,
        key=key,
        value=value,
        block_tables=(slots // bs).to(torch.int32)[:, None],
        k_scales=case.layer._k_scale,
        v_scales=case.layer._v_scale,
        cu_seqlens_kv=torch.arange(n + 1, dtype=torch.int32, device=DEVICE),
        token_to_batch=torch.arange(n, dtype=torch.int32, device=DEVICE),
        seq_starts=(slots % bs).to(torch.int32),
        dequant=quantized,
        kv_cache_layout=(
            "SHUFFLE" if rocm_aiter_ops.is_shuffle_kv_cache_enabled() else "NHD"
        ),
        total_tokens=n,
    )
    return key, value


@pytest.mark.parametrize("kv_cache_dtype", KV_DTYPES)
@pytest.mark.parametrize("layout", [KVCacheLayout.LBHNC, KVCacheLayout.LHBNC])
@pytest.mark.parametrize(
    "shape", [MUSE_SPARK_FULL, QWEN3_235B, QWEN3_8_27B], ids=lambda s: s.id
)
def test_write_then_read_back(shape, layout, kv_cache_dtype, shuffle):
    if layout not in AiterFlashAttentionBackend.supported_kv_cache_layouts():
        pytest.skip(f"{layout.name} is not listed for this layout mode")
    case = FaCase(shape, kv_cache_dtype=kv_cache_dtype, layout=layout).setup(32)
    num_slots = case.num_blocks * case.block_size
    slots = torch.randperm(num_slots, device=DEVICE)[:100]
    key, value = _read_back(case, slots)
    tol = {"atol": 0, "rtol": 0}
    if is_quantized_kv_cache(kv_cache_dtype):
        tol = {"atol": 1e-6, "rtol": 0}
    torch.testing.assert_close(key, case.k_all[slots], **tol)
    torch.testing.assert_close(value, case.v_all[slots], **tol)


def test_padding_slots_are_skipped(shuffle):
    case = FaCase(QWEN3_235B, kv_cache_dtype="fp8").setup(8)
    before = case.kv_cache.view(torch.uint8).clone()
    n = 16
    key = torch.randn(
        n,
        QWEN3_235B.num_kv_heads,
        QWEN3_235B.head_size,
        dtype=case.dtype,
        device=DEVICE,
    )
    slot_mapping = torch.full((n,), -1, dtype=torch.int64, device=DEVICE)
    slot_mapping[:4] = torch.tensor([3, 17, 40, 99])
    case.layer.impl.do_kv_cache_update(
        case.layer, key, key.clone(), case.kv_cache, slot_mapping
    )
    k_back, _ = _read_back(case, slot_mapping[:4])
    expected = (key[:4] / FP8_K_SCALE).to(current_platform.fp8_dtype()).to(
        case.dtype
    ) * FP8_K_SCALE
    torch.testing.assert_close(k_back, expected, atol=1e-6, rtol=0)
    # Everything outside the four written slots is byte-identical.
    case.layer.impl.do_kv_cache_update(
        case.layer,
        case.k_all[slot_mapping[:4]],
        case.v_all[slot_mapping[:4]],
        case.kv_cache,
        slot_mapping[:4],
    )
    torch.testing.assert_close(case.kv_cache.view(torch.uint8), before, atol=0, rtol=0)


@pytest.mark.parametrize(
    "query_lens, seq_lens",
    [([1, 1], [32, 40]), ([16, 9], [80, 9])],
    ids=["decode", "extend+prefill"],
)
def test_unquantized_cache_ignores_layer_kv_scales(query_lens, seq_lens):
    """A checkpoint may ship KV scales while the cache stays bf16; the
    backend must not apply them (AITER treats non-null descales as dequant)."""
    case = FaCase(QWEN3_235B).setup(blocks_for(seq_lens, 16))
    case.layer._k_scale.fill_(FP8_K_SCALE)
    case.layer._v_scale.fill_(FP8_V_SCALE)
    case.layer._k_scale_float = FP8_K_SCALE
    case.layer._v_scale_float = FP8_V_SCALE
    batch = case.batch(query_lens, seq_lens)
    query = case.query(batch.num_tokens)
    case.check(batch, query, case.run(batch, query))


def test_every_listed_layout_gives_identical_attention(shuffle):
    q_lens, s_lens = [1, 1, 40, 64], [100, 300, 200, 64]
    outputs = []
    for layout in AiterFlashAttentionBackend.supported_kv_cache_layouts():
        case = FaCase(QWEN3_235B, layout=layout).setup(blocks_for(s_lens, 16))
        batch = case.batch(q_lens, s_lens)
        torch.manual_seed(1)
        query = case.query(batch.num_tokens)
        outputs.append(case.run(batch, query))
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], atol=0, rtol=0)
