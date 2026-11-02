# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the AVX-512 UltraQuant store kernel on CPU.

Same contract as the Triton kernel in ``test_store_vs_reference.py`` -- codes
and scale bytes match ``reference.ultraquant_encode`` bit for bit -- except
that the kernel's bf16 fast path rotates K in bf16, so there K is held to a
one-level bound instead. The CPU cache is head-major inside a block.
"""

from __future__ import annotations

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.ultraquant.format import (
    DEFAULT_CONSTANT_C,
    k_codes_offset,
    k_scales_offset,
    n_groups,
    slot_size,
    v_codes_offset,
    v_scales_offset,
)
from vllm.v1.attention.ops.ultraquant.reference import (
    UltraQuantEncoded,
    hadamard_matrix,
    ultraquant_dequant,
    ultraquant_encode,
)

pytestmark = pytest.mark.skipif(
    not torch.cpu._is_avx512_supported(), reason="requires AVX-512"
)


def _read_codes_and_scales(kv_cache, slot_mapping, num_kv_heads, head_dim):
    """Pull (k_codes, k_scale_bytes, v_codes, v_scale_bytes) per (token, head)
    out of the AoS slot, with the CPU cache's [block, head, token] interior."""
    block_size = kv_cache.shape[2]
    n_tokens = slot_mapping.shape[0]
    groups = n_groups(head_dim)
    codes_bytes = head_dim // 2

    k_off, ks_off = k_codes_offset(head_dim), k_scales_offset(head_dim)
    v_off, vs_off = v_codes_offset(head_dim), v_scales_offset(head_dim)

    k_codes = torch.zeros(n_tokens, num_kv_heads, codes_bytes, dtype=torch.uint8)
    k_scale_bytes = torch.zeros(n_tokens, num_kv_heads, groups, dtype=torch.uint8)
    v_codes = torch.zeros_like(k_codes)
    v_scale_bytes = torch.zeros_like(k_scale_bytes)

    for ti in range(n_tokens):
        slot = int(slot_mapping[ti].item())
        if slot < 0:
            continue
        blk, off = divmod(slot, block_size)
        for h in range(num_kv_heads):
            slot_bytes = kv_cache[blk, h, off]
            k_codes[ti, h] = slot_bytes[k_off : k_off + codes_bytes]
            k_scale_bytes[ti, h] = slot_bytes[ks_off : ks_off + groups]
            v_codes[ti, h] = slot_bytes[v_off : v_off + codes_bytes]
            v_scale_bytes[ti, h] = slot_bytes[vs_off : vs_off + groups]

    return k_codes, k_scale_bytes, v_codes, v_scale_bytes


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("num_kv_heads", [1, 8])
@pytest.mark.parametrize("num_tokens", [1, 16, 17])
def test_store_matches_reference(head_dim, num_kv_heads, num_tokens):
    set_random_seed(0xF8)
    dtype = torch.float32
    block_size = 32
    num_blocks = max(2, (num_tokens + block_size - 1) // block_size + 1)

    key = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype)
    value = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    if num_tokens >= 4:
        slot_mapping[1] = -1  # skip-path test

    kv_cache = torch.zeros(
        num_blocks, num_kv_heads, block_size, slot_size(head_dim), dtype=torch.uint8
    )
    rotation = hadamard_matrix(head_dim, key.device, torch.float32)
    ops.ultraquant_store(
        key, value, kv_cache, slot_mapping, rotation, DEFAULT_CONSTANT_C
    )

    ref_k = ultraquant_encode(key, rotate=True)
    ref_v = ultraquant_encode(value, rotate=False)
    k_codes, k_scales, v_codes, v_scales = _read_codes_and_scales(
        kv_cache, slot_mapping, num_kv_heads, head_dim
    )

    if num_tokens >= 4:
        assert (kv_cache[0, :, 1, :] == 0).all(), (
            "skipped slot's physical cache region was unexpectedly written"
        )

    for ti in range(num_tokens):
        if int(slot_mapping[ti].item()) < 0:
            continue
        assert torch.equal(k_codes[ti], ref_k.codes_packed[ti]), (
            f"K codes mismatch at token {ti}"
        )
        assert torch.equal(k_scales[ti], ref_k.scale_bytes[ti]), (
            f"K scale bytes mismatch at token {ti}"
        )
        assert torch.equal(v_codes[ti], ref_v.codes_packed[ti]), (
            f"V codes mismatch at token {ti}"
        )
        assert torch.equal(v_scales[ti], ref_v.scale_bytes[ti]), (
            f"V scale bytes mismatch at token {ti}"
        )


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_store_bf16_rotation_costs_little_accuracy(head_dim):
    """The bf16 fast path rotates K in bf16 to keep the GEMM on the bf16 units.

    V is never rotated, so it stays bit-exact. K picks up rotation error that
    can move a value to the next FP4 level, or a group's absmax across a
    power-of-two edge, so it is held to the reconstruction it is there for.
    """
    set_random_seed(0xF8)
    num_kv_heads, num_tokens, block_size = 4, 64, 32
    num_blocks = (num_tokens + block_size - 1) // block_size + 1

    key = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16)
    value = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    kv_cache = torch.zeros(
        num_blocks, num_kv_heads, block_size, slot_size(head_dim), dtype=torch.uint8
    )
    ops.ultraquant_store(
        key,
        value,
        kv_cache,
        slot_mapping,
        hadamard_matrix(head_dim, key.device, torch.float32),
        DEFAULT_CONSTANT_C,
    )

    ref_k = ultraquant_encode(key, rotate=True)
    ref_v = ultraquant_encode(value, rotate=False)
    k_codes, k_scales, v_codes, v_scales = _read_codes_and_scales(
        kv_cache, slot_mapping, num_kv_heads, head_dim
    )

    assert torch.equal(v_codes, ref_v.codes_packed)
    assert torch.equal(v_scales, ref_v.scale_bytes)

    rotated = key.float() @ hadamard_matrix(head_dim, key.device, torch.float32).T
    got = ultraquant_dequant(UltraQuantEncoded(k_codes, k_scales), head_dim)
    want = ultraquant_dequant(
        UltraQuantEncoded(ref_k.codes_packed, ref_k.scale_bytes), head_dim
    )

    def rel(x):
        return ((x - rotated).norm() / rotated.norm()).item()

    assert rel(got) <= rel(want) * 1.02, (
        f"bf16 rotation cost {rel(got) / rel(want):.4f}x the reference's error"
    )


@pytest.mark.parametrize("magnitude", [0.0, 1e-6, 1e4])
def test_store_handles_degenerate_magnitudes(magnitude):
    """Zero-amax groups encode byte 0, and the UE8M0 exponent has to saturate
    rather than wrap at the ends of the range."""
    set_random_seed(0xF9)
    head_dim, num_kv_heads, num_tokens, block_size = 128, 2, 4, 32

    key = torch.full((num_tokens, num_kv_heads, head_dim), magnitude).bfloat16()
    value = key.clone()
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    kv_cache = torch.zeros(
        1, num_kv_heads, block_size, slot_size(head_dim), dtype=torch.uint8
    )
    rotation = hadamard_matrix(head_dim, key.device, torch.float32)
    ops.ultraquant_store(
        key, value, kv_cache, slot_mapping, rotation, DEFAULT_CONSTANT_C
    )

    ref_v = ultraquant_encode(value, rotate=False)
    _, _, v_codes, v_scales = _read_codes_and_scales(
        kv_cache, slot_mapping, num_kv_heads, head_dim
    )
    assert torch.equal(v_codes, ref_v.codes_packed)
    assert torch.equal(v_scales, ref_v.scale_bytes)
