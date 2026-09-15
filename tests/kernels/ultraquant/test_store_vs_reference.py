# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Triton UltraQuant store kernel.

The Triton kernel must produce bit-identical codes and scale bytes vs
the PyTorch reference (``reference.ultraquant_encode``).
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.ops.ultraquant.format import (
    k_codes_offset,
    k_scales_offset,
    n_groups,
    slot_size,
    v_codes_offset,
    v_scales_offset,
)
from vllm.v1.attention.ops.ultraquant.reference import ultraquant_encode
from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/HIP device"
)


def _make_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    slot_b = slot_size(head_dim)
    return torch.zeros(
        num_blocks, block_size, num_kv_heads, slot_b, dtype=torch.uint8, device=device
    )


def _read_codes_and_scales(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
):
    """Pull (k_codes, k_scale_bytes, v_codes, v_scale_bytes) for each
    (token, head) from the AoS slot layout. Scales come back as uint8
    (one E8M0 byte per group)."""
    block_size = kv_cache.shape[1]
    N = slot_mapping.shape[0]
    Gk = n_groups(head_dim)

    k_codes_off = k_codes_offset(head_dim)
    k_scales_off = k_scales_offset(head_dim)
    v_codes_off = v_codes_offset(head_dim)
    v_scales_off = v_scales_offset(head_dim)
    codes_bytes = head_dim // 2

    k_codes = torch.zeros(
        N, num_kv_heads, codes_bytes, dtype=torch.uint8, device=kv_cache.device
    )
    k_scale_bytes = torch.zeros(
        N, num_kv_heads, Gk, dtype=torch.uint8, device=kv_cache.device
    )
    v_codes = torch.zeros_like(k_codes)
    v_scale_bytes = torch.zeros_like(k_scale_bytes)

    for ti in range(N):
        slot = int(slot_mapping[ti].item())
        if slot < 0:
            continue
        blk = slot // block_size
        off = slot % block_size
        for h in range(num_kv_heads):
            slot_bytes = kv_cache[blk, off, h]  # [slot_size_aligned] uint8

            k_codes[ti, h] = slot_bytes[k_codes_off : k_codes_off + codes_bytes]
            k_scale_bytes[ti, h] = slot_bytes[k_scales_off : k_scales_off + Gk]
            v_codes[ti, h] = slot_bytes[v_codes_off : v_codes_off + codes_bytes]
            v_scale_bytes[ti, h] = slot_bytes[v_scales_off : v_scales_off + Gk]

    return k_codes, k_scale_bytes, v_codes, v_scale_bytes


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_tokens", [1, 16, 17])
def test_store_matches_reference(dtype, head_dim, num_kv_heads, num_tokens):
    """Triton store output (codes + E8M0 scale bytes) matches the reference
    bit-exactly."""
    device = torch.device("cuda")
    torch.manual_seed(0xF8)

    block_size = 32
    num_blocks = max(2, (num_tokens + block_size - 1) // block_size + 1)

    key = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)

    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    if num_tokens >= 4:
        slot_mapping[1] = -1  # skip-path test

    kv_cache = _make_cache(num_blocks, block_size, num_kv_heads, head_dim, device)

    ultraquant_store(key, value, kv_cache, slot_mapping)
    torch.accelerator.synchronize()

    ref_k = ultraquant_encode(key, rotate=True)
    ref_v = ultraquant_encode(value, rotate=False)

    k_codes, k_scale_bytes, v_codes, v_scale_bytes = _read_codes_and_scales(
        kv_cache, slot_mapping, num_kv_heads, head_dim
    )

    if num_tokens >= 4:
        assert (kv_cache[0, 1, :, :] == 0).all(), (
            "skipped slot's physical cache region was unexpectedly written"
        )

    for ti in range(num_tokens):
        slot = int(slot_mapping[ti].item())
        if slot < 0:
            continue

        assert torch.equal(k_codes[ti], ref_k.codes_packed[ti]), (
            f"K codes mismatch at token {ti}\n"
            f"got:      {k_codes[ti].cpu().numpy().tolist()[:16]}\n"
            f"expected: {ref_k.codes_packed[ti].cpu().numpy().tolist()[:16]}"
        )
        assert torch.equal(k_scale_bytes[ti], ref_k.scale_bytes[ti]), (
            f"K scale bytes mismatch at token {ti}\n"
            f"got:      {k_scale_bytes[ti].cpu().numpy().tolist()}\n"
            f"expected: {ref_k.scale_bytes[ti].cpu().numpy().tolist()}"
        )

        assert torch.equal(v_codes[ti], ref_v.codes_packed[ti]), (
            f"V codes mismatch at token {ti}"
        )
        assert torch.equal(v_scale_bytes[ti], ref_v.scale_bytes[ti]), (
            f"V scale bytes mismatch at token {ti}\n"
            f"got:      {v_scale_bytes[ti].cpu().numpy().tolist()}\n"
            f"expected: {ref_v.scale_bytes[ti].cpu().numpy().tolist()}"
        )


def test_zero_input_produces_zero_codes_and_zero_scale_byte():
    """All-zero K/V → all-zero FP4 codes AND scale byte = 0 (zero sentinel)."""
    device = torch.device("cuda")
    D, H, N = 128, 4, 8
    block_size = 16
    num_blocks = 2

    key = torch.zeros(N, H, D, dtype=torch.bfloat16, device=device)
    value = torch.zeros(N, H, D, dtype=torch.bfloat16, device=device)
    slot_mapping = torch.arange(N, dtype=torch.int64, device=device)
    kv_cache = _make_cache(num_blocks, block_size, H, D, device)

    ultraquant_store(key, value, kv_cache, slot_mapping)
    torch.accelerator.synchronize()

    k_codes, k_scale_bytes, v_codes, v_scale_bytes = _read_codes_and_scales(
        kv_cache, slot_mapping, H, D
    )
    assert (k_codes == 0).all()
    assert (k_scale_bytes == 0).all(), (
        "zero K must encode E8M0 scale byte = 0 (zero sentinel)"
    )
    assert (v_codes == 0).all()
    assert (v_scale_bytes == 0).all()


def test_scale_is_power_of_two():
    """Decoded E8M0 scales must always be exact powers of two (or zero)."""
    device = torch.device("cuda")
    D, H, N = 128, 2, 32
    block_size = 32
    num_blocks = 2

    torch.manual_seed(7)
    key = torch.randn(N, H, D, dtype=torch.bfloat16, device=device) * 3.0
    value = torch.randn(N, H, D, dtype=torch.bfloat16, device=device) * 0.1
    slot_mapping = torch.arange(N, dtype=torch.int64, device=device)
    kv_cache = _make_cache(num_blocks, block_size, H, D, device)

    ultraquant_store(key, value, kv_cache, slot_mapping)
    torch.accelerator.synchronize()

    _, k_scale_bytes, _, v_scale_bytes = _read_codes_and_scales(
        kv_cache, slot_mapping, H, D
    )

    # Decode: nonzero bytes must map to 2^(byte-127); zero byte → 0.0.
    from vllm.v1.attention.ops.ultraquant.format import UE8M0_BIAS

    for bytes_t in (k_scale_bytes, v_scale_bytes):
        b = bytes_t.to(torch.int32)
        nonzero = b > 0
        if nonzero.any():
            exps = (b[nonzero] - UE8M0_BIAS).cpu().tolist()
            for e in exps:
                assert -127 <= e <= 127, f"E8M0 exponent out of range: {e}"
            # 2^e is always exactly representable, so just check no NaN/inf
            # would appear after decoding.
            decoded = torch.exp2((b[nonzero] - UE8M0_BIAS).float())
            assert torch.isfinite(decoded).all()
