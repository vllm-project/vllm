# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the UltraQuant full-KV dequant kernel (continuation prefill).

The kernel reconstructs cached K/V from FP4 codes plus UE8M0 group scales.
Because dequant is an exact table lookup times a power of two, the result
must match the reference to within the output dtype's rounding.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.ops.ultraquant.format import slot_size
from vllm.v1.attention.ops.ultraquant.reference import (
    ultraquant_dequant,
    ultraquant_encode,
)
from vllm.v1.attention.ops.ultraquant.triton_dequant import (
    ultraquant_full_dequant_kv,
)
from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/HIP device"
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("cached_len", [32, 96])
def test_full_dequant_matches_reference(dtype, head_dim, num_kv_heads, cached_len):
    device = torch.device("cuda")
    torch.manual_seed(0xD4)

    block_size = 32
    alloc_len = cached_len
    num_blocks = alloc_len // block_size + 1

    key = torch.randn(alloc_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn(alloc_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        slot_size(head_dim),
        dtype=torch.uint8,
        device=device,
    )
    # Leave block 0 unused so the block table exercises a non-identity mapping.
    slot_mapping = torch.arange(
        block_size, block_size + alloc_len, dtype=torch.int64, device=device
    )
    ultraquant_store(key, value, kv_cache, slot_mapping)

    block_table = torch.arange(
        1, num_blocks, dtype=torch.int32, device=device
    ).unsqueeze(0)

    k_out = torch.zeros(
        1, num_kv_heads, alloc_len, head_dim, dtype=dtype, device=device
    )
    v_out = torch.zeros_like(k_out)
    ultraquant_full_dequant_kv(
        kv_cache=kv_cache,
        block_table=block_table,
        k_out=k_out,
        v_out=v_out,
        alloc_len=alloc_len,
    )
    torch.accelerator.synchronize()

    # K is stored Hadamard-rotated, V unrotated; the kernel returns them as
    # stored, so the reference must not inverse-rotate either.
    ref_k = ultraquant_dequant(ultraquant_encode(key, rotate=True), head_dim)
    ref_v = ultraquant_dequant(ultraquant_encode(value, rotate=False), head_dim)

    got_k = k_out[0].transpose(0, 1).to(torch.float32)
    got_v = v_out[0].transpose(0, 1).to(torch.float32)

    # Reference is fp32; the kernel rounds to the output dtype once.
    tol = 1e-2 if dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(got_k, ref_k.to(torch.float32), rtol=tol, atol=tol)
    torch.testing.assert_close(got_v, ref_v.to(torch.float32), rtol=tol, atol=tol)


def test_full_dequant_zero_slots_decode_to_zero():
    """Slots never written stay all-zero, which must dequant to exact zero."""
    device = torch.device("cuda")
    head_dim, num_kv_heads = 256, 1
    block_size, num_blocks = 32, 3
    alloc_len = block_size * (num_blocks - 1)

    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        slot_size(head_dim),
        dtype=torch.uint8,
        device=device,
    )
    block_table = torch.arange(
        1, num_blocks, dtype=torch.int32, device=device
    ).unsqueeze(0)

    k_out = torch.full(
        (1, num_kv_heads, alloc_len, head_dim),
        7.0,
        dtype=torch.bfloat16,
        device=device,
    )
    v_out = torch.full_like(k_out, 7.0)
    ultraquant_full_dequant_kv(
        kv_cache=kv_cache,
        block_table=block_table,
        k_out=k_out,
        v_out=v_out,
        alloc_len=alloc_len,
    )
    torch.accelerator.synchronize()

    assert (k_out == 0).all()
    assert (v_out == 0).all()
