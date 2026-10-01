# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SplitQ HIP kernels against the PyTorch reference in
vllm/v1/attention/ops/rocm_splitq.py."""

import math

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops import rocm_splitq as sq

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or not hasattr(torch.ops._C, "splitq_decode"),
    reason="SplitQ kernels are ROCm-only",
)

HEAD, ROPE, BLOCK = 256, 64, 16


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _token_slots(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """(blocks, H, block_size, slot) cache -> (T, H, slot) for flat slots."""
    flat = cache.permute(0, 2, 1, 3).reshape(-1, cache.shape[1], cache.shape[3])
    return flat[slots]


def _fill(cache, block_ids, k, v, nope_signs, v_signs, bits):
    n = k.shape[0]
    slots = (
        block_ids.long().repeat_interleave(BLOCK)[:n] * BLOCK
        + torch.arange(n, device=k.device) % BLOCK
    )
    torch.ops._C.splitq_cache_store(k, v, cache, slots, nope_signs, v_signs, bits)
    return slots


@pytest.mark.parametrize("bits", [4, 3])
def test_store_matches_reference(bits):
    torch.manual_seed(0)
    dev, fmt = "cuda", sq.SplitQFormat(HEAD, ROPE, bits)
    nsg = sq.sign_bits(fmt.nope_dim).to(dev)
    vsg = sq.sign_bits(HEAD).to(dev)
    t, hkv = 1024, 2
    k = torch.randn(t, hkv, HEAD, device=dev)
    k[..., 100] *= 8  # an outlier channel is spread by the rotation
    k = k.bfloat16()
    v = torch.randn(t, hkv, HEAD, device=dev).bfloat16()
    nblk = t // BLOCK + 4
    cache = torch.zeros(nblk, hkv, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=dev)
    slots = _fill(cache, torch.randperm(nblk, device=dev), k, v, nsg, vsg, bits)

    k_got, v_got = sq.reference_dequantize(_token_slots(cache, slots), fmt)
    ref = sq.reference_quantize(k.float(), v.float(), fmt)
    k_ref, v_ref = sq.reference_dequantize(ref, fmt)
    # Same codes up to rounding ties from a different summation order.
    assert _rel(k_got, k_ref) < 1e-2 and _rel(v_got, v_ref) < 1e-2
    tol = 0.12 if bits == 4 else 0.22
    assert _rel(k_got, k) < tol and _rel(v_got, v) < tol


@pytest.mark.parametrize("bits", [4, 3])
@pytest.mark.parametrize("query_group", [1, 4])
@pytest.mark.parametrize("num_splits", [1, 7, 64])
def test_decode_matches_reference(bits, query_group, num_splits):
    """Covers MTP verification: up to 4 consecutive query tokens per request,
    each with its own causal length, plus a padded query with no request."""
    torch.manual_seed(1)
    dev, fmt = "cuda", sq.SplitQFormat(HEAD, ROPE, bits)
    nsg = sq.sign_bits(fmt.nope_dim).to(dev)
    vsg = sq.sign_bits(HEAD).to(dev)
    hkv, group = 1, 6
    lens = [1, 16, 37, 700, 3000]
    cache = torch.zeros(
        512, hkv, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=dev
    )
    max_blocks = max(math.ceil(n / BLOCK) for n in lens)
    block_table = torch.zeros(len(lens), max_blocks, dtype=torch.int32, device=dev)
    free = torch.randperm(512, device=dev)
    q_to_req, q_to_klen, used = [], [], 0
    for i, n in enumerate(lens):
        nb = math.ceil(n / BLOCK)
        block_table[i, :nb] = free[used : used + nb].int()
        used += nb
        k = torch.randn(n, hkv, HEAD, device=dev).bfloat16()
        v = torch.randn(n, hkv, HEAD, device=dev).bfloat16()
        _fill(cache, block_table[i, :nb], k, v, nsg, vsg, bits)
        nq = min(query_group, n)
        for j in range(nq):
            q_to_req.append(i)
            q_to_klen.append(n - nq + 1 + j)
    q_to_req.append(len(lens) + 3)  # cudagraph padding: no request
    q_to_klen.append(0)
    q_to_req = torch.tensor(q_to_req, dtype=torch.int32, device=dev)
    q_to_klen = torch.tensor(q_to_klen, dtype=torch.int32, device=dev)

    num_q = q_to_req.numel()
    q = (torch.randn(num_q, hkv * group, HEAD, device=dev) * 2).bfloat16()
    out = torch.zeros_like(q)
    mid = torch.empty(
        num_q, hkv * group, num_splits, HEAD + 2, dtype=torch.float32, device=dev
    )
    torch.ops._C.splitq_decode(
        out, q, cache, block_table, q_to_req, q_to_klen, mid, nsg, vsg,
        1 / 16, num_splits, bits, query_group,
    )
    ref = sq.reference_attention(
        q.float(), cache, block_table, q_to_req.clamp(max=len(lens) - 1),
        q_to_klen, fmt, 1 / 16,
    )
    # The kernel quantizes the query to int8; nothing else differs.
    assert _rel(out[:-1], ref[:-1]) < 2e-2
    assert out[-1].abs().max().item() == 0


@pytest.mark.parametrize("bits", [4, 3])
def test_to_int8_matches_reference(bits):
    torch.manual_seed(2)
    dev, fmt = "cuda", sq.SplitQFormat(HEAD, ROPE, bits)
    nsg = sq.sign_bits(fmt.nope_dim).to(dev)
    vsg = sq.sign_bits(HEAD).to(dev)
    n, hkv = 300, 2
    cache = torch.zeros(64, hkv, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=dev)
    blocks = torch.randperm(64, device=dev)[: math.ceil(n / BLOCK)].int()
    k = torch.randn(n, hkv, HEAD, device=dev).bfloat16()
    v = torch.randn(n, hkv, HEAD, device=dev).bfloat16()
    slots = _fill(cache, blocks, k, v, nsg, vsg, bits)

    k8 = torch.empty(n, hkv, HEAD, dtype=torch.int8, device=dev)
    v8 = torch.empty_like(k8)
    ks = torch.empty(n, hkv, dtype=torch.float32, device=dev)
    vs = torch.empty_like(ks)
    torch.ops._C.splitq_to_int8(cache, blocks, n, k8, v8, ks, vs, bits)
    k_ref, v_ref = sq.reference_dequantize(
        _token_slots(cache, slots), fmt, rotated=True
    )
    assert _rel(k8.float() * ks[..., None], k_ref) < 1e-2
    assert _rel(v8.float() * vs[..., None], v_ref) < 1e-2
