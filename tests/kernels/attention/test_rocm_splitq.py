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
CACHE_DTYPES = ["splitq_k3v4", "splitq_k3v3", "splitq_k3v3_compact"]


def _format(cache_dtype: str) -> sq.SplitQFormat:
    return sq.SplitQFormat.from_cache_dtype(cache_dtype, HEAD, ROPE)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _token_slots(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """(blocks, H, block_size, slot) cache -> (T, H, slot) for flat slots."""
    flat = cache.permute(0, 2, 1, 3).reshape(-1, cache.shape[1], cache.shape[3])
    return flat[slots]


def _fill(cache, block_ids, k, v, k_signs, v_signs, fmt):
    n = k.shape[0]
    slots = (
        block_ids.long().repeat_interleave(BLOCK)[:n] * BLOCK
        + torch.arange(n, device=k.device) % BLOCK
    )
    torch.ops._C.splitq_cache_store(
        k, v, cache, slots, k_signs, v_signs, fmt.kernel_code
    )
    return slots


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES)
def test_store_matches_reference(cache_dtype):
    torch.manual_seed(0)
    dev, fmt = "cuda", _format(cache_dtype)
    ksg = vsg = sq.sign_bits(HEAD).to(dev)
    t, hkv = 1024, 2
    k = torch.randn(t, hkv, HEAD, device=dev)
    k[..., 100] *= 8  # an outlier channel is spread by the rotation
    k = k.bfloat16()
    v = torch.randn(t, hkv, HEAD, device=dev).bfloat16()
    nblk = t // BLOCK + 4
    cache = torch.zeros(nblk, hkv, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=dev)
    slots = _fill(cache, torch.randperm(nblk, device=dev), k, v, ksg, vsg, fmt)

    k_got, v_got = sq.reference_dequantize(_token_slots(cache, slots), fmt)
    ref = sq.reference_quantize(k.float(), v.float(), fmt)
    k_ref, v_ref = sq.reference_dequantize(ref, fmt)
    # Same codes up to rounding ties from a different summation order.
    assert _rel(k_got, k_ref) < 1e-2 and _rel(v_got, v_ref) < 1e-2
    # Lloyd-Max on Gaussian data: 3-bit codes (4-bit for block K's RoPE
    # block) keep ~19% of the norm as error, 4-bit ~10%.
    assert _rel(k_got, k) < 0.2
    assert _rel(v_got, v) < (0.12 if fmt.v_bits == 4 else 0.22)


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES)
@pytest.mark.parametrize("query_group", [1, 4])
@pytest.mark.parametrize("num_splits", [1, 7, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_wmma", [True, False])
def test_decode_matches_reference(
    cache_dtype, query_group, num_splits, dtype, use_wmma
):
    """Covers MTP verification: up to 4 consecutive query tokens per request,
    each with its own causal length, plus a padded query with no request.
    ``use_wmma=False`` forces the portable kernel every architecture runs."""
    torch.manual_seed(1)
    dev, fmt = "cuda", _format(cache_dtype)
    ksg = vsg = sq.sign_bits(HEAD).to(dev)
    hkv, group = 1, 6
    lens = [1, 16, 37, 700, 3000]
    cache = torch.zeros(512, hkv, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=dev)
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
        _fill(cache, block_table[i, :nb], k, v, ksg, vsg, fmt)
        nq = min(query_group, n)
        for j in range(nq):
            q_to_req.append(i)
            q_to_klen.append(n - nq + 1 + j)
    q_to_req.append(len(lens) + 3)  # cudagraph padding: no request
    q_to_klen.append(0)
    q_to_req = torch.tensor(q_to_req, dtype=torch.int32, device=dev)
    q_to_klen = torch.tensor(q_to_klen, dtype=torch.int32, device=dev)

    num_q = q_to_req.numel()
    q = (torch.randn(num_q, hkv * group, HEAD, device=dev) * 2).to(dtype)
    out = torch.zeros_like(q)
    mid = torch.empty(
        num_q, hkv * group, num_splits, HEAD + 2, dtype=torch.float32, device=dev
    )
    torch.ops._C.splitq_decode(
        out,
        q,
        cache,
        block_table,
        q_to_req,
        q_to_klen,
        mid,
        ksg,
        vsg,
        1 / 16,
        num_splits,
        fmt.kernel_code,
        query_group,
        use_wmma,
    )
    ref = sq.reference_attention(
        q.float(),
        cache,
        block_table,
        q_to_req.clamp(max=len(lens) - 1),
        q_to_klen,
        fmt,
        1 / 16,
    )
    # Only the kernel's query precision (fp16 or int8) differs.
    assert _rel(out[:-1], ref[:-1]) < 2e-2
    assert out[-1].abs().max().item() == 0


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES)
@pytest.mark.parametrize("query_lens", [[300], [129, 1, 517]])
def test_prefill_matches_reference(cache_dtype, query_lens):
    """Chunked prefill: each request's chunk attends causally to itself
    (unquantized K/V) and to its cached prefix (packed cache), in the rotated
    space the backend runs it in."""
    torch.manual_seed(3)
    dev, fmt = "cuda", _format(cache_dtype)
    ksg = vsg = sq.sign_bits(HEAD).to(dev)
    hkv, group = 1, 6
    ctx_lens = [2500, 0, 4100][: len(query_lens)]
    seq_lens = [c + n for c, n in zip(ctx_lens, query_lens)]
    max_blocks = max(math.ceil(n / BLOCK) for n in seq_lens)
    nblk = sum(math.ceil(n / BLOCK) for n in seq_lens) + 8
    cache = torch.zeros(nblk, hkv, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=dev)
    block_table = torch.zeros(len(seq_lens), max_blocks, dtype=torch.int32, device=dev)
    free, used = torch.randperm(nblk, device=dev), 0
    ks, vs = [], []
    for i, n in enumerate(seq_lens):
        nb = math.ceil(n / BLOCK)
        block_table[i, :nb] = free[used : used + nb].int()
        used += nb
        k = torch.randn(n, hkv, HEAD, device=dev).half()
        v = torch.randn(n, hkv, HEAD, device=dev).half()
        _fill(cache, block_table[i, :nb], k, v, ksg, vsg, fmt)
        ks.append(k)
        vs.append(v)

    qsl = torch.tensor([0, *query_lens], device=dev).cumsum(0).int()
    total = int(qsl[-1])
    q = (torch.randn(total, hkv * group, HEAD, device=dev) * 2).half()
    k_new = torch.cat([k[c:] for k, c in zip(ks, ctx_lens)])
    v_new = torch.cat([v[c:] for v, c in zip(vs, ctx_lens)])
    q_rot, k_rot, v_rot = q.clone(), k_new.clone(), v_new.clone()
    torch.ops._C.splitq_rotate(q_rot, ksg, not fmt.compact, False)
    torch.ops._C.splitq_rotate(k_rot, ksg, not fmt.compact, False)
    torch.ops._C.splitq_rotate(v_rot, vsg, False, False)
    out = torch.empty_like(q_rot)
    torch.ops._C.splitq_prefill(
        out,
        q_rot,
        k_rot,
        v_rot,
        cache,
        block_table,
        qsl,
        torch.tensor(seq_lens, dtype=torch.int32, device=dev),
        max(query_lens),
        1 / 16,
        fmt.kernel_code,
    )
    torch.ops._C.splitq_rotate(out, vsg, False, True)

    ref = torch.empty(total, hkv * group, HEAD, device=dev)
    for i, (c, n) in enumerate(zip(ctx_lens, query_lens)):
        k_all, v_all = ks[i][c:].float(), vs[i][c:].float()
        if c > 0:
            pos = torch.arange(c, device=dev)
            slots = cache[block_table[i, pos // BLOCK].long(), :, pos % BLOCK]
            k_pre, v_pre = sq.reference_dequantize(slots, fmt)
            k_all, v_all = torch.cat([k_pre, k_all]), torch.cat([v_pre, v_all])
        k_all = k_all.repeat_interleave(group, 1)
        v_all = v_all.repeat_interleave(group, 1)
        qi = q[int(qsl[i]) : int(qsl[i + 1])].float()
        s = torch.einsum("qhd,thd->hqt", qi, k_all) / 16
        mask = (
            torch.arange(c + n, device=dev)[None]
            > (c + torch.arange(n, device=dev))[:, None]
        )
        p = torch.softmax(s.masked_fill(mask, float("-inf")), -1)
        ref[int(qsl[i]) : int(qsl[i + 1])] = torch.einsum("hqt,thd->qhd", p, v_all)
    # Only the kernel's int8 query and fp16 P/V rounding differ.
    assert _rel(out, ref) < 2e-2
