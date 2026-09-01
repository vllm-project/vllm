# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical tests for the fp8_e4m3 and nvfp4 KV cache branches of the QSA
sparse paged attention kernel.

The kernel is compared against a float64 PyTorch reference computed on the
*dequantized* cache contents, so the comparison measures the kernel, not the
quantization error. The bf16 branch must stay bitwise identical with and
without the scale arguments (the quantized branches are compiled out).
"""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import (
    model as _qwen4_exp_model,  # noqa: F401
)
from vllm.models.qwen4_exp.nvidia import qsa as qsa_backend
from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim, nvfp4_split_data_scale

requires_qsa_kernels = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA kernels require CUDA and Triton",
)

NUM_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
PAGE_SIZE = 64
TOPK = 2051  # indexer budget 2048 + compress ratio 4 - 1
NUM_REQ = 2
SEQ_LEN = 4096
# Query-row counts that exercise every tile profile of the wrapper
# (BLOCK_N=16 decode profiles and the wide prefill profiles).
TILE_PROFILE_ROWS = (2, 5, 64, 200, 300)

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _reference(q, k_cache, v_cache, indices, block_table, token_to_req):
    """Same computation as the kernel, in float64 with plain PyTorch."""
    rows, heads, dim = q.shape
    group = heads // k_cache.shape[2]
    out = torch.zeros(rows, heads, dim, dtype=torch.float64, device=q.device)
    max_pages = block_table.shape[1]
    for r in range(rows):
        req = int(token_to_req[r])
        tok = indices[r]
        valid = tok >= 0
        tok = tok.clamp(min=0)
        page = tok // PAGE_SIZE
        valid &= page < max_pages
        offset = tok % PAGE_SIZE
        phys = block_table[req, page.clamp(max=max_pages - 1)]
        valid &= (phys >= 0) & (phys < k_cache.shape[0])
        idx = valid.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        p = phys[idx].long()
        o = offset[idx].long()
        for kvh in range(k_cache.shape[2]):
            keys = k_cache[p, o, kvh].to(torch.float64)
            values = v_cache[p, o, kvh].to(torch.float64)
            for h in range(kvh * group, (kvh + 1) * group):
                scores = (q[r, h].to(torch.float64) @ keys.T) * (dim**-0.5)
                w = torch.softmax(scores, dim=0)
                out[r, h] = w @ values
    return out


def _rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.to(torch.float64) - b.to(torch.float64)).abs().mean()
    return (d / b.to(torch.float64).abs().mean().clamp(min=1e-12)).item()


def _e4m3_scale_to_float(bits: torch.Tensor) -> torch.Tensor:
    payload = bits.to(torch.int32) & 0x7F
    exp = (payload >> 3) & 0x0F
    mant = payload & 0x07
    normal = torch.pow(2.0, exp.float() - 7.0) * (1.0 + mant.float() / 8.0)
    subnormal = mant.float() / 512.0
    value = torch.where(exp == 0, subnormal, normal)
    return torch.where(payload == 0, torch.zeros_like(value), value)


def _scale_coords(page_size: int, scale_dim: int, swizzled: bool, device):
    t = torch.arange(page_size, device=device).view(-1, 1)
    s = torch.arange(scale_dim, device=device).view(1, -1)
    if not swizzled:
        return t.expand(page_size, scale_dim), s.expand(page_size, scale_dim)
    g = scale_dim // 4
    return (t // 4) * 4 + (s // g), (s % g) * 4 + (t % 4)


def _dequantize_nvfp4(kv_cache, num_kv_heads, head_dim, k_swizzled, v_swizzled):
    """Unpack an nvfp4 cache in PyTorch. Returns bf16 K and V of shape
    [B, N, H, head_dim], the layout the bf16 kernel expects."""
    dev = kv_cache.device
    table = torch.tensor(_E2M1 + [-v for v in _E2M1], dtype=torch.float32, device=dev)
    scale_dim = head_dim // 16
    page_size = kv_cache.shape[2]
    sides = (
        (kv_cache[:, :num_kv_heads].transpose(1, 2), k_swizzled),
        (kv_cache[:, num_kv_heads:].transpose(1, 2), v_swizzled),
    )
    result = []
    for side, swizzled in sides:
        data, scales = nvfp4_split_data_scale(side)
        data = data.contiguous()
        scales = scales.view(torch.uint8)
        b, n, h, _ = data.shape
        nibbles = torch.stack((data & 0x0F, (data >> 4) & 0x0F), dim=-1)
        values = table[nibbles.reshape(b, n, h, head_dim).long()]
        t_idx, s_idx = _scale_coords(page_size, scale_dim, swizzled, dev)
        picked = scales[:, t_idx.reshape(-1), :, :]
        picked = picked.reshape(b, page_size, scale_dim, h, scale_dim)
        s_sel = s_idx.view(1, page_size, scale_dim, 1, 1).expand(
            b, page_size, scale_dim, h, 1
        )
        picked = torch.gather(picked, 4, s_sel).squeeze(4).permute(0, 1, 3, 2)
        factor = _e4m3_scale_to_float(picked).repeat_interleave(16, dim=3)
        result.append((values * factor).to(torch.bfloat16))
    return result[0], result[1]


def _make_problem(dev, num_rows: int, seed: int = 0):
    torch.manual_seed(seed)
    pages_per_req = (SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    num_blocks = NUM_REQ * pages_per_req + 3
    q = (torch.randn(num_rows, NUM_HEADS, HEAD_DIM, device=dev) * 0.5).bfloat16()
    block_table = torch.arange(
        NUM_REQ * pages_per_req, dtype=torch.int32, device=dev
    ).view(NUM_REQ, pages_per_req)
    token_to_req = torch.randint(0, NUM_REQ, (num_rows,), dtype=torch.int32, device=dev)
    indices = torch.randint(0, SEQ_LEN, (num_rows, TOPK), dtype=torch.int32, device=dev)
    # Part of the selection is invalid (-1), as in operation when the indexer
    # fills less than the full width.
    indices[:, TOPK // 2 :] = -1
    return num_blocks, q, block_table, token_to_req, indices


def _assert_kernel_matches(out_quant, ref_quant, out_bf16, ref_bf16) -> None:
    r_bf16 = _rel_error(out_bf16, ref_bf16)
    r_kernel = _rel_error(out_quant, ref_quant)
    assert r_kernel <= max(5 * r_bf16, 5e-3), (r_kernel, r_bf16)


@requires_qsa_kernels
def test_qsa_fp8_kv_matches_dequantized_reference() -> None:
    dev = torch.device("cuda")
    num_blocks, q, block_table, token_to_req, indices = _make_problem(dev, 5)
    k_cache = (
        torch.randn(num_blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.5
    ).bfloat16()
    v_cache = (
        torch.randn(num_blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.5
    ).bfloat16()
    one = torch.tensor(1.0, dtype=torch.float32, device=dev)

    out_bf16 = qsa_sparse_paged_attention(
        q, k_cache, v_cache, indices, block_table, token_to_req
    )
    out_bf16_scaled = qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        indices,
        block_table,
        token_to_req,
        k_scale=one,
        v_scale=one,
    )
    # The bf16 branch must not change with the scale arguments present.
    assert torch.equal(out_bf16, out_bf16_scaled)

    # Quantized caches are allocated as uint8 by vLLM and reinterpreted right
    # before the kernel; the test takes the same detour.
    k_fp8 = k_cache.to(torch.float8_e4m3fn).view(torch.uint8).view(torch.float8_e4m3fn)
    v_fp8 = v_cache.to(torch.float8_e4m3fn).view(torch.uint8).view(torch.float8_e4m3fn)
    ref_bf16 = _reference(q, k_cache, v_cache, indices, block_table, token_to_req)
    ref_fp8 = _reference(
        q,
        k_fp8.to(torch.bfloat16),
        v_fp8.to(torch.bfloat16),
        indices,
        block_table,
        token_to_req,
    )
    out_fp8 = qsa_sparse_paged_attention(
        q, k_fp8, v_fp8, indices, block_table, token_to_req, k_scale=one, v_scale=one
    )
    _assert_kernel_matches(out_fp8, ref_fp8, out_bf16, ref_bf16)


@requires_qsa_kernels
@pytest.mark.parametrize("num_rows", TILE_PROFILE_ROWS)
def test_qsa_fp8_kv_tile_profiles(num_rows: int) -> None:
    dev = torch.device("cuda")
    num_blocks, q, block_table, token_to_req, indices = _make_problem(dev, num_rows)
    k_cache = (
        torch.randn(num_blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.5
    ).bfloat16()
    v_cache = (
        torch.randn(num_blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.5
    ).bfloat16()
    one = torch.tensor(1.0, dtype=torch.float32, device=dev)
    k_fp8 = k_cache.to(torch.float8_e4m3fn)
    v_fp8 = v_cache.to(torch.float8_e4m3fn)
    out_deq = qsa_sparse_paged_attention(
        q,
        k_fp8.to(torch.bfloat16),
        v_fp8.to(torch.bfloat16),
        indices,
        block_table,
        token_to_req,
    )
    out_fp8 = qsa_sparse_paged_attention(
        q, k_fp8, v_fp8, indices, block_table, token_to_req, k_scale=one, v_scale=one
    )
    assert (out_fp8.double() - out_deq.double()).abs().max().item() < 5e-3


def _write_nvfp4_cache(kv_cache, key, value, slots, scale):
    from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash

    kv_cache.zero_()
    reshape_and_cache_flash(
        key,
        value,
        kv_cache[:, :NUM_KV_HEADS].transpose(1, 2),
        kv_cache[:, NUM_KV_HEADS:].transpose(1, 2),
        slots,
        "nvfp4",
        scale,
        scale,
    )
    torch.cuda.synchronize()


@requires_qsa_kernels
def test_qsa_nvfp4_kv_block_scale_layout_matches_writer() -> None:
    """The read kernel assumes linear K block scales and swizzled V block
    scales, the layout reshape_and_cache_flash writes. Measure it instead of
    assuming it: each 16-value group gets its own amplitude so a wrong scale
    lookup is off by factors, not by rounding."""
    dev = torch.device("cuda")
    torch.manual_seed(0)
    num_blocks = 8
    num_slots = num_blocks * PAGE_SIZE
    t_i = torch.arange(num_slots, device=dev).view(-1, 1, 1)
    h_i = torch.arange(NUM_KV_HEADS, device=dev).view(1, -1, 1)
    g_i = torch.arange(HEAD_DIM // 16, device=dev).view(1, 1, -1)
    amp = torch.pow(4.0, ((t_i * 7 + h_i * 5 + g_i * 3) % 5).float()) * 0.05
    amp = amp.repeat_interleave(16, dim=2)
    key = (torch.randn(num_slots, NUM_KV_HEADS, HEAD_DIM, device=dev) * amp).bfloat16()
    value = (
        torch.randn(num_slots, NUM_KV_HEADS, HEAD_DIM, device=dev) * amp
    ).bfloat16()
    slots = torch.arange(num_slots, dtype=torch.int64, device=dev)
    kv_cache = torch.zeros(
        num_blocks,
        2 * NUM_KV_HEADS,
        PAGE_SIZE,
        nvfp4_kv_cache_full_dim(HEAD_DIM),
        dtype=torch.uint8,
        device=dev,
    )
    one = torch.tensor(1.0, dtype=torch.float32, device=dev)
    _write_nvfp4_cache(kv_cache, key, value, slots, one)

    errors = {}
    for k_swz in (False, True):
        for v_swz in (False, True):
            k_deq, v_deq = _dequantize_nvfp4(
                kv_cache, NUM_KV_HEADS, HEAD_DIM, k_swz, v_swz
            )
            errors[(k_swz, v_swz)] = (
                _rel_error(k_deq.reshape(num_slots, NUM_KV_HEADS, HEAD_DIM), key),
                _rel_error(v_deq.reshape(num_slots, NUM_KV_HEADS, HEAD_DIM), value),
            )
    # Correct hypothesis: pure nvfp4 rounding noise (about 0.09 on this
    # pattern); every wrong hypothesis lands above 1.
    k_err, v_err = errors[(False, qsa_backend._NVFP4_V_SCALE_SWIZZLED)]
    assert k_err < 0.3 and v_err < 0.3, errors
    assert errors[(True, not qsa_backend._NVFP4_V_SCALE_SWIZZLED)][0] > 0.3, errors


@requires_qsa_kernels
@pytest.mark.parametrize("num_rows", (5,) + TILE_PROFILE_ROWS)
def test_qsa_nvfp4_kv_matches_dequantized_reference(num_rows: int) -> None:
    dev = torch.device("cuda")
    num_blocks, q, block_table, token_to_req, indices = _make_problem(dev, num_rows)
    num_slots = num_blocks * PAGE_SIZE
    key = (torch.randn(num_slots, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.5).bfloat16()
    value = (
        torch.randn(num_slots, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.5
    ).bfloat16()
    slots = torch.arange(num_slots, dtype=torch.int64, device=dev)
    kv_cache = torch.zeros(
        num_blocks,
        2 * NUM_KV_HEADS,
        PAGE_SIZE,
        nvfp4_kv_cache_full_dim(HEAD_DIM),
        dtype=torch.uint8,
        device=dev,
    )
    one = torch.tensor(1.0, dtype=torch.float32, device=dev)
    _write_nvfp4_cache(kv_cache, key, value, slots, one)

    v_swizzled = qsa_backend._NVFP4_V_SCALE_SWIZZLED
    k_deq, v_deq = _dequantize_nvfp4(
        kv_cache, NUM_KV_HEADS, HEAD_DIM, False, v_swizzled
    )
    k_deq = k_deq.contiguous()
    v_deq = v_deq.contiguous()
    k_data, k_sf, v_data, v_sf = qsa_backend._nvfp4_cache_views(kv_cache, NUM_KV_HEADS)

    out_deq = qsa_sparse_paged_attention(
        q, k_deq, v_deq, indices, block_table, token_to_req
    )
    out_nvfp4 = qsa_sparse_paged_attention(
        q,
        k_data,
        v_data,
        indices,
        block_table,
        token_to_req,
        k_scale=one,
        v_scale=one,
        k_scale_cache=k_sf,
        v_scale_cache=v_sf,
        v_scale_swizzled=v_swizzled,
    )
    # Kernel against kernel on identical dequantized values: measures only the
    # nvfp4 read branch across all tile profiles.
    assert (out_nvfp4.double() - out_deq.double()).abs().max().item() < 5e-3

    if num_rows == 5:
        kb = key.reshape(num_blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM).contiguous()
        vb = value.reshape(num_blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM).contiguous()
        out_bf16 = qsa_sparse_paged_attention(
            q, kb, vb, indices, block_table, token_to_req
        )
        ref_bf16 = _reference(q, kb, vb, indices, block_table, token_to_req)
        ref_nvfp4 = _reference(q, k_deq, v_deq, indices, block_table, token_to_req)
        _assert_kernel_matches(out_nvfp4, ref_nvfp4, out_bf16, ref_bf16)
