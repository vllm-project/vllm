# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CDNA INT4 per-token-head decode kernel."""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.platforms.rocm import on_mi3xx  # noqa: E402

if not on_mi3xx():
    pytest.skip("CDNA only", allow_module_level=True)

if not (hasattr(torch.ops, "_C")
        and hasattr(torch.ops._C, "pth_decode_int4_cdna")):
    pytest.skip("pth_decode_int4_cdna op not registered",
                allow_module_level=True)

from vllm.v1.attention.ops.cdna_int4_prefill import pack_scale_zp  # noqa: E402


def _quant_int4_pth(t_fp):
    fp32 = t_fp.to(torch.float32)
    lo = fp32.amin(dim=-1)
    hi = fp32.amax(dim=-1)
    rng = (hi - lo).clamp_min(1e-8)
    scale = (rng / 15.0).clamp_min(1e-8)
    zp = (-lo / scale).round().clamp(0, 15).to(torch.int32)
    qf = (fp32 / scale.unsqueeze(-1) + zp.unsqueeze(-1)).round()
    q = qf.clamp(0, 15).to(torch.uint8)
    even = q[..., 0::2]
    odd = q[..., 1::2]
    packed = (even | (odd << 4)).contiguous()
    return packed, scale, zp


def _deq_int4(packed, scale, zp, head_size, dtype):
    even = packed & 0xF
    odd = (packed >> 4) & 0xF
    out_shape = packed.shape[:-1] + (head_size,)
    out = torch.empty(out_shape, device=packed.device, dtype=torch.float32)
    out[..., 0::2] = (even.to(torch.float32)
                      - zp.unsqueeze(-1).to(torch.float32))
    out[..., 1::2] = (odd.to(torch.float32)
                      - zp.unsqueeze(-1).to(torch.float32))
    return (out * scale.unsqueeze(-1).to(torch.float32)).to(dtype)


def _ref_decode(q, k, v, sm_scale):
    _, Hq, D = q.shape
    _, Hkv, _ = k.shape
    g = Hq // Hkv
    k = k.repeat_interleave(g, dim=1)
    v = v.repeat_interleave(g, dim=1)
    qf = q.to(torch.float32).transpose(0, 1)
    kf = k.to(torch.float32).transpose(0, 1)
    vf = v.to(torch.float32).transpose(0, 1)
    scores = torch.einsum("hmd,hnd->hmn", qf, kf) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("hmn,hnd->hmd", weights, vf).transpose(0, 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("seq_len", [128, 1024, 8192])
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(8, 1), (32, 8)])
def test_cdna_int4_decode(dtype, head_size, seq_len, num_q_heads,
                          num_kv_heads):
    torch.manual_seed(0)
    device = "cuda"
    block_size = 16
    nb = (seq_len + block_size - 1) // block_size
    nt = max(8, nb * 2)
    q_fp = (0.1 * torch.randn(1, num_q_heads, head_size, device=device,
                              dtype=dtype))
    full_k = (0.1 * torch.randn(seq_len, num_kv_heads, head_size,
                                device=device, dtype=dtype))
    full_v = (0.1 * torch.randn(seq_len, num_kv_heads, head_size,
                                device=device, dtype=dtype))
    k_pack, k_sc, k_zp = _quant_int4_pth(full_k)
    v_pack, v_sc, v_zp = _quant_int4_pth(full_v)
    k_steg = pack_scale_zp(k_sc, k_zp)
    v_steg = pack_scale_zp(v_sc, v_zp)

    k_cache = torch.zeros(nt, block_size, num_kv_heads, head_size // 2,
                          dtype=torch.uint8, device=device)
    v_cache = torch.zeros_like(k_cache)
    k_scale_cache = torch.zeros(nt, block_size, num_kv_heads,
                                dtype=torch.float32, device=device)
    v_scale_cache = torch.zeros_like(k_scale_cache)
    block_table = torch.zeros(1, nb, dtype=torch.int32, device=device)
    for i in range(nb):
        block_table[0, i] = i
    for t in range(seq_len):
        blk = t // block_size
        slot = t % block_size
        k_cache[blk, slot] = k_pack[t]
        v_cache[blk, slot] = v_pack[t]
        k_scale_cache[blk, slot] = k_steg[t]
        v_scale_cache[blk, slot] = v_steg[t]

    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    out = torch.empty_like(q_fp)
    sm_scale = float(head_size) ** -0.5
    torch.ops._C.pth_decode_int4_cdna(out, q_fp, k_cache, v_cache,
                                      k_scale_cache, v_scale_cache,
                                      block_table, seq_lens, sm_scale)
    ref_k = _deq_int4(k_pack, k_sc, k_zp, head_size, dtype)
    ref_v = _deq_int4(v_pack, v_sc, v_zp, head_size, dtype)
    ref = _ref_decode(q_fp, ref_k, ref_v, sm_scale)
    torch.testing.assert_close(out.to(torch.float32), ref, rtol=4e-2,
                               atol=4e-2)
