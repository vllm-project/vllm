# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CDNA INT8 per-token-head decode kernel."""

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
        and hasattr(torch.ops._C, "pth_decode_int8_cdna")):
    pytest.skip("pth_decode_int8_cdna op not registered",
                allow_module_level=True)


def _quantize_int8_pth(t_fp):
    absmax = t_fp.abs().amax(dim=-1).clamp_min(1e-8)
    scale = (absmax / 127.0).to(torch.float32)
    q = (t_fp.to(torch.float32) / scale.unsqueeze(-1)).round()
    q = q.clamp(-128, 127).to(torch.int8)
    return q, scale


def _ref_attention_decode(q, k, v, sm_scale):
    """[1, Hq, D], [N, Hkv, D], [N, Hkv, D] -> [1, Hq, D]."""
    _, Hq, D = q.shape
    N, Hkv, _ = k.shape
    g = Hq // Hkv
    k = k.repeat_interleave(g, dim=1)
    v = v.repeat_interleave(g, dim=1)
    qf = q.to(torch.float32).transpose(0, 1)  # [Hq, 1, D]
    kf = k.to(torch.float32).transpose(0, 1)  # [Hq, N, D]
    vf = v.to(torch.float32).transpose(0, 1)
    scores = torch.einsum("hmd,hnd->hmn", qf, kf) * sm_scale  # [Hq, 1, N]
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("hmn,hnd->hmd", weights, vf)
    return out.transpose(0, 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("seq_len", [128, 1024, 8192])
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(8, 1), (32, 8)])
def test_cdna_int8_decode_matches_reference(
    dtype, head_size, seq_len, num_q_heads, num_kv_heads
):
    torch.manual_seed(0)
    device = "cuda"
    block_size = 16
    num_blocks_needed = (seq_len + block_size - 1) // block_size
    num_blocks_total = max(8, num_blocks_needed * 2)

    q_fp = (0.1 * torch.randn(1, num_q_heads, head_size,
                              device=device, dtype=dtype))
    full_k_fp = (0.1 * torch.randn(seq_len, num_kv_heads, head_size,
                                   device=device, dtype=dtype))
    full_v_fp = (0.1 * torch.randn(seq_len, num_kv_heads, head_size,
                                   device=device, dtype=dtype))
    k_q, k_sc = _quantize_int8_pth(full_k_fp)
    v_q, v_sc = _quantize_int8_pth(full_v_fp)

    k_cache = torch.zeros(num_blocks_total, block_size, num_kv_heads,
                          head_size, dtype=torch.int8, device=device)
    v_cache = torch.zeros_like(k_cache)
    k_scale_cache = torch.zeros(num_blocks_total, block_size, num_kv_heads,
                                dtype=torch.float32, device=device)
    v_scale_cache = torch.zeros_like(k_scale_cache)
    block_table = torch.zeros(1, num_blocks_needed, dtype=torch.int32,
                              device=device)
    for i in range(num_blocks_needed):
        block_table[0, i] = i
    for t in range(seq_len):
        blk = t // block_size
        slot = t % block_size
        k_cache[blk, slot] = k_q[t]
        v_cache[blk, slot] = v_q[t]
        k_scale_cache[blk, slot] = k_sc[t]
        v_scale_cache[blk, slot] = v_sc[t]

    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    out = torch.empty_like(q_fp)
    sm_scale = float(head_size) ** -0.5
    torch.ops._C.pth_decode_int8_cdna(out, q_fp, k_cache, v_cache,
                                      k_scale_cache, v_scale_cache,
                                      block_table, seq_lens, sm_scale)

    ref_k = (k_q.to(torch.float32) * k_sc.unsqueeze(-1)).to(dtype)
    ref_v = (v_q.to(torch.float32) * v_sc.unsqueeze(-1)).to(dtype)
    ref = _ref_attention_decode(q_fp, ref_k, ref_v, sm_scale)
    # Tolerance rationale: the KV cache is symmetric INT8-quantized with
    # scale = amax / 127, so the worst-case rounding error is scale / 2 and
    # the per-element RMS relative quant error is
    # (scale / sqrt(12)) / signal_rms ≈ sqrt(3) / (127 * sqrt(12)) ≈ 0.4%.
    # Averaging over the head_size-wide Q·K / P·V dot products decorrelates
    # these errors, while fp16 / MFMA accumulation and the softmax set the
    # floor, leaving an end-to-end output error of ~1%. rtol = atol = 2e-2
    # is ~2x that headroom (confirmed across the swept shapes): tight enough
    # to catch a real regression, loose enough to avoid quant-noise flakes.
    torch.testing.assert_close(out.to(torch.float32), ref, rtol=2e-2,
                               atol=2e-2)
