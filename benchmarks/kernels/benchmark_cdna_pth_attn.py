# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for the CDNA per-token-head INT8/INT4 attention kernels.

Measures, on MI300-class (gfx942/gfx950) hardware:
  * kernel latency (CUDA/HIP events, warmup + timed iters),
  * accuracy vs an fp16 dequantized reference (max-abs / mean-rel error), and
  * KV-cache footprint vs an fp16 cache (memory saving).

A PyTorch SDPA fp16 run (full-precision K/V materialized) is timed at the same
shapes as a familiar reference point. The INT8/INT4 kernels read a 4x / 8x
smaller KV cache, which is the win at long context.

Run inside the ROCm container:
    python benchmarks/kernels/benchmark_cdna_pth_attn.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

assert current_platform.is_rocm(), "ROCm only"
from vllm.platforms.rocm import on_mi3xx

assert on_mi3xx(), "CDNA (gfx942/gfx950/gfx90a) only"

from vllm.v1.attention.ops.cdna_int4_prefill import pack_scale_zp

DEVICE = "cuda"
BLOCK = 16


# --------------------------------------------------------------------------- #
# Quantization helpers (mirror the kernel tests)
# --------------------------------------------------------------------------- #
def quant_int8_pth(t_fp):
    absmax = t_fp.abs().amax(dim=-1).clamp_min(1e-8)
    scale = (absmax / 127.0).to(torch.float32)
    q = (t_fp.to(torch.float32) / scale.unsqueeze(-1)).round()
    return q.clamp(-128, 127).to(torch.int8), scale


def quant_int4_pth(t_fp):
    fp32 = t_fp.to(torch.float32)
    lo, hi = fp32.amin(dim=-1), fp32.amax(dim=-1)
    rng = (hi - lo).clamp_min(1e-8)
    scale = (rng / 15.0).clamp_min(1e-8)
    zp = (-lo / scale).round().clamp(0, 15).to(torch.int32)
    qf = (fp32 / scale.unsqueeze(-1) + zp.unsqueeze(-1)).round()
    q = qf.clamp(0, 15).to(torch.uint8)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()
    return packed, scale, zp


def deq_int4(packed, scale, zp, head_size, dtype):
    even, odd = packed & 0xF, (packed >> 4) & 0xF
    out = torch.empty(packed.shape[:-1] + (head_size,),
                      device=packed.device, dtype=torch.float32)
    zpf = zp.unsqueeze(-1).to(torch.float32)
    out[..., 0::2] = even.to(torch.float32) - zpf
    out[..., 1::2] = odd.to(torch.float32) - zpf
    return (out * scale.unsqueeze(-1).to(torch.float32)).to(dtype)


def fill_cache(cache, scale_cache, q_src, scale_src, nb):
    for t in range(q_src.shape[0]):
        cache[t // BLOCK, t % BLOCK] = q_src[t]
        scale_cache[t // BLOCK, t % BLOCK] = scale_src[t]


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # microseconds


def err_stats(out, ref):
    out, ref = out.to(torch.float32), ref.to(torch.float32)
    abs_err = (out - ref).abs()
    rel = abs_err / ref.abs().clamp_min(1e-4)
    return abs_err.max().item(), rel.mean().item()


def sdpa_fp16(q, k, v, sm_scale, attn_mask=None):
    # q:[M,Hq,D] k/v:[N,Hkv,D] -> [M,Hq,D]; expand GQA. Causal masking uses
    # bottom-right alignment (the M query tokens sit at the END of the N-token
    # context), matching the kernel — NOT torch's is_causal, which aligns
    # top-left for non-square q/kv and would mis-score chunked prefill.
    g = q.shape[1] // k.shape[1]
    kk = k.repeat_interleave(g, dim=1)
    vv = v.repeat_interleave(g, dim=1)
    q2 = q.transpose(0, 1).unsqueeze(0)
    k2 = kk.transpose(0, 1).unsqueeze(0)
    v2 = vv.transpose(0, 1).unsqueeze(0)
    o = F.scaled_dot_product_attention(q2, k2, v2, scale=sm_scale,
                                       attn_mask=attn_mask)
    return o.squeeze(0).transpose(0, 1)


def causal_mask_br(m, n, device, dtype):
    """Additive bottom-right causal mask [M, N]: query i (0..M-1) may attend to
    key j (0..N-1) iff j <= (N - M) + i. None when M == 1 (decode)."""
    if m <= 1:
        return None
    ctx = n - m
    i = torch.arange(m, device=device).unsqueeze(1)
    j = torch.arange(n, device=device).unsqueeze(0)
    mask = torch.zeros(m, n, device=device, dtype=dtype)
    mask.masked_fill_((ctx + i) < j, float("-inf"))
    return mask


def kv_mem_ratio(n_tokens, hkv, d, which):
    """fp16 KV bytes / quantized KV bytes for ``n_tokens`` actual tokens (K+V),
    measured per token (no block padding). The quantized cache adds two fp32
    per-token-head scales (K and V)."""
    fp16 = n_tokens * hkv * d * 2 * 2          # K+V, 2 bytes/elem
    scales = n_tokens * hkv * 2 * 4            # K+V per-token-head fp32 scale
    if which == "int8":
        q = n_tokens * hkv * d * 2 * 1 + scales
    else:  # int4: packed nibbles, 0.5 byte/elem
        q = n_tokens * hkv * (d // 2) * 2 + scales
    return fp16 / q


# --------------------------------------------------------------------------- #
# Decode benchmarks (qlen = 1, large context)
# --------------------------------------------------------------------------- #
def bench_decode(dtype, hq, hkv, d, seq_len, which):
    torch.manual_seed(0)
    nb = (seq_len + BLOCK - 1) // BLOCK
    nt = max(8, nb * 2)
    q = 0.1 * torch.randn(1, hq, d, device=DEVICE, dtype=dtype)
    k = 0.1 * torch.randn(seq_len, hkv, d, device=DEVICE, dtype=dtype)
    v = 0.1 * torch.randn(seq_len, hkv, d, device=DEVICE, dtype=dtype)
    block_table = torch.arange(nb, dtype=torch.int32,
                               device=DEVICE).view(1, nb)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
    out = torch.empty_like(q)
    sm = float(d) ** -0.5

    if which == "int8":
        kq, ksc = quant_int8_pth(k)
        vq, vsc = quant_int8_pth(v)
        kc = torch.zeros(nt, BLOCK, hkv, d, dtype=torch.int8, device=DEVICE)
        vc = torch.zeros_like(kc)
        ksc_c = torch.zeros(nt, BLOCK, hkv, dtype=torch.float32, device=DEVICE)
        vsc_c = torch.zeros_like(ksc_c)
        fill_cache(kc, ksc_c, kq, ksc, nb)
        fill_cache(vc, vsc_c, vq, vsc, nb)
        op = torch.ops._C.pth_decode_int8_cdna
        fn = lambda: op(out, q, kc, vc, ksc_c, vsc_c, block_table,
                        seq_lens, sm)
        ref_k = (kq.float() * ksc.unsqueeze(-1)).to(dtype)
        ref_v = (vq.float() * vsc.unsqueeze(-1)).to(dtype)
    else:  # int4
        kp, ksc, kzp = quant_int4_pth(k)
        vp, vsc, vzp = quant_int4_pth(v)
        ksteg, vsteg = pack_scale_zp(ksc, kzp), pack_scale_zp(vsc, vzp)
        kc = torch.zeros(nt, BLOCK, hkv, d // 2, dtype=torch.uint8,
                         device=DEVICE)
        vc = torch.zeros_like(kc)
        ksc_c = torch.zeros(nt, BLOCK, hkv, dtype=torch.float32, device=DEVICE)
        vsc_c = torch.zeros_like(ksc_c)
        fill_cache(kc, ksc_c, kp, ksteg, nb)
        fill_cache(vc, vsc_c, vp, vsteg, nb)
        op = torch.ops._C.pth_decode_int4_cdna
        fn = lambda: op(out, q, kc, vc, ksc_c, vsc_c, block_table,
                        seq_lens, sm)
        ref_k = deq_int4(kp, ksc, kzp, d, dtype)
        ref_v = deq_int4(vp, vsc, vzp, d, dtype)

    fn()
    torch.cuda.synchronize()
    ref = sdpa_fp16(q, ref_k, ref_v, sm)
    max_abs, mean_rel = err_stats(out, ref)
    t_kernel = bench(fn)
    t_sdpa = bench(lambda: sdpa_fp16(q, ref_k, ref_v, sm))
    return dict(regime="decode", which=which, dtype=str(dtype).split(".")[-1],
                hq=hq, hkv=hkv, d=d, seq=seq_len, qlen=1,
                t_kernel=t_kernel, t_sdpa=t_sdpa, max_abs=max_abs,
                mean_rel=mean_rel,
                mem_ratio=kv_mem_ratio(seq_len, hkv, d, which))


# --------------------------------------------------------------------------- #
# Prefill benchmarks (qlen tokens, ctxlen = 0, causal)
# --------------------------------------------------------------------------- #
def bench_prefill(dtype, hq, hkv, d, qlen, ctxlen, which):
    """Chunked-prefill: ``qlen`` fp16 query/chunk tokens attending over a
    ``ctxlen``-token *quantized* paged context (the regime the kernel targets).
    ctxlen=0 is the degenerate pure-prefill case (no quantization benefit)."""
    torch.manual_seed(0)
    seq_len = qlen + ctxlen
    nb = (seq_len + BLOCK - 1) // BLOCK
    nt = max(8, nb * 2)
    q = 0.1 * torch.randn(qlen, hq, d, device=DEVICE, dtype=dtype)
    full_k = 0.1 * torch.randn(seq_len, hkv, d, device=DEVICE, dtype=dtype)
    full_v = 0.1 * torch.randn(seq_len, hkv, d, device=DEVICE, dtype=dtype)
    k_chunk = full_k[ctxlen:].contiguous()
    v_chunk = full_v[ctxlen:].contiguous()
    block_table = torch.arange(nb, dtype=torch.int32,
                               device=DEVICE).view(1, nb)
    cu = torch.tensor([0, qlen], dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
    out = torch.empty_like(q)
    sm = float(d) ** -0.5

    if which == "int8":
        kc = torch.zeros(nt, BLOCK, hkv, d, dtype=torch.int8, device=DEVICE)
        vc = torch.zeros_like(kc)
        ksc_c = torch.zeros(nt, BLOCK, hkv, dtype=torch.float32, device=DEVICE)
        vsc_c = torch.zeros_like(ksc_c)
        op = torch.ops._C.paged_prefill_attn_cdna_int8
        if ctxlen > 0:
            kq, ksc = quant_int8_pth(full_k[:ctxlen])
            vq, vsc = quant_int8_pth(full_v[:ctxlen])
            fill_cache(kc, ksc_c, kq, ksc, nb)
            fill_cache(vc, vsc_c, vq, vsc, nb)
            ctx_k = (kq.float() * ksc.unsqueeze(-1)).to(dtype)
            ctx_v = (vq.float() * vsc.unsqueeze(-1)).to(dtype)
    else:
        kc = torch.zeros(nt, BLOCK, hkv, d // 2, dtype=torch.uint8,
                         device=DEVICE)
        vc = torch.zeros_like(kc)
        ksc_c = torch.zeros(nt, BLOCK, hkv, dtype=torch.float32, device=DEVICE)
        vsc_c = torch.zeros_like(ksc_c)
        op = torch.ops._C.paged_prefill_attn_cdna_int4
        if ctxlen > 0:
            kp, ksc, kzp = quant_int4_pth(full_k[:ctxlen])
            vp, vsc, vzp = quant_int4_pth(full_v[:ctxlen])
            fill_cache(kc, ksc_c, kp, pack_scale_zp(ksc, kzp), nb)
            fill_cache(vc, vsc_c, vp, pack_scale_zp(vsc, vzp), nb)
            ctx_k = deq_int4(kp, ksc, kzp, d, dtype)
            ctx_v = deq_int4(vp, vsc, vzp, d, dtype)

    if ctxlen > 0:
        ref_k = torch.cat([ctx_k, k_chunk], dim=0)
        ref_v = torch.cat([ctx_v, v_chunk], dim=0)
    else:
        ref_k, ref_v = k_chunk, v_chunk

    mask = causal_mask_br(qlen, seq_len, DEVICE, q.dtype)
    fn = lambda: op(out, q, k_chunk, v_chunk, kc, vc, ksc_c, vsc_c,
                    block_table, cu, seq_lens, qlen, sm, True)
    fn()
    torch.cuda.synchronize()
    ref = sdpa_fp16(q, ref_k, ref_v, sm, attn_mask=mask)
    max_abs, mean_rel = err_stats(out, ref)
    t_kernel = bench(fn)
    t_sdpa = bench(lambda: sdpa_fp16(q, ref_k, ref_v, sm, attn_mask=mask))
    return dict(regime="prefill", which=which, dtype=str(dtype).split(".")[-1],
                hq=hq, hkv=hkv, d=d, seq=seq_len, qlen=qlen,
                t_kernel=t_kernel, t_sdpa=t_sdpa, max_abs=max_abs,
                mean_rel=mean_rel,
                mem_ratio=kv_mem_ratio(seq_len, hkv, d, which))


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}  "
          f"torch {torch.__version__}  hip {getattr(torch.version,'hip',None)}")
    rows = []
    dtype = torch.float16
    # (Hq, Hkv, D)
    shapes = [(32, 8, 128), (8, 1, 128), (64, 8, 128)]
    for which in ("int8", "int4"):
        for hq, hkv, d in shapes:
            for seq in (1024, 4096, 8192):
                rows.append(bench_decode(dtype, hq, hkv, d, seq, which))
            # Chunked prefill: short fp16 chunk over a long quantized context
            # (qlen, ctxlen) — plus a pure-prefill (ctx=0) reference point.
            for qlen, ctxlen in ((512, 0), (512, 4096), (256, 8192)):
                rows.append(bench_prefill(dtype, hq, hkv, d, qlen, ctxlen,
                                          which))

    hdr = (f"{'regime':8} {'kv':5} {'Hq/Hkv':8} {'D':4} {'qlen':>5} "
           f"{'seq':6} {'kernel_us':>10} {'sdpa_us':>9} {'speedup':>8} "
           f"{'max_abs':>9} {'mean_rel':>9} {'mem_x':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['regime']:8} {r['which']:5} "
              f"{str(r['hq'])+'/'+str(r['hkv']):8} {r['d']:<4} "
              f"{r['qlen']:>5} {r['seq']:<6} "
              f"{r['t_kernel']:10.1f} {r['t_sdpa']:9.1f} "
              f"{r['t_sdpa']/r['t_kernel']:7.2f}x "
              f"{r['max_abs']:9.2e} {r['mean_rel']:9.2e} "
              f"{r['mem_ratio']:5.1f}x")


if __name__ == "__main__":
    main()
