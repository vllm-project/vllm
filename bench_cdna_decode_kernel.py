"""Kernel-level microbench for CDNA INT8 / INT4 paged decode.

Drives torch.ops._C.pth_decode_int{8,4}_cdna directly so we can get a
number for INT4 without needing the (still-missing) end-to-end wiring.
Mirrors the shape parametrization used in
tests/kernels/attention/test_cdna_int{4,8}_decode.py.
"""
import argparse
import time

import torch
import vllm._C  # noqa: F401 — registers torch.ops._C.pth_decode_int{8,4}_cdna

NUM_WARMUP = 20
NUM_ITERS = 100


def _q_int8(t_fp):
    absmax = t_fp.abs().amax(dim=-1).clamp_min(1e-8)
    scale = (absmax / 127.0).to(torch.float32)
    q = (t_fp.to(torch.float32) / scale.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def _q_int4(t_fp):
    # Per-token-head INT4 with zero-point. Pack two nibbles per byte; store
    # (scale, zp) steganographed into a single fp32 (low 4 bits = zp).
    n, h, d = t_fp.shape
    assert d % 2 == 0
    fmin = t_fp.amin(dim=-1).to(torch.float32)
    fmax = t_fp.amax(dim=-1).to(torch.float32)
    scale = ((fmax - fmin) / 15.0).clamp_min(1e-8)
    zp_f = (-fmin / scale).round().clamp(0, 15)
    zp = zp_f.to(torch.int32)
    q = (t_fp.to(torch.float32) / scale.unsqueeze(-1) + zp_f.unsqueeze(-1)).round().clamp(0, 15).to(torch.int32)
    # Pack two nibbles per byte.
    lo = q[..., 0::2]
    hi = q[..., 1::2]
    packed_bytes = (lo | (hi << 4)).to(torch.uint8)
    # Steganograph scale + zp into one fp32 word per (token, kv_head).
    sb = scale.view(torch.int32)
    packed_scale_zp = ((sb & ~0xF) | zp).view(torch.float32)
    return packed_bytes, packed_scale_zp


def run_int8(head_size, seq_len, num_q_heads, num_kv_heads, dtype):
    device = "cuda"
    block_size = 16
    nblk = (seq_len + block_size - 1) // block_size
    nblk_total = max(8, nblk * 2)

    q = (0.1 * torch.randn(1, num_q_heads, head_size, device=device, dtype=dtype))
    k_fp = (0.1 * torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype))
    v_fp = (0.1 * torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype))
    k_q, k_sc = _q_int8(k_fp)
    v_q, v_sc = _q_int8(v_fp)

    k_cache = torch.zeros(nblk_total, block_size, num_kv_heads, head_size, dtype=torch.int8, device=device)
    v_cache = torch.zeros_like(k_cache)
    k_scale_cache = torch.zeros(nblk_total, block_size, num_kv_heads, dtype=torch.float32, device=device)
    v_scale_cache = torch.zeros_like(k_scale_cache)
    block_table = torch.arange(nblk, dtype=torch.int32, device=device).view(1, nblk)
    for t in range(seq_len):
        blk, slot = t // block_size, t % block_size
        k_cache[blk, slot] = k_q[t]
        v_cache[blk, slot] = v_q[t]
        k_scale_cache[blk, slot] = k_sc[t]
        v_scale_cache[blk, slot] = v_sc[t]
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    sm_scale = head_size ** -0.5

    def call():
        torch.ops._C.pth_decode_int8_cdna(out, q, k_cache, v_cache,
                                          k_scale_cache, v_scale_cache,
                                          block_table, seq_lens, sm_scale)

    for _ in range(NUM_WARMUP):
        call()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NUM_ITERS):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / NUM_ITERS


def run_int4(head_size, seq_len, num_q_heads, num_kv_heads, dtype):
    assert head_size == 128, "INT4 kernel only supports HS=128"
    device = "cuda"
    block_size = 16
    nblk = (seq_len + block_size - 1) // block_size
    nblk_total = max(8, nblk * 2)

    q = (0.1 * torch.randn(1, num_q_heads, head_size, device=device, dtype=dtype))
    k_fp = (0.1 * torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype))
    v_fp = (0.1 * torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype))
    k_q, k_sc_packed = _q_int4(k_fp)
    v_q, v_sc_packed = _q_int4(v_fp)

    bytes_per_row = head_size // 2
    k_cache = torch.zeros(nblk_total, block_size, num_kv_heads, bytes_per_row, dtype=torch.uint8, device=device)
    v_cache = torch.zeros_like(k_cache)
    k_scale_cache = torch.zeros(nblk_total, block_size, num_kv_heads, dtype=torch.float32, device=device)
    v_scale_cache = torch.zeros_like(k_scale_cache)
    block_table = torch.arange(nblk, dtype=torch.int32, device=device).view(1, nblk)
    for t in range(seq_len):
        blk, slot = t // block_size, t % block_size
        k_cache[blk, slot] = k_q[t]
        v_cache[blk, slot] = v_q[t]
        k_scale_cache[blk, slot] = k_sc_packed[t]
        v_scale_cache[blk, slot] = v_sc_packed[t]
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    sm_scale = head_size ** -0.5

    def call():
        torch.ops._C.pth_decode_int4_cdna(out, q, k_cache, v_cache,
                                          k_scale_cache, v_scale_cache,
                                          block_table, seq_lens, sm_scale)

    for _ in range(NUM_WARMUP):
        call()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NUM_ITERS):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / NUM_ITERS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="default")
    args = p.parse_args()

    # (head_size, seq_len, num_q, num_kv, label)
    shapes = [
        (128,  128,  32,  8, "GQA32/8 hs128 sl128"),
        (128, 1024,  32,  8, "GQA32/8 hs128 sl1024"),
        (128, 8192,  32,  8, "GQA32/8 hs128 sl8192"),
        ( 64,  128,  32,  8, "GQA32/8 hs64  sl128"),
        ( 64, 1024,  32,  8, "GQA32/8 hs64  sl1024"),
        ( 64, 8192,  32,  8, "GQA32/8 hs64  sl8192"),
    ]
    print(f"{'shape':28s}  {'INT8 (us)':>10s}  {'INT4 (us)':>10s}")
    print("-" * 56)
    for hs, sl, nq, nkv, label in shapes:
        t8 = run_int8(hs, sl, nq, nkv, torch.bfloat16) * 1e6
        if hs == 128:
            t4 = run_int4(hs, sl, nq, nkv, torch.bfloat16) * 1e6
            t4s = f"{t4:10.2f}"
        else:
            t4s = "      n/a"
        print(f"{label:28s}  {t8:10.2f}  {t4s}")


if __name__ == "__main__":
    main()
