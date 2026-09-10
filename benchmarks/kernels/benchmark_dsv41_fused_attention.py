# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashMLA fused sparse attention vs the split-KV decode pipeline (DSv4.1).

Decode: topk_swa=128 (V4 584 B cache) + topk_extra=512, h_q=64. The unfused
pipeline is Q RoPE (torch) + flash_mla_with_kvcache + fused_inv_rope_fp8_quant.
Run: .venv/bin/python benchmarks/kernels/benchmark_dsv41_fused_attention.py
"""

import argparse

import torch

import vllm.v1.attention.ops.flashmla as fm
from vllm.models.deepseek_v4_1.common.ops import (
    fused_inv_rope_fp8_quant,
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4_1.common.ops.fused_layout import permute_q_to_fused
from vllm.utils.math_utils import round_up

HEAD_DIM, ROPE_DIM, NOPE_DIM = 512, 64, 448


def make_cos_sin_cache(max_pos, device):
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, ROPE_DIM, 2, device=device).float() / ROPE_DIM)
    )
    freqs = torch.outer(torch.arange(max_pos, device=device).float(), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], -1)


def rope_gptj(x, positions, cos_sin):
    cs = cos_sin[positions].float()
    cos, sin = cs[:, None, :32], cs[:, None, 32:]
    r = x[..., NOPE_DIM:].float().unflatten(-1, (32, 2))
    x0, x1 = r[..., 0], r[..., 1]
    rot = torch.stack([x0 * cos - x1 * sin, x1 * cos + x0 * sin], -1).flatten(-2)
    return torch.cat([x[..., :NOPE_DIM].float(), rot], -1).to(x.dtype)


def build_v4_cache(k, block_size):
    n = k.shape[0]
    num_blocks = (n + block_size - 1) // block_size + 1
    cache = torch.zeros(
        num_blocks, round_up(block_size * 584, 576), dtype=torch.uint8, device=k.device
    )
    quantize_and_insert_k_cache(
        k, cache, torch.arange(n, device=k.device), block_size=block_size
    )
    return cache[:, : block_size * 584].unflatten(1, (block_size, 1, 584))


def random_indices(s_q, topk, num_slots, device):
    idx = torch.randint(0, num_slots, (s_q, topk), device=device, dtype=torch.int32)
    lens = torch.full((s_q,), topk, device=device, dtype=torch.int32)
    return idx, lens


def time_fn(fn, iters=50, warmup=10, graph=False):
    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        fn = g.replay
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def bench_decode(s_q, graph, device, topk_extra):
    h_q, groups, scale = 64, 8, HEAD_DIM**-0.5
    cos_sin = make_cos_sin_cache(65536, device)
    positions = torch.randint(0, 65536, (s_q,), device=device)
    q = torch.randn(s_q, h_q, HEAD_DIM, device=device, dtype=torch.bfloat16)
    q_fused = permute_q_to_fused(q)
    n_swa = s_q * 128 + 128
    swa = build_v4_cache(
        torch.randn(n_swa, HEAD_DIM, device=device, dtype=torch.bfloat16), 32
    )
    extra = build_v4_cache(
        torch.randn(65536, HEAD_DIM, device=device, dtype=torch.bfloat16), 128
    )
    swa_idx, swa_len = random_indices(s_q, 128, n_swa, device)
    ex_idx, ex_len = random_indices(s_q, topk_extra, 65536, device)
    sink = torch.zeros(h_q, device=device)
    sched = fm.FlashMLASchedMeta()
    pos32 = positions.to(torch.int32)

    def fused():
        fm.flash_mla_fused_sparse_decode(
            q_fused,
            swa,
            swa_idx,
            scale,
            pos32,
            cos_sin,
            groups,
            attn_sink=sink,
            topk_length=swa_len,
            extra_k_cache=extra,
            extra_indices=ex_idx,
            extra_topk_length=ex_len,
        )

    def splitkv_attn(q_in):
        return fm.flash_mla_with_kvcache(
            q=q_in.unsqueeze(1),
            k_cache=swa,
            block_table=None,
            head_dim_v=HEAD_DIM,
            tile_scheduler_metadata=sched,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_idx.view(s_q, 1, -1),
            topk_length=swa_len,
            softmax_scale=scale,
            attn_sink=sink,
            extra_k_cache=extra,
            extra_indices_in_kvcache=ex_idx.view(s_q, 1, -1),
            extra_topk_length=ex_len,
        )[0]

    def splitkv_total():
        o = splitkv_attn(rope_gptj(q, positions, cos_sin))
        fused_inv_rope_fp8_quant(
            o.squeeze(1),
            positions,
            cos_sin,
            n_groups=groups,
            heads_per_group=8,
            quant_group_size=32,
            tma_aligned_scales=True,
        )

    splitkv_attn(q)  # plans the tile scheduler once
    return (
        time_fn(fused, graph=graph),
        time_fn(lambda: splitkv_attn(q), graph=graph),
        time_fn(splitkv_total, graph=graph),
    )


def bench_prefill(s_q, device):
    h_q, groups, topk, s_kv, scale = 64, 8, 640, 8192, HEAD_DIM**-0.5
    cos_sin = make_cos_sin_cache(65536, device)
    positions = torch.randint(0, 65536, (s_q,), device=device)
    q = torch.randn(s_q, h_q, HEAD_DIM, device=device, dtype=torch.bfloat16)
    q_fused = permute_q_to_fused(q)
    kv = torch.randn(s_kv, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    idx, lens = random_indices(s_q, topk, s_kv, device)
    pos32 = positions.to(torch.int32)

    def fused():
        fm.flash_mla_fused_sparse_prefill(
            q_fused,
            kv,
            idx.view(s_q, 1, -1),
            scale,
            pos32,
            cos_sin,
            groups,
            topk_length=lens,
        )

    def unfused():
        o = fm.flash_mla_sparse_fwd(
            rope_gptj(q, positions, cos_sin),
            kv,
            idx.view(s_q, 1, -1),
            scale,
            topk_length=lens,
        )[0]
        fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin,
            n_groups=groups,
            heads_per_group=8,
            quant_group_size=32,
            tma_aligned_scales=True,
        )

    return time_fn(fused), time_fn(unfused)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decode-s-q", nargs="+", type=int, default=[1, 6, 16, 64, 256, 1024]
    )
    parser.add_argument("--prefill-s-q", nargs="+", type=int, default=[184, 2123])
    parser.add_argument("--topk-extra", type=int, default=512)
    parser.add_argument("--cudagraph", action="store_true")
    args = parser.parse_args()
    ok, reason = fm.is_flashmla_fused_sparse_supported()
    if not ok:
        raise SystemExit(reason)
    device = torch.device("cuda")
    torch.manual_seed(0)
    print(
        f"decode (topk_swa=128, topk_extra={args.topk_extra}, "
        f"cudagraph={args.cudagraph})"
    )
    print("  s_q  fused_us  splitkv_attn_us  splitkv_total_us")
    for s_q in args.decode_s_q:
        f, a, t = bench_decode(s_q, args.cudagraph, device, args.topk_extra)
        print(f"{s_q:>5d} {f:9.1f} {a:16.1f} {t:17.1f}")
    print("prefill (topk=640)")
    print("  s_q  fused_us  sparse_fwd_total_us")
    for s_q in args.prefill_s_q:
        f, u = bench_prefill(s_q, device)
        print(f"{s_q:>5d} {f:9.1f} {u:20.1f}")


if __name__ == "__main__":
    main()
