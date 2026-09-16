# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashMLA mega attention vs the split-KV decode pipeline (DeepSeek V4.1).

Times the full set of per-layer ops each path runs in a decode step, captured
in a CUDA graph -- eager timing is dominated by launch overhead and flatters
whichever path launches fewer kernels.

    mega:     fused_qnorm_rope_kv_rope_quant_insert (Q pad + KV insert)
              + mega kernel
    split-KV: fused_qnorm_rope_kv_rope_quant_insert (Q RoPE + pad + KV insert)
              + flash_mla_with_kvcache + fused_inv_rope_fp8_quant

Both arms fuse their Q preparation into the KV insert, so that op is charged
to both; the mega arm's Q side is only a zero-pad (nothing at all when the
shard is already at the kernel head count) because its kernel does the Q RoPE,
the inverse RoPE and the FP8 cast.

``--local-heads`` is the variable that decides the outcome: the mega kernel
absorbs work proportional to the live head count while paying for the padded
one, so it wins at 64 live heads and is a wash at 16 (TP4 on a 64-head model).

Both arms use the same KV records so the comparison is kernel-for-kernel:
a V4.1 fp8 (528 B) sliding-window cache and a compressed cache that is either
V4.1 fp8 or V4.1 NVFP4 (288 B) -- FlashMLA's SM100 sparse decode reads the
NVFP4 compressed record in both the fused and unfused kernels.

Run: .venv/bin/python benchmarks/kernels/benchmark_dsv41_mega_attn.py
"""

import argparse

import torch

import vllm.v1.attention.ops.flashmla as fm
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.models.deepseek_v41.common.ops import quantize_and_insert_k_cache
from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
    rope_quant_insert,
)
from vllm.models.deepseek_v41.common.ops.fused_layout import permute_q_to_fused
from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import (
    alloc_mega_attn_output,
    is_flashmla_mega_attn_supported,
)
from vllm.utils.math_utils import round_up

HEAD_DIM, ROPE_DIM = 512, 64
V41_BYTES, V41_FP4_BYTES = 528, 288
SWA_BLOCK, COMPRESSED_BLOCK = 32, 128


def make_cos_sin_cache(max_pos, device):
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, ROPE_DIM, 2, device=device).float() / ROPE_DIM)
    )
    freqs = torch.outer(torch.arange(max_pos, device=device).float(), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], -1)


def empty_paged_cache(num_rows, block_size, bytes_per_token, device):
    """A zeroed paged cache view [num_blocks, block_size, bytes_per_token]."""
    num_blocks = (num_rows + block_size - 1) // block_size + 1
    page = round_up(block_size * bytes_per_token, 512)
    backing = torch.zeros(num_blocks, page, dtype=torch.uint8, device=device)
    return backing, backing.as_strided(
        (num_blocks, block_size, bytes_per_token), (page, bytes_per_token, 1)
    )


def fill_cache(view, bytes_per_token, cos_sin, device):
    n = view.shape[0] * view.shape[1]
    rows = torch.randn(n, HEAD_DIM, device=device, dtype=torch.bfloat16)
    slots = torch.arange(n, dtype=torch.int64, device=device)
    if bytes_per_token == V41_FP4_BYTES:
        # A page-aligned cache has more slots than the cos_sin table has rows;
        # wrap so the insert never indexes past it.
        positions = slots % cos_sin.shape[0]
        rope_quant_insert(rows, positions, cos_sin, view, slots, 1)
    else:
        quantize_and_insert_k_cache(
            rows,
            view.reshape(view.shape[0], -1),
            slots,
            block_size=view.shape[1],
            bytes_per_token=bytes_per_token,
        )


def time_graph(fn, iters=100, warmup=5):
    """Capture ``fn`` in a CUDA graph and time its replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup):
        g.replay()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def bench_decode(s_q, device, topk_extra, extra_bytes, local_heads):
    padded_heads, scale = 64, HEAD_DIM**-0.5
    n_wv_group = padded_heads // 8
    cos_sin = make_cos_sin_cache(65536, device)
    positions = torch.randint(0, 65536, (s_q,), device=device, dtype=torch.int64)
    pos32 = positions.to(torch.int32)

    # Per-step inputs: the wq_b output (local heads) and this step's KV row.
    q_local = torch.randn(
        s_q, local_heads, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    q_fused = permute_q_to_fused(q_local)
    kv = torch.randn(s_q, HEAD_DIM, device=device, dtype=torch.bfloat16)

    n_swa = s_q * 128 + 128
    swa_backing, swa = empty_paged_cache(n_swa, SWA_BLOCK, V41_BYTES, device)
    fill_cache(swa, V41_BYTES, cos_sin, device)
    _, extra = empty_paged_cache(65536, COMPRESSED_BLOCK, extra_bytes, device)
    fill_cache(extra, extra_bytes, cos_sin, device)

    slot_mapping = torch.arange(s_q, dtype=torch.int64, device=device)
    swa_idx = torch.randint(0, n_swa, (s_q, 128), device=device, dtype=torch.int32)
    swa_len = torch.full((s_q,), 128, device=device, dtype=torch.int32)
    ex_idx = torch.randint(
        0, 65536, (s_q, topk_extra), device=device, dtype=torch.int32
    )
    ex_len = torch.full((s_q,), topk_extra, device=device, dtype=torch.int32)
    sink = torch.zeros(padded_heads, device=device)
    sched = fm.FlashMLASchedMeta()
    out = alloc_mega_attn_output(s_q, n_wv_group, device)
    swa4 = swa.unsqueeze(-2)
    extra4 = extra.unsqueeze(-2)
    swa_2d = swa_backing

    def mega():
        # One launch zero-pads the fused-layout Q and inserts the SWA KV.
        q_pad = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q_fused,
            kv,
            swa_2d,
            slot_mapping,
            positions,
            cos_sin,
            0 if padded_heads == local_heads else padded_heads,
            1e-6,
            SWA_BLOCK,
            False,  # apply_q_norm
            True,  # kv_mxfp8
            False,  # apply_q_rope: the mega kernel rotates Q itself
            True,  # is_q_interleaved
        )
        torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_decode(
            q_fused if padded_heads == local_heads else q_pad,
            swa4,
            swa_idx,
            scale,
            HEAD_DIM,
            sink,
            swa_len,
            extra4,
            ex_idx,
            ex_len,
            False,
            0.0,
            pos32,
            False,
            64,
            cos_sin,
            n_wv_group,
            32,
            True,
            True,
            True,
            out.data,
            out.scale,
        )

    def split_kv():
        q_pad = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q_local,
            kv,
            swa_2d,
            slot_mapping,
            positions,
            cos_sin,
            padded_heads,
            1e-6,
            SWA_BLOCK,
            False,
            True,
        )
        o = fm.flash_mla_with_kvcache(
            q=q_pad.unsqueeze(1),
            k_cache=swa4,
            block_table=None,
            head_dim_v=HEAD_DIM,
            tile_scheduler_metadata=sched,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_idx.view(s_q, 1, -1),
            topk_length=swa_len,
            softmax_scale=scale,
            attn_sink=sink,
            extra_k_cache=extra4,
            extra_indices_in_kvcache=ex_idx.view(s_q, 1, -1),
            extra_topk_length=ex_len,
        )[0]
        fused_inv_rope_fp8_quant(
            o.squeeze(1)[:, :local_heads],
            positions,
            cos_sin,
            n_groups=local_heads // 8,
            heads_per_group=8,
            quant_group_size=32,
            tma_aligned_scales=True,
        )

    return time_graph(mega), time_graph(split_kv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decode-s-q",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    )
    parser.add_argument("--topk-extra", type=int, default=512)
    parser.add_argument("--local-heads", type=int, default=16)
    parser.add_argument(
        "--extra-bytes",
        type=int,
        choices=[V41_BYTES, V41_FP4_BYTES],
        default=V41_BYTES,
        help="compressed-cache record: 528 (V4.1 fp8) or 288 (V4.1 NVFP4)",
    )
    args = parser.parse_args()
    ok, reason = is_flashmla_mega_attn_supported()
    if not ok:
        raise SystemExit(reason)
    device = torch.device("cuda")
    torch.manual_seed(0)
    print(
        f"decode, cudagraph (topk_swa=128, topk_extra={args.topk_extra}, "
        f"local_heads={args.local_heads}, extra={args.extra_bytes}B)"
    )
    print(f"{'s_q':>6} {'mega_us':>9} {'split_kv_us':>12} {'speedup':>8}")
    for s_q in args.decode_s_q:
        m, s = bench_decode(
            s_q, device, args.topk_extra, args.extra_bytes, args.local_heads
        )
        print(f"{s_q:>6d} {m:9.1f} {s:12.1f} {s / m:7.2f}x")


if __name__ == "__main__":
    main()
