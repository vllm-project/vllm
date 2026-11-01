# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA3 paged attention on mixed decode+prefill batches: one varlen call vs a
decode call plus a prefill call.

A single call with num_splits=0 lets FA3 size split-KV from max_seqlen_q, so a
large prefill chunk forces one split and a long decode row is read by one CTA.
"""

import argparse
import functools
import statistics

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.platforms import current_platform
from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata

WARMUP = 10
K = 1024

# (name, decode seq_lens, [(prefill query_len, prefill seq_len)])
CASES = [
    ("1x278K+30x8K dec, 2K chunk", [278 * K] + [8 * K] * 30, [(2 * K, 2 * K)]),
    ("1x278K+30x8K dec, 8K chunk", [278 * K] + [8 * K] * 30, [(8 * K, 8 * K)]),
    ("1x128K+30x4K dec, 2K chunk", [128 * K] + [4 * K] * 30, [(2 * K, 2 * K)]),
    ("1x32K+30x2K dec, 2K chunk", [32 * K] + [2 * K] * 30, [(2 * K, 2 * K)]),
    ("1x16K+30x2K dec, 2K chunk", [16 * K] + [2 * K] * 30, [(2 * K, 2 * K)]),
    ("1x8K+30x1K dec, 2K chunk", [8 * K] + [1 * K] * 30, [(2 * K, 2 * K)]),
    ("1x8K+30x1K dec, 8K chunk", [8 * K] + [1 * K] * 30, [(8 * K, 8 * K)]),
    ("1x4K+30x1K dec, 2K chunk", [4 * K] + [1 * K] * 30, [(2 * K, 2 * K)]),
    ("16x4K dec, 2K chunk", [4 * K] * 16, [(2 * K, 2 * K)]),
    ("31x8K dec, 8K chunk", [8 * K] * 31, [(8 * K, 8 * K)]),
    ("31x8K dec, 2K chunk @32K", [8 * K] * 31, [(2 * K, 32 * K)]),
    ("64x1K dec, 2K chunk", [1 * K] * 64, [(2 * K, 2 * K)]),
    ("128x2K dec, 2K chunk", [2 * K] * 128, [(2 * K, 2 * K)]),
    ("8x512 dec, 2K chunk", [512] * 8, [(2 * K, 2 * K)]),
    ("4x1K dec, 8K chunk @32K", [1 * K] * 4, [(8 * K, 32 * K)]),
    ("16x4K dec, 4x512 chunks", [4 * K] * 16, [(512, 4 * K)] * 4),
]


class Batch:
    def __init__(self, args, decode_lens, prefills, device):
        self.args = args
        q_lens = [1] * len(decode_lens) + [q for q, _ in prefills]
        seq_lens = decode_lens + [s for _, s in prefills]
        self.num_decodes = len(decode_lens)
        self.max_seq_len = max(seq_lens)
        pages = [-(-s // args.block_size) for s in seq_lens]
        shape = (sum(pages), args.block_size, args.num_kv_heads, args.head_size)
        self.k_cache = torch.randn(shape, dtype=args.dtype, device=device)
        self.v_cache = torch.randn_like(self.k_cache)
        self.block_table = torch.zeros(
            len(seq_lens), max(pages), dtype=torch.int32, device=device
        )
        start = 0
        for i, n in enumerate(pages):
            self.block_table[i, :n] = torch.arange(start, start + n)
            start += n
        cu = [0]
        for q in q_lens:
            cu.append(cu[-1] + q)
        self.num_tokens = cu[-1]
        self.q = torch.randn(
            self.num_tokens,
            args.num_q_heads,
            args.head_size,
            dtype=args.dtype,
            device=device,
        )
        self.out = torch.empty_like(self.q)
        cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
        seq_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        nd = self.num_decodes
        nt = cu[nd]
        # (token slice, request slice, cu_seqlens_q, seqused_k, max_seqlen_q)
        self.parts = {
            "one": [(slice(None), slice(None), cu_t, seq_t, max(q_lens))],
            "split": [
                (slice(0, nt), slice(0, nd), cu_t[: nd + 1], seq_t[:nd], 1),
                (
                    slice(nt, None),
                    slice(nd, None),
                    cu_t[nd:] - nt,
                    seq_t[nd:],
                    max(q_lens[nd:]),
                ),
            ],
        }
        self.sched = {
            (arm, i, splits): self._sched(p, splits)
            for arm, parts in self.parts.items()
            for i, p in enumerate(parts)
            for splits in (0, 32)
        }

    def _sched(self, part, num_splits):
        _, reqs, cu, seqused, max_q = part
        a = self.args
        return get_scheduler_metadata(
            batch_size=seqused.shape[0],
            max_seqlen_q=max_q,
            max_seqlen_k=self.max_seq_len,
            num_heads_q=a.num_q_heads,
            num_heads_kv=a.num_kv_heads,
            headdim=a.head_size,
            cache_seqlens=seqused,
            qkv_dtype=a.dtype,
            cu_seqlens_q=cu,
            page_size=a.block_size,
            causal=True,
            num_splits=num_splits,
        )

    def run(self, arm, num_splits=0):
        for i, (toks, reqs, cu, seqused, max_q) in enumerate(self.parts[arm]):
            flash_attn_varlen_func(
                q=self.q[toks],
                k=self.k_cache,
                v=self.v_cache,
                out=self.out[toks],
                cu_seqlens_q=cu,
                max_seqlen_q=max_q,
                seqused_k=seqused,
                max_seqlen_k=self.max_seq_len,
                softmax_scale=self.args.head_size**-0.5,
                causal=True,
                block_table=self.block_table[reqs],
                scheduler_metadata=self.sched[(arm, i, num_splits)],
                num_splits=num_splits,
                fa_version=3,
            )


def peak_extra_mib(fn):
    fn()
    torch.accelerator.synchronize()
    base = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()
    fn()
    torch.accelerator.synchronize()
    return (torch.accelerator.max_memory_allocated() - base) / 2**20


def bench_us(fn, use_cuda_graph):
    for _ in range(WARMUP):
        fn()
    torch.accelerator.synchronize()
    times = bench_gpu_time_with_cupti(
        fn, cold_l2_cache=True, use_cuda_graph=use_cuda_graph
    )
    return statistics.median(times) * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-q-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--eager",
        action="store_true",
        help="time eager calls (adds the host launch gap between the two "
        "split calls) instead of CUDA graph replays",
    )
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    device = torch.device("cuda")

    print(
        f"{current_platform.get_device_name()}, hq={args.num_q_heads} "
        f"hkv={args.num_kv_heads} d={args.head_size} page={args.block_size} "
        f"{args.dtype}, {'eager' if args.eager else 'CUDA graph'}"
    )
    header = (
        f"{'case':<28}{'one (us)':>10}{'split (us)':>12}{'speedup':>9}"
        f"{'one+32spl (us)':>16}{'extra MiB one/split/32spl':>28}"
    )
    print(header)
    for name, decode_lens, prefills in CASES:
        b = Batch(args, decode_lens, prefills, device)
        b.run("one")
        ref = b.out.clone()
        b.run("split")
        torch.testing.assert_close(b.out, ref, atol=2e-2, rtol=2e-2)

        arms = [("one", 0), ("split", 0), ("one", 32)]
        runs = [functools.partial(b.run, a, s) for a, s in arms]
        us = [bench_us(run, not args.eager) for run in runs]
        mib = [peak_extra_mib(run) for run in runs]
        print(
            f"{name:<28}{us[0]:>10.1f}{us[1]:>12.1f}{us[0] / us[1]:>8.2f}x"
            f"{us[2]:>16.1f}{'/'.join(f'{m:.0f}' for m in mib):>28}"
        )
        del b, runs
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
