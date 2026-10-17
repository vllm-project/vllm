# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the fused Muse-Spark QK-norm + RoPE + KV-insert kernel vs unfused.

``tuned_us`` forces the cp.async schedule ``fused_qk_norm_rope`` only auto-selects
on SM 9.0; ``unfused_us`` is what production runs.
"""

import argparse
import itertools

import torch

import vllm._custom_ops as ops

HEAD_DIM = 64
DEFAULT_TOKENS = [1, 8, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536]


def make_cos_sin_cache(max_pos, rotary_dim, base, device):
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


class Case:
    """Everything a single (num_tokens, config) measurement needs."""

    def __init__(self, num_tokens, nq, nkv, dtype, kv_cache_dtype, device):
        self.num_tokens = num_tokens
        self.nq, self.nkv = nq, nkv
        self.dtype = dtype
        self.kv_cache_dtype = kv_cache_dtype
        self.eps = 1e-5
        self.block_size = 128
        self.rotary_dim = HEAD_DIM

        self.qsz, self.kvsz = nq * HEAD_DIM, nkv * HEAD_DIM
        self.row = self.qsz + 2 * self.kvsz + self.qsz  # [q | k | v | o_gate]
        self.qkv = torch.randn(num_tokens, self.row, dtype=dtype, device=device)
        self.pristine = self.qkv.clone()

        max_pos = 32768
        self.cos_sin = make_cos_sin_cache(max_pos, self.rotary_dim, 500_000.0, device)
        self.positions = torch.randint(
            0, max_pos, (num_tokens,), dtype=torch.int64, device=device
        )

        num_blocks = (num_tokens + self.block_size - 1) // self.block_size + 1
        storage = torch.uint8 if kv_cache_dtype != "auto" else dtype
        self.kv_cache = torch.zeros(
            num_blocks,
            nkv,
            self.block_size,
            2 * HEAD_DIM,
            dtype=storage,
            device=device,
        )
        self.k_cache, self.v_cache = self.kv_cache.transpose(1, 2).split(
            HEAD_DIM, dim=-1
        )
        self.slot_mapping = torch.randperm(
            num_blocks * self.block_size, dtype=torch.int64, device=device
        )[:num_tokens]
        self.k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
        self.v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
        self.ones = torch.ones(HEAD_DIM, dtype=dtype, device=device)
        self.q_fp8 = torch.empty(
            num_tokens, self.qsz, dtype=torch.float8_e4m3fn, device=device
        )
        self.q_scale = torch.tensor(0.37, dtype=torch.float32, device=device)

    def views(self):
        q, k, v, _ = self.qkv.split([self.qsz, self.kvsz, self.kvsz, self.qsz], dim=-1)
        return q, k, v

    def fused(self, use_rope=True, quant_query=False):
        ops.fused_muse_spark_qknorm_rope_kv_insert(
            self.qkv,
            self.positions,
            self.nq,
            self.nkv,
            HEAD_DIM,
            self.eps,
            cos_sin_cache=self.cos_sin if use_rope else None,
            rotary_dim=self.rotary_dim if use_rope else 0,
            slot_mapping=self.slot_mapping,
            kv_cache=self.kv_cache,
            block_size=self.block_size,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=self.k_scale if self.kv_cache_dtype != "auto" else None,
            v_scale=self.v_scale if self.kv_cache_dtype != "auto" else None,
            q_fp8_out=self.q_fp8 if quant_query else None,
            q_scale=self.q_scale if quant_query else None,
        )

    def separate_query_quant(self, use_rope=True):
        self.fused(use_rope)
        ops.scaled_fp8_quant(self.qkv[:, : self.qsz], scale=self.q_scale)

    def _insert(self):
        _, k, v = self.views()
        ops.reshape_and_cache_flash(
            k.view(-1, self.nkv, HEAD_DIM),
            v.view(-1, self.nkv, HEAD_DIM),
            self.k_cache,
            self.v_cache,
            self.slot_mapping,
            self.kv_cache_dtype,
            self.k_scale,
            self.v_scale,
        )

    def unfused_rope(self, token_heads_per_warp: int = -1):
        ops.fused_qk_norm_rope(
            self.qkv,
            self.nq,
            self.nkv,
            self.nkv + self.nq,  # v heads plus the o-gate
            HEAD_DIM,
            self.eps,
            self.ones,
            self.ones,
            self.cos_sin,
            False,
            self.positions,
            token_heads_per_warp,
        )
        self._insert()

    def fused_bytes(self):
        el = self.qkv.element_size()
        cache_el = self.kv_cache.element_size()
        read = (self.nq + 2 * self.nkv) * HEAD_DIM * el
        write = (self.nq + self.nkv) * HEAD_DIM * el
        cache = 2 * self.nkv * HEAD_DIM * cache_el
        return self.num_tokens * (read + write + cache)


def _l2_bytes(device) -> int:
    return torch.cuda.get_device_properties(device).L2_cache_size


def _peak_bandwidth(device) -> float:
    """Measured large-copy bandwidth, in GB/s -- the practical ceiling."""
    n = 1 << 28  # 256 Mi elements = 512 MiB each way at 2 bytes
    a = torch.empty(n, dtype=torch.bfloat16, device=device)
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.accelerator.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    iters = 20
    for _ in range(iters):
        b.copy_(a)
    end.record()
    torch.accelerator.synchronize()
    ms = start.elapsed_time(end) / iters
    return (2 * a.numel() * a.element_size()) / (ms * 1e-3) / 1e9


def time_eager(fn, warmup=20, iters=100) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # us


def time_graph(fn, warmup=10, iters=50, inner=32) -> float:
    """GPU time per call; ``inner`` calls per graph amortize the ~2 us replay."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / (iters * inner) * 1e3  # us


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nq", type=int, default=32, help="query heads per rank")
    p.add_argument("--nkv", type=int, default=4, help="kv heads per rank")
    p.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--tokens", type=int, nargs="*", default=DEFAULT_TOKENS)
    p.add_argument("--csv", default=None)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument(
        "--query-fp8", action="store_true", help="compare fused Q quantization"
    )
    args = p.parse_args()

    device = "cuda"
    dtype = getattr(torch, args.dtype)
    if args.query_fp8:
        if args.kv_cache_dtype != "fp8":
            p.error("--query-fp8 requires --kv-cache-dtype fp8")
        timer = time_eager if args.no_graph else time_graph
        print(f"device: {torch.cuda.get_device_name()}; nq={args.nq}, nkv={args.nkv}")
        print("layer,tokens,separate_us,fused_us,speedup")
        rows = []
        for use_rope, nt in itertools.product([True, False], args.tokens):
            case = Case(nt, args.nq, args.nkv, dtype, args.kv_cache_dtype, device)
            separate_us = timer(lambda c=case, r=use_rope: c.separate_query_quant(r))
            case.qkv.copy_(case.pristine)
            fused_us = timer(lambda c=case, r=use_rope: c.fused(r, quant_query=True))
            row = (
                "rope" if use_rope else "nope",
                nt,
                separate_us,
                fused_us,
                separate_us / fused_us,
            )
            rows.append(row)
            print(f"{row[0]},{nt},{separate_us:.3f},{fused_us:.3f},{row[4]:.3f}")
        if args.csv:
            import csv

            with open(args.csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["layer", "tokens", "separate_us", "fused_us", "speedup"]
                )
                writer.writerows(rows)
        return
    peak = _peak_bandwidth(device)
    print(
        f"device: {torch.cuda.get_device_name()}   measured copy peak: {peak:.0f} GB/s"
    )
    print(
        f"config: nq={args.nq} nkv={args.nkv} head_dim={HEAD_DIM} "
        f"dtype={args.dtype} kv_cache={args.kv_cache_dtype}\n"
    )

    modes = ["rope"]
    l2 = _l2_bytes(device)
    print(
        f"rows marked * have a working set inside the {l2 / 2**20:.0f} MiB L2, "
        "so their bandwidth is not an HBM number\n"
    )
    header = (
        f"{'layer':>5} {'tokens':>7} {'fused_us':>9} {'unfused_us':>11} "
        f"{'tuned_us':>9} {'speedup':>8} {'GB/s':>8} {'SOL%':>6}"
    )
    rows = []
    for mode, use_graph in itertools.product(modes, [False, True]):
        if use_graph and args.no_graph:
            continue
        label = "graph" if use_graph else "eager"
        print(f"── {mode} layer, {label} ──")
        print(header)
        timer = time_graph if use_graph else time_eager
        for nt in args.tokens:
            case = Case(nt, args.nq, args.nkv, dtype, args.kv_cache_dtype, device)
            use_rope = mode == "rope"
            fused_us = timer(lambda c=case, r=use_rope: c.fused(r))
            baseline = case.unfused_rope
            unfused_us = timer(baseline)
            tuned_us = (
                min(timer(lambda c=case, t=t: c.unfused_rope(t)) for t in (4, 8))
                if use_rope
                else unfused_us
            )
            nbytes = case.fused_bytes()
            gbps = nbytes / (fused_us * 1e-6) / 1e9
            sol = 100.0 * gbps / peak
            resident = "*" if nbytes < l2 else " "
            print(
                f"{mode:>5} {nt:>7} {fused_us:>9.2f} {unfused_us:>11.2f} "
                f"{tuned_us:>9.2f} {tuned_us / fused_us:>7.2f}x {gbps:>8.0f} "
                f"{sol:>5.1f}%{resident}"
            )
            rows.append((mode, label, nt, fused_us, unfused_us, tuned_us, gbps, sol))
        print()

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "layer",
                    "timing",
                    "tokens",
                    "fused_us",
                    "unfused_us",
                    "tuned_us",
                    "gbps",
                    "sol",
                ]
            )
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
