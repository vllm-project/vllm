# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-1 microbenchmark for the ReplaySSM Mamba2 AR-decode kernels.

Compares the stored-state baseline ``selective_state_update`` against the
ReplaySSM replay kernels ("output_only" / "state_and_output") for a single
decode step, at batch size 1, on representative per-layer Mamba2 shapes.

Reports per-step latency (averaged over a full ring window so flush steps are
amortized) and the analytical per-step HBM traffic for the recurrent state +
ring buffer. The core claim is the write-traffic reduction: the baseline writes
the full SSM state every step; ReplaySSM writes it only once per ``buffer_len``
steps and otherwise appends the (much smaller) input buffer.

Usage:
    python benchmarks/replayssm/bench_decode_ssu.py
    python benchmarks/replayssm/bench_decode_ssu.py --buffer-len 16 --dtype bfloat16
"""

import argparse

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_output_only import (  # noqa: E501
    selective_state_update_replayssm_output_only,
)
from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_state_and_output import (  # noqa: E501
    selective_state_update_replayssm_state_and_output,
)

# (name, nheads, headdim, dstate, ngroups) — representative hybrid-SSM configs.
CONFIGS = [
    ("mamba2-small", 64, 64, 128, 1),
    ("mamba2-ng8", 80, 64, 128, 8),
    ("mamba2-large", 128, 64, 128, 8),
]


def _bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def analytical_traffic(nheads, headdim, dstate, ngroups, buffer_len, route, itype):
    """Per-step HBM bytes (steady state) for state + buffer, batch=1.

    The state read happens every step in all paths (the checkpoint readout needs
    the current C). The difference is the *write* side.
    """
    eb = _bytes(itype)
    fb = _bytes(torch.float32)
    state_elems = nheads * headdim * dstate
    # Baseline: read state + write state, every step.
    base = state_elems * eb * 2
    # Replay: read state every step; write state only on flush (1/buffer_len);
    # append x_cache (nheads*headdim) + dt_cache (nheads, fp32) + B_cache
    # (ngroups*dstate) every non-flush step; read those back over the window.
    buf_append = (nheads * headdim) * eb + nheads * fb + (ngroups * dstate) * eb
    # Window reads: x_cache + B_cache scale with write_pos; average ~buffer_len/2.
    avg_pos = (buffer_len - 1) / 2.0
    win_read = ((nheads * headdim) * eb + (ngroups * dstate) * eb) * avg_pos
    state_write = state_elems * eb / buffer_len  # amortized flush write
    replay = state_elems * eb + state_write + buf_append + win_read
    return base, replay


def _alloc(batch, nheads, ngroups, headdim, dstate, buffer_len, dev, itype):
    x_cache = torch.zeros(batch, nheads, buffer_len, headdim, device=dev, dtype=itype)
    dt_cache = torch.zeros(batch, nheads, buffer_len, device=dev, dtype=torch.float32)
    B_cache = torch.zeros(batch, ngroups, buffer_len, dstate, device=dev, dtype=itype)
    write_pos = torch.zeros(batch, dtype=torch.int32, device=dev)
    return x_cache, dt_cache, B_cache, write_pos


def bench_step(fn, n_warmup=20, n_iter=200):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter * 1e3  # microseconds/step


def bench_cudagraph(fn, n_warmup=5, n_iter=500):
    """Per-step latency with launch overhead removed (vLLM's decode regime).

    Captures a single step into a CUDA graph and replays it, so the measurement
    reflects kernel execution (memory/compute) rather than Python+launch cost.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter * 1e3  # microseconds/step


def run(nheads, headdim, dstate, ngroups, buffer_len, itype, dev, batch=1):
    state = torch.randn(batch, nheads, headdim, dstate, dtype=itype, device=dev)
    A = (
        (-torch.rand(nheads, device=dev) - 1.0)
        .view(nheads, 1, 1)
        .expand(nheads, headdim, dstate)
    )
    dt_bias = (
        (torch.rand(nheads, device=dev) - 4.0).view(nheads, 1).expand(nheads, headdim)
    )
    D = torch.randn(nheads, headdim, device=dev)
    x = torch.randn(batch, nheads, headdim, device=dev, dtype=itype)
    dt = (
        torch.randn(batch, nheads, device=dev, dtype=itype)
        .unsqueeze(-1)
        .expand(batch, nheads, headdim)
    )
    B = torch.randn(batch, ngroups, dstate, device=dev, dtype=itype)
    C = torch.randn(batch, ngroups, dstate, device=dev, dtype=itype)
    out = torch.empty_like(x)

    def baseline():
        selective_state_update(
            state, x, dt, A, B, C, D=D, dt_bias=dt_bias, dt_softplus=True, out=out
        )

    # Cycle write_pos through the full window so the flush step is amortized.
    pos = [0]

    def make_replay(route):
        xc, dtc, Bc, wp = _alloc(
            batch, nheads, ngroups, headdim, dstate, buffer_len, dev, itype
        )
        bc = torch.empty(batch, ngroups, buffer_len, device=dev, dtype=torch.float32)

        is_flush = torch.zeros(batch, dtype=torch.int8, device=dev)

        def step():
            wp.fill_(pos[0])
            is_flush.fill_(1 if pos[0] == buffer_len - 1 else 0)
            if route == "output_only":
                selective_state_update_replayssm_output_only(
                    state,
                    x,
                    dt,
                    A,
                    B,
                    C,
                    D=D,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    x_cache=xc,
                    dt_cache=dtc,
                    B_cache=Bc,
                    bc_pre=bc,
                    write_pos=wp,
                    is_flush=is_flush,
                    max_cache_len=buffer_len,
                    out=out,
                )
            else:
                selective_state_update_replayssm_state_and_output(
                    state,
                    x,
                    dt,
                    A,
                    B,
                    C,
                    D=D,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    x_cache=xc,
                    dt_cache=dtc,
                    B_cache=Bc,
                    write_pos=wp,
                    is_flush=is_flush,
                    max_cache_len=buffer_len,
                    out=out,
                )
            pos[0] = (pos[0] + 1) % buffer_len

        return step

    base_us = bench_step(baseline)
    oo_us = bench_step(make_replay("output_only"))
    so_us = bench_step(make_replay("state_and_output"))
    base_b, oo_b = analytical_traffic(
        nheads, headdim, dstate, ngroups, buffer_len, "output_only", itype
    )
    return base_us, oo_us, so_us, base_b, oo_b


def run_cudagraph(nheads, headdim, dstate, ngroups, buffer_len, itype, dev, batch):
    """CUDA-graph per-step latency for a representative non-flush step.

    write_pos is fixed at buffer_len//2 (average ring depth); is_flush=0. This is
    the common-case step (15/16 at buffer_len=16) and the regime vLLM decode runs
    in (graph-captured, launch overhead removed).
    """
    state = torch.randn(batch, nheads, headdim, dstate, dtype=itype, device=dev)
    A = (
        (-torch.rand(nheads, device=dev) - 1.0)
        .view(nheads, 1, 1)
        .expand(nheads, headdim, dstate)
    )
    dt_bias = (
        (torch.rand(nheads, device=dev) - 4.0).view(nheads, 1).expand(nheads, headdim)
    )
    D = torch.randn(nheads, headdim, device=dev)
    x = torch.randn(batch, nheads, headdim, device=dev, dtype=itype)
    dt = (
        torch.randn(batch, nheads, device=dev, dtype=itype)
        .unsqueeze(-1)
        .expand(batch, nheads, headdim)
    )
    B = torch.randn(batch, ngroups, dstate, device=dev, dtype=itype)
    C = torch.randn(batch, ngroups, dstate, device=dev, dtype=itype)
    out = torch.empty_like(x)
    xc, dtc, Bc, wp = _alloc(
        batch, nheads, ngroups, headdim, dstate, buffer_len, dev, itype
    )
    bc = torch.empty(batch, ngroups, buffer_len, device=dev, dtype=torch.float32)
    wp.fill_(buffer_len // 2)
    is_flush = torch.zeros(batch, dtype=torch.int8, device=dev)

    def baseline():
        selective_state_update(
            state, x, dt, A, B, C, D=D, dt_bias=dt_bias, dt_softplus=True, out=out
        )

    def oo():
        selective_state_update_replayssm_output_only(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            x_cache=xc,
            dt_cache=dtc,
            B_cache=Bc,
            bc_pre=bc,
            write_pos=wp,
            is_flush=is_flush,
            max_cache_len=buffer_len,
            out=out,
        )

    def so():
        selective_state_update_replayssm_state_and_output(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            x_cache=xc,
            dt_cache=dtc,
            B_cache=Bc,
            write_pos=wp,
            is_flush=is_flush,
            max_cache_len=buffer_len,
            out=out,
        )

    return (bench_cudagraph(baseline), bench_cudagraph(oo), bench_cudagraph(so))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer-len", type=int, default=16)
    ap.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    args = ap.parse_args()
    itype = getattr(torch, args.dtype)
    dev = "cuda"

    print(
        f"\nReplaySSM batch-1 decode microbench  (dtype={args.dtype}, "
        f"buffer_len={args.buffer_len}, device={torch.cuda.get_device_name()})\n"
    )
    hdr = (
        f"{'config':<16}{'base us':>9}{'oo us':>9}{'so us':>9}"
        f"{'state KB/step':>15}{'oo KB/step':>12}{'write-traffic':>15}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, nh, hd, ds, ng in CONFIGS:
        base_us, oo_us, so_us, base_b, oo_b = run(
            nh, hd, ds, ng, args.buffer_len, itype, dev
        )
        # Write-traffic reduction: baseline writes |S|/step; replay writes
        # |S|/buffer_len/step (amortized) + small buffer appends.
        eb = _bytes(itype)
        state_w = nh * hd * ds * eb
        replay_w = state_w / args.buffer_len + (
            (nh * hd) * eb + nh * 4 + (ng * ds) * eb
        )
        print(
            f"{name:<16}{base_us:>9.2f}{oo_us:>9.2f}{so_us:>9.2f}"
            f"{base_b / 1024:>15.1f}{oo_b / 1024:>12.1f}"
            f"{state_w / replay_w:>13.1f}x"
        )
    print("\noo = output_only route, so = state_and_output route")
    print("state KB/step = baseline total state HBM (read+write) per step")
    print(
        "write-traffic = baseline state write / replay state write+append "
        "(per step, amortized over the ring window)"
    )

    # Batch sweep: at batch-1 the isolated kernel is launch/overhead-bound, so the
    # IO reduction does not show up as latency. Sweep batch to find where the
    # per-step SSM update becomes memory-bound and the replay path crosses over.
    name, nh, hd, ds, ng = CONFIGS[-1]
    print(f"\nBatch sweep ({name}, dtype={args.dtype}, buffer_len={args.buffer_len}):")
    sweep_hdr = (
        f"{'batch':>6}{'base us':>10}{'oo us':>10}{'so us':>10}"
        f"{'oo speedup':>12}{'so speedup':>12}"
    )
    print(sweep_hdr)
    print("-" * len(sweep_hdr))
    for batch in [1, 8, 32, 128, 256, 512]:
        base_us, oo_us, so_us, _, _ = run(
            nh, hd, ds, ng, args.buffer_len, itype, dev, batch=batch
        )
        print(
            f"{batch:>6}{base_us:>10.2f}{oo_us:>10.2f}{so_us:>10.2f}"
            f"{base_us / oo_us:>11.2f}x{base_us / so_us:>11.2f}x"
        )

    # CUDA-graph sweep: launch overhead removed (vLLM's actual decode regime),
    # representative non-flush step. This isolates the memory-bound kernel time.
    print(f"\nCUDA-graph sweep ({name}, non-flush step, launch overhead removed):")
    cg_hdr = (
        f"{'batch':>6}{'base us':>10}{'oo us':>10}{'so us':>10}"
        f"{'oo speedup':>12}{'so speedup':>12}"
    )
    print(cg_hdr)
    print("-" * len(cg_hdr))
    for batch in [1, 8, 32, 128]:
        base_us, oo_us, so_us = run_cudagraph(
            nh, hd, ds, ng, args.buffer_len, itype, dev, batch=batch
        )
        print(
            f"{batch:>6}{base_us:>10.2f}{oo_us:>10.2f}{so_us:>10.2f}"
            f"{base_us / oo_us:>11.2f}x{base_us / so_us:>11.2f}x"
        )
    print()


if __name__ == "__main__":
    main()
