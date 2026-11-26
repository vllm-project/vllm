# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA prefill short conv: causal_conv1d_fn vs the planar kernel.

Inputs follow the model: q|k|v is a row-strided slice of the merged in_proj
output (row = 3P + H + 2 * 128), decode rows come first, every sequence owns a
scattered slot of a bf16 conv-state pool, decode rows and every other prompt
carry an initial conv state (continued chunked prefills). `stock` is
causal_conv1d_fn with precomputed metadata (as in serving); `stock+copies`
adds the q/k/v `.contiguous()` copies that the Triton chunk path makes of
the stock kernel's column slices (FlashKDA takes them strided, so it pays
only `stock`); `planar-<impl>` is causal_conv1d_planar, labelled with
the kernel that actually ran. Outputs and conv states are compared with the
stock kernel before timing and any difference is reported (count, fraction,
largest distance in bf16 ulps); the `ms` column holds wall-clock medians from
triton's do_bench with L2 flushed between runs, and `host us` is the cost of one
call on the host (launches, descriptors, metadata), measured without
synchronizing: prefill steps run eagerly, so the planar path's three launches
(and, on the TMA path, three descriptor constructions) per layer are paid
there, next to the stock kernel's one launch over precomputed metadata.
`--gpu-time` adds the GPU time of one call (the sum of the path's kernel
durations from torch.profiler, L2 flushed between calls) and the speedup in
it: the planar path's three launches cost about 40 us more host time per
call than the stock kernel's one, so below a few thousand prefill tokens its
wall-clock here is launch-bound (a floor near its host time), while a prefill
step that keeps the host ahead of the GPU pays the GPU time. The dispatch
guard's constants are set on GPU time. `--layout` picks the conv-state pool
layout, SD (vLLM's default) unless given: the per-sequence kernels' cost per
decode row depends on it.

The default shapes cover the prefill-dominated regime (8K prompts), mixed
chunked-prefill steps with decode rows and a decode-heavy step where the
planar kernel loses; the default head counts are the TP1 to TP8 shards of
GLM-5.3-Flash plus 12, whose rows are not 16-byte aligned (register kernel).
"""

import argparse
import time
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.models.glm5next.nvidia.ops.kda_prefill_conv import (
    causal_conv1d_planar,
    planar_conv_impl,
)
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

HEAD_DIM = 128
WIDTH = 4
DEFAULT_BATCHES = [
    "8x8192+194x1",
    "8x8192",
    "1x8192",
    "1x16384+500x1",
    "2x4096+500x1",
    "8x1024+500x1",
    "1x1024+1000x1",
]
DEFAULT_HEADS = [64, 32, 16, 12, 8]
_FLUSH: torch.Tensor | None = None


def parse_batch(spec: str) -> list[int]:
    """`"8x8192+194x1"` -> lengths, decode rows (length 1) first."""
    lens: list[int] = []
    for term in spec.split("+"):
        count, length = (int(v) for v in term.split("x"))
        lens += [length] * count
    return sorted(lens, key=lambda n: n != 1)


def make_inputs(num_heads: int, lens: list[int], dim_first: bool, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dim = 3 * num_heads * HEAD_DIM
    num_tokens = sum(lens)
    row = dim + num_heads + 2 * HEAD_DIM
    proj = torch.randn(
        num_tokens, row, generator=gen, device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(dim, WIDTH, generator=gen, device="cuda") * 0.5
    num_slots = len(lens) + 1
    shape = (num_slots, dim, WIDTH - 1) if dim_first else (num_slots, WIDTH - 1, dim)
    pool = torch.randn(shape, generator=gen, device="cuda", dtype=torch.bfloat16)
    qsl_cpu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()]).int()
    slots = torch.randperm(len(lens), generator=torch.Generator().manual_seed(seed))
    # Decode rows continue a sequence; prompts alternate continued and fresh.
    prompts = 0
    has_initial_state = []
    for n in lens:
        if n == 1:
            has_initial_state.append(True)
        else:
            has_initial_state.append(prompts % 2 == 1)
            prompts += 1
    return SimpleNamespace(
        x=proj[:, :dim],
        weight=weight,
        pool=pool,
        dim_first=dim_first,
        query_start_loc=qsl_cpu.cuda(),
        query_start_loc_cpu=qsl_cpu,
        cache_indices=(slots + 1).int().cuda(),
        has_initial_state=torch.tensor(has_initial_state, device="cuda"),
    )


def conv_states(c, pool: torch.Tensor) -> torch.Tensor:
    return pool if c.dim_first else pool.transpose(-1, -2)


def run_stock(c, pool: torch.Tensor, metadata, copies: bool):
    out = causal_conv1d_fn(
        c.x.transpose(0, 1),
        c.weight,
        None,
        conv_states=conv_states(c, pool),
        query_start_loc=c.query_start_loc,
        cache_indices=c.cache_indices,
        has_initial_state=c.has_initial_state,
        activation="silu",
        metadata=metadata,
    ).transpose(0, 1)
    q, k, v = out.split(c.x.shape[1] // 3, dim=-1)
    if copies:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    return out


def run_planar(c, pool: torch.Tensor, **kwargs):
    return causal_conv1d_planar(
        c.x,
        c.weight,
        None,
        conv_states(c, pool),
        query_start_loc=c.query_start_loc,
        cache_indices=c.cache_indices,
        has_initial_state=c.has_initial_state,
        **kwargs,
    )


def describe_mismatch(out: torch.Tensor, ref: torch.Tensor) -> str:
    """Count, fraction and largest distance in ulps of `ref` (its dtype)."""
    diff = out != ref
    n = int(diff.sum())
    if n == 0:
        return "identical"
    o, r = out[diff], ref[diff]
    # One ulp at each reference value in the direction of the difference (the
    # two differ at powers of two), zero included (the smallest subnormal).
    up = torch.nextafter(r, torch.full_like(r, torch.inf)).float() - r.float()
    down = r.float() - torch.nextafter(r, torch.full_like(r, -torch.inf)).float()
    ulp = torch.where(o < r, down, up)
    max_ulps = float(((o.float() - r.float()).abs() / ulp).max())
    return (
        f"{n} of {ref.numel()} differ ({n / ref.numel():.2e}), max {max_ulps:.1f} ulp"
    )


def time_ms(fn, warmup_ms: int, rep_ms: int) -> float:
    return triton.testing.do_bench(
        fn, warmup=warmup_ms, rep=rep_ms, return_mode="median"
    )


def host_us(fn, iters: int = 100) -> float:
    """Wall-clock host time of one call, launches queued without a sync.

    Valid while `iters` x launches per call stays well below the stream's
    launch queue depth (on the order of a thousand entries): beyond that the
    host blocks on the queue and the figure turns into GPU time. The paths
    here launch at most four kernels per call.
    """
    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - start
    torch.accelerator.synchronize()
    return elapsed / iters * 1e6


def gpu_us(fn, iters: int = 20) -> float:
    """GPU time of one call: the sum of the path's kernel durations from
    torch.profiler over `iters` calls with the L2 cache flushed between them
    (the flush kernel excluded). Unlike `time_ms`, a launch-bound path does
    not show its host time here (see the module docstring)."""
    from torch.profiler import ProfilerActivity, profile

    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(iters):
            _FLUSH.zero_()
            fn()
        torch.accelerator.synchronize()
    total = 0.0
    for ev in prof.events():
        if ev.device_type != torch.autograd.DeviceType.CUDA:
            continue
        if "zero_" in ev.name or "fill" in ev.name.lower() or "Memset" in ev.name:
            continue  # the L2 flush
        total += ev.time_range.elapsed_us()
    return total / iters


def bench_case(num_heads: int, spec: str, args: argparse.Namespace) -> None:
    c = make_inputs(num_heads, parse_batch(spec), args.layout == "DS")
    nums_dict, batch_ptr, offset_ptr = compute_causal_conv1d_metadata(
        c.query_start_loc_cpu, device=c.x.device
    )
    metadata = SimpleNamespace(
        nums_dict=nums_dict, batch_ptr=batch_ptr, token_chunk_offset_ptr=offset_ptr
    )

    # Which kernel each requested impl runs on these rows; strict "tma" on
    # rows it cannot address is reported and skipped.
    impls: dict[str, str] = {}
    for impl in args.impl:
        try:
            impls[impl] = planar_conv_impl(impl, c.x)
        except ValueError as e:
            print(f"{num_heads:>3} {spec:>14} planar-{impl}: skipped ({e})")

    ref_pool, pool = c.pool.clone(), c.pool.clone()
    ref = run_stock(c, ref_pool, metadata, copies=False)
    for impl in impls:
        pool.copy_(c.pool)
        out = run_planar(c, pool, impl=impl)
        outputs = describe_mismatch(torch.cat(out.unbind(0), dim=1), ref)
        states = describe_mismatch(pool, ref_pool)
        if outputs != "identical" or states != "identical":
            print(
                f"{num_heads:>3} {spec:>14} planar-{impl} vs stock: outputs "
                f"{outputs}; conv states {states}"
            )

    # x read once, y written once.
    nbytes = 2 * c.x.numel() * c.x.element_size()
    paths = {
        "stock": lambda: run_stock(c, pool, metadata, copies=False),
        "stock+copies": lambda: run_stock(c, pool, metadata, copies=True),
    }
    for impl, used in impls.items():
        label = f"planar-{impl}" if impl == used else f"planar-{impl}({used})"
        paths[label] = lambda impl=impl: run_planar(c, pool, impl=impl)
    base = base_gpu = None
    for name, fn in paths.items():
        ms = time_ms(fn, args.warmup_ms, args.rep_ms)
        base = base or ms
        line = (
            f"{num_heads:>3} {spec:>14} {name:>16} {ms:>8.3f} "
            f"{nbytes / ms / 1e9:>6.2f} {base / ms:>7.2f}x {host_us(fn):>8.1f}"
        )
        if args.gpu_time:
            gpu = gpu_us(fn)
            base_gpu = base_gpu or gpu
            line += f" {gpu:>8.1f} {base_gpu / gpu:>7.2f}x"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, nargs="+", default=DEFAULT_HEADS)
    parser.add_argument("--batches", nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument(
        "--layout",
        choices=["DS", "SD"],
        default="SD",
        help="conv-state pool layout; SD (state positions outer, channels inner) is "
        "vLLM's default, DS the other (VLLM_SSM_CONV_STATE_LAYOUT)",
    )
    parser.add_argument(
        "--impl",
        nargs="+",
        choices=["auto", "tma", "regs"],
        default=["auto", "regs"],
        help="planar kernels to time; 'auto' is labelled with the kernel it ran",
    )
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=200)
    parser.add_argument(
        "--gpu-time",
        action="store_true",
        help="also report the GPU time of one call (kernel durations from "
        "torch.profiler, L2 flushed between calls) and the speedup in it",
    )
    args = parser.parse_args()

    print(f"{torch.cuda.get_device_name()}, conv-state layout {args.layout}")
    header = ["H", "batch", "path", "ms", "TB/s", "vs stock", "host us"]
    fmt = "{:>3} {:>14} {:>16} {:>8} {:>6} {:>8} {:>8}"
    if args.gpu_time:
        header += ["gpu us", "gpu x"]
        fmt += " {:>8} {:>8}"
    print(fmt.format(*header))
    with torch.inference_mode():
        for num_heads in args.heads:
            for spec in args.batches:
                bench_case(num_heads, spec, args)


if __name__ == "__main__":
    main()
