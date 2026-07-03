#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark and tuning script for the Mamba selective_state_update kernel.

Mirrors the fused MoE tuning workflow: sweeps (BLOCK_SIZE_M, num_warps) across
an effective_batch grid for a given (headdim, dstate, ngroups, cache_dtype) and
saves the best config per effective_batch to JSON. Generated configs are picked
up by selective_state_update at runtime.

Usage:
    python -m benchmarks.kernels.benchmark_selective_state_update \
        --all-dstates --save-configs --compare
"""

import argparse
import json
import os
import sys
from io import StringIO
from itertools import product
from typing import Any

import torch

from tests.kernels.mamba.utils import selective_state_update_ref
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    _CONFIGS_DIR,
    _canonical_cache_dtype,
    _get_default_ssm_launch_config,
    get_ssm_config_file_name,
    get_ssm_device_name,
    override_ssm_config,
    selective_state_update,
)
from vllm.triton_utils import triton

# bf16 shares configs with fp16 - same bit width.
_SSM_CACHE_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.float16,
}

_RESULTS_DIR = os.path.dirname(os.path.realpath(__file__))

# ---------------------------------------------------------------------------
# Tuning search space
# ---------------------------------------------------------------------------

_BSM_CHOICES_ALL = [4, 8, 16, 32, 64, 128, 256]

NUM_WARPS_CHOICES = [1, 2, 4, 8]


def _block_size_m_choices(headdim: int) -> list[int]:
    """BLOCK_SIZE_M candidates worth sweeping for a given headdim.

    BLOCK_SIZE_M > next_pow2(headdim) wastes >=50% of each tile via masking
    (offs_m >= dim rows are zeroed out), so we cap the sweep there.
    """
    ceiling = 1
    while ceiling < headdim:
        ceiling <<= 1
    return [b for b in _BSM_CHOICES_ALL if b <= ceiling]


# Default deployment shapes. effective_batch = batch * nheads scales the
# kernel grid, so configs transfer across (model, TP) combos sharing
# (headdim, dstate, cache_dtype).
DEFAULT_BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048]
DEFAULT_NHEADS = [128, 256]

ALL_DSTATES = [16, 32, 64, 128, 256]

# Default tuning shape — matches Nemotron-3-Super and Nemotron-3-Nano Mamba layers.
# Override with CLI flags for other architectures.
DEFAULT_HEADDIM = 64
DEFAULT_NGROUPS = 8


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------


def _make_inputs(
    batch: int,
    nheads: int,
    dim: int,
    dstate: int,
    ngroups: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype | None = None,
    device: str = "cuda",
):
    if state_dtype is None:
        state_dtype = dtype
    state = torch.randn(batch, nheads, dim, dstate, dtype=state_dtype, device=device)
    x = torch.randn(batch, nheads, dim, dtype=dtype, device=device)
    dt = torch.randn(batch, nheads, dim, dtype=dtype, device=device)
    A = -torch.rand(nheads, dim, dstate, dtype=torch.float32, device=device)
    B = torch.randn(batch, ngroups, dstate, dtype=dtype, device=device)
    C = torch.randn(batch, ngroups, dstate, dtype=dtype, device=device)
    D = torch.randn(nheads, dim, dtype=dtype, device=device)
    dt_bias = torch.randn(nheads, dim, dtype=dtype, device=device)
    out = torch.zeros(batch, nheads, dim, dtype=dtype, device=device)
    return state, x, dt, A, B, C, D, dt_bias, out


def benchmark_config(
    batch: int,
    nheads: int,
    dim: int,
    dstate: int,
    ngroups: int,
    block_size_m: int,
    num_warps_val: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype | None = None,
    num_iters: int = 100,
    num_warmup: int = 20,
    graph_batch_size: int = 10,
) -> float | None:
    """
    Time one (BLOCK_SIZE_M, num_warps) config for selective_state_update.
    Returns elapsed time in microseconds, or None on error.

    Uses CUDA graph capture-and-replay to isolate kernel time from Python
    eager-mode dispatch / kwarg-resolution overhead, mirroring the timing
    methodology in benchmarks/kernels/benchmark_moe.py.
    """
    state, x, dt, A, B, C, D, dt_bias, out = _make_inputs(
        batch, nheads, dim, dstate, ngroups, dtype, state_dtype=state_dtype
    )

    def _call_kernel() -> None:
        selective_state_update(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=None,
            dt_bias=dt_bias,
            dt_softplus=True,
            out=out,
        )

    try:
        with override_ssm_config((block_size_m, num_warps_val)):
            # Eager-mode warmup: triggers Triton autotune / JIT, primes caches.
            for _ in range(num_warmup):
                _call_kernel()
            torch.accelerator.synchronize()

            # Capture graph_batch_size invocations into a CUDA graph so the
            # timed region runs without Python dispatch overhead per call.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(graph_batch_size):
                    _call_kernel()
            torch.accelerator.synchronize()

            # Warmup graph replays (let the runtime stabilize).
            for _ in range(5):
                graph.replay()
            torch.accelerator.synchronize()

            start = torch.Event(enable_timing=True)
            end = torch.Event(enable_timing=True)
            latencies: list[float] = []
            for _ in range(num_iters):
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                latencies.append(start.elapsed_time(end))
            graph.reset()
        # elapsed_time returns ms; each replay runs graph_batch_size kernels,
        # so divide by (num_iters * graph_batch_size) and convert ms -> us.
        return sum(latencies) / (num_iters * graph_batch_size) * 1000
    except Exception as e:
        if "OutOfResources" not in str(e):
            print(
                f"    Warning: config M={block_size_m},w={num_warps_val} "
                f"raised {type(e).__name__}: {e}"
            )
        return None


# ---------------------------------------------------------------------------
# Tuning loop
# ---------------------------------------------------------------------------


# CUDA grid Y/Z dim limit — both `batch` and `nheads` must fit individually.
_CUDA_MAX_GRID_DIM = 65535

# Above this, kernel state-offset arithmetic (batch * nheads * headdim * dstate)
# overflows int32 and the launch raises cudaErrorIllegalAddress.
# 262144 covers Nemotron Super TP1 BS=2048.
_MAX_EFFECTIVE_BATCH = 262144


def expand_batch_x_nheads(
    batch_sizes: list[int],
    nheads_list: list[int],
    ngroups: int,
) -> list[tuple[int, int, int]]:
    """Cross-product batch_sizes × nheads_list → sorted [(effective_batch,
    batch, nheads)], deduped by effective_batch. Filters pairs that exceed
    the CUDA grid dim limit, the effective_batch ceiling, or where nheads is
    not a positive multiple of ngroups.
    """
    seen: dict[int, tuple[int, int]] = {}
    skipped_grid: list[tuple[int, int]] = []
    skipped_ngroups: list[tuple[int, int]] = []
    skipped_eb: list[tuple[int, int]] = []
    for b, n in product(batch_sizes, nheads_list):
        if b <= 0 or n <= 0:
            continue
        if b > _CUDA_MAX_GRID_DIM or n > _CUDA_MAX_GRID_DIM:
            skipped_grid.append((b, n))
            continue
        if n % ngroups != 0:
            skipped_ngroups.append((b, n))
            continue
        if b * n > _MAX_EFFECTIVE_BATCH:
            skipped_eb.append((b, n))
            continue
        seen.setdefault(b * n, (b, n))
    if skipped_grid:
        print(
            f"  Note: skipping (batch, nheads) pairs exceeding CUDA grid dim "
            f"{_CUDA_MAX_GRID_DIM}: {skipped_grid}"
        )
    if skipped_ngroups:
        print(
            f"  Note: skipping (batch, nheads) pairs where nheads % ngroups != 0 "
            f"for ngroups={ngroups}: {skipped_ngroups}"
        )
    if skipped_eb:
        print(
            f"  Note: skipping (batch, nheads) pairs whose effective_batch "
            f"exceeds {_MAX_EFFECTIVE_BATCH}: {skipped_eb}"
        )
    return sorted((eb, b, n) for eb, (b, n) in seen.items())


def tune_dstate(
    dstate: int,
    headdim: int,
    ngroups: int,
    dtype: torch.dtype,
    num_iters: int,
    verbose: bool,
    active: list[tuple[int, int, int]],
    state_dtype: torch.dtype | None = None,
) -> tuple[dict[int, dict], dict[int, dict[tuple[int, int], float]]]:
    """For each (effective_batch, batch, nheads) in *active*, sweep
    (BLOCK_SIZE_M, num_warps) and return
    ({effective_batch: best_config}, {effective_batch: {(bsm, nw): us}}).
    The second map is the full timing grid, used downstream so we don't
    re-measure the same config in the comparison phase.
    """
    best_per_eb: dict[int, dict] = {}
    timings: dict[int, dict[tuple[int, int], float]] = {}

    print(f"\n{'=' * 74}")
    effective_state_dtype = state_dtype if state_dtype is not None else dtype
    print(
        f"Tuning  headdim={headdim}  dstate={dstate}  ngroups={ngroups}  "
        f"dtype={dtype}  ssm_cache_dtype={effective_state_dtype}"
    )
    print(f"{'=' * 74}")

    bsm_choices = _block_size_m_choices(headdim)
    print(f"BSM candidates (capped at next_pow2(headdim={headdim})): {bsm_choices}")

    hdr = f"{'EffBatch':>8} | {'BLOCK_M':>7} | {'warps':>5} | {'us':>10} | note"
    print(hdr)
    print("-" * 52)

    for eb, batch, nheads in active:
        best_time = float("inf")
        best_cfg: dict = {}
        eb_timings: dict[tuple[int, int], float] = {}

        for bsm, nw in product(bsm_choices, NUM_WARPS_CHOICES):
            t = benchmark_config(
                batch=batch,
                nheads=nheads,
                dim=headdim,
                dstate=dstate,
                ngroups=ngroups,
                block_size_m=bsm,
                num_warps_val=nw,
                dtype=dtype,
                state_dtype=state_dtype,
                num_iters=num_iters,
            )
            if t is None:
                continue
            eb_timings[(bsm, nw)] = t
            is_best = t < best_time
            if is_best:
                best_time = t
                best_cfg = {"BLOCK_SIZE_M": bsm, "num_warps": nw}
            if verbose:
                marker = " <-- best" if is_best else ""
                print(f"{eb:>8} | {bsm:>7} | {nw:>5} | {t:>10.2f} |{marker}")

        timings[eb] = eb_timings

        if not best_cfg:
            print(
                f"{eb:>8} | {'-':>7} | {'-':>5} | {'-':>10} | "
                f"no working config (skipped)"
            )
            continue

        if not verbose:
            print(
                f"{eb:>8} | {best_cfg['BLOCK_SIZE_M']:>7} | "
                f"{best_cfg['num_warps']:>5} | {best_time:>10.2f} | best"
            )

        best_per_eb[eb] = best_cfg

    return best_per_eb, timings


# ---------------------------------------------------------------------------
# Correctness validation
# ---------------------------------------------------------------------------


def validate_configs(
    dstate: int,
    headdim: int,
    ngroups: int,
    tuned: dict[int, dict],
    active: list[tuple[int, int, int]],
    dtype: torch.dtype,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    state_dtype: torch.dtype | None = None,
) -> dict[int, bool]:
    """
    For every (effective_batch, batch, nheads) in *active* that has a tuned
    config, run the kernel with that config and compare against the reference.
    Returns {effective_batch: passed}.
    """
    # Disable TF32 so the reference's matmul matches the Triton kernel's
    # fp32 accumulation; otherwise large ebs show bf16 rounding mismatches.
    torch.set_float32_matmul_precision("highest")

    print(f"\n{'=' * 74}")
    effective_state_dtype = state_dtype if state_dtype is not None else dtype
    print(
        f"Validation  headdim={headdim}  dstate={dstate}  ngroups={ngroups}  "
        f"dtype={dtype}  ssm_cache_dtype={effective_state_dtype}  atol={atol}"
    )
    print(f"{'=' * 74}")
    print(f"{'EffBatch':>8} | {'MaxAbsErr':>12} | {'Status':>8}")
    print("-" * 36)

    results: dict[int, bool] = {}

    for eb, batch, nheads in active:
        cfg = tuned.get(eb)
        if cfg is None:
            continue
        state, x, dt, A, B, C, D, dt_bias, out = _make_inputs(
            batch=batch,
            nheads=nheads,
            dim=headdim,
            dstate=dstate,
            ngroups=ngroups,
            dtype=dtype,
            state_dtype=state_dtype,
        )
        # Clone state before GPU kernel modifies it in-place
        state_ref = state.clone()

        with override_ssm_config((cfg["BLOCK_SIZE_M"], cfg["num_warps"])):
            selective_state_update(
                state,
                x,
                dt,
                A,
                B,
                C,
                D=D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
                out=out,
            )
        torch.accelerator.synchronize()
        gpu_out = out.detach().cpu()

        # Reference uses the original (unmodified) state
        # Upcast to fp32 so the reference sums in fp32 (matches the Triton
        # kernel); summing in bf16 over `dstate` blows up the error.
        ref_out = (
            selective_state_update_ref(
                state_ref.float(),
                x.float(),
                dt.float(),
                A.float(),
                B.float(),
                C.float(),
                D=D.float(),
                dt_bias=dt_bias.float(),
                dt_softplus=True,
            )
            .to(out.dtype)
            .cpu()
        )

        passed = torch.allclose(gpu_out.float(), ref_out.float(), atol=atol, rtol=rtol)
        max_err = (gpu_out.float() - ref_out.float()).abs().max().item()
        status = "PASS" if passed else "FAIL"
        results[eb] = passed
        print(f"{eb:>8} | {max_err:>12.6f} | {status:>8}")

    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} configs passed validation for dstate={dstate}")
    return results


# ---------------------------------------------------------------------------
# Save configs
# ---------------------------------------------------------------------------


def save_configs(
    headdim: int,
    dstate: int,
    cache_dtype: str,
    configs: dict[int, dict],
    save_dir: str | None = None,
) -> str:
    # bf16 shares configs with fp16, use common filename for both
    cache_dtype = _canonical_cache_dtype(cache_dtype)

    base_dir = save_dir if save_dir else _CONFIGS_DIR
    os.makedirs(base_dir, exist_ok=True)
    file_path = os.path.join(
        base_dir,
        get_ssm_config_file_name(headdim, dstate, cache_dtype, get_ssm_device_name()),
    )
    # triton_version is informational only, the loader ignores it
    payload: dict[str, Any] = {
        "triton_version": triton.__version__,
        **{str(k): v for k, v in sorted(configs.items())},
    }
    with open(file_path, "w") as f:
        json.dump(payload, f, indent=4)
    return file_path


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def current_heuristic(dstate: int, is_blackwell: bool = False) -> dict:
    """Return the current hard-coded BLOCK_SIZE_M / num_warps for dstate."""
    bsm, nw = _get_default_ssm_launch_config(dstate, is_blackwell)
    return {"BLOCK_SIZE_M": bsm, "num_warps": nw}


def compare_heuristic_vs_tuned(
    dstate: int,
    headdim: int,
    ngroups: int,
    tuned: dict[int, dict],
    timings: dict[int, dict[tuple[int, int], float]],
    active: list[tuple[int, int, int]],
    dtype: torch.dtype,
    num_iters: int,
    is_blackwell: bool,
    state_dtype: torch.dtype | None = None,
):
    heur_cfg = current_heuristic(dstate, is_blackwell)
    heur_key = (heur_cfg["BLOCK_SIZE_M"], heur_cfg["num_warps"])

    print(f"\n{'=' * 74}")
    print(
        f"Comparison  headdim={headdim}  dstate={dstate}  "
        f"ngroups={ngroups}  —  heuristic vs tuned"
    )
    print(
        f"Heuristic: BLOCK_SIZE_M={heur_cfg['BLOCK_SIZE_M']}, "
        f"num_warps={heur_cfg['num_warps']}"
    )
    print(f"{'=' * 74}")
    hdr = (
        f"{'EffBatch':>8} | {'Heur(us)':>10} | {'Tuned(us)':>10} | "
        f"{'Speedup':>8} | Best config"
    )
    print(hdr)
    print("-" * len(hdr))

    for eb, batch, nheads in active:
        eb_timings = timings.get(eb, {})

        # Heuristic timing: reuse the tuning measurement if the heuristic
        # config was in the swept grid; otherwise measure it once.
        t_h = eb_timings.get(heur_key)
        if t_h is None:
            t_h = benchmark_config(
                batch=batch,
                nheads=nheads,
                dim=headdim,
                dstate=dstate,
                ngroups=ngroups,
                block_size_m=heur_cfg["BLOCK_SIZE_M"],
                num_warps_val=heur_cfg["num_warps"],
                dtype=dtype,
                state_dtype=state_dtype,
                num_iters=num_iters,
            )

        # `tuned[eb]` may be missing if all configs failed in tune_dstate;
        # in that case fall back to the heuristic so the table still prints.
        best = tuned.get(eb) or heur_cfg
        t_t = eb_timings.get((best["BLOCK_SIZE_M"], best["num_warps"]))

        if t_h is None or t_t is None:
            print(f"{eb:>8} | {'N/A':>10} | {'N/A':>10} | {'N/A':>8} |")
            continue
        speedup = t_h / t_t
        marker = " <--" if speedup > 1.05 else ""
        print(
            f"{eb:>8} | {t_h:>10.2f} | {t_t:>10.2f} | "
            f"{speedup:>7.2f}x | "
            f"M={best['BLOCK_SIZE_M']},w={best['num_warps']}{marker}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def save_results(device_name: str, output: str, results_file: str | None = None) -> str:
    """Save the full benchmark output to a results text file."""
    if results_file is None:
        results_file = os.path.join(
            _RESULTS_DIR, f"ssm_benchmark_results_{device_name}.txt"
        )
    with open(results_file, "w") as f:
        f.write(output)
    return results_file


def main():
    parser = argparse.ArgumentParser(
        description="Tune selective_state_update kernel for Mamba SSM"
    )
    parser.add_argument(
        "--dstate",
        type=int,
        default=128,
        help="SSM state size to tune for (default: 128)",
    )
    parser.add_argument(
        "--all-dstates",
        action="store_true",
        help="Tune all common dstate values: " + str(ALL_DSTATES),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Activation / input data type (default: bfloat16)",
    )
    parser.add_argument(
        "--mamba-ssm-cache-dtype",
        type=str,
        default="float32",
        choices=list(_SSM_CACHE_DTYPE_MAP.keys()),
        help="SSM state cache dtype (default: float32)",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=100,
        help="Number of timing iterations (default: 100)",
    )
    parser.add_argument(
        "--save-configs",
        action="store_true",
        help=f"Save best configs to JSON in {_CONFIGS_DIR}",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show comparison table: heuristic vs tuned",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every (BLOCK_SIZE_M, num_warps) result, not just best",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to save the benchmark results text file "
        "(default: ssm_benchmark_results_<device>.txt alongside this script)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help=f"Directory to save JSON configs (default: {_CONFIGS_DIR})",
    )
    parser.add_argument(
        "--headdim",
        type=int,
        default=DEFAULT_HEADDIM,
        help=f"Per-head feature dim (default: {DEFAULT_HEADDIM})",
    )
    parser.add_argument(
        "--ngroups",
        type=int,
        default=DEFAULT_NGROUPS,
        help=f"Number of B/C groups (default: {DEFAULT_NGROUPS})",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        metavar="B",
        help=f"Decoder batch sizes to sweep (default: {DEFAULT_BATCH_SIZES})",
    )
    parser.add_argument(
        "--nheads",
        type=int,
        nargs="+",
        default=DEFAULT_NHEADS,
        metavar="N",
        help=f"Number of heads per rank to sweep (default: {DEFAULT_NHEADS}). "
        "effective_batch = batch * nheads; cross-product is deduped by eb.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="After tuning, verify each best config against a CPU reference "
        "implementation. Configs that fail are flagged in the output.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for --validate (default: 1e-2)",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    state_dtype = _SSM_CACHE_DTYPE_MAP[args.mamba_ssm_cache_dtype]
    device_name = get_ssm_device_name()
    cap = torch.cuda.get_device_capability()
    is_blackwell = cap[0] >= 10

    # Mirror all output to a results file (like Unix tee).
    buf = StringIO()

    class _Tee:
        """Writes to both the original stdout and an in-memory buffer."""

        def write(self, s):
            buf.write(s)
            sys.__stdout__.write(s)

        def flush(self):
            sys.__stdout__.flush()

    sys.stdout = _Tee()  # type: ignore[assignment]

    try:
        print(f"Device : {device_name}  (sm_{cap[0]}{cap[1]})")
        print(f"Blackwell: {is_blackwell}")
        print(f"dtype  : {args.dtype}")
        print(f"ssm_cache_dtype: {args.mamba_ssm_cache_dtype}")
        print(f"headdim: {args.headdim}")
        print(f"ngroups: {args.ngroups}")
        print(f"triton : {triton.__version__}")

        dstates = ALL_DSTATES if args.all_dstates else [args.dstate]
        active = expand_batch_x_nheads(args.batch_sizes, args.nheads, args.ngroups)

        for dstate in dstates:
            tuned, timings = tune_dstate(
                dstate=dstate,
                headdim=args.headdim,
                ngroups=args.ngroups,
                dtype=dtype,
                num_iters=args.num_iters,
                verbose=args.verbose,
                active=active,
                state_dtype=state_dtype,
            )

            if args.compare:
                compare_heuristic_vs_tuned(
                    dstate=dstate,
                    headdim=args.headdim,
                    ngroups=args.ngroups,
                    tuned=tuned,
                    timings=timings,
                    active=active,
                    dtype=dtype,
                    num_iters=args.num_iters,
                    is_blackwell=is_blackwell,
                    state_dtype=state_dtype,
                )

            if args.validate:
                validity = validate_configs(
                    dstate=dstate,
                    headdim=args.headdim,
                    ngroups=args.ngroups,
                    tuned=tuned,
                    active=active,
                    dtype=dtype,
                    atol=args.atol,
                    state_dtype=state_dtype,
                )
                # Filter out any configs that failed correctness check
                failed = [eb for eb, ok in validity.items() if not ok]
                if failed:
                    print(
                        f"\n  WARNING: {len(failed)} config(s) failed validation "
                        f"for dstate={dstate}: effective_batches {failed}"
                    )
                    print("  These will NOT be saved even with --save-configs.")
                    tuned = {
                        eb: cfg for eb, cfg in tuned.items() if validity.get(eb, True)
                    }

            if args.save_configs:
                path = save_configs(
                    headdim=args.headdim,
                    dstate=dstate,
                    cache_dtype=args.mamba_ssm_cache_dtype,
                    configs=tuned,
                    save_dir=args.save_dir,
                )
                print(f"\nSaved: {path}")
            else:
                print(f"\nBest configs for dstate={dstate}:")
                for eb, cfg in sorted(tuned.items()):
                    print(f"  effective_batch={eb:>6}: {cfg}")
                print("\n(Re-run with --save-configs to persist to JSON)")
    finally:
        sys.stdout = sys.__stdout__
        results_path = save_results(device_name, buf.getvalue(), args.results_file)
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
