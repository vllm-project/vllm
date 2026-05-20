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
from unittest.mock import patch

import torch

import vllm.model_executor.layers.mamba.ops.mamba_ssm as mamba_ssm_module
from tests.kernels.mamba.test_mamba_ssm import selective_state_update_ref
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    _get_default_ssm_launch_config,
    get_ssm_config_file_name,
    get_ssm_device_name,
    selective_state_update,
)
from vllm.triton_utils import triton

# MambaDType subset: bf16 is excluded (not commonly used)
_SSM_CACHE_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
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


# effective_batch = batch * nheads_per_rank — the kernel grid scales with
# the product, so configs transfer across (model, TP) combos sharing
# (headdim, dstate, cache_dtype).
# Ceiling 262144 covers 256-head at TP1, max BS=1024 (256 * 1024).
EFFECTIVE_BATCH_SIZES = [
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
    196608,
    262144,
]

ALL_DSTATES = [16, 32, 64, 128, 256]

# Default tuning shape — matches Nemotron-3-Super and Nemotron-3-Nano Mamba layers.
# Override with CLI flags for other architectures.
DEFAULT_HEADDIM = 64
DEFAULT_NGROUPS = 8


# ---------------------------------------------------------------------------
# Config file naming
# ---------------------------------------------------------------------------


def get_ssm_configs_dir() -> str:
    return os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../vllm/model_executor/layers/mamba/configs",
        )
    )


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

    # Monkeypatch try_get_optimal_ssm_config to return the specific config
    # without affecting the lru_cache on get_ssm_configs.
    def _fixed_launch_config(*_args, **_kwargs):
        return block_size_m, num_warps_val

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
        with patch.object(
            mamba_ssm_module, "try_get_optimal_ssm_config", _fixed_launch_config
        ):
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

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
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


# CUDA grid Y/Z dim limit — both `batch` and `nheads` must fit individually,
# so effective_batch > 65535 has to be split across the two.
_CUDA_MAX_GRID_DIM = 65535


def _factor_effective_batch(
    effective_batch: int, ngroups: int
) -> tuple[int, int] | None:
    """Return (batch, nheads) with batch*nheads == effective_batch such that
    both fit the CUDA grid Y/Z dim limit and nheads is a positive multiple of
    ngroups. Prefers batch=1 (the cheapest split) when it fits.

    Returns None if no valid factorization exists.
    """
    for batch in range(1, _CUDA_MAX_GRID_DIM + 1):
        if batch > effective_batch or effective_batch % batch != 0:
            continue
        nheads = effective_batch // batch
        if nheads > _CUDA_MAX_GRID_DIM:
            continue
        if nheads % ngroups != 0:
            continue
        return batch, nheads
    return None


def _resolve_effective_batches(
    user_supplied: list[int] | None,
    ngroups: int,
) -> list[tuple[int, int, int]]:
    """Return [(effective_batch, batch, nheads)] for each valid sweep point.

    Drops any effective_batch with no valid (batch, nheads) factorization
    that satisfies both the CUDA grid dim limit and nheads % ngroups == 0.
    """
    candidates = user_supplied if user_supplied is not None else EFFECTIVE_BATCH_SIZES
    valid: list[tuple[int, int, int]] = []
    skipped: list[int] = []
    for eb in candidates:
        if eb <= 0:
            skipped.append(eb)
            continue
        factored = _factor_effective_batch(eb, ngroups)
        if factored is None:
            skipped.append(eb)
            continue
        batch, nheads = factored
        valid.append((eb, batch, nheads))
    if skipped:
        print(
            f"  Note: skipping effective_batch values with no valid "
            f"(batch, nheads) factorization for ngroups={ngroups} "
            f"under CUDA grid dim {_CUDA_MAX_GRID_DIM}: {skipped}"
        )
    return valid


def tune_dstate(
    dstate: int,
    headdim: int,
    ngroups: int,
    dtype: torch.dtype,
    num_iters: int,
    verbose: bool,
    effective_batches: list[int] | None = None,
    state_dtype: torch.dtype | None = None,
) -> dict[int, dict]:
    """For each effective_batch, sweep (BLOCK_SIZE_M, num_warps) and return
    {effective_batch: best_config}. effective_batch is factored into
    (batch, nheads) by `_factor_effective_batch`.
    """
    active = _resolve_effective_batches(effective_batches, ngroups)

    best_per_eb: dict[int, dict] = {}

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
            is_best = t < best_time
            if is_best:
                best_time = t
                best_cfg = {"BLOCK_SIZE_M": bsm, "num_warps": nw}
            if verbose:
                marker = " <-- best" if is_best else ""
                print(f"{eb:>8} | {bsm:>7} | {nw:>5} | {t:>10.2f} |{marker}")

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

    return best_per_eb


# ---------------------------------------------------------------------------
# Correctness validation
# ---------------------------------------------------------------------------


def validate_configs(
    dstate: int,
    headdim: int,
    ngroups: int,
    tuned: dict[int, dict],
    dtype: torch.dtype,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    state_dtype: torch.dtype | None = None,
) -> dict[int, bool]:
    """
    For every effective_batch in *tuned*, run the kernel with the tuned config
    and compare against the CPU reference. Returns {effective_batch: passed}.
    """
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

    for eb, cfg in sorted(tuned.items()):
        factored = _factor_effective_batch(eb, ngroups)
        if factored is None:
            continue
        batch, nheads = factored
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

        # GPU kernel output
        def _fixed(*_args, _cfg=cfg, **_kwargs):
            return _cfg["BLOCK_SIZE_M"], _cfg["num_warps"]

        with patch.object(mamba_ssm_module, "try_get_optimal_ssm_config", _fixed):
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
        ref_out = selective_state_update_ref(
            state_ref, x, dt, A, B, C, D=D, dt_bias=dt_bias, dt_softplus=True
        ).cpu()

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
    base_dir = save_dir if save_dir else get_ssm_configs_dir()
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
    dtype: torch.dtype,
    num_iters: int,
    is_blackwell: bool,
    effective_batches: list[int] | None = None,
    state_dtype: torch.dtype | None = None,
):
    active = _resolve_effective_batches(effective_batches, ngroups)
    heur_cfg = current_heuristic(dstate, is_blackwell)

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
        t_t = benchmark_config(
            batch=batch,
            nheads=nheads,
            dim=headdim,
            dstate=dstate,
            ngroups=ngroups,
            block_size_m=best["BLOCK_SIZE_M"],
            num_warps_val=best["num_warps"],
            dtype=dtype,
            state_dtype=state_dtype,
            num_iters=num_iters,
        )
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
        help="Save best configs to JSON in mamba/configs/",
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
        help="Directory to save JSON configs. "
        "(default: vllm/model_executor/layers/mamba/configs/)",
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
        "--effective-batches",
        type=int,
        nargs="+",
        default=None,
        metavar="EB",
        help="Tune only these effective_batch values (default: full sweep)",
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

        for dstate in dstates:
            tuned = tune_dstate(
                dstate=dstate,
                headdim=args.headdim,
                ngroups=args.ngroups,
                dtype=dtype,
                num_iters=args.num_iters,
                verbose=args.verbose,
                effective_batches=args.effective_batches,
                state_dtype=state_dtype,
            )

            if args.compare:
                compare_heuristic_vs_tuned(
                    dstate=dstate,
                    headdim=args.headdim,
                    ngroups=args.ngroups,
                    tuned=tuned,
                    dtype=dtype,
                    num_iters=args.num_iters,
                    is_blackwell=is_blackwell,
                    effective_batches=args.effective_batches,
                    state_dtype=state_dtype,
                )

            if args.validate:
                validity = validate_configs(
                    dstate=dstate,
                    headdim=args.headdim,
                    ngroups=args.ngroups,
                    tuned=tuned,
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
