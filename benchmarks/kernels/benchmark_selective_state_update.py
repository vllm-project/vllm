#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark and tuning script for the Mamba selective_state_update kernel.

This script mirrors the fused MoE tuning workflow in vLLM:
  - Sweeps BLOCK_SIZE_M x num_warps across all batch sizes for a given dstate
  - Finds the best launch config per (batch, dstate) combination
  - Optionally saves configs to JSON in vllm/model_executor/layers/mamba/configs/
  - Optionally compares tuned configs against the existing heuristic baseline
  - Always saves a human-readable results file alongside this script

Usage (tune all dstates, save configs + compare vs heuristic):
    python benchmarks/kernels/benchmark_selective_state_update.py \\
        --all-dstates --save-configs --compare

Usage (single dstate, show results only):
    python benchmarks/kernels/benchmark_selective_state_update.py --dstate 128

Generated JSON configs are loaded automatically by selective_state_update
at runtime when a matching device config file is found.
"""

import argparse
import json
import os
import sys
from io import StringIO
from itertools import product
from unittest.mock import patch

import torch

import vllm.model_executor.layers.mamba.ops.mamba_ssm as mamba_ssm_module
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_state_update,
)
from vllm.platforms import current_platform

_RESULTS_DIR = os.path.dirname(os.path.realpath(__file__))

# ---------------------------------------------------------------------------
# Tuning search space
# ---------------------------------------------------------------------------

BLOCK_SIZE_M_CHOICES = [4, 8, 16, 32, 64]
NUM_WARPS_CHOICES = [1, 2, 4, 8]

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

ALL_DSTATES = [16, 32, 64, 128, 256]


# ---------------------------------------------------------------------------
# Config file naming (mirrors fused_moe pattern)
# ---------------------------------------------------------------------------


def get_ssm_config_file_name(dstate: int) -> str:
    return f"dstate={dstate}.json"


def get_device_name() -> str:
    return current_platform.get_device_name().replace(" ", "_")


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
    device: str = "cuda",
):
    state = torch.randn(batch, nheads, dim, dstate, dtype=dtype, device=device)
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
    num_iters: int = 100,
    num_warmup: int = 20,
) -> float | None:
    """
    Time one (BLOCK_SIZE_M, num_warps) config for selective_state_update.
    Returns elapsed time in microseconds, or None on error.
    """
    state, x, dt, A, B, C, D, dt_bias, out = _make_inputs(
        batch, nheads, dim, dstate, ngroups, dtype
    )

    # Monkeypatch _get_ssm_launch_config to return the specific config
    # without affecting the lru_cache on get_ssm_configs.
    def _fixed_launch_config(dstate_, batch_, is_blackwell_):
        return block_size_m, num_warps_val

    try:
        with patch.object(
            mamba_ssm_module, "_get_ssm_launch_config", _fixed_launch_config
        ):
            # Warmup
            for _ in range(num_warmup):
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

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(num_iters):
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
            end.record()
            torch.accelerator.synchronize()
        return start.elapsed_time(end) / num_iters * 1000  # ms -> us
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


def tune_dstate(
    dstate: int,
    dtype: torch.dtype,
    num_iters: int,
    verbose: bool,
    batch_sizes: list[int] | None = None,
) -> dict[int, dict]:
    """
    For each batch size, sweep all (BLOCK_SIZE_M, num_warps) combos and
    return a dict mapping batch_size -> best_config.
    """
    # Use a representative shape for tuning (Mamba-2 style, common case).
    nheads, dim, ngroups = 64, 64, 1
    active_batches = batch_sizes if batch_sizes is not None else BATCH_SIZES

    best_per_batch: dict[int, dict] = {}

    print(f"\n{'=' * 74}")
    print(f"Tuning  dstate={dstate}  nheads={nheads}  dim={dim}  dtype={dtype}")
    print(f"{'=' * 74}")

    hdr = f"{'Batch':>7} | {'BLOCK_M':>7} | {'warps':>5} | {'us':>10} | note"
    print(hdr)
    print("-" * 50)

    for batch in active_batches:
        best_time = float("inf")
        best_cfg: dict = {}

        for bsm, nw in product(BLOCK_SIZE_M_CHOICES, NUM_WARPS_CHOICES):
            t = benchmark_config(
                batch=batch,
                nheads=nheads,
                dim=dim,
                dstate=dstate,
                ngroups=ngroups,
                block_size_m=bsm,
                num_warps_val=nw,
                dtype=dtype,
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
                print(f"{batch:>7} | {bsm:>7} | {nw:>5} | {t:>10.2f} |{marker}")

        if not verbose and best_cfg:
            print(
                f"{batch:>7} | {best_cfg['BLOCK_SIZE_M']:>7} | "
                f"{best_cfg['num_warps']:>5} | {best_time:>10.2f} | best"
            )

        best_per_batch[batch] = best_cfg

    return best_per_batch


# ---------------------------------------------------------------------------
# Correctness validation
# ---------------------------------------------------------------------------


def _selective_state_update_ref(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Pure-PyTorch CPU reference for selective_state_update (dt_softplus=True).

    Shapes (all moved to CPU float32 internally):
        state  : (batch, nheads, dim, dstate)
        x      : (batch, nheads, dim)
        dt     : (batch, nheads, dim)
        A      : (nheads, dim, dstate)
        B      : (batch, ngroups, dstate)
        C      : (batch, ngroups, dstate)
        D      : (nheads, dim)
        dt_bias: (nheads, dim)
    Returns:
        out    : (batch, nheads, dim)  in the original dtype
    """
    orig_dtype = x.dtype
    state = state.clone().cpu().float()
    x = x.cpu().float()
    dt = dt.cpu().float()
    A = A.cpu().float()
    B = B.cpu().float()
    C = C.cpu().float()
    D = D.cpu().float()
    dt = dt + dt_bias.cpu().float()
    dt = torch.nn.functional.softplus(dt)  # (batch, nheads, dim)

    nheads, _, _ = A.shape
    ngroups = B.shape[1]

    dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # (batch, nheads, dim, dstate)
    B_exp = B.repeat_interleave(nheads // ngroups, dim=1)  # (batch, nheads, dstate)
    C_exp = C.repeat_interleave(nheads // ngroups, dim=1)
    dB = dt.unsqueeze(-1) * B_exp.unsqueeze(2)  # (batch, nheads, dim, dstate)

    state_new = state * dA + dB * x.unsqueeze(-1)
    out = (state_new * C_exp.unsqueeze(2)).sum(-1)  # (batch, nheads, dim)
    out = out + x * D.unsqueeze(0)
    return out.to(orig_dtype)


def validate_configs(
    dstate: int,
    tuned: dict[int, dict],
    dtype: torch.dtype,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict[int, bool]:
    """
    For every batch size in *tuned*, run the kernel with the tuned config and
    compare against the CPU reference.  Returns {batch: passed}.
    """
    nheads, dim, ngroups = 64, 64, 1

    print(f"\n{'=' * 74}")
    print(f"Validation  dstate={dstate}  dtype={dtype}  atol={atol}")
    print(f"{'=' * 74}")
    print(f"{'Batch':>7} | {'MaxAbsErr':>12} | {'Status':>8}")
    print("-" * 36)

    results: dict[int, bool] = {}

    for batch, cfg in sorted(tuned.items()):
        state, x, dt, A, B, C, D, dt_bias, out = _make_inputs(
            batch, nheads, dim, dstate, ngroups, dtype
        )
        # Clone state before GPU kernel modifies it in-place
        state_ref = state.clone()

        # GPU kernel output
        def _fixed(dstate_, batch_, is_blackwell_, _cfg=cfg):
            return _cfg["BLOCK_SIZE_M"], _cfg["num_warps"]

        with patch.object(mamba_ssm_module, "_get_ssm_launch_config", _fixed):
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

        # CPU reference uses the original (unmodified) state
        ref_out = _selective_state_update_ref(state_ref, x, dt, A, B, C, D, dt_bias)

        passed = torch.allclose(gpu_out.float(), ref_out.float(), atol=atol, rtol=rtol)
        max_err = (gpu_out.float() - ref_out.float()).abs().max().item()
        status = "PASS" if passed else "FAIL"
        results[batch] = passed
        print(f"{batch:>7} | {max_err:>12.6f} | {status:>8}")

    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} configs passed validation for dstate={dstate}")
    return results


# ---------------------------------------------------------------------------
# Save configs
# ---------------------------------------------------------------------------


def save_configs(
    dstate: int, configs: dict[int, dict], save_dir: str | None = None
) -> str:
    base_dir = save_dir if save_dir else get_ssm_configs_dir()
    # Place configs in a per-GPU subfolder for easy multi-GPU organisation.
    configs_dir = os.path.join(base_dir, get_device_name())
    os.makedirs(configs_dir, exist_ok=True)
    file_path = os.path.join(configs_dir, get_ssm_config_file_name(dstate))
    payload = {str(k): v for k, v in sorted(configs.items())}
    with open(file_path, "w") as f:
        json.dump(payload, f, indent=4)
    return file_path


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def current_heuristic(dstate: int, is_blackwell: bool = False) -> dict:
    """Return the current hard-coded BLOCK_SIZE_M / num_warps for dstate."""
    if dstate <= 16:
        return {"BLOCK_SIZE_M": 32, "num_warps": 4}
    elif dstate <= 32:
        return {"BLOCK_SIZE_M": 16, "num_warps": 4}
    elif dstate <= 64:
        return {"BLOCK_SIZE_M": 8, "num_warps": 4}
    else:
        if is_blackwell:
            return {"BLOCK_SIZE_M": 32, "num_warps": 8}
        elif dstate <= 128:
            return {"BLOCK_SIZE_M": 4, "num_warps": 4}
        else:
            return {"BLOCK_SIZE_M": 4, "num_warps": 8}


def compare_heuristic_vs_tuned(
    dstate: int,
    tuned: dict[int, dict],
    dtype: torch.dtype,
    num_iters: int,
    is_blackwell: bool,
):
    nheads, dim, ngroups = 64, 64, 1
    heur_cfg = current_heuristic(dstate, is_blackwell)

    print(f"\n{'=' * 74}")
    print(f"Comparison  dstate={dstate}  —  heuristic vs tuned")
    print(
        f"Heuristic: BLOCK_SIZE_M={heur_cfg['BLOCK_SIZE_M']}, "
        f"num_warps={heur_cfg['num_warps']}"
    )
    print(f"{'=' * 74}")
    hdr = (
        f"{'Batch':>7} | {'Heur(us)':>10} | {'Tuned(us)':>10} | "
        f"{'Speedup':>8} | Best config"
    )
    print(hdr)
    print("-" * len(hdr))

    for batch in BATCH_SIZES:
        t_h = benchmark_config(
            batch,
            nheads,
            dim,
            dstate,
            ngroups,
            heur_cfg["BLOCK_SIZE_M"],
            heur_cfg["num_warps"],
            dtype,
            num_iters,
        )
        best = tuned.get(batch, heur_cfg)
        t_t = benchmark_config(
            batch,
            nheads,
            dim,
            dstate,
            ngroups,
            best["BLOCK_SIZE_M"],
            best["num_warps"],
            dtype,
            num_iters,
        )
        if t_h is None or t_t is None:
            print(f"{batch:>7} | {'N/A':>10} | {'N/A':>10} | {'N/A':>8} |")
            continue
        speedup = t_h / t_t
        marker = " <--" if speedup > 1.05 else ""
        print(
            f"{batch:>7} | {t_h:>10.2f} | {t_t:>10.2f} | "
            f"{speedup:>7.2f}x | "
            f"M={best['BLOCK_SIZE_M']},w={best['num_warps']}{marker}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def save_results(device_name: str, output: str, results_file: str | None = None) -> str:
    """Save the full benchmark output to a results text file."""
    if results_file is None:
        safe_name = device_name.replace(" ", "_")
        results_file = os.path.join(
            _RESULTS_DIR, f"ssm_benchmark_results_{safe_name}.txt"
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
        help="Data type (default: bfloat16)",
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
        help="Base directory to save JSON configs. Configs are placed in a "
        "per-GPU subfolder: <save-dir>/<device_name>/. "
        "(default: vllm/model_executor/layers/mamba/configs/)",
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=None,
        metavar="B",
        help="Only tune these specific batch sizes, e.g. --batches 2 16 256. "
        "Useful for stability re-checks on flagged configs.",
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
    device_name = current_platform.get_device_name()
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
        print(f"Device : {device_name}  (sm_{cap[0]}{cap[1]:02d})")
        print(f"Blackwell: {is_blackwell}")
        print(f"dtype  : {args.dtype}")

        dstates = ALL_DSTATES if args.all_dstates else [args.dstate]

        for dstate in dstates:
            tuned = tune_dstate(
                dstate, dtype, args.num_iters, args.verbose, args.batches
            )

            if args.compare:
                compare_heuristic_vs_tuned(
                    dstate, tuned, dtype, args.num_iters, is_blackwell
                )

            if args.validate:
                validity = validate_configs(dstate, tuned, dtype, args.atol)
                # Filter out any configs that failed correctness check
                failed = [b for b, ok in validity.items() if not ok]
                if failed:
                    print(
                        f"\n  WARNING: {len(failed)} config(s) failed "
                        f"validation for dstate={dstate}: batches {failed}"
                    )
                    print("  These will NOT be saved even with --save-configs.")
                    tuned = {
                        b: cfg for b, cfg in tuned.items() if validity.get(b, True)
                    }

            if args.save_configs:
                path = save_configs(dstate, tuned, args.save_dir)
                print(f"\nSaved: {path}")
            else:
                print(f"\nBest configs for dstate={dstate}:")
                for batch, cfg in sorted(tuned.items()):
                    print(f"  batch={batch:>5}: {cfg}")
                print("\n(Re-run with --save-configs to persist to JSON)")
    finally:
        sys.stdout = sys.__stdout__
        results_path = save_results(device_name, buf.getvalue(), args.results_file)
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
