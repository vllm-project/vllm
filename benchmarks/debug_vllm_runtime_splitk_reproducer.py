#!/usr/bin/env python3
"""Staged runtime reproducer for the vLLM BlockScale SplitK zero-init bug.

Background
----------
The full Qwen3-Next server corrupts generation when
``fuse_blockscale_splitk_zero_init`` is enabled, even though:

* the direct custom-op contract is bit-for-bit correct
  (``debug_splitk_zero_init_correctness.py``), and
* a single compiled GEMM block through vLLM's real pass pipeline + a manual
  CUDA graph replay is also bit-for-bit correct
  (``debug_mini_vllm_splitk_zero_init_graph.py`` -- "Stage A").

This script implements the next stages of the
"Better isolated reproducer plan" in
``docs/issues/vllm-splitk-fusion/status-2026-05-28.md``:

Stage B (this file, ``--scenario seq|parallel``):
    Put *several* rewritten ``preop -> quant -> blockscale_gemm`` blocks in
    one compiled graph and check that repeated
    ``auto_functionalized(...with_zero_init...)`` + splitK sites still match
    baseline across eager / compiled / CUDA-graph replay. Two block
    topologies are exercised:

    * ``seq``      -- a deep chain (block i+1 consumes block i's output).
                      Stresses output lifetimes / buffer reuse between
                      consecutive rewritten sites (N == K so outputs chain).
    * ``parallel`` -- one quantized activation fans out to several GEMMs whose
                      outputs are summed. Stresses the *shared producer*
                      situation and allocator pressure.

The harness reuses Stage A's exact vLLM pass wiring (pre-grad IR
functionalization + PostGradPassManager in the compile-range context, same
Inductor options) by importing the helpers from
``debug_mini_vllm_splitk_zero_init_graph``.

Two correctness probes go beyond Stage A and target the human-flagged
landmines (CSV-lookup staleness during capture; whether the producer's
grid-strided zero-init really re-runs on replay so the splitK atomic-add
does not drift):

1. Multi-replay drift test -- after a single capture, the graph is replayed
   over several *different* inputs (and one repeated input). Each replayed
   output is compared against a fresh compiled call on the same input. If the
   captured graph has a stale buffer, or the producer zero-init does not re-run
   before the splitK atomic accumulate, the fused replay will drift away from
   the fresh reference while the baseline replay stays correct.
2. Dirty-buffer capture -- the static input buffer is pre-filled with garbage
   before capture, mirroring vLLM's persistent-buffer reuse.

Exit code is non-zero if any compared tensor pair diverges.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_TARGET_DEVICE", "rocm")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER_TRITON_GEMM", "1")

import torch
import torch.nn as nn

import vllm.ir.ops  # noqa: F401  Registers vllm_ir.rms_norm.
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.utils import Range

# Reuse Stage A's known-good vLLM pass wiring verbatim so the only variable
# here is the *graph shape* (multiple rewritten blocks), not the harness.
from debug_mini_vllm_splitk_zero_init_graph import (  # noqa: E402
    EPS,
    GROUP_SIZE,
    _compile_module,
    _make_vllm_config,
    _max_abs,
    _sync,
)


# ---------------------------------------------------------------------------
# Multi-block modules. Weights are baked as module buffers (like a real
# model), so forward() takes only the dynamic activation ``x`` -- this keeps
# the CUDA-graph capture/replay to a single dynamic input, matching how vLLM
# feeds persistent activation buffers into the compiled backbone.
# ---------------------------------------------------------------------------


def _quantized_weight(n: int, k: int, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Make an FP8 block-scale weight (B, Bs) from a bounded bf16 init."""
    w_bf16 = (torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * scale)
    b, b_scale = torch.ops.vllm.rocm_aiter_group_fp8_quant(w_bf16, GROUP_SIZE)
    return b, b_scale


def _quant(producer: str, h: torch.Tensor, norm_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the chosen producer -> (fp8, scales).

    ``rmsnorm`` feeds ``vllm_ir.rms_norm`` into ``rocm_aiter_group_fp8_quant``;
    vLLM's upstream norm_quant fusion rewrites that pair into
    ``rocm_aiter_rmsnorm_fp8_group_quant`` before the blockscale pass runs.
    ``group`` uses the bare group quant (no norm)."""
    if producer == "rmsnorm":
        normed = torch.ops.vllm_ir.rms_norm(h, norm_w, EPS)
        return torch.ops.vllm.rocm_aiter_group_fp8_quant(normed, GROUP_SIZE)
    if producer == "group":
        return torch.ops.vllm.rocm_aiter_group_fp8_quant(h, GROUP_SIZE)
    raise ValueError(f"unknown producer: {producer}")


class SeqBlocks(nn.Module):
    """Deep chain: out_{i+1} = block_i(out_i), with N == K so outputs chain.

    Each block is ``producer -> blockscale_gemm``. Every block has its *own*
    producer, so the fusion should rewrite all ``num_blocks`` sites. With the
    ``rmsnorm`` producer the per-block normalization also keeps activations
    bounded across many fp8 GEMMs."""

    def __init__(self, k: int, num_blocks: int, producer: str) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.producer = producer
        scale = 1.0 / math.sqrt(k)
        for i in range(num_blocks):
            self.register_buffer(
                f"norm_w_{i}", torch.ones((k,), dtype=torch.bfloat16)
            )
            b, b_scale = _quantized_weight(k, k, scale)
            self.register_buffer(f"w_{i}", b)
            self.register_buffer(f"ws_{i}", b_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i in range(self.num_blocks):
            a, a_scale = _quant(self.producer, h, getattr(self, f"norm_w_{i}"))
            h = torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
                a, getattr(self, f"w_{i}"), a_scale, getattr(self, f"ws_{i}"),
                torch.bfloat16,
            )
        return h


class ParallelBlocks(nn.Module):
    """Fan-out: one shared producer feeds ``num_blocks`` GEMMs.

    Outputs (M, N) are summed. This is the *shared producer* situation: a
    single quantized activation is consumed by several blockscale GEMMs. The
    Inductor pattern matcher applies non-overlapping matches, so only ONE GEMM
    can adopt the producer's ``gemm_out_zero_init`` buffer; the rest stay
    functional. The harness records the realized site counts rather than
    assuming all fuse."""

    def __init__(self, k: int, n: int, num_blocks: int, producer: str) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.producer = producer
        scale = 1.0 / math.sqrt(k)
        self.register_buffer("norm_w", torch.ones((k,), dtype=torch.bfloat16))
        for i in range(num_blocks):
            b, b_scale = _quantized_weight(n, k, scale)
            self.register_buffer(f"w_{i}", b)
            self.register_buffer(f"ws_{i}", b_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, a_scale = _quant(self.producer, x, self.norm_w)
        out = None
        for i in range(self.num_blocks):
            y = torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
                a, getattr(self, f"w_{i}"), a_scale, getattr(self, f"ws_{i}"),
                torch.bfloat16,
            )
            out = y if out is None else out + y
        return out


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    scenario: str
    producer: str
    m: int
    n: int
    k: int
    num_blocks: int
    dynamic: bool
    replays: int
    allow_buffer_reuse: bool
    # Rewritten-site bookkeeping read off the fused FX dump.
    fused_producer_sites: int
    fused_gemm_sites: int
    expected_producer_sites: int
    expected_gemm_sites: int
    fusion_fired: bool
    sites_as_expected: bool
    # Output scale + intrinsic atomic-add noise floor (baseline vs itself).
    output_rms: float
    baseline_self_noise: float
    baseline_cg_noise: float
    noise_floor: float
    tolerance: float
    # Single-call sanity (informational; nonzero is expected under splitK
    # atomic nondeterminism and the eager-vs-AITER-fused norm difference).
    baseline_vs_eager_max_abs: float
    fused_vs_baseline_compiled_max_abs: float
    # The headline corruption signals.
    fused_cg_vs_baseline_ref_max_abs: float
    fused_cg_vs_baseline_ref_rel: float
    fused_replay_self_drift_max_abs: float
    fused_has_nonfinite: bool
    # Detail per replay.
    per_replay: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = False


# ---------------------------------------------------------------------------
# CUDA-graph replay over multiple inputs (the drift probe)
# ---------------------------------------------------------------------------


def _rms(t: torch.Tensor) -> float:
    f = t.float()
    return float(f.pow(2).mean().sqrt().item()) if f.numel() else 0.0


def _has_nonfinite(t: torch.Tensor) -> bool:
    return bool((~torch.isfinite(t.float())).any().item()) if t.numel() else False


def _capture_and_replay(
    compiled: Any,
    capture_x: torch.Tensor,
    replay_inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Capture one CUDA graph then replay it over several inputs.

    ``capture_x`` is intentionally distinct garbage from the replay inputs and
    the static buffer is pre-dirtied, so a stale-capture bug cannot hide behind
    "the captured input happened to be the replay input".
    """
    static_x = capture_x.clone()
    # Pre-dirty: ensure nothing downstream silently relies on a zeroed buffer.
    static_x.add_(7.0)
    _ = compiled(static_x)
    _sync()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = compiled(static_x)
    _sync()

    outputs: list[torch.Tensor] = []
    for xr in replay_inputs:
        static_x.copy_(xr)
        graph.replay()
        _sync()
        outputs.append(static_out.clone())
    return outputs


# ---------------------------------------------------------------------------
# FX dump introspection: count rewritten sites in the fused graph
# ---------------------------------------------------------------------------

_PRODUCER_RE = re.compile(
    r"auto_functionalized\(torch\.ops\.vllm\.\S*_with_zero_init"
)
_GEMM_RE = re.compile(
    r"auto_functionalized\(torch\.ops\.vllm\.\S*_blockscale_splitk"
)


def _count_sites(dump_dir: Path) -> tuple[int, int]:
    """Return (producer_sites, gemm_sites) from the most-rewritten after-dump.

    The blockscale pass writes ``*.after.<seq>.<i>.fx.txt`` files; we take the
    max over all of them so a single fully-rewritten graph is reported even if
    torch.compile invoked the pass more than once.
    """
    best_prod = 0
    best_gemm = 0
    for fx_file in dump_dir.rglob("*.after.*.fx.txt"):
        text = fx_file.read_text(encoding="utf-8", errors="ignore")
        # Restrict to the readable class section to avoid double counting the
        # "raw graph" repetition at the bottom of each dump.
        readable = text.split("# ---- raw graph ----", 1)[0]
        prod = len(_PRODUCER_RE.findall(readable))
        gemm = len(_GEMM_RE.findall(readable))
        best_prod = max(best_prod, prod)
        best_gemm = max(best_gemm, gemm)
    return best_prod, best_gemm


# ---------------------------------------------------------------------------
# One scenario run
# ---------------------------------------------------------------------------


# How far the fused path may diverge from baseline before we call it
# corruption. Calibrated against baseline's own atomic-add noise floor:
# legitimate splitK atomic reordering keeps fused within a small multiple of
# baseline's run-to-run variation; the real serving bug produces garbage
# (relative error O(1) or non-finite), which is orders of magnitude beyond.
_DRIFT_FACTOR = 8.0
_REL_CORRUPTION = 0.10  # >10% relative divergence == corruption, not atomics


def _build_module(
    scenario: str, m: int, n: int, k: int, num_blocks: int, producer: str
) -> nn.Module:
    if scenario == "seq":
        # N == K for chaining; the provided n is ignored.
        return SeqBlocks(k, num_blocks, producer).cuda()
    if scenario == "parallel":
        return ParallelBlocks(k, n, num_blocks, producer).cuda()
    raise ValueError(f"unknown scenario: {scenario}")


def _run_one(
    scenario: str,
    producer: str,
    m: int,
    n: int,
    k: int,
    num_blocks: int,
    dynamic: bool,
    replays: int,
    dump_root: Path,
    allow_buffer_reuse: bool = True,
) -> RunResult:
    module = _build_module(scenario, m, n, k, num_blocks, producer)
    effective_n = k if scenario == "seq" else n
    tag = f"{scenario}_{producer}"

    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    # Distinct replay inputs; index 0 and the final index are the SAME tensor
    # so we can isolate replay-to-replay drift on a fixed input.
    replay_inputs = [torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
                     for _ in range(replays)]
    if replays >= 2:
        replay_inputs[-1] = replay_inputs[0].clone()

    with torch.no_grad():
        eager = module(x)
        _sync()

        compile_range = Range(1, max(8192, m))
        baseline_cfg = _make_vllm_config(
            fuse_zero_init=False, debug_dump_path=dump_root / f"{tag}_baseline"
        )
        fused_cfg = _make_vllm_config(
            fuse_zero_init=True, debug_dump_path=dump_root / f"{tag}_fused"
        )
        if not allow_buffer_reuse:
            # Root-cause probe: if disabling Inductor buffer reuse makes the
            # fused chain correct, the bug is the producer's zero-init buffer
            # (gemm_out_zero_init) being assigned storage that aliases the
            # producer's own input under buffer reuse.
            for cfg in (baseline_cfg, fused_cfg):
                cfg.compilation_config.inductor_compile_config[
                    "allow_buffer_reuse"
                ] = False
        baseline = _compile_module(module, baseline_cfg, compile_range, dynamic)
        fused = _compile_module(module, fused_cfg, compile_range, dynamic)

        # Two fresh baseline calls on x establish the intrinsic atomic-add
        # noise floor (splitK accumulates in a nondeterministic order).
        baseline_out = baseline(x).clone()
        baseline_out2 = baseline(x).clone()
        fused_out = fused(x).clone()
        _sync()

        # Fresh (non-cudagraph) compiled references per replay input, computed
        # before any capture mutates allocator state.
        baseline_refs = [baseline(xr).clone() for xr in replay_inputs]
        _sync()

        # Capture with garbage distinct from every replay input.
        capture_x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) + 3.0
        baseline_cg = _capture_and_replay(baseline, capture_x, replay_inputs)
        fused_cg = _capture_and_replay(fused, capture_x, replay_inputs)

    output_rms = _rms(baseline_out)
    baseline_self_noise = _max_abs(baseline_out, baseline_out2)
    baseline_vs_eager = _max_abs(baseline_out, eager)
    fused_vs_baseline_compiled = _max_abs(fused_out, baseline_out)

    per_replay: list[dict[str, Any]] = []
    baseline_cg_noise = 0.0
    fused_cg_vs_ref = 0.0
    for r in range(replays):
        b_cg = _max_abs(baseline_cg[r], baseline_refs[r])
        f_cg = _max_abs(fused_cg[r], baseline_refs[r])
        baseline_cg_noise = max(baseline_cg_noise, b_cg)
        fused_cg_vs_ref = max(fused_cg_vs_ref, f_cg)
        per_replay.append({
            "replay": r,
            "baseline_cg_vs_baseline_ref_max_abs": b_cg,
            "fused_cg_vs_baseline_ref_max_abs": f_cg,
        })

    # Replay-to-replay drift on a fixed input (replay[0] == replay[-1]). If the
    # producer zero-init stops re-running before the splitK atomic-add, this
    # grows; if it just reflects atomic reordering it stays near the noise floor.
    fused_self_drift = _max_abs(fused_cg[0], fused_cg[-1]) if replays >= 2 else 0.0

    fused_nonfinite = (
        _has_nonfinite(fused_out)
        or any(_has_nonfinite(t) for t in fused_cg)
    )

    # Self-calibrating tolerance: max of (a multiple of the measured atomic
    # noise floor) and (a relative-to-output-scale corruption threshold).
    noise_floor = max(baseline_self_noise, baseline_cg_noise)
    tolerance = max(_DRIFT_FACTOR * noise_floor, _REL_CORRUPTION * output_rms)
    fused_rel = fused_cg_vs_ref / output_rms if output_rms > 0 else float("inf")

    producer_sites, gemm_sites = _count_sites(dump_root / f"{tag}_fused")
    if scenario == "seq":
        expected_producer_sites = num_blocks
        expected_gemm_sites = num_blocks
    else:  # parallel: shared producer -> non-overlapping matcher fuses one
        expected_producer_sites = 1
        expected_gemm_sites = 1
    fusion_fired = producer_sites >= 1 and gemm_sites >= 1
    sites_as_expected = (
        producer_sites == expected_producer_sites
        and gemm_sites == expected_gemm_sites
    )

    passed = (
        fusion_fired
        and not fused_nonfinite
        and fused_cg_vs_ref <= tolerance
        and fused_self_drift <= max(tolerance, _DRIFT_FACTOR * noise_floor)
    )

    return RunResult(
        scenario=scenario,
        producer=producer,
        m=m,
        n=effective_n,
        k=k,
        num_blocks=num_blocks,
        dynamic=dynamic,
        replays=replays,
        allow_buffer_reuse=allow_buffer_reuse,
        fused_producer_sites=producer_sites,
        fused_gemm_sites=gemm_sites,
        expected_producer_sites=expected_producer_sites,
        expected_gemm_sites=expected_gemm_sites,
        fusion_fired=fusion_fired,
        sites_as_expected=sites_as_expected,
        output_rms=output_rms,
        baseline_self_noise=baseline_self_noise,
        baseline_cg_noise=baseline_cg_noise,
        noise_floor=noise_floor,
        tolerance=tolerance,
        baseline_vs_eager_max_abs=baseline_vs_eager,
        fused_vs_baseline_compiled_max_abs=fused_vs_baseline_compiled,
        fused_cg_vs_baseline_ref_max_abs=fused_cg_vs_ref,
        fused_cg_vs_baseline_ref_rel=fused_rel,
        fused_replay_self_drift_max_abs=fused_self_drift,
        fused_has_nonfinite=fused_nonfinite,
        per_replay=per_replay,
        passed=passed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=["seq", "parallel"],
        choices=["seq", "parallel"],
    )
    parser.add_argument(
        "--producer",
        nargs="+",
        default=["rmsnorm"],
        choices=["rmsnorm", "group"],
    )
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--n", type=int, default=64, help="GEMM N (parallel only)")
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--replays", type=int, default=4)
    # Dynamic (symbolic M) is the realistic vLLM mode AND the one where the
    # fusion's K>=2048 gate fires; static M with K<4096 never fuses.
    parser.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="symbolic batch dim (default: on)",
    )
    parser.add_argument(
        "--disable-buffer-reuse",
        action="store_true",
        help="set inductor allow_buffer_reuse=False (root-cause probe)",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--dump-root",
        type=Path,
        default=Path("benchmarks/zero_init_demo_results/runtime_reproducer"),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.k % GROUP_SIZE != 0:
        raise ValueError(f"k must be divisible by {GROUP_SIZE}, got {args.k}")

    torch.manual_seed(args.seed)
    rocm_aiter_ops.register_ops_once()
    args.dump_root.mkdir(parents=True, exist_ok=True)
    for child in args.dump_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)

    results = [
        _run_one(
            scenario,
            producer,
            args.m,
            args.n,
            args.k,
            args.num_blocks,
            args.dynamic,
            args.replays,
            args.dump_root,
            not args.disable_buffer_reuse,
        )
        for scenario in args.scenario
        for producer in args.producer
    ]
    payload = [asdict(result) for result in results]
    print(json.dumps(payload, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # Compact human-readable summary table.
    print("\n=== summary ===")
    for r in results:
        verdict = "PASS" if r.passed else "FAIL"
        print(
            f"[{verdict}] {r.scenario}/{r.producer} "
            f"blocks={r.num_blocks} sites(prod/gemm)={r.fused_producer_sites}/"
            f"{r.fused_gemm_sites}(exp {r.expected_producer_sites}/"
            f"{r.expected_gemm_sites}) "
            f"noise={r.noise_floor:.4g} tol={r.tolerance:.4g} "
            f"fused_cg_vs_ref={r.fused_cg_vs_baseline_ref_max_abs:.4g} "
            f"(rel={r.fused_cg_vs_baseline_ref_rel:.3g}) "
            f"drift={r.fused_replay_self_drift_max_abs:.4g} "
            f"nonfinite={r.fused_has_nonfinite}"
        )

    all_passed = all(result.passed for result in results)
    print(f"\n{'PASS' if all_passed else 'FAIL'}: "
          f"{sum(r.passed for r in results)}/{len(results)} configs passed")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
