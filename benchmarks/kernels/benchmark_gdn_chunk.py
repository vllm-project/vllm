# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness-first RDNA3.5 GDN prefill benchmark.

The primary comparison times the same public API with HIP enabled/disabled,
including wrapper allocations. Optional raw HIP timing excludes output allocation
and argument staging. Model presets use full head counts on a single GPU;
tensor-parallel and distributed benchmarks are not supported. Without arguments
every preset runs the same grid of batches, so the head shapes stay comparable;
--seqlens takes an explicit workload instead, with --hv/--hg for its head counts.

Timing uses `triton.testing.do_bench`, as the rest of the kernel benchmarks in
this repository do: it sizes the repetition count from a first measurement and
flushes the L2 cache before every run, so the reported medians are cold-cache
latencies of the whole public API call.

Examples::

    python benchmarks/kernels/benchmark_gdn_chunk.py
    python benchmarks/kernels/benchmark_gdn_chunk.py --dry-run --num-sms 20
    python benchmarks/kernels/benchmark_gdn_chunk.py \
        --preset dense --dtype fp16 --jsonl results.jsonl
    python benchmarks/kernels/benchmark_gdn_chunk.py \
        --hv 48 --hg 16 --seqlens 941 --seqlens 1,4095 --raw-hip
    python benchmarks/kernels/benchmark_gdn_chunk.py --check --exact
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import itertools
import json
import math
import os
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

DEFAULT_HV = 32
DEFAULT_HG = 16
HEAD_DIM = 128
DTYPES = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
}

# Every head shape runs the same batches, so the shapes stay comparable against
# each other rather than against a different set of cases each.
GRID = (
    (32,),
    (128,),
    (512,),
    (2048,),
    (32768,),
    (128,) * 2,
    (512,) * 2,
    (1, 4095),
    (3584, 512),
    (128,) * 4,
    (512,) * 4,
    (128,) * 8,
    (512,) * 8,
    (8192,) * 8,
    (2048,) * 32,
    (256,) * 256,
)


@dataclass(frozen=True)
class Preset:
    name: str
    hv: int
    hk: int
    models: str


PRESETS = (
    Preset("small", 16, 16, "Qwen3.5-0.8B/2B"),
    Preset("medium", 32, 16, "Qwen3.5-9B/35B/Next"),
    Preset("dense", 48, 16, "Qwen3.6-27B"),
    Preset("large", 64, 16, "Qwen3.5-122B/397B"),
)


@dataclass(frozen=True)
class Case:
    seqlens: tuple[int, ...]
    hv: int
    hk: int
    seed: int
    preset: str


def generate_cases(args) -> list[Case]:
    """CPU-only case planning; no torch or device discovery."""
    if args.seqlens:
        return [
            Case(
                tuple(int(s) for s in batch.split(",")),
                args.hv,
                args.hg,
                args.seed,
                "manual",
            )
            for batch in args.seqlens
        ]
    return [
        Case(lengths, preset.hv, preset.hk, args.seed, preset.name)
        for preset in PRESETS
        if not args.preset or preset.name in args.preset
        for lengths in GRID
    ]


def case_record(case: Case, index: int, args, num_sms: int, sms_source: str) -> dict:
    chunks = {b: sum((s + b - 1) // b for s in case.seqlens) for b in (32, 64)}
    total, n = sum(case.seqlens), len(case.seqlens)
    return {
        "type": "case",
        "index": index,
        "seqlens": case.seqlens,
        "T": total,
        "N": n,
        "Hk": case.hk,
        "Hv": case.hv,
        "preset": case.preset,
        "seed": case.seed,
        "config": {
            "head_dim": HEAD_DIM,
            "dtype": args.dtype,
            "g_scale": args.g_scale,
        },
        "chunks32": chunks[32],
        "chunks64": chunks[64],
        "padding_tokens": {b: count * b - total for b, count in chunks.items()},
        "activation_bytes": total
        * (2 * case.hk * 128 * 2 + case.hv * 128 * 2 + case.hv * 8),
        "initial_state_bytes": n * case.hv * 128 * 128 * 4,
        "hip_blocks": n * case.hv,
        "mode": {
            "value": "CU" if n * case.hv > num_sms else "WGP",
            "source": "inferred from N*Hv > num_sms",
            "num_sms": num_sms,
            "num_sms_source": sms_source,
        },
        "status": "planned",
        "numerics": {},
        "timings": {},
    }


def load_runtime():
    """Keep planning/help independent of torch, vLLM, and device discovery."""
    global torch, triton, envs, chunk_module, rocm_gdn, current_platform
    global chunk_gated_delta_rule, prepare_chunk_indices
    global prepare_chunk_offsets, FLA_CHUNK_SIZE
    import torch
    import triton.testing

    import vllm.envs as envs
    import vllm.third_party.flash_linear_attention.ops.chunk as chunk_module
    from vllm.platforms import current_platform
    from vllm.third_party.flash_linear_attention.ops import (
        rocm_rdna35_gdn_chunked as rocm_gdn,
    )
    from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
    from vllm.third_party.flash_linear_attention.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

    if not current_platform.is_rocm() or not torch.version.hip:
        raise RuntimeError("This benchmark requires ROCm, not CUDA/CPU")
    from vllm.platforms.rocm import on_gfx115x

    if not on_gfx115x():
        raise RuntimeError("This benchmark requires gfx115x (RDNA3.5)")
    with force_backend(True):
        if not rocm_gdn._available():
            raise RuntimeError("ROCm gdn_chunked op is unavailable in this build")
    if FLA_CHUNK_SIZE != 64:
        raise RuntimeError("The reported chunk counts assume Triton FLA_CHUNK_SIZE=64")


@contextmanager
def force_backend(enabled: bool):
    """Never enter this context inside a timed callable."""
    saved = envs.VLLM_GDN_HIP
    try:
        envs.VLLM_GDN_HIP = enabled
        rocm_gdn._available.cache_clear()
        yield
    finally:
        envs.VLLM_GDN_HIP = saved
        rocm_gdn._available.cache_clear()


class Inputs:
    """One prefill batch, laid out exactly as the model hands it to the op."""

    def __init__(
        self,
        seqlens: list[int],
        hv: int,
        hg: int,
        dtype: torch.dtype,
        device: torch.device,
        seed: int = 0,
        g_scale: float = 0.05,
    ):
        gen = torch.Generator(device=device).manual_seed(seed)
        total = sum(seqlens)
        self.seqlens = seqlens
        self.n_seqs = len(seqlens)

        def randn(*shape, dt=dtype):
            return torch.randn(*shape, generator=gen, device=device, dtype=dt)

        # q and k reach the op l2-normalised, and the conditioning of (I + A)
        # depends on it.
        self.q = torch.nn.functional.normalize(randn(1, total, hg, HEAD_DIM), dim=-1)
        self.k = torch.nn.functional.normalize(randn(1, total, hg, HEAD_DIM), dim=-1)
        self.v = randn(1, total, hv, HEAD_DIM)
        # g is log-space decay: g = -exp(A_log) * softplus(a + dt_bias) <= 0.
        # g_scale sets how fast the state forgets.  Large values crush the
        # within-chunk decay exp(g_i - g_j) to zero, degenerating Tinv to the
        # identity; real gates sit near zero, which is what exercises the
        # triangular inverse and the chunk-to-chunk carry.
        self.g = -g_scale * torch.nn.functional.softplus(
            randn(1, total, hv, dt=torch.float32)
        )
        self.beta = torch.sigmoid(randn(1, total, hv, dt=torch.float32))
        self.h0 = randn(self.n_seqs, hv, HEAD_DIM, HEAD_DIM, dt=torch.float32)

        self.cu_seqlens = torch.tensor(
            [0, *itertools.accumulate(seqlens)], dtype=torch.int32, device=device
        )
        self.chunk_indices = prepare_chunk_indices(self.cu_seqlens, FLA_CHUNK_SIZE)
        self.chunk_offsets = prepare_chunk_offsets(self.cu_seqlens, FLA_CHUNK_SIZE)
        self.scale = HEAD_DIM**-0.5

    def baseline(self):
        """Public API, including its wrapper allocations but no benchmark copies."""
        return chunk_gated_delta_rule(
            q=self.q,
            k=self.k,
            v=self.v,
            g=self.g,
            beta=self.beta,
            scale=self.scale,
            initial_state=self.h0,
            output_final_state=True,
            cu_seqlens=self.cu_seqlens,
            chunk_indices=self.chunk_indices,
            chunk_offsets=self.chunk_offsets,
            use_qk_l2norm_in_kernel=False,
        )

    def reference(self):
        """Token-by-token gated delta rule in fp64.

        The chunked formulation collapses to this at chunk size 1::

            S <- S * exp(g_t)
            u <- beta_t * (v_t - S k_t)
            S <- S + outer(u, k_t)
            o <- scale * S q_t

        Slow, but it is the only thing here that is not itself bf16, so it is
        what decides whether a kernel is better or worse rather than merely
        different.
        """
        q = self.q[0].double()
        k = self.k[0].double()
        v = self.v[0].double()
        g = self.g[0].double()
        beta = self.beta[0].double()
        h = self.h0.double().clone()  # [N, H, V, K]
        hv = v.shape[1]
        rep = hv // k.shape[1]
        o = torch.empty_like(v)
        for i_n in range(self.n_seqs):
            bos = int(self.cu_seqlens[i_n])
            eos = int(self.cu_seqlens[i_n + 1])
            s = h[i_n]  # [H, V, K]
            for t in range(bos, eos):
                kt = k[t].repeat_interleave(rep, dim=0)  # [H, K]
                qt = q[t].repeat_interleave(rep, dim=0)  # [H, K]
                s = s * torch.exp(g[t])[:, None, None]
                u = beta[t][:, None] * (v[t] - torch.einsum("hvk,hk->hv", s, kt))
                s = s + u[:, :, None] * kt[:, None, :]
                o[t] = self.scale * torch.einsum("hvk,hk->hv", s, qt)
            h[i_n] = s
        return o.unsqueeze(0), h


def raw_hip_call(inp: Inputs):
    """Stage once; time only the raw torch op with reusable output/state buffers."""
    out, state = torch.empty_like(inp.v), torch.empty_like(inp.h0)
    arguments = (
        inp.q.squeeze(0).contiguous(),
        inp.k.squeeze(0).contiguous(),
        inp.v.squeeze(0).contiguous(),
        inp.g.squeeze(0).float().contiguous(),
        inp.beta.squeeze(0).float().contiguous(),
        inp.h0.float().contiguous(),
        inp.cu_seqlens.to(torch.int32).contiguous(),
        out.squeeze(0),
        state,
        float(inp.scale),
    )

    def call():
        torch.ops._rocm_C.gdn_chunked(*arguments)
        return out, state

    return call


def safe_ratio(numerator: float, denominator: float):
    return numerator / denominator if denominator else (0.0 if numerator == 0 else None)


def compare(got, ref, seqlens, *, state=False, rtol=2e-2, atol=1e-5) -> dict:
    """Global and every sequence/head RMS with bounded fp64 scratch.

    Acceptance is RMS(error) <= atol + rtol * RMS(reference), both globally and
    for every sequence/head. Max-absolute error is diagnostic, not an elementwise
    relative test near zero. A null relative error means a nonzero error/zero ref.
    """
    if got is None or ref is None or got.shape != ref.shape:
        return {"passed": False, "error": "missing tensor or shape mismatch"}
    hv = got.shape[1] if state else got.shape[2]
    sums = [0.0, 0.0]
    elements, max_abs, ref_peak, start = 0, 0.0, 0.0, 0
    worst_rms = worst_scaled = None
    passed = True
    for seq, length in enumerate(seqlens):
        acc = torch.zeros((2, hv), dtype=torch.float64, device=got.device)
        peaks = torch.zeros(2, dtype=torch.float64, device=got.device)
        finite = torch.ones((), dtype=torch.bool, device=got.device)
        for offset in range(0, 1 if state else length, 256):
            if state:
                a, b = got[seq].double(), ref[seq].double()
                axes = (1, 2)
            else:
                end = min(offset + 256, length)
                a = got[0, start + offset : start + end].double()
                b = ref[0, start + offset : start + end].double()
                axes = (0, 2)
            finite &= torch.isfinite(a).all() & torch.isfinite(b).all()
            error = a - b
            acc[0] += error.square().sum(dim=axes)
            acc[1] += b.square().sum(dim=axes)
            peaks = torch.maximum(
                peaks, torch.stack((error.abs().max(), b.abs().max()))
            )
        if not finite.item():
            return {"passed": False, "finite": False, "sequence": seq}
        count = 128 * 128 if state else length * 128
        err_sums, ref_sums = acc.tolist()
        peak_error, peak_ref = peaks.tolist()
        max_abs, ref_peak = max(max_abs, peak_error), max(ref_peak, peak_ref)
        for head, (es, rs) in enumerate(zip(err_sums, ref_sums)):
            rms, ref_rms = math.sqrt(es / count), math.sqrt(rs / count)
            scaled = rms / (atol + rtol * ref_rms)
            entry = {
                "sequence": seq,
                "head": head,
                "rms": rms,
                "reference_rms": ref_rms,
                "relative_rms": safe_ratio(rms, ref_rms),
                "tolerance_ratio": scaled,
            }
            if worst_rms is None or rms > worst_rms["rms"]:
                worst_rms = entry
            if worst_scaled is None or scaled > worst_scaled["tolerance_ratio"]:
                worst_scaled = entry
            passed &= scaled <= 1
        sums[0] += sum(err_sums)
        sums[1] += sum(ref_sums)
        elements += count * hv
        start += length
    rms, ref_rms = (math.sqrt(s / elements) for s in sums)
    return {
        "passed": bool(passed and rms <= atol + rtol * ref_rms),
        "finite": True,
        "rms": rms,
        "reference_rms": ref_rms,
        "relative_rms": safe_ratio(rms, ref_rms),
        "max_abs": max_abs,
        "reference_peak": ref_peak,
        "max_abs_over_peak": safe_ratio(max_abs, ref_peak),
        "rtol": rtol,
        "atol": atol,
        "worst_sequence_head_rms": worst_rms,
        "worst_sequence_head_tolerance": worst_scaled,
    }


def correctness(inp: Inputs, args, record: dict, raw_call) -> None:
    """Compile/verify both public paths once, before any timing or warmup."""
    numerics = record["numerics"]
    snapshot = inp.h0.clone()
    results = {}
    for name, enabled in (("triton_api", False), ("hip_api", True)):
        with (
            force_backend(enabled),
            patch.object(
                chunk_module, "chunk_gdn_hip_fwd", wraps=chunk_module.chunk_gdn_hip_fwd
            ) as probe,
        ):
            results[name] = inp.baseline()
            count = probe.call_count
        numerics[f"{name}_hip_dispatch_count"] = count
        unchanged = torch.equal(inp.h0, snapshot)
        numerics[f"{name}_initial_state_unchanged"] = unchanged
        if count != int(enabled) or not unchanged:
            raise RuntimeError(
                f"{name}: dispatch count={count}, h0 unchanged={unchanged}"
            )
    if raw_call is not None:
        results["raw_hip_op"] = raw_call()
        unchanged = torch.equal(inp.h0, snapshot)
        numerics["raw_hip_initial_state_unchanged"] = unchanged
        if not unchanged:
            raise RuntimeError("raw HIP mutated the initial state")
    del snapshot
    for name in results:
        if name == "triton_api":
            continue
        for i, label in enumerate(("out", "state")):
            numerics[f"{name}_vs_triton_{label}"] = compare(
                results[name][i], results["triton_api"][i], inp.seqlens, state=bool(i)
            )
    if args.exact:
        reference = inp.reference()
        for name, tensors in results.items():
            for i, label in enumerate(("out", "state")):
                numerics[f"{name}_vs_fp64_{label}"] = compare(
                    tensors[i],
                    reference[i],
                    inp.seqlens,
                    state=bool(i),
                    rtol=1e-4 if i and name != "triton_api" else 2e-2,
                )
        for label in ("out", "state"):
            numerics[f"hip_over_triton_fp64_{label}_rms"] = (
                safe_ratio(
                    numerics[f"hip_api_vs_fp64_{label}"]["rms"],
                    numerics[f"triton_api_vs_fp64_{label}"]["rms"],
                )
                if all(
                    "rms" in numerics[f"{name}_vs_fp64_{label}"]
                    for name in ("hip_api", "triton_api")
                )
                else None
            )
    failed = [k for k, v in numerics.items() if isinstance(v, dict) and not v["passed"]]
    for key, value in numerics.items():
        if isinstance(value, dict):
            print(f"  {key}: {json.dumps(value, allow_nan=False)}")
    if failed:
        raise RuntimeError(f"Numerical checks failed: {', '.join(failed)}")


def measure(inp: Inputs, args, record: dict, raw_call) -> None:
    """Time each path with triton.testing.do_bench, which flushes L2 per run."""
    quantiles = [0.5, 0.1, 0.9]
    backends = {"triton_api": inp.baseline, "hip_api": inp.baseline}
    if raw_call is not None:
        backends["raw_hip_op"] = raw_call
    timings = record["timings"]
    for name, fn in backends.items():
        with force_backend(name != "triton_api"):
            median, p10, p90 = triton.testing.do_bench(fn, quantiles=quantiles)
        timings[name] = {"median_ms": median, "p10_ms": p10, "p90_ms": p90}
        print(f"  {name}: {json.dumps(timings[name])}")
    timings["public_api_speedup"] = (
        timings["triton_api"]["median_ms"] / timings["hip_api"]["median_ms"]
    )
    print(f"  public API speedup={timings['public_api_speedup']:.3f}x")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seqlens", action="append", help="comma-separated batch; repeatable"
    )
    p.add_argument(
        "--hv", type=int, default=DEFAULT_HV, help="value heads, manual runs"
    )
    p.add_argument("--hg", type=int, default=DEFAULT_HG, help="key heads, manual runs")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    p.add_argument("--g-scale", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check", action="store_true", help="correctness only, no timing")
    p.add_argument(
        "--exact", action="store_true", help="fp64 reference (T<=4096,N<=16)"
    )
    p.add_argument(
        "--raw-hip", action="store_true", help="also time preallocated raw torch op"
    )
    p.add_argument("--preset", action="append", choices=[p.name for p in PRESETS])
    p.add_argument(
        "--num-sms", type=int, help="launch planning override; dry-run default 20"
    )
    p.add_argument("--limit", type=int, help="case count to run")
    p.add_argument(
        "--dry-run", action="store_true", help="CPU-only planning; no torch/vLLM"
    )
    p.add_argument(
        "--jsonl", type=Path, help="exclusive-create output; never overwrite"
    )
    args = p.parse_args()
    if args.hv <= 0 or args.hg <= 0 or args.hv % args.hg:
        p.error("Hv and Hk must be positive, with an integral positive Hv/Hk ratio")
    if args.limit is not None and args.limit <= 0:
        p.error("--limit must be positive")
    if args.num_sms is not None and args.num_sms <= 0:
        p.error("--num-sms must be positive")
    if not math.isfinite(args.g_scale) or args.g_scale < 0:
        p.error("--g-scale must be finite and nonnegative")
    if not 0 <= args.seed < 2**63:
        p.error("--seed must be in [0, 2**63)")
    if args.preset and args.seqlens:
        p.error("--preset and --seqlens are mutually exclusive")
    if args.exact and not args.check:
        p.error("--exact requires --check")
    args.dtype = DTYPES[args.dtype]
    try:
        for batch in args.seqlens or []:
            if any(int(s) <= 0 for s in batch.split(",")):
                raise ValueError
    except ValueError:
        p.error("--seqlens requires comma-separated positive integers")
    return args


def run_metadata(args, num_sms, sms_source, gpu):
    root = Path(__file__).resolve().parents[2]

    def git(*command):
        result = subprocess.run(
            ["git", "-C", str(root), *command],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    versions = {}
    for package in ("torch", "vllm", "triton"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "type": "run",
        "started_at_unix": time.time(),
        "args": vars(args),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
        "versions": versions,
        "gpu": gpu,
        "num_sms": num_sms,
        "num_sms_source": sms_source,
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(
                (
                    "VLLM_GDN",
                    "GDN_",
                    "FLA_",
                    "HIP_VISIBLE",
                    "ROCR_VISIBLE",
                    "CUDA_VISIBLE",
                )
            )
        },
        "presets": [vars(p) for p in PRESETS],
    }


def run(args, output) -> int:
    run_start = time.monotonic()

    def emit(record):
        if output is not None:
            output.write(json.dumps(record, default=str, allow_nan=False) + "\n")
            output.flush()

    num_sms, sms_source = args.num_sms or 20, "explicit" if args.num_sms else "assumed"
    gpu = None
    if not args.dry_run:
        load_runtime()
        from vllm.triton_utils import triton

        actual_sms = current_platform.num_compute_units()
        if args.num_sms is not None and args.num_sms != actual_sms:
            raise RuntimeError(
                f"--num-sms={args.num_sms} differs from device {actual_sms}"
            )
        num_sms, sms_source = actual_sms, "device"
        gpu = {
            "name": current_platform.get_device_name(),
            "family": "gfx115x",
            "architecture": triton.runtime.driver.active.get_current_target().arch,
            "total_memory_bytes": current_platform.get_device_total_memory(),
            "hip_version": torch.version.hip,
        }
    emit(run_metadata(args, num_sms, sms_source, gpu))
    cases = generate_cases(args)
    selected = list(enumerate(cases))
    if args.limit is not None:
        selected = selected[: args.limit]
    if args.exact and any(
        sum(c.seqlens) > 4096 or len(c.seqlens) > 16 for _, c in selected
    ):
        raise ValueError("--exact requires small selected cases: T<=4096 and N<=16")
    print(
        f"num_sms={num_sms} ({sms_source}); "
        f"generated={len(cases)}, selected={len(selected)}"
    )
    counts = Counter()
    completed_seconds = 0.0
    quick_results = []
    eligible = len(selected)
    for index, case in selected:
        case_start = time.monotonic()
        record = case_record(case, index, args, num_sms, sms_source)
        record["wall_seconds"] = {}
        record["started_at_unix"] = time.time()
        print(
            f"case {index}: T={record['T']} N={record['N']} Hv={case.hv} Hk={case.hk} "
            f"preset={case.preset} dtype={args.dtype} "
            f"mode={record['mode']['value']} (inferred)"
        )
        if args.dry_run:
            record["status"] = "dry_run"
            print(
                f"  seqlens={list(case.seqlens)} chunks32={record['chunks32']} "
                f"chunks64={record['chunks64']} padding={record['padding_tokens']}"
            )
        else:
            try:
                with torch.inference_mode():
                    inp = Inputs(
                        list(case.seqlens),
                        case.hv,
                        case.hk,
                        getattr(torch, args.dtype),
                        torch.device("cuda"),
                        case.seed,
                        args.g_scale,
                    )
                    raw_call = raw_hip_call(inp) if args.raw_hip else None
                    torch.accelerator.synchronize()
                    record["wall_seconds"]["setup"] = time.monotonic() - case_start
                    phase_start = time.monotonic()
                    correctness(inp, args, record, raw_call)
                    torch.accelerator.synchronize()
                    record["wall_seconds"]["correctness"] = (
                        time.monotonic() - phase_start
                    )
                    if not args.check:
                        phase_start = time.monotonic()
                        measure(inp, args, record, raw_call)
                        record["wall_seconds"]["measurement"] = (
                            time.monotonic() - phase_start
                        )
                    record["status"] = "checked" if args.check else "ok"
                    del raw_call, inp
            except Exception as exc:
                record["status"] = "error"
                record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                print(f"  ERROR: {type(exc).__name__}: {exc}")
        record["wall_seconds"]["total"] = time.monotonic() - case_start
        counts[record["status"]] += 1
        if record["status"] in ("ok", "checked"):
            completed_seconds += record["wall_seconds"]["total"]
            completed = counts["ok"] + counts["checked"]
            record["eta_seconds"] = (
                completed_seconds / completed * (eligible - completed)
            )
            print(
                f"  elapsed={time.monotonic() - run_start:.1f}s "
                f"remaining~{record['eta_seconds']:.1f}s ({completed}/{eligible})"
            )
            speedup = record["timings"].get("public_api_speedup")
            if speedup is not None:
                quick_results.append(
                    {
                        "index": index,
                        "speedup": speedup,
                        "observation": "faster" if speedup > 1 else "not faster",
                    }
                )
        emit(record)
        if record["status"] == "error":
            break  # OOM/dispatch/numerical errors are never successful skips.
    summary = {
        "type": "summary",
        "generated": len(cases),
        "selected": len(selected),
        "processed": sum(counts.values()),
        "counts": dict(counts),
        "no_cases_selected": not selected,
        "not_processed": len(selected) - sum(counts.values()),
        "elapsed_seconds": time.monotonic() - run_start,
        "speedup_observations": quick_results,
    }
    print(f"Summary: {json.dumps(summary)}")
    emit(summary)
    return int(bool(counts["error"]))


def main() -> int:
    args = parse_args()
    try:
        if args.jsonl:
            with args.jsonl.open("x", encoding="utf-8") as output:
                return run(args, output)
        return run(args, None)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
