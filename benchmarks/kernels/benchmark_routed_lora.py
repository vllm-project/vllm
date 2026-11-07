# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare complete routed-LoRA updates, without loading a model.

Run from the repository root:
    .venv/bin/python -m benchmarks.kernels.benchmark_routed_lora --suite smoke

Metadata construction (per-route groups, LoRAKernelMeta, route plans) is
outside operation timing, as in vLLM where it runs once per step. The fused
op's scratch is op-owned and self-clearing; baselines clear theirs inside
timing. No baseline imports tests or changes the production dispatch path.
"""

import argparse
import csv
import itertools
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from benchmarks.kernels.utils import Bench


@dataclass(frozen=True)
class Case:
    tokens: int
    hidden: int
    output: int
    rank: int = 32
    routes: int = 2
    adapters: int = 8
    dtype: str = "bfloat16"
    distribution: str = "balanced"


class Experiment:
    def __init__(self, case: Case, seed: int):
        from vllm.lora.ops.triton_ops import LoRAKernelMeta

        self.case = case
        generator = torch.Generator().manual_seed(seed)
        dtype = getattr(torch, case.dtype)
        t, h, o, r, k, e = (
            case.tokens,
            case.hidden,
            case.output,
            case.rank,
            case.routes,
            case.adapters,
        )
        requests = min(t, 8)
        plans = torch.stack(
            [torch.randperm(e, generator=generator)[:k] for _ in range(requests)]
        )
        if case.distribution == "hot":
            for row in range(requests):
                if row % 5 != 0:
                    others = plans[row][plans[row] != 0]
                    plans[row] = torch.cat((torch.tensor([0]), others))[:k]
        self.slots_cpu = plans[torch.arange(t) * requests // t]
        self.weights = torch.rand(t, k, generator=generator)
        self.weights /= self.weights.sum(-1, keepdim=True)
        self.weights = self.weights.cuda()
        self.x = torch.randn(t, h, generator=generator).to("cuda", dtype)
        self.a = (torch.randn(e, r, h, generator=generator) / h**0.5).to("cuda", dtype)
        self.b = (torch.randn(e, o, r, generator=generator) / r**0.5).to("cuda", dtype)
        self.initial = torch.randn(t, o, generator=generator).to("cuda", dtype)
        self.y = self.initial.clone()
        self.scratch = torch.empty(k, t, r, device="cuda", dtype=torch.float32)
        self.shrink = torch.empty(1, t, r, device="cuda", dtype=torch.float32)
        self.metas = []
        self.groups = []
        sorted_rows, offsets = [], []
        torch.accelerator.synchronize()
        metadata_start = time.perf_counter()
        for route in range(k):
            ids = self.slots_cpu[:, route]
            order = torch.argsort(ids, stable=True)
            counts = torch.bincount(ids, minlength=e)
            sorted_rows.append(order)
            offsets.append(
                torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
            )
            meta = LoRAKernelMeta.make(e, t, "cuda")
            meta.prepare_tensors(ids.cuda())
            self.metas.append(meta)
            self.groups.append(
                [
                    (adapter, torch.where(ids == adapter)[0].cuda())
                    for adapter in range(e)
                    if torch.any(ids == adapter)
                ]
            )
        self.indices = torch.stack(sorted_rows).cuda()
        self.offsets = torch.stack(offsets).cuda()
        from vllm.lora.punica_wrapper.utils import make_route_plans

        self.plans, self.plan_tokens, self.plan_starts_loc = (
            torch.from_numpy(array).cuda()
            for array in make_route_plans(self.slots_cpu.numpy())
        )
        torch.accelerator.synchronize()
        self.metadata_setup_us = (time.perf_counter() - metadata_start) * 1e6

    def run(self, provider: str):
        if provider == "candidate":
            from vllm.lora.ops.triton_ops.routed_lora_op import routed_lora

            routed_lora(
                self.x,
                self.a,
                self.b,
                self.y,
                self.weights,
                self.plans,
                self.plan_tokens,
                self.plan_starts_loc,
            )
        elif provider == "eager":
            for route, groups in enumerate(self.groups):
                for adapter, rows in groups:
                    z = self.x[rows] @ self.a[adapter].T
                    delta = z @ self.b[adapter].T
                    self.y[rows] += delta * self.weights[rows, route, None].to(
                        delta.dtype
                    )
        elif provider == "composed":
            from vllm.lora.ops.triton_ops import lora_expand, lora_shrink

            # Strongest existing-kernel composition: scale the low-rank shrink
            # by the route coefficient, then expand straight into the output.
            for route, meta in enumerate(self.metas):
                args = meta.meta_args(self.case.tokens, False)
                lora_shrink(self.x, [self.a], self.shrink, *args, 1.0)
                self.shrink.mul_(self.weights[:, route].view(1, -1, 1))
                lora_expand(self.shrink, [self.b], self.y, *args, add_inputs=True)
        else:
            raise ValueError(provider)

    def reference(self):
        expected = self.initial.float()
        for route, groups in enumerate(self.groups):
            for adapter, rows in groups:
                delta = (self.x[rows].float() @ self.a[adapter].float().T) @ self.b[
                    adapter
                ].float().T
                expected[rows] += delta * self.weights[rows, route, None]
        return expected.to(self.y.dtype)

    def validate(self, provider: str):
        self.y.copy_(self.initial)
        self.run(provider)
        expected = self.reference()
        torch.testing.assert_close(self.y, expected, atol=6e-2, rtol=6e-2)
        error = (self.y.float() - expected.float()).abs()
        return {
            "max_abs_error": error.max().item(),
            "max_relative_error": (error / expected.float().abs().clamp_min(1e-6))
            .max()
            .item(),
        }


def cases(suite: str):
    if suite == "smoke":
        return [
            Case(t, 4096, 4096, dtype=d)
            for t in (1, 128)
            for d in ("float16", "bfloat16")
        ]
    dimensions = ((4096, 4096), (4096, 14336), (14336, 4096))
    result = []
    for (h, o), dtype, distribution in itertools.product(
        dimensions, ("float16", "bfloat16"), ("balanced", "hot")
    ):
        result.extend(
            Case(t, h, o, 32, k, dtype=dtype, distribution=distribution)
            for t, k in itertools.product((1, 16, 128, 1024, 8192), (1, 2, 4))
        )
        result.extend(
            Case(t, h, o, r, 2, dtype=dtype, distribution=distribution)
            for t, r in itertools.product((1, 128, 8192), (8, 128))
        )
    return result


def measure(experiment: Experiment, provider: str, mode: str, cache: str):
    from importlib.metadata import version

    from cupti import cupti  # noqa: F401
    from flashinfer.testing import bench_gpu_time_with_cupti

    if int(version("cupti-python").split(".")[0]) < 13:
        raise RuntimeError("CUPTI >=13 is required for this experiment")

    def run():
        experiment.run(provider)

    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    experiment.y.copy_(experiment.initial)
    samples = [
        value * 1000
        for value in bench_gpu_time_with_cupti(
            run,
            dry_run_iters=3,
            repeat_iters=20,
            use_cuda_graph=mode == "graph",
            cold_l2_cache=cache == "cold",
        )
    ]
    host_us = None
    if mode == "direct":
        # The existing shared helper measures host-visible latency separately.
        experiment.y.copy_(experiment.initial)
        timer = Bench(None, "routed_lora", provider, "host-visible", run)
        timer.min_run_time = 0.05
        host_us = timer.run().median * 1e6
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "host_visible_us": host_us,
        "timing_backend": "flashinfer_cupti",
        "cold_cache_method": "L2_flush" if cache == "cold" else "none",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=("eager", "composed", "candidate"),
        default=["eager", "composed", "candidate"],
    )
    parser.add_argument(
        "--modes", nargs="+", choices=("direct", "graph"), default=["direct", "graph"]
    )
    parser.add_argument(
        "--caches", nargs="+", choices=("warm", "cold"), default=["warm", "cold"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("/tmp/routed-lora-results")
    )
    parser.add_argument("--max-seconds", type=int, default=3300)
    parser.add_argument("--tokens", nargs="+", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--output", type=int)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--routes", type=int, default=2)
    parser.add_argument("--adapters", type=int, default=8)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--distribution", choices=("balanced", "hot"), default="balanced"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("This experiment requires an NVIDIA CUDA GPU")
    selected = cases(args.suite)
    if args.tokens:
        if args.hidden is None or args.output is None or args.routes > args.adapters:
            parser.error(
                "custom cases require --hidden, --output, and routes <= adapters"
            )
        selected = [
            Case(
                t,
                args.hidden,
                args.output,
                args.rank,
                args.routes,
                args.adapters,
                args.dtype,
                args.distribution,
            )
            for t in args.tokens
        ]
    args.output_directory.mkdir(parents=True, exist_ok=True)
    import triton

    manifest = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "arguments": vars(args) | {"output_directory": str(args.output_directory)},
        "timing_boundary": (
            "complete adapter update; base GEMM and metadata preparation excluded"
        ),
        "status": "running",
    }
    manifest_path = args.output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    deadline = time.monotonic() + args.max_seconds
    measurements = []
    try:
        with (args.output_directory / "measurements.jsonl").open("w") as results:
            for case in selected:
                if time.monotonic() >= deadline:
                    manifest["status"] = "time_limit"
                    break
                if (
                    min(case.tokens, case.hidden, case.output, case.rank, case.routes)
                    < 1
                ):
                    raise ValueError("benchmark dimensions must be positive")
                experiment = Experiment(case, args.seed)
                errors = {
                    provider: experiment.validate(provider)
                    for provider in args.providers
                }
                for repetition in range(3):
                    order = (
                        args.providers
                        if repetition % 2 == 0
                        else list(reversed(args.providers))
                    )
                    for mode, cache, provider in itertools.product(
                        args.modes, args.caches, order
                    ):
                        row = asdict(case) | {
                            "provider": provider,
                            "mode": mode,
                            "cache": cache,
                            "repetition": repetition,
                            "explicit_workspace_bytes": (
                                experiment.scratch.numel() * 4
                                if provider == "candidate"
                                else experiment.shrink.numel() * 4
                                if provider == "composed"
                                else None
                            ),
                            "metadata_setup_us_including_allocation": (
                                experiment.metadata_setup_us
                            ),
                            **errors[provider],
                            **measure(experiment, provider, mode, cache),
                        }
                        row["useful_tflops"] = (
                            2
                            * case.tokens
                            * case.routes
                            * case.rank
                            * (case.hidden + case.output)
                            / (row["median_us"] * 1e6)
                        )
                        measurements.append(row)
                        results.write(json.dumps(row) + "\n")
                        results.flush()
                        print(json.dumps(row), flush=True)
                del experiment
            else:
                manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2))
        if measurements:
            with (args.output_directory / "measurements.csv").open("w") as output:
                writer = csv.DictWriter(output, fieldnames=list(measurements[0]))
                writer.writeheader()
                writer.writerows(measurements)
            groups = {}
            for row in measurements:
                key = tuple(row[field] for field in Case.__dataclass_fields__) + (
                    row["mode"],
                    row["cache"],
                )
                groups.setdefault(key, {}).setdefault(row["provider"], []).append(
                    row["median_us"]
                )
            summaries = []
            for key, providers in groups.items():
                medians = {
                    name: statistics.median(values)
                    for name, values in providers.items()
                }
                summaries.append(
                    {
                        "case": dict(zip(Case.__dataclass_fields__, key)),
                        "mode": key[-2],
                        "cache": key[-1],
                        "median_us": medians,
                        "candidate_speedup": {
                            name: value / medians["candidate"]
                            for name, value in medians.items()
                            if name != "candidate"
                        }
                        if "candidate" in medians
                        else {},
                    }
                )
            (args.output_directory / "summary.json").write_text(
                json.dumps(summaries, indent=2)
            )


if __name__ == "__main__":
    main()
