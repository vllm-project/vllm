# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune or benchmark BatchedTritonExperts, the Triton kernel behind the
batched expert-parallel MoE activation format (E x capacity x K).

``benchmark_moe.py`` only ever instantiates ``TritonOrDeepGemmExperts``, so
this is the tuning entry point for the batched kernel. It reuses the same
mechanisms: ``override_config()`` forces a candidate through the real code
path, and results use the ``E=...,N=...,device_name=...json`` naming so they
drop straight into ``vllm/model_executor/layers/fused_moe/configs/`` or
``VLLM_TUNED_CONFIG_FOLDER``.

What differs from ``benchmark_moe.py``:

* The lookup key is the per-expert *capacity*, ``hidden_states.size(1)`` of the
  ``[E_local, capacity, K]`` tensor, not a token count. With the
  ``alltoall_batched`` dispatch it is ``max_num_batched_tokens * ep_size``.
* ``E`` is the number of *local* experts (``E_global / ep_size``) and ``N`` is
  the full ``moe_intermediate_size``: with expert parallelism the MoE layers are
  not tensor-parallel.
* Only some keys reach the kernel: ``BLOCK_SIZE_M/N/K`` everywhere, plus
  ``num_warps``, ``num_stages`` and ``grf_mode`` on XPU.
* The batch is routed over ``E_global = E_local * ep_size`` experts and only the
  rows landing on local experts are kept, so each local expert holds roughly
  ``tokens * topk / E_global`` rows, as in a real EP run. Each candidate is
  timed at several fill levels (share of the capacity that is populated), and
  ranked by the geometric mean, so a config that wins on full prefill batches
  does not lose on decode-sized ones.
* Every candidate is checked against ``torch_experts`` before it may win.

Examples (run from the vllm repo root)::

    # Tune Qwen3-30B-A3B (128 experts, K=2048, N=768, topk=8) for EP=4
    python3 benchmarks/kernels/benchmark_batched_moe.py tune \\
        --num-experts 32 --ep-size 4 --hidden-size 2048 \\
        --intermediate-size 768 --topk 8 \\
        --capacity 1024 2048 4096 --save-dir ./tuned

    # Latency of whatever the lookup resolves to (built-in heuristic, or the
    # configs in VLLM_TUNED_CONFIG_FOLDER), with a correctness check
    VLLM_TUNED_CONFIG_FOLDER=./tuned/merged \\
    python3 benchmarks/kernels/benchmark_batched_moe.py bench \\
        --num-experts 32 --ep-size 4 --hidden-size 2048 \\
        --intermediate-size 768 --topk 8 --capacity 1024 2048 4096
"""

import argparse
import gc
import json
import math
import os
import statistics
import time
from datetime import datetime
from itertools import product
from unittest import mock

import torch

from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_moe as fused_moe_module
from vllm.model_executor.layers.fused_moe import fused_topk, override_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
    BatchedTritonExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import get_config_file_name
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize.batched import (
    BatchedPrepareAndFinalize,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

DEVICE = current_platform.device_type
VLLM_CONFIG = VllmConfig()

# Share of the capacity that is populated with tokens. 1.0 is a full batch;
# the small values mimic decode steps, where most of the slots stay empty.
DEFAULT_FILLS = (1.0, 0.1, 0.01)


def get_search_space() -> list[dict[str, int | str]]:
    """Only parameters the batched kernel actually consumes."""
    if current_platform.is_xpu():
        # Mirrors benchmark_moe.get_xpu_tuning_space. GROUP_SIZE_M is left out:
        # the batched kernel has no such parameter.
        param_ranges: dict[str, list[int] | list[str]] = {
            "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_N": [64, 128],
            "BLOCK_SIZE_K": [32, 64],
            "num_warps": [4, 8, 16],
            "num_stages": [3, 4],
            "grf_mode": ["128", "256"],
        }
    else:
        # num_warps/num_stages are not read by the kernel on CUDA.
        param_ranges = {
            "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_N": [32, 64, 128, 256],
            "BLOCK_SIZE_K": [64, 128, 256],
        }
    keys, values = zip(*param_ranges.items())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def sync() -> None:
    torch.accelerator.synchronize()


def clear_triton_cache() -> None:
    gc.collect()
    torch.accelerator.empty_cache()
    try:
        if hasattr(triton.runtime, "cache") and hasattr(triton.runtime.cache, "clear"):
            triton.runtime.cache.clear()
    except Exception as e:  # best effort only
        print(f"warning: failed to clear Triton cache: {e}")
    gc.collect()


def make_dummy_moe_config(max_num_tokens: int) -> FusedMoEConfig:
    """Most fields are unused by the config lookup; the constructor of the
    modular kernel just needs the object (see tests/kernels/moe/utils.py)."""
    return FusedMoEConfig(
        num_experts=1,
        experts_per_token=1,
        hidden_dim=1,
        intermediate_size=1,
        num_local_experts=1,
        num_logical_experts=1,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device=DEVICE,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=max_num_tokens,
    )


def build_kernel(capacity: int, num_local_experts: int) -> FusedMoEKernel:
    quant_config = FusedMoEQuantConfig.make(
        None, per_act_token_quant=False, block_shape=None
    )
    return FusedMoEKernel(
        BatchedPrepareAndFinalize(
            capacity,
            num_dispatchers=1,
            num_local_experts=num_local_experts,
            rank=0,
        ),
        BatchedTritonExperts(
            max_num_tokens=capacity,
            num_dispatchers=1,
            quant_config=quant_config,
            moe_config=make_dummy_moe_config(capacity),
        ),
    )


class Problem:
    """Weights plus one routed batch per fill level for a given capacity.

    ``BatchedPrepareAndFinalize`` sizes the expert tensor as
    ``min(capacity, num_tokens)`` rows, and that size is the config lookup key.
    To keep the key at ``capacity`` for every fill level, the batch always has
    ``capacity`` tokens and the unused ones are routed to a non-existent expert
    id, so they are dropped by the dispatch.
    """

    def __init__(
        self,
        capacity: int,
        num_local_experts: int,
        ep_size: int,
        N: int,
        K: int,
        topk: int,
        fills: tuple[float, ...],
    ):
        self.capacity = capacity
        self.E = num_local_experts
        E_global = num_local_experts * ep_size
        self.fills = fills

        self.a = torch.randn((capacity, K), device=DEVICE, dtype=torch.bfloat16) / 10
        # w1 is the fused gate+up projection [E, 2N, K]; w2 is [E, K, N].
        self.w1 = (
            torch.randn(
                (num_local_experts, 2 * N, K), device=DEVICE, dtype=torch.bfloat16
            )
            / 15
        )
        self.w2 = (
            torch.randn((num_local_experts, K, N), device=DEVICE, dtype=torch.bfloat16)
            / 15
        )

        score = torch.randn((capacity, E_global), device=DEVICE, dtype=torch.bfloat16)
        topk_weight, topk_ids, _ = fused_topk(self.a, score, topk, False)

        self.batches = []
        for fill in fills:
            num_active = max(1, int(round(capacity * fill)))
            active = torch.arange(capacity, device=DEVICE) < num_active
            ids = torch.where(active[:, None], topk_ids, E_global)
            local = ids < num_local_experts
            self.batches.append(
                {
                    "fill": fill,
                    "weight": topk_weight,
                    "ids": ids,
                    "has_local": local.any(dim=1),
                    "reference": torch_experts(
                        self.a,
                        self.w1,
                        self.w2,
                        torch.where(local, topk_weight, 0),
                        torch.where(local, ids, 0),
                    ),
                    "rows_per_expert": local.sum().item() / num_local_experts,
                }
            )


def run_once(kernel: FusedMoEKernel, problem: Problem, batch: dict) -> torch.Tensor:
    return kernel.apply(
        problem.a,
        problem.w1,
        problem.w2,
        batch["weight"],
        batch["ids"],
        # Only the local experts exist on this rank; ids >= E_local are dropped
        # by the dispatch. The activation step on CUDA checks this count.
        global_num_experts=problem.E,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        expert_map=None,
    )


def time_us(fn, num_iters: int, num_warmup: int) -> float:
    for _ in range(num_warmup):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(num_iters):
        fn()
    sync()
    return (time.perf_counter() - start) / num_iters * 1e6


def measure(
    kernel: FusedMoEKernel,
    problem: Problem,
    config: dict | None,
    num_iters: int,
    num_warmup: int,
    atol: float,
    rtol: float,
) -> list[float]:
    """Latency (us) per fill level, or raises if the config is invalid."""
    latencies = []
    for batch in problem.batches:

        def run(batch=batch):
            if config is None:
                return run_once(kernel, problem, batch)
            with override_config(config):
                return run_once(kernel, problem, batch)

        output = run()
        rows = batch["has_local"]
        torch.testing.assert_close(
            output[rows], batch["reference"][rows], atol=atol, rtol=rtol
        )
        latencies.append(time_us(run, num_iters, num_warmup))
    return latencies


def _no_tuned(*args, **kwargs):
    """Stands in for get_moe_configs so that the built-in default is used."""
    return None


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def tune_one_capacity(args, capacity: int) -> dict | None:
    with set_current_vllm_config(VLLM_CONFIG):
        # torch_experts instantiates a CustomOp, which needs the config.
        problem = Problem(
            capacity,
            args.num_experts,
            args.ep_size,
            args.intermediate_size,
            args.hidden_size,
            args.topk,
            tuple(args.fills),
        )
        kernel = build_kernel(capacity, args.num_experts)

        # The built-in default for this shape, for the regression check.
        baseline = measure(
            kernel, problem, None, args.num_iters, args.num_warmup, args.atol, args.rtol
        )

        best: dict | None = None
        # With --max-regression: the best candidate that is at most that much
        # slower than the default at every fill, else the least-regressing one.
        best_ok: dict | None = None
        least_bad: dict | None = None
        search_space = get_search_space()
        for idx, config in enumerate(search_space):
            try:
                latencies = measure(
                    kernel,
                    problem,
                    config,
                    args.num_iters,
                    args.num_warmup,
                    args.atol,
                    args.rtol,
                )
            except triton.runtime.autotuner.OutOfResources:
                continue
            except AssertionError as e:
                print(f"  capacity={capacity} config#{idx} failed correctness: {e}")
                continue
            except Exception as e:
                print(
                    f"  capacity={capacity} config#{idx} raised {type(e).__name__}: {e}"
                )
                continue

            score = geomean(latencies)
            candidate = {"config": config, "score_us": score, "latencies_us": latencies}
            if best is None or score < best["score_us"]:
                best = candidate
            if args.max_regression is not None:
                worst = max(t / b for t, b in zip(latencies, baseline))
                candidate["worst_ratio"] = worst
                if worst <= 1 + args.max_regression and (
                    best_ok is None or score < best_ok["score_us"]
                ):
                    best_ok = candidate
                if least_bad is None or worst < least_bad["worst_ratio"]:
                    least_bad = candidate

            if (
                args.cache_clear_interval > 0
                and idx
                and idx % args.cache_clear_interval == 0
            ):
                clear_triton_cache()
        clear_triton_cache()

    if best is None:
        return None
    if args.max_regression is not None:
        best = best_ok or least_bad
        best["constraint_met"] = best_ok is not None
    best["baseline_us"] = baseline
    best["rows_per_expert"] = [b["rows_per_expert"] for b in problem.batches]
    return best


def save_result(
    save_dir: str, E: int, N: int, capacity: int, result: dict, args
) -> str:
    filename = get_config_file_name(E, N, dtype=None, block_shape=None)
    out_dir = os.path.join(save_dir, f"capacity_{capacity}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    payload = {"triton_version": triton.__version__, str(capacity): result["config"]}
    with open(path + ".tmp", "w") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")
    os.replace(path + ".tmp", path)

    # Not read by vLLM; kept next to the config for review.
    report = {
        "device": current_platform.get_device_name(),
        "triton_version": triton.__version__,
        "E_local": E,
        "ep_size": args.ep_size,
        "N": N,
        "K": args.hidden_size,
        "topk": args.topk,
        "capacity": capacity,
        "fills": args.fills,
        "rows_per_expert": result["rows_per_expert"],
        "tuned_us": result["latencies_us"],
        "baseline_us": result["baseline_us"],
        "config": result["config"],
        "constraint_met": result.get("constraint_met"),
    }
    with open(os.path.join(out_dir, filename + ".report.json"), "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return path


def cmd_tune(args) -> None:
    for capacity in args.capacity:
        t0 = datetime.now()
        try:
            result = tune_one_capacity(args, capacity)
        except Exception as e:  # e.g. OOM at the largest capacities
            print(f"capacity={capacity}: SKIPPED, {type(e).__name__}: {e}")
            clear_triton_cache()
            continue
        elapsed = (datetime.now() - t0).total_seconds()
        if result is None:
            print(f"capacity={capacity}: NO VALID CONFIG FOUND ({elapsed:.0f}s)")
            continue
        path = save_result(
            args.save_dir,
            args.num_experts,
            args.intermediate_size,
            capacity,
            result,
            args,
        )
        pairs = ", ".join(
            f"fill={f}: {t:.0f}us vs default {b:.0f}us ({(b - t) / b * 100:+.1f}%)"
            for f, t, b in zip(
                args.fills, result["latencies_us"], result["baseline_us"]
            )
        )
        regress = any(
            t > 1.05 * b for t, b in zip(result["latencies_us"], result["baseline_us"])
        )
        print(
            f"capacity={capacity}: {pairs} -> {path} ({elapsed:.0f}s)"
            + ("  [REGRESSION >5% at some fill]" if regress else "")
            + (
                "  [NO CONFIG MEETS --max-regression, kept the least-regressing]"
                if result.get("constraint_met") is False
                else ""
            )
        )


def cmd_bench(args) -> None:
    """Latency through the real config lookup (tuned files if present) next to the
    built-in default, measured alternately in one process so that host/GPU drift
    cancels out."""
    out = {}
    for capacity in args.capacity:
        with set_current_vllm_config(VLLM_CONFIG):
            problem = Problem(
                capacity,
                args.num_experts,
                args.ep_size,
                args.intermediate_size,
                args.hidden_size,
                args.topk,
                tuple(args.fills),
            )
            kernel = build_kernel(capacity, args.num_experts)
            runs = (args.num_iters, args.num_warmup, args.atol, args.rtol)
            tuned_rounds, default_rounds = [], []
            for _ in range(args.rounds):
                tuned_rounds.append(measure(kernel, problem, None, *runs))
                with mock.patch.object(fused_moe_module, "get_moe_configs", _no_tuned):
                    default_rounds.append(measure(kernel, problem, None, *runs))
        tuned = [statistics.median(col) for col in zip(*tuned_rounds)]
        default = [statistics.median(col) for col in zip(*default_rounds)]
        out[capacity] = {
            "fills": list(args.fills),
            "tuned_us": tuned,
            "default_us": default,
        }
        print(
            f"capacity={capacity}: "
            + ", ".join(
                f"fill={f}: {t:.0f}us vs default {d:.0f}us ({(d - t) / d * 100:+.1f}%)"
                for f, t, d in zip(args.fills, tuned, default)
            )
        )
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(
                {
                    "device": current_platform.get_device_name(),
                    "triton_version": triton.__version__,
                    "tuned_config_folder": os.environ.get("VLLM_TUNED_CONFIG_FOLDER"),
                    "rounds": args.rounds,
                    "results": out,
                },
                f,
                indent=2,
            )
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name, fn in (("tune", cmd_tune), ("bench", cmd_bench)):
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
        p.add_argument(
            "--num-experts",
            "-E",
            type=int,
            required=True,
            help="local experts per rank (E_global / ep_size)",
        )
        p.add_argument("--ep-size", type=int, default=1)
        p.add_argument("--hidden-size", "-K", type=int, required=True)
        p.add_argument(
            "--intermediate-size",
            "-N",
            type=int,
            required=True,
            help="moe_intermediate_size (not divided by the EP size)",
        )
        p.add_argument("--topk", type=int, required=True)
        p.add_argument(
            "--capacity",
            type=int,
            nargs="+",
            required=True,
            help="per-expert rows (max_num_batched_tokens * ep_size)",
        )
        p.add_argument("--fills", type=float, nargs="+", default=list(DEFAULT_FILLS))
        p.add_argument("--num-iters", type=int, default=20)
        p.add_argument("--num-warmup", type=int, default=5)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--atol", type=float, default=3e-2)
        p.add_argument("--rtol", type=float, default=2e-2)
        if name == "tune":
            p.add_argument("--save-dir", type=str, default="./tuned_configs")
            p.add_argument("--cache-clear-interval", type=int, default=50)
            p.add_argument(
                "--max-regression",
                type=float,
                default=None,
                help="only accept configs at most this fraction slower than the "
                "default at every fill level (e.g. 0.02)",
            )
        else:
            p.add_argument("--out-json", type=str, default=None)
            p.add_argument(
                "--rounds",
                type=int,
                default=3,
                help="alternating tuned/default rounds; the median is reported",
            )
    args = parser.parse_args()

    assert current_platform.is_cuda_alike() or current_platform.is_xpu(), (
        "BatchedTritonExperts requires CUDA or XPU"
    )
    set_random_seed(args.seed)
    init_workspace_manager(torch.device(f"{DEVICE}:0"))
    print(
        f"device={current_platform.get_device_name()} triton={triton.__version__} "
        f"E_local={args.num_experts} ep={args.ep_size} N={args.intermediate_size} "
        f"K={args.hidden_size} topk={args.topk} capacities={args.capacity} "
        f"fills={args.fills}"
    )
    args.func(args)


if __name__ == "__main__":
    main()
