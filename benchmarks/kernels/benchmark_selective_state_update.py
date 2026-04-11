# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark and tune the Mamba selective_state_update Triton kernel.

Usage examples:

    # Benchmark with default model parameters:
    python benchmarks/kernels/benchmark_selective_state_update.py \
        --model state-spaces/mamba-2.8b-slimpj

    # Tune and save optimal configs:
    python benchmarks/kernels/benchmark_selective_state_update.py \
        --model state-spaces/mamba-2.8b-slimpj --tune

    # Tune with a specific output directory:
    python benchmarks/kernels/benchmark_selective_state_update.py \
        --model state-spaces/mamba-2.8b-slimpj --tune \
        --save-dir ./tuned_configs
"""

import argparse
import json
import os
import time
from itertools import product
from typing import Any, TypedDict

import ray
import torch
from ray.experimental.tqdm_ray import tqdm

from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    get_ssu_config_file_name,
    get_ssu_default_config,
    selective_state_update,
)
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed


class SSUBenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    num_warps: int


def benchmark_config(
    config: SSUBenchmarkConfig,
    dim: int,
    dstate: int,
    dtype: torch.dtype,
    has_z: bool = True,
    num_iters: int = 100,
) -> float:
    """Benchmark a single kernel config and return the average latency in us.

    Args:
        config: Kernel parameters to benchmark.
        dim: Intermediate dimension of the SSM.
        dstate: State dimension of the SSM.
        dtype: Data type for inputs.
        has_z: Whether to include gating (z) tensor.
        num_iters: Number of iterations for timing.

    Returns:
        Average kernel latency in microseconds.
    """
    batch_size = 1
    state = torch.randn(batch_size, dim, dstate, dtype=dtype, device="cuda")
    x = torch.randn(batch_size, dim, device="cuda", dtype=dtype)
    out = torch.empty_like(x)
    dt = torch.randn(batch_size, dim, device="cuda", dtype=dtype)
    dt_bias = torch.rand(dim, device="cuda") - 4.0
    A = -torch.rand(dim, dstate, device="cuda") - 1.0
    B = torch.randn(batch_size, dstate, device="cuda")
    C = torch.randn(batch_size, dstate, device="cuda")
    D = torch.randn(dim, device="cuda")
    z = torch.randn_like(x) if has_z else None

    def run():
        selective_state_update(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            out=out,
        )

    # Warmup + JIT compile
    for _ in range(5):
        run()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: list[float] = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start_event.record()
        run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    avg = sum(latencies) / num_iters * 1000  # convert ms to us
    return avg


def get_search_space() -> list[SSUBenchmarkConfig]:
    """Generate the search space of kernel configurations.

    Returns:
        A list of config dicts to benchmark.
    """
    block_size_m_range = [2, 4, 8, 16, 32, 64]
    num_warps_range = [1, 2, 4, 8]

    configs: list[SSUBenchmarkConfig] = []
    for block_size_m, num_warps in product(block_size_m_range, num_warps_range):
        configs.append({"BLOCK_SIZE_M": block_size_m, "num_warps": num_warps})
    return configs


@ray.remote(num_gpus=1)
class BenchmarkWorker:
    """Ray actor that runs benchmarks on a single GPU."""

    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        set_random_seed(seed)
        self.seed = seed

    def benchmark(
        self,
        dim: int,
        dstate: int,
        dtype: torch.dtype,
        has_z: bool,
    ) -> tuple[dict[str, int], float]:
        """Run with current best or default config and report timing.

        Args:
            dim: Intermediate dimension of the SSM.
            dstate: State dimension of the SSM.
            dtype: Data type for inputs.
            has_z: Whether to include gating (z) tensor.

        Returns:
            A tuple of (config, kernel_time_us).
        """
        set_random_seed(self.seed)
        config = get_ssu_default_config(dstate)
        kernel_time = benchmark_config(
            config, dim, dstate, dtype, has_z=has_z, num_iters=100
        )
        return config, kernel_time

    def tune(
        self,
        dim: int,
        dstate: int,
        dtype: torch.dtype,
        has_z: bool,
        search_space: list[SSUBenchmarkConfig],
    ) -> SSUBenchmarkConfig:
        """Find the best config by exhaustive search.

        Args:
            dim: Intermediate dimension of the SSM.
            dstate: State dimension of the SSM.
            dtype: Data type for inputs.
            has_z: Whether to include gating (z) tensor.
            search_space: List of configs to try.

        Returns:
            The best performing config dict.
        """
        set_random_seed(self.seed)

        best_config: SSUBenchmarkConfig | None = None
        best_time = float("inf")

        for config in tqdm(search_space):
            try:
                kernel_time = benchmark_config(
                    config, dim, dstate, dtype, has_z=has_z, num_iters=20
                )
            except Exception:
                continue

            if kernel_time < best_time:
                best_time = kernel_time
                best_config = config

        assert best_config is not None
        return best_config


def save_configs(
    configs: dict[int, SSUBenchmarkConfig],
    dim: int,
    dstate: int,
    save_dir: str,
) -> None:
    """Save tuned configs to a JSON file.

    Args:
        configs: Mapping of dstate values to best configs.
        dim: Intermediate dimension of the SSM.
        dstate: State dimension of the SSM.
        save_dir: Directory to write the JSON file to.
    """
    filename = get_ssu_config_file_name(dim, dstate)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    print(f"Writing best config to {filepath}...")
    with open(filepath, "w") as f:
        json.dump({"triton_version": triton.__version__, **configs}, f, indent=4)
        f.write("\n")


def get_model_mamba_params(
    config: Any,
) -> tuple[int, int]:
    """Extract (intermediate_size, ssm_state_size) from an HF model config.

    Args:
        config: A HuggingFace model config object.

    Returns:
        A tuple of (intermediate_size, ssm_state_size).
    """
    architectures = getattr(config, "architectures", None) or [type(config).__name__]
    architecture = architectures[0]

    if architecture in ("MambaForCausalLM", "MambaModel"):
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
    elif architecture == "JambaForCausalLM":
        intermediate_size = config.mamba_expand * config.hidden_size
        ssm_state_size = config.mamba_d_state
    elif architecture in (
        "FalconMambaForCausalLM",
        "FalconMambaModel",
    ):
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
    else:
        intermediate_size = getattr(config, "intermediate_size", config.hidden_size * 2)
        ssm_state_size = getattr(config, "state_size", 16)

    return intermediate_size, ssm_state_size


def main(args: argparse.Namespace) -> None:
    """Run benchmark or tuning based on CLI args."""
    print(args)

    if args.dim is not None and args.dstate is not None:
        intermediate_size = args.dim
        ssm_state_size = args.dstate
    else:
        from vllm.transformers_utils.config import get_config

        config = get_config(model=args.model, trust_remote_code=args.trust_remote_code)
        intermediate_size, ssm_state_size = get_model_mamba_params(config)

    tp_size = args.tp_size
    dim = intermediate_size // tp_size

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    if args.dstate_values is None:
        dstate_values = [ssm_state_size]
    else:
        dstate_values = args.dstate_values

    print(
        f"Benchmarking selective_state_update: dim={dim}, "
        f"dstate_values={dstate_values}, dtype={dtype}"
    )

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]

    def _distribute(method: str, inputs: list[tuple[Any, ...]]) -> list[Any]:
        outputs = []
        worker_idx = 0
        for input_args in inputs:
            worker = workers[worker_idx]
            worker_method = getattr(worker, method)
            output = worker_method.remote(*input_args)
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs)

    if args.tune:
        search_space = get_search_space()
        print(
            f"Start tuning over {len(search_space)} configurations "
            f"for {len(dstate_values)} dstate values..."
        )
        start = time.time()
        configs = _distribute(
            "tune",
            [(dim, dstate, dtype, True, search_space) for dstate in dstate_values],
        )
        best_configs = {
            dstate: config for dstate, config in zip(dstate_values, configs)
        }
        for dstate_val, config in best_configs.items():
            save_configs({str(dstate_val): config}, dim, dstate_val, args.save_dir)
        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        outputs = _distribute(
            "benchmark",
            [(dim, dstate, dtype, True) for dstate in dstate_values],
        )
        for dstate_val, (config, kernel_time) in zip(dstate_values, outputs):
            print(f"dstate: {dstate_val}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark and tune the Mamba selective_state_update Triton kernel."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="state-spaces/mamba-2.8b-slimpj",
        help="HuggingFace model name to extract SSM parameters from.",
    )
    parser.add_argument(
        "--tp-size",
        "-tp",
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Data type for benchmarking.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Override intermediate dimension (skip model config lookup).",
    )
    parser.add_argument(
        "--dstate",
        type=int,
        default=None,
        help="Override SSM state size (skip model config lookup).",
    )
    parser.add_argument(
        "--dstate-values",
        type=int,
        nargs="+",
        required=False,
        help="Specific dstate values to benchmark.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./",
        help="Directory to save tuned results.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run exhaustive tuning instead of benchmarking defaults.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    main(args)
