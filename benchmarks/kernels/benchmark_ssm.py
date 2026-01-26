# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark and tuning script for Mamba SSM selective_state_update kernel.

This script benchmarks the selective_state_update Triton kernel and can
generate optimal configuration files for different GPU architectures.

Similar to benchmark_moe.py, this script:
1. Benchmarks different kernel configurations
2. Saves optimal configs to JSON files
3. Can be used to auto-tune for specific GPU architectures

Usage:
    # Benchmark with default parameters
    python benchmark_ssm.py

    # Tune and save optimal configs
    python benchmark_ssm.py --tune --save-dir ./

    # Benchmark with specific parameters
    python benchmark_ssm.py --dim 2048 --dstate 64 --batch-size 1 2 4 8
"""

import argparse
import json
import os
import time
from datetime import datetime
from itertools import product
from typing import Any, TypedDict

import ray
import torch
from ray.experimental.tqdm_ray import tqdm

from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_state_update,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed


class SSMBenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    num_warps: int


def get_ssm_config_file_name(
    dim: int,
    dstate: int,
    nheads: int | None = None,
) -> str:
    """Generate config file name for SSM kernel tuning results.

    Args:
        dim: The dimension size (head_dim * nheads for multi-head case)
        dstate: The state dimension
        nheads: Number of heads (optional, for multi-head Mamba)
    """
    device_name = current_platform.get_device_name().replace(" ", "_")
    # Set device_name to H200 if a device from the H200 family is detected
    if "H200" in device_name.split("_"):
        device_name = "NVIDIA_H200"

    if nheads is not None:
        return f"dim={dim},dstate={dstate},nheads={nheads},device_name={device_name}.json"
    return f"dim={dim},dstate={dstate},device_name={device_name}.json"


def get_ssm_configs_search_space() -> list[dict[str, int]]:
    """Get the search space for SSM kernel configurations.

    Returns a list of configurations to try during tuning.
    """
    # Based on the current hardcoded heuristics in mamba_ssm.py
    # BLOCK_SIZE_M options based on dstate
    block_m_range = [4, 8, 16, 32, 64]
    num_warps_range = [2, 4, 8]

    configs = []
    for block_m, num_warps in product(block_m_range, num_warps_range):
        configs.append({
            "BLOCK_SIZE_M": block_m,
            "num_warps": num_warps,
        })
    return configs


def benchmark_ssm_config(
    config: SSMBenchmarkConfig,
    batch_size: int,
    dim: int,
    dstate: int,
    nheads: int,
    ngroups: int,
    dtype: torch.dtype,
    has_z: bool = True,
    num_iters: int = 100,
) -> float:
    """Benchmark a specific SSM configuration.

    Args:
        config: Configuration to benchmark
        batch_size: Batch size
        dim: Head dimension
        dstate: State dimension
        nheads: Number of heads
        ngroups: Number of groups
        dtype: Data type
        has_z: Whether to use z (gating)
        num_iters: Number of iterations for benchmarking

    Returns:
        Average kernel time in microseconds
    """
    device = "cuda"

    # Create input tensors
    state = torch.randn(batch_size, nheads, dim, dstate, dtype=dtype, device=device)
    x = torch.randn(batch_size, nheads, dim, device=device, dtype=dtype)
    out = torch.empty_like(x)
    dt = torch.randn(batch_size, nheads, dim, device=device, dtype=dtype)
    dt_bias = torch.rand(nheads, dim, device=device) - 4.0
    A = -torch.rand(nheads, dim, dstate, device=device) - 1.0
    B = torch.randn(batch_size, ngroups, dstate, device=device)
    C = torch.randn(batch_size, ngroups, dstate, device=device)
    D = torch.randn(nheads, dim, device=device)
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

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Warmup runs
    for _ in range(10):
        run()
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    latencies: list[float] = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start_event.record()
        run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    avg = sum(latencies) / num_iters * 1000  # Convert to microseconds
    return avg


def get_default_ssm_config(dstate: int) -> SSMBenchmarkConfig:
    """Get default SSM configuration based on dstate.

    This matches the current hardcoded heuristics in mamba_ssm.py.
    """
    if dstate <= 16:
        return {"BLOCK_SIZE_M": 32, "num_warps": 4}
    elif dstate <= 32:
        return {"BLOCK_SIZE_M": 16, "num_warps": 4}
    elif dstate <= 64:
        return {"BLOCK_SIZE_M": 8, "num_warps": 4}
    elif dstate <= 128:
        return {"BLOCK_SIZE_M": 4, "num_warps": 4}
    else:
        return {"BLOCK_SIZE_M": 4, "num_warps": 8}


@ray.remote(num_gpus=1)
class SSMBenchmarkWorker:
    """Ray worker for distributed SSM benchmarking."""

    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        set_random_seed(seed)
        self.seed = seed

    def benchmark(
        self,
        batch_size: int,
        dim: int,
        dstate: int,
        nheads: int,
        ngroups: int,
        dtype: torch.dtype,
        has_z: bool = True,
    ) -> tuple[dict[str, int], float]:
        """Benchmark with default config."""
        set_random_seed(self.seed)
        config = get_default_ssm_config(dstate)
        kernel_time = benchmark_ssm_config(
            config,
            batch_size,
            dim,
            dstate,
            nheads,
            ngroups,
            dtype,
            has_z,
            num_iters=100,
        )
        return config, kernel_time

    def tune(
        self,
        batch_size: int,
        dim: int,
        dstate: int,
        nheads: int,
        ngroups: int,
        dtype: torch.dtype,
        search_space: list[dict[str, int]],
        has_z: bool = True,
    ) -> dict[str, int]:
        """Tune to find optimal config for given parameters."""
        set_random_seed(self.seed)
        best_config = None
        best_time = float("inf")

        for config in tqdm(search_space):
            try:
                kernel_time = benchmark_ssm_config(
                    config,
                    batch_size,
                    dim,
                    dstate,
                    nheads,
                    ngroups,
                    dtype,
                    has_z,
                    num_iters=20,
                )

                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config

            except triton.runtime.autotuner.OutOfResources:
                # Some configurations may be invalid
                continue
            except Exception as e:
                print(f"Config {config} failed: {e}")
                continue

        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={batch_size}")
        assert best_config is not None, "No valid configuration found"
        return best_config


def save_ssm_configs(
    configs: dict[int, SSMBenchmarkConfig],
    dim: int,
    dstate: int,
    nheads: int | None,
    save_dir: str,
) -> None:
    """Save tuned SSM configurations to a JSON file."""
    filename = get_ssm_config_file_name(dim, dstate, nheads)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    print(f"Writing best config to {filepath}...")
    with open(filepath, "w") as f:
        json.dump(
            {"triton_version": triton.__version__, **configs},
            f,
            indent=4,
        )
        f.write("\n")


def main(args: argparse.Namespace):
    print(args)

    dim = args.dim
    dstate = args.dstate
    nheads = args.nheads
    ngroups = args.ngroups
    dtype = torch.float16 if current_platform.is_rocm() else torch.bfloat16
    has_z = args.has_z

    if args.batch_size is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    else:
        batch_sizes = args.batch_size

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [SSMBenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]

    def _distribute(method: str, inputs: list[Any]) -> list[Any]:
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
        search_space = get_ssm_configs_search_space()
        print(f"Start tuning over {len(search_space)} configurations...")
        start = time.time()
        configs = _distribute(
            "tune",
            [
                (
                    batch_size,
                    dim,
                    dstate,
                    nheads,
                    ngroups,
                    dtype,
                    search_space,
                    has_z,
                )
                for batch_size in batch_sizes
            ],
        )
        best_configs = {M: config for M, config in zip(batch_sizes, configs)}
        save_ssm_configs(
            best_configs,
            dim,
            dstate,
            nheads if nheads > 1 else None,
            args.save_dir,
        )
        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        outputs = _distribute(
            "benchmark",
            [
                (
                    batch_size,
                    dim,
                    dstate,
                    nheads,
                    ngroups,
                    dtype,
                    has_z,
                )
                for batch_size in batch_sizes
            ],
        )

        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark and tune Mamba SSM selective_state_update kernel"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="Head dimension (default: 64)",
    )
    parser.add_argument(
        "--dstate",
        type=int,
        default=64,
        help="State dimension (default: 64)",
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=32,
        help="Number of heads (default: 32)",
    )
    parser.add_argument(
        "--ngroups",
        type=int,
        default=1,
        help="Number of groups for B and C (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        required=False,
        help="Batch sizes to benchmark (default: 1,2,4,8,16,32,64,128,256,512)",
    )
    parser.add_argument(
        "--has-z",
        action="store_true",
        default=True,
        help="Include gating (z) in benchmark (default: True)",
    )
    parser.add_argument(
        "--no-z",
        dest="has_z",
        action="store_false",
        help="Exclude gating (z) from benchmark",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable tuning mode to find optimal configurations",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./",
        help="Directory to save tuned results (default: ./)",
    )
    args = parser.parse_args()

    main(args)
