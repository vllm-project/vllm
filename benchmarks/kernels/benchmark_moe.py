import argparse
from typing import Dict, List, Tuple

import ray
import torch
import torch.nn.functional as F
import triton.language as tl
from transformers import AutoConfig

from vllm.model_executor.layers.fused_moe.fused_moe import *


def benchmark_config(
    config: Dict[str, int],
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: torch.dtype,
    num_iters: int = 100,
) -> float:
    x = torch.randn(M, K, dtype=dtype)
    w = torch.randn(E, N, K, dtype=dtype)
    o = torch.empty(M, topk, N, dtype=dtype)
    gating = torch.randn(num_iters, M, E, dtype=dtype)

    compute_type = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16
    routing_weights = F.softmax(gating, dim=-1, dtype=torch.float32)
    topk_weights, input_topk_ids = torch.topk(routing_weights, topk, dim=-1)
    topk_ids = input_topk_ids[0]

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E)

    def prepare(i: int):
        topk_ids.copy_(input_topk_ids[i])
        outputs = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E)
        sorted_token_ids.copy_(outputs[0])
        expert_ids.copy_(outputs[1])
        num_tokens_post_padded.copy_(outputs[2])

    def run():
        invoke_fused_moe_kernel(
            x,
            w,
            o,
            None,
            None,
            topk_weights[0],
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            topk,
            config,
            compute_type=compute_type,
            use_fp8=False,  # TODO(woosuk): Support FP8.
        )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    for i in range(num_iters):
        prepare(i)
        torch.cuda.synchronize()

        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg


def get_legacy_config_file_name(E: int, N: int, dtype: Optional[str]) -> str:
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    return f"E={E},N={N},device_name={device_name}{dtype_selector}.json"


@ray.remote(num_gpus=1)
class BenchmarkWorker:

    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        torch.cuda.manual_seed_all(seed)
        self.seed = seed

    def benchmark(
        self,
        M: int,
        E: int,
        N: int,
        K: int,
        topk: int,
        dtype: torch.dtype,
    ) -> Tuple[Dict[str, int], float]:
        torch.cuda.manual_seed_all(self.seed)

        op_config = get_op_config(E, N, K, topk, str(dtype))
        if op_config is None:
            config = get_default_config(M, E, N, K, topk, str(dtype))
        else:
            config = op_config[min(op_config.keys(), key=lambda x: abs(x - M))]
        kernel_time = benchmark_config(config, M, E, N, K, topk, dtype)
        return config, kernel_time

    def tune(
        self,
        M: int,
        E: int,
        N: int,
        K: int,
        topk: int,
        dtype: torch.dtype,
        search_space: List[Dict[str, int]],
    ) -> Dict[str, int]:
        best_config = None
        best_time = float("inf")
        for config in search_space:
            try:
                kernel_time = benchmark_config(config, M, E, N, K, topk, dtype, num_iters=10)
            except triton.runtime.autotuner.OutOfResources:
                # Some configurations may be invalid due to resource constraints.
                continue

            if kernel_time < best_time:
                best_time = kernel_time
                best_config = config
        return best_config


def save_configs(
    configs: Dict[int, Dict[str, int]],
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str,
) -> None:
    filename = get_config_file_name(E, N, K, topk, dtype)
    print(f"writing config to file {filename}")
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def main(args: argparse.Namespace):
    print(args)

    config = AutoConfig.from_pretrained(args.model)
    E = config.num_local_experts
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    shard_intermediate_size = intermediate_size // args.tp_size
    topk = config.num_experts_per_tok
    dtype = config.torch_dtype

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]

    if args.batch_size is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    else:
        batch_sizes = [args.batch_size]

    search_space = [
        # Compute-bound configurations.
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8},
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8},
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 5, "num_warps": 2},
    ]

    def _tune(Ms: List[int], N: int, K: int, topk_experts: int):
        outputs = []
        worker_idx = 0
        for M in Ms:
            worker = workers[worker_idx]
            output = worker.tune.remote(
                M,
                E,
                N,
                K,
                topk_experts,
                dtype,
                search_space,
            )
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        configs = ray.get(outputs)
        best_configs = {M: config for M, config in zip(Ms, configs)}
        save_configs(best_configs, E, N, K, topk_experts, str(dtype))

    def _benchmark(Ms: List[int], N: int, K: int, topk_experts: int):
        outputs = []
        worker_idx = 0
        for M in Ms:
            worker = workers[worker_idx]
            output = worker.benchmark.remote(
                M,
                E,
                N,
                K,
                topk_experts,
                dtype,
            )
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs)

    if args.tune:
        # w1
        _tune(batch_sizes, 2 * shard_intermediate_size, hidden_size, topk)
        # w2
        _tune(batch_sizes, hidden_size, shard_intermediate_size, 1)
    else:
        # w1
        outputs = _benchmark(batch_sizes, 2 * shard_intermediate_size, hidden_size, topk)
        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"W1 batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")

        # w2
        outputs = _benchmark(
            [batch_size * topk for batch_size in batch_sizes],
            hidden_size,
            shard_intermediate_size,
            1,
        )
        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"W2 batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    main(args)
