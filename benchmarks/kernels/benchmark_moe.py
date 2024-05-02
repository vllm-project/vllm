import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple

import ray
import torch
import torch.nn.functional as F
import triton.language as tl
from transformers import AutoConfig

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe import *

logger = init_logger(__name__)


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


def get_configs_compute_bound():
    # Adapted from https://github.com/openai/triton/blob/22af8d80458ee4e6269779dae0a3c34b755aade2/python/triton/ops/matmul.py#L56
    configs = [
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "num_warps": 8, "num_stages": 3},
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "num_warps": 8, "num_stages": 3},
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "num_warps": 4, "num_stages": 4},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "num_warps": 4, "num_stages": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "num_warps": 4, "num_stages": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "num_warps": 4, "num_stages": 4},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "num_warps": 4, "num_stages": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,  "num_warps": 2, "num_stages": 5},
    ]
    for config in configs:
        config["BLOCK_SIZE_K"] = 32
        config["GROUP_SIZE_M"] = 8
        config["SPLIT_K"] = 1
    return configs


def get_configs_io_bound():
    # Adapted from https://github.com/openai/triton/blob/22af8d80458ee4e6269779dae0a3c34b755aade2/python/triton/ops/matmul.py#L36
    # TODO(woosuk): Implement a performance model to prune the search space.
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append({
                        "BLOCK_SIZE_M": block_m,
                        "BLOCK_SIZE_N": block_n,
                        "BLOCK_SIZE_K": block_k,
                        "GROUP_SIZE_M": 8,
                        "SPLIT_K": 1,
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                    })
                    # Split-K
                    for split_k in [2, 4, 8, 16]:
                        configs.append({
                        "BLOCK_SIZE_M": block_m,
                        "BLOCK_SIZE_N": block_n,
                        "BLOCK_SIZE_K": block_k,
                        "GROUP_SIZE_M": 8,
                        "SPLIT_K": split_k,
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                    })
    return configs


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
        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for M={M}")
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
    logger.info(f"writing config to file {filename}")
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def main(args: argparse.Namespace):
    logger.info(args)

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

    search_space = get_configs_compute_bound() + get_configs_io_bound()
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

    w2_batch_sizes = [batch_size * topk for batch_size in batch_sizes]
    if args.tune:
        logger.info(f"Start tuning over {len(search_space)} configurations...")
        # w1
        start = time.time()
        _tune(batch_sizes, 2 * shard_intermediate_size, hidden_size, topk)
        end = time.time()
        logger.info(f"W1 tuning took {end - start:.2f} seconds")
        # w2
        start = time.time()
        _tune(w2_batch_sizes, hidden_size, shard_intermediate_size, 1)
        end = time.time()
        logger.info(f"W2 tuning took {end - start:.2f} seconds")
    else:
        # w1
        outputs = _benchmark(batch_sizes, 2 * shard_intermediate_size, hidden_size, topk)
        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            logger.info(f"W1 batch size: {batch_size}, config: {config}")
            logger.info(f"Kernel time: {kernel_time:.2f} us")

        # w2
        outputs = _benchmark(w2_batch_sizes, hidden_size, shard_intermediate_size, 1)
        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            # NOTE(woosuk): Here the batch size is the number of input tokens
            # to the MoE block. This is not the batch size of the w2 layer.
            # The actual batch size of the w2 layer is batch_size * topk.
            logger.info(f"W2 batch size: {batch_size}, config: {config}")
            logger.info(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    main(args)
