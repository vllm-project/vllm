import argparse
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import ray
import torch
import triton
from ray.experimental.tqdm_ray import tqdm
from transformers import AutoConfig

from vllm.model_executor.layers.fused_moe.fused_moe import *


def benchmark_config(
    config: Dict[str, int],
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8: bool,
    num_iters: int = 100,
) -> float:
    init_dtype = torch.float16 if use_fp8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    w1 = torch.randn(num_experts,
                     shard_intermediate_size,
                     hidden_size,
                     dtype=init_dtype)
    w2 = torch.randn(num_experts,
                     hidden_size,
                     shard_intermediate_size // 2,
                     dtype=init_dtype)
    gating_output = torch.randn(num_iters,
                                num_tokens,
                                num_experts,
                                dtype=torch.float32)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_fp8:
        w1_scale = torch.randn(num_experts, dtype=torch.float32)
        w2_scale = torch.randn(num_experts, dtype=torch.float32)
        a1_scale = torch.randn(1, dtype=torch.float32)
        a2_scale = torch.randn(1, dtype=torch.float32)

        w1 = w1.to(torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fn)

    input_gating = torch.empty(num_tokens, num_experts, dtype=torch.float32)

    def prepare(i: int):
        input_gating.copy_(gating_output[i])

    def run():
        fused_moe(
            x,
            w1,
            w2,
            input_gating,
            topk,
            renormalize=True,
            inplace=True,
            override_config=config,
            use_fp8=use_fp8,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
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


def get_configs_compute_bound() -> List[Dict[str, int]]:
    # Reduced search space for faster tuning.
    # TODO(woosuk): Increase the search space and use a performance model to
    # prune the search space.
    configs = []
    for num_stages in [2, 3, 4, 5]:
        for block_m in [16, 32, 64, 128, 256]:
            for block_k in [64, 128, 256]:
                for block_n in [32, 64, 128, 256]:
                    for num_warps in [4, 8]:
                        for group_size in [1, 16, 32, 64]:
                            configs.append({
                                "BLOCK_SIZE_M": block_m,
                                "BLOCK_SIZE_N": block_n,
                                "BLOCK_SIZE_K": block_k,
                                "GROUP_SIZE_M": group_size,
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
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8: bool,
    ) -> Tuple[Dict[str, int], float]:
        torch.cuda.manual_seed_all(self.seed)

        dtype_str = "float8" if use_fp8 else None
        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        op_config = get_moe_configs(num_experts, shard_intermediate_size // 2,
                                    dtype_str)
        if op_config is None:
            config = get_default_config(num_tokens, num_experts,
                                        shard_intermediate_size, hidden_size,
                                        topk, dtype_str)
        else:
            config = op_config[min(op_config.keys(),
                                   key=lambda x: abs(x - num_tokens))]
        kernel_time = benchmark_config(config, num_tokens, num_experts,
                                       shard_intermediate_size, hidden_size,
                                       topk, dtype, use_fp8)
        return config, kernel_time

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8: bool,
        search_space: List[Dict[str, int]],
    ) -> Dict[str, int]:
        best_config = None
        best_time = float("inf")
        for config in tqdm(search_space):
            try:
                kernel_time = benchmark_config(config,
                                               num_tokens,
                                               num_experts,
                                               shard_intermediate_size,
                                               hidden_size,
                                               topk,
                                               dtype,
                                               use_fp8,
                                               num_iters=10)
            except triton.runtime.autotuner.OutOfResources:
                # Some configurations may be invalid and fail to compile.
                continue

            if kernel_time < best_time:
                best_time = kernel_time
                best_config = config
        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        return best_config


def sort_config(config: Dict[str, int]) -> Dict[str, int]:
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
    }


def save_configs(
    configs: Dict[int, Dict[str, int]],
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8: bool,
) -> None:
    dtype_str = "float8" if use_fp8 else None
    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    filename = get_config_file_name(num_experts, shard_intermediate_size // 2,
                                    dtype_str)
    print(f"Writing best config to {filename}...")
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def main(args: argparse.Namespace):
    print(args)

    config = AutoConfig.from_pretrained(args.model)
    if config.architectures[0] == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    else:
        # Default: Mixtral.
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size

    hidden_size = config.hidden_size
    dtype = config.torch_dtype
    use_fp8 = args.dtype == "fp8"

    if args.batch_size is None:
        batch_sizes = [
            1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536,
            2048, 3072, 4096
        ]
    else:
        batch_sizes = [args.batch_size]

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]

    def _distribute(method: str, inputs: List[Any]) -> List[Any]:
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
        search_space = get_configs_compute_bound()
        print(f"Start tuning over {len(search_space)} configurations...")

        start = time.time()
        configs = _distribute(
            "tune", [(batch_size, E, shard_intermediate_size, hidden_size,
                      topk, dtype, use_fp8, search_space)
                     for batch_size in batch_sizes])
        best_configs = {
            M: sort_config(config)
            for M, config in zip(batch_sizes, configs)
        }
        save_configs(best_configs, E, shard_intermediate_size, hidden_size,
                     topk, dtype, use_fp8)
        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        outputs = _distribute("benchmark",
                              [(batch_size, E, shard_intermediate_size,
                                hidden_size, topk, dtype, use_fp8)
                               for batch_size in batch_sizes])

        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--tp-size", "-tp", type=int, default=2)
    parser.add_argument("--dtype",
                        type=str,
                        choices=["auto", "fp8"],
                        default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    main(args)
