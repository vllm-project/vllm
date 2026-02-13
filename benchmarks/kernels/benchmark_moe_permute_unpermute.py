# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from typing import Any, TypedDict

import ray
import torch
from transformers import AutoConfig

from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute,
    moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.utils import _fp8_quantize
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed

FP8_DTYPE = current_platform.fp8_dtype()


class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def benchmark_permute(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    num_iters: int = 100,
) -> float:
    # init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype)
    # output_hidden_states = torch.empty_like(hidden_states)
    if use_fp8_w8a8:
        qhidden_states, scale = _fp8_quantize(hidden_states, None, None)
    else:
        qhidden_states = hidden_states

    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    topk_weights, topk_ids, token_expert_indices = fused_topk(
        qhidden_states, input_gating, topk, False
    )

    def prepare(i: int):
        input_gating.copy_(gating_output[i])

    def run():
        moe_permute(
            qhidden_states,
            a1q_scale=None,
            topk_ids=topk_ids,
            n_expert=num_experts,
            expert_map=None,
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

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    latencies: list[float] = []
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


def benchmark_unpermute(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    num_iters: int = 100,
) -> float:
    # init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype)
    if use_fp8_w8a8:
        qhidden_states, scale = _fp8_quantize(hidden_states, None, None)
    else:
        qhidden_states = hidden_states

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)

    topk_weights, topk_ids, token_expert_indices = fused_topk(
        qhidden_states, input_gating, topk, False
    )

    def prepare():
        (
            permuted_hidden_states,
            _,
            first_token_off,
            inv_perm_idx,
            _,
        ) = moe_permute(
            qhidden_states,
            a1q_scale=None,
            topk_ids=topk_ids,
            n_expert=num_experts,
            expert_map=None,
        )
        # convert to fp16/bf16 as gemm output
        return (
            permuted_hidden_states.to(dtype),
            first_token_off,
            inv_perm_idx,
        )

    def run(input: tuple):
        (permuted_hidden_states, first_token_off, inv_perm_idx) = input
        output = torch.empty_like(hidden_states)
        moe_unpermute(
            output,
            permuted_hidden_states,
            topk_weights,
            inv_perm_idx,
            first_token_off,
        )

    # JIT compilation & warmup
    input = prepare()
    run(input)
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run(input)
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    latencies: list[float] = []
    for i in range(num_iters):
        torch.cuda.synchronize()
        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg


@ray.remote(num_gpus=1)
class BenchmarkWorker:
    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        set_random_seed(seed)
        self.seed = seed
        # Get the device ID to allocate tensors and kernels
        # on the respective GPU. This is required for Ray to work
        # correctly with multi-GPU tuning on the ROCm platform.
        self.device_id = int(ray.get_gpu_ids()[0])

    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
    ) -> tuple[float, float]:
        set_random_seed(self.seed)

        permute_time = benchmark_permute(
            num_tokens,
            num_experts,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a16,
            num_iters=100,
        )
        unpermute_time = benchmark_unpermute(
            num_tokens,
            num_experts,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a16,
            num_iters=100,
        )
        return permute_time, unpermute_time


def get_weight_block_size_safety(config, default_value=None):
    quantization_config = getattr(config, "quantization_config", {})
    if isinstance(quantization_config, dict):
        return quantization_config.get("weight_block_size", default_value)
    return default_value


def main(args: argparse.Namespace):
    print(args)

    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if config.architectures[0] == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts
        topk = config.ffn_config.moe_top_k
    elif config.architectures[0] == "JambaForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
    elif (
        config.architectures[0] == "DeepseekV3ForCausalLM"
        or config.architectures[0] == "DeepseekV2ForCausalLM"
        or config.architectures[0] == "Glm4MoeForCausalLM"
        or config.architectures[0] == "Glm4MoeLiteForCausalLM"
    ):
        E = config.n_routed_experts
        topk = config.num_experts_per_tok
    elif config.architectures[0] in ["Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM"]:
        E = config.num_experts
        topk = config.num_experts_per_tok

    else:
        # Support for llama4
        config = config.get_text_config()
        # Default: Mixtral.
        E = config.num_local_experts
        topk = config.num_experts_per_tok

    hidden_size = config.hidden_size
    dtype = torch.float16 if current_platform.is_rocm() else config.dtype
    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"

    if args.batch_size is None:
        batch_sizes = [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
        ]
    else:
        batch_sizes = [args.batch_size]

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]

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

    outputs = _distribute(
        "benchmark",
        [
            (
                batch_size,
                E,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a16,
            )
            for batch_size in batch_sizes
        ],
    )

    for batch_size, (permute, unpermute) in zip(batch_sizes, outputs):
        print(f"Batch size: {batch_size}")
        print(f"Permute time: {permute:.2f} us")
        print(f"Unpermute time: {unpermute:.2f} us")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument(
        "--dtype", type=str, choices=["auto", "fp8_w8a8", "int8_w8a16"], default="auto"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    main(args)
