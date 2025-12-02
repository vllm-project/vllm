# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import time
from contextlib import nullcontext
from datetime import datetime
from itertools import product
from typing import Any, TypedDict

import ray
import torch
from ray.experimental.tqdm_ray import tqdm

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    _get_config_dtype_str,
)
from vllm.model_executor.layers.fused_moe.fused_moe import *
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser

FP8_DTYPE = current_platform.fp8_dtype()


def ensure_divisibility(numerator, denominator, text):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} {} is not divisible by tp {}.".format(
        text, numerator, denominator
    )


class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    num_iters: int = 100,
    block_quant_shape: list[int] = None,
    use_deep_gemm: bool = False,
) -> float:
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    if use_int8_w8a16:
        w1 = torch.randint(
            -127,
            127,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size,
            ),
            dtype=torch.int8,
        )
        w2 = torch.randint(
            -127,
            127,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 2,
            ),
            dtype=torch.int8,
        )
    else:
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype
        )
    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn(
            (num_experts, 2 * shard_intermediate_size), dtype=torch.float32
        )
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_deep_gemm:
        # we use the default block shape for deepgemm
        block_quant_shape = [128, 128]
    if use_fp8_w8a8:
        if block_quant_shape:
            block_n, block_k = block_quant_shape[0], block_quant_shape[1]
            E = num_experts
            N = shard_intermediate_size // 2
            K = hidden_size
            factor_for_scale = 1e-2
            n_tiles_w1 = (2 * N + block_n - 1) // block_n
            n_tiles_w2 = (K + block_n - 1) // block_n
            k_tiles_w1 = (K + block_k - 1) // block_k
            k_tiles_w2 = (N + block_k - 1) // block_k
            w1_scale = (
                torch.rand((E, n_tiles_w1, k_tiles_w1), dtype=torch.float32)
                * factor_for_scale
            )
            w2_scale = (
                torch.rand((E, n_tiles_w2, k_tiles_w2), dtype=torch.float32)
                * factor_for_scale
            )
        else:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)

        a1_scale = torch.randn(1, dtype=torch.float32)
        a2_scale = torch.randn(1, dtype=torch.float32)

        w1 = w1.to(FP8_DTYPE)
        w2 = w2.to(FP8_DTYPE)

    input_gating = torch.empty(num_tokens, num_experts, dtype=torch.float32)

    def prepare(i: int):
        input_gating.copy_(gating_output[i])

    def run():
        from vllm.model_executor.layers.fused_moe import override_config

        if use_fp8_w8a8:
            quant_dtype = torch.float8_e4m3fn
        elif use_int8_w8a16:
            quant_dtype = torch.int8
        else:
            quant_dtype = None

        quant_config = FusedMoEQuantConfig.make(
            quant_dtype=quant_dtype,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_quant_shape,
        )

        with override_config(config):
            topk_weights, topk_ids, token_expert_indices = fused_topk(
                x, input_gating, topk, renormalize=not use_deep_gemm
            )
            return fused_experts(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                inplace=True,
                quant_config=quant_config,
                allow_deep_gemm=use_deep_gemm,
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


def get_rocm_tuning_space(use_fp16):
    block_mn_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256]
    if not use_fp16:
        block_k_range.remove(16)  # BLOCK_K=16 not supported for fp8
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [1, 4, 8, 16, 32]
    num_stage_range = [2]
    waves_per_eu_range = [0, 1, 2, 4]
    matrix_instr_nonkdim_range = [16, 32] if use_fp16 else []
    kpack_range = [1, 2] if use_fp16 else []

    param_ranges = {
        "BLOCK_SIZE_M": block_mn_range,
        "BLOCK_SIZE_N": block_mn_range,
        "BLOCK_SIZE_K": block_k_range,
        "GROUP_SIZE_M": group_m_range,
        "num_warps": num_warps_range,
        "num_stages": num_stage_range,
        "waves_per_eu": waves_per_eu_range,
    }
    if use_fp16:
        param_ranges["matrix_instr_nonkdim"] = matrix_instr_nonkdim_range
        param_ranges["kpack"] = kpack_range

    return param_ranges


def get_configs_compute_bound(use_fp16, block_quant_shape) -> list[dict[str, int]]:
    configs: list[BenchmarkConfig] = []

    if current_platform.is_rocm():
        param_ranges = get_rocm_tuning_space(use_fp16)
    else:
        # Reduced search space for faster tuning.
        # TODO(woosuk): Increase the search space and use a performance model to
        # prune the search space.
        block_m_range = [16, 32, 64, 128, 256]
        block_n_range = [32, 64, 128, 256]
        block_k_range = [64, 128, 256]
        num_warps_range = [4, 8]
        group_m_range = [1, 16, 32, 64]
        num_stage_range = [2, 3, 4, 5]

        param_ranges = {
            "BLOCK_SIZE_M": block_m_range,
            "BLOCK_SIZE_N": block_n_range,
            "BLOCK_SIZE_K": block_k_range,
            "GROUP_SIZE_M": group_m_range,
            "num_warps": num_warps_range,
            "num_stages": num_stage_range,
        }

    keys, values = zip(*param_ranges.items())
    for config_values in product(*values):
        config = dict(zip(keys, config_values))
        configs.append(config)

    # Remove configs that are not compatible with fp8 block quantization
    # BLOCK_SIZE_K must be a multiple of block_k
    # BLOCK_SIZE_N must be a multiple of block_n
    if block_quant_shape is not None and not use_fp16:
        block_n, block_k = block_quant_shape[0], block_quant_shape[1]
        for config in configs[:]:
            if (
                config["BLOCK_SIZE_K"] % block_k != 0
                or config["BLOCK_SIZE_N"] % block_n != 0
            ):
                configs.remove(config)
    return configs


def prune_rocm_search_space(
    num_tokens, shard_intermediate_size, hidden_size, search_space, is_fp16, topk
):
    N1, K1 = shard_intermediate_size, hidden_size
    N2, K2 = hidden_size, shard_intermediate_size // 2
    pruned_space_1 = prune_rocm_configs(
        num_tokens * topk, N1, K1, search_space, is_fp16
    )
    pruned_space_2 = prune_rocm_configs(
        num_tokens * topk, N2, K2, search_space, is_fp16
    )
    search_space = merge_unique_dicts(pruned_space_1, pruned_space_2)
    return search_space


# The following code is inspired by ROCm/Triton GEMM tuning script:
# https://github.com/ROCm/triton/blob/triton-mlir/scripts/amd/gemm/tune_gemm.py#L89
def prune_rocm_configs(M, N, K, configs, is_fp16=True):
    pruned_configs = []
    elemBytes_a = 2 if is_fp16 else 1
    elemBytes_b = 2 if is_fp16 else 1

    mfma = 16 if M < 32 or N < 32 else 32

    # TODO (zhanglx): figure out the boundary between large and small gemms
    large_gemm = False
    if M >= 2048 and N >= 2048:
        large_gemm = True

    for config in configs:
        BLOCK_SIZE_M = config.get("BLOCK_SIZE_M")
        BLOCK_SIZE_N = config.get("BLOCK_SIZE_N")
        BLOCK_SIZE_K = config.get("BLOCK_SIZE_K")
        num_warps = config.get("num_warps")

        if is_fp16:
            matrix_instr_nonkdim = config.get("matrix_instr_nonkdim")
            if matrix_instr_nonkdim > mfma:
                continue
        if mfma == 4 and BLOCK_SIZE_K < 64:
            continue
        # some layouts could not work properly in case
        # number elements per thread is less 1
        if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
            continue
        SPLIT_K = config.get("SPLIT_K", 1)
        GROUP_M = config.get("GROUP_SIZE_M")
        if is_fp16:
            if (
                matrix_instr_nonkdim > BLOCK_SIZE_M
                or matrix_instr_nonkdim > BLOCK_SIZE_N
            ):
                continue
            if matrix_instr_nonkdim >= M and matrix_instr_nonkdim != BLOCK_SIZE_M:
                continue
            if matrix_instr_nonkdim >= N and matrix_instr_nonkdim != BLOCK_SIZE_N:
                continue
        # Skip BLOCK_SIZE that is too large compare to M/N
        # unless BLOCK_SIZE is already small enough
        if M * 2 < BLOCK_SIZE_M and BLOCK_SIZE_M != 16:
            continue
        if N * 2 < BLOCK_SIZE_N and BLOCK_SIZE_N != 16:
            continue
        # skip large split_k when not necessary
        if SPLIT_K != 1 and not need_split_k(M, N, K):
            continue
        # skip split_k that leads to EVEN_K = false
        leap = SPLIT_K * BLOCK_SIZE_K
        modv = K % leap
        if modv != 0:
            continue
        # skip large GROUP_M
        if GROUP_M * BLOCK_SIZE_M > M and GROUP_M != 1:
            continue
        # out of shared memory resource
        # TODO (zhanglx): This does not consider the LDS usage in the epilogue
        LDS = (
            BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a
            + BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b
        )
        if LDS > 65536:
            continue
        # Skip small block sizes and num_warps for large gemm
        # For fp16 and f8, we want to only use BLOCK_SIZE >= 64
        if large_gemm:
            if BLOCK_SIZE_M < 64 or BLOCK_SIZE_N < 64:
                continue
            if BLOCK_SIZE_K < 64:
                continue
            if num_warps < 4:
                continue

        pruned_configs.append(config)

    return pruned_configs


def need_split_k(SIZE_M, SIZE_N, SIZE_K):
    return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024


def merge_unique_dicts(list1, list2):
    result = []
    combined_list = list1.copy()
    combined_list.extend(list2)
    for dictionary in combined_list:
        if dictionary not in result:
            result.append(dictionary)
    return result


@ray.remote(num_gpus=1)
class BenchmarkWorker:
    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        current_platform.seed_everything(seed)
        self.seed = seed
        # Get the device ID to allocate tensors and kernels
        # on the respective GPU. This is required for Ray to work
        # correctly with multi-GPU tuning on the ROCm platform.
        self.device_id = int(ray.get_gpu_ids()[0])

    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        block_quant_shape: list[int] = None,
        use_deep_gemm: bool = False,
    ) -> tuple[dict[str, int], float]:
        current_platform.seed_everything(self.seed)
        dtype_str = _get_config_dtype_str(
            dtype, use_int8_w8a16=use_int8_w8a16, use_fp8_w8a8=use_fp8_w8a8
        )
        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        block_n = block_quant_shape[0] if block_quant_shape else None
        block_k = block_quant_shape[1] if block_quant_shape else None
        op_config = get_moe_configs(
            num_experts, shard_intermediate_size // 2, dtype_str, block_n, block_k
        )
        if op_config is None:
            config = get_default_config(
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype_str,
                block_quant_shape,
            )
        else:
            config = op_config[min(op_config.keys(), key=lambda x: abs(x - num_tokens))]
        kernel_time = benchmark_config(
            config,
            num_tokens,
            num_experts,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a16,
            num_iters=100,
            block_quant_shape=block_quant_shape,
            use_deep_gemm=use_deep_gemm,
        )
        return config, kernel_time

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        search_space: list[dict[str, int]],
        block_quant_shape: list[int],
        use_deep_gemm: bool,
    ) -> dict[str, int]:
        best_config = None
        best_time = float("inf")
        if current_platform.is_rocm():
            is_fp16 = not (use_fp8_w8a8 or use_int8_w8a16)
            search_space = prune_rocm_search_space(
                num_tokens,
                shard_intermediate_size,
                hidden_size,
                search_space,
                is_fp16,
                topk,
            )

        need_device_guard = False
        if current_platform.is_rocm():
            visible_device = os.environ.get("ROCR_VISIBLE_DEVICES", None)
            if visible_device != f"{self.device_id}":
                need_device_guard = True

        with torch.cuda.device(self.device_id) if need_device_guard else nullcontext():
            for config in tqdm(search_space):
                try:
                    kernel_time = benchmark_config(
                        config,
                        num_tokens,
                        num_experts,
                        shard_intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a16,
                        num_iters=20,
                        block_quant_shape=block_quant_shape,
                        use_deep_gemm=use_deep_gemm,
                    )
                except triton.runtime.autotuner.OutOfResources:
                    # Some configurations may be invalid and fail to compile.
                    continue

                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config
        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        assert best_config is not None
        return best_config


def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
        **(
            {"waves_per_eu": config["waves_per_eu"]} if "waves_per_eu" in config else {}
        ),
        **(
            {"matrix_instr_nonkdim": config["matrix_instr_nonkdim"]}
            if "matrix_instr_nonkdim" in config
            else {}
        ),
        **({"kpack": config["kpack"]} if "kpack" in config else {}),
    }


def save_configs(
    configs: dict[int, BenchmarkConfig],
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    block_quant_shape: list[int],
    save_dir: str,
) -> None:
    dtype_str = _get_config_dtype_str(
        dtype, use_int8_w8a16=use_int8_w8a16, use_fp8_w8a8=use_fp8_w8a8
    )

    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    filename = get_config_file_name(
        num_experts, shard_intermediate_size // 2, dtype_str, block_quant_shape
    )
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, filename)
    print(f"Writing best config to {filename}...")
    with open(filename, "w") as f:
        json.dump({"triton_version": triton.__version__, **configs}, f, indent=4)
        f.write("\n")


def get_weight_block_size_safety(config, default_value=None):
    quantization_config = getattr(config, "quantization_config", {})
    if isinstance(quantization_config, dict):
        return quantization_config.get("weight_block_size", default_value)
    return default_value


def main(args: argparse.Namespace):
    print(args)

    config = get_config(model=args.model, trust_remote_code=args.trust_remote_code)
    if args.model_prefix:
        config = getattr(config, args.model_prefix)

    if config.architectures[0] == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
        hidden_size = config.hidden_size
    elif config.architectures[0] == "JambaForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        hidden_size = config.hidden_size
    elif config.architectures[0] in (
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "Glm4MoeForCausalLM",
        "NemotronHForCausalLM",
    ):
        E = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        hidden_size = config.hidden_size
    elif config.architectures[0] in (
        "Qwen2MoeForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3NextForCausalLM",
    ):
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        hidden_size = config.hidden_size
    elif config.architectures[0] == "Qwen3VLMoeForConditionalGeneration":
        text_config = config.get_text_config()
        E = text_config.num_experts
        topk = text_config.num_experts_per_tok
        intermediate_size = text_config.moe_intermediate_size
        hidden_size = text_config.hidden_size
    elif config.architectures[0] in ("HunYuanMoEV1ForCausalLM"):
        E = config.num_experts
        topk = config.moe_topk[0]
        intermediate_size = config.moe_intermediate_size[0]
        hidden_size = config.hidden_size
    elif config.architectures[0] in ["Qwen3OmniMoeForConditionalGeneration"]:
        E = config.thinker_config.text_config.num_experts
        topk = config.thinker_config.text_config.num_experts_per_tok
        intermediate_size = config.thinker_config.text_config.moe_intermediate_size
        hidden_size = config.thinker_config.text_config.hidden_size
    else:
        # Support for llama4
        config = config.get_text_config()
        # Default: Mixtral.
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        hidden_size = config.hidden_size
    enable_ep = bool(args.enable_expert_parallel)
    if enable_ep:
        ensure_divisibility(E, args.tp_size, "Number of experts")
        E = E // args.tp_size
        shard_intermediate_size = 2 * intermediate_size
    else:
        ensure_divisibility(intermediate_size, args.tp_size, "intermediate_size")
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    dtype = torch.float16 if current_platform.is_rocm() else config.dtype
    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"
    block_quant_shape = get_weight_block_size_safety(config)

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
        batch_sizes = args.batch_size

    use_deep_gemm = bool(args.use_deep_gemm)

    if current_platform.is_rocm() and "HIP_VISIBLE_DEVICES" in os.environ:
        # Ray will set ROCR_VISIBLE_DEVICES for device visibility
        logger.warning(
            "Ray uses ROCR_VISIBLE_DEVICES to control device accessibility."
            "Replacing HIP_VISIBLE_DEVICES with ROCR_VISIBLE_DEVICES."
        )
        val = os.environ["HIP_VISIBLE_DEVICES"]
        os.environ["ROCR_VISIBLE_DEVICES"] = val
        del os.environ["HIP_VISIBLE_DEVICES"]

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

    if args.tune:
        is_fp16 = not (use_fp8_w8a8 or use_int8_w8a16)
        search_space = get_configs_compute_bound(is_fp16, block_quant_shape)
        print(f"Start tuning over {len(search_space)} configurations...")
        if use_deep_gemm:
            raise ValueError(
                "Tuning with --use-deep-gemm is not supported as it only tunes Triton "
                "kernels. Please remove the flag."
            )
        start = time.time()
        configs = _distribute(
            "tune",
            [
                (
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a16,
                    search_space,
                    block_quant_shape,
                    use_deep_gemm,
                )
                for batch_size in batch_sizes
            ],
        )
        best_configs = {
            M: sort_config(config) for M, config in zip(batch_sizes, configs)
        }
        save_configs(
            best_configs,
            E,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a16,
            block_quant_shape,
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
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a16,
                    block_quant_shape,
                    use_deep_gemm,
                )
                for batch_size in batch_sizes
            ],
        )

        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument(
        "--tp-size", "-tp", "--tensor-parallel-size", type=int, default=2
    )
    parser.add_argument("--enable-expert-parallel", "-enable-ep", action="store_true")
    parser.add_argument(
        "--dtype", type=str, choices=["auto", "fp8_w8a8", "int8_w8a16"], default="auto"
    )
    parser.add_argument("--use-deep-gemm", action="store_true")
    parser.add_argument(
        "--save-dir", type=str, default="./", help="Directory to save tuned results"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, nargs="+", required=False)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--model-prefix", type=str, required=False)
    args = parser.parse_args()

    main(args)
