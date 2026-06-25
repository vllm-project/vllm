# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import os
import time

import torch

from vllm import LLM, SamplingParams


def run_benchmark(
    model_id, kv_cache_dtype, batch_size, input_len, output_len, gpu_util
):
    # Setup dummy rotation paths if not provided
    if not os.environ.get("VLLM_OSCAR_K_ROTATION_PATH"):
        os.environ["VLLM_OSCAR_K_ROTATION_PATH"] = "dummy_k.pt"
        os.environ["VLLM_OSCAR_V_ROTATION_PATH"] = "dummy_v.pt"
        # Create dummy tensors for testing
        torch.save(torch.eye(128, dtype=torch.float16), "dummy_k.pt")
        torch.save(torch.eye(128, dtype=torch.float16), "dummy_v.pt")

    print(
        f"Loading {model_id} with kv_cache_dtype={kv_cache_dtype} "
        f"and gpu_util={gpu_util}..."
    )

    llm = LLM(
        model=model_id,
        kv_cache_dtype=kv_cache_dtype,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_util,
    )

    prompts = [
        "The quick brown fox jumps over the lazy dog. " * (input_len // 10)
    ] * batch_size
    sampling_params = SamplingParams(max_tokens=output_len, ignore_eos=True)

    print(
        f"\nStarting benchmark: batch_size={batch_size}, "
        f"input_len={input_len}, output_len={output_len}"
    )

    llm.generate(prompts[:1], sampling_params, use_tqdm=False)

    torch.accelerator.synchronize()
    start_time = time.perf_counter()

    llm.generate(prompts, sampling_params, use_tqdm=True)

    torch.accelerator.synchronize()
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    total_tokens = batch_size * (input_len + output_len)
    throughput = total_tokens / elapsed

    print("\n" + "=" * 50)
    print(f"Benchmark Results for {model_id} ({kv_cache_dtype})")
    print("=" * 50)
    print(f"Throughput: {throughput:.2f} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--kv-cache-dtype", type=str, default="oscar_int2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    args = parser.parse_args()
    run_benchmark(
        args.model,
        args.kv_cache_dtype,
        args.batch_size,
        args.input_len,
        args.output_len,
        args.gpu_memory_utilization,
    )
