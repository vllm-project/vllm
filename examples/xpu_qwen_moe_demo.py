# SPDX-License-Identifier: Apache-2.0
# Demo: Qwen1.5-MoE-A2.7B on 2x Intel Arc Pro B60 with TP=2

import os

# Triton is not available on XPU; disable torch.compile to avoid errors
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
# XPU model loading can be slow; increase the engine startup timeout
os.environ.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "1800")

from vllm import LLM, SamplingParams


def main():
    model_path = "/home/media/Hongbo/models/Qwen3-30B-A3B"

    print("=" * 60)
    print("Qwen3-30B-A3B Demo with Tensor Parallelism = 4")
    print("Device: 4x Intel Arc Pro B60 (XPU)")
    print("=" * 60)

    # Initialize the model with TP=2
    # enforce_eager=True since we are not using Triton
    # num_gpu_blocks_override limits KV cache to avoid OOM on GPU 1
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.95,
        num_gpu_blocks_override=100,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
    )

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Explain quantum computing in simple terms:",
        "Write a short poem about artificial intelligence:",
    ]

    print("\nGenerating responses...\n")
    outputs = llm.generate(prompts, sampling_params)

    print("=" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Response:  {generated_text!r}")
        print("-" * 60)

    print("\nDone!")


if __name__ == "__main__":
    main()
