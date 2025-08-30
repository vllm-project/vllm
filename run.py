# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8)


def main():
    # Create an LLM.
    llm = LLM(model="Qwen/Qwen3-0.6B",
              enable_sleep_mode=True,
              gpu_memory_utilization=0.5,
              compilation_config=CompilationConfig(cudagraph_mode="FULL"))
    outputs = llm.generate(prompts, sampling_params)
    time.sleep(3)
    from vllm.device_allocator.cumem import CuMemAllocator
    allocator = CuMemAllocator.get_instance()
    print(f"total CuMemAllocator memory: {allocator.get_current_usage()}")
    for tag in ["weights", "kv_cache", "cuda_graph"]:
        print(f"[{tag}] count={allocator.get_current_items_by_tag(tag)}, "
              f"memory={allocator.get_current_usage_by_tag(tag)}")
    print(f"vocab size: {llm.get_tokenizer().vocab_size}")
    llm.sleep(level=2)
    time.sleep(3)
    llm.wake_up(tags=["weights"])
    llm.collective_rpc("reload_weights")
    llm.wake_up(tags=["kv_cache"])
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # logprobs = output.outputs[0].logprobs
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        # print(f"Logprobs:  {len(logprobs)=}, {len(logprobs[0])=}")
        print("-" * 60)

    print(f"total CuMemAllocator memory: {allocator.get_current_usage()}")
    for tag in ["weights", "kv_cache", "cuda_graph"]:
        print(f"[{tag}] count={allocator.get_current_items_by_tag(tag)}, "
              f"memory={allocator.get_current_usage_by_tag(tag)}")


if __name__ == "__main__":
    main()
