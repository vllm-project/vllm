#!/usr/bin/env python3
"""
Test script to actually instantiate XPUWorker and test sleep/wakeup functionality.
This test creates a real XPUWorker instance and calls the methods.
"""

import sys
import os
import tempfile
import shutil

def test_xpu_worker_instantiation():

    from vllm import LLM, SamplingParams
    model_path = "facebook/opt-125m"  # Small model for testing
    tensor_parallel_size = 1
    max_model_len = 1024
    load_format = "auto"
    trust_remote_code = False
    max_num_batched_tokens = 1024
    free_cache_engine = True
    dtype = "float16"
    enforce_eager = False
    enable_chunked_prefill = False
    gpu_memory_utilization = 0.8

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=False,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=0,
        )


    outputs = inference_engine.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)
    
    for i in range(5):
        inference_engine.sleep()
        inference_engine.wake_up(["weights"])
        inference_engine.wake_up(["kv_cache"])

        outputs = inference_engine.generate(prompts, sampling_params)
        # Print the outputs.
        print("\nGenerated Outputs:\n" + "-" * 60)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt:    {prompt!r}")
            print(f"Output:    {generated_text!r}")
            print("-" * 60)

if __name__ == "__main__":
    test_xpu_worker_instantiation()
