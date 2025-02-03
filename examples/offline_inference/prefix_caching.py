# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

# NOTE: This is just a running example. For benchmarking purpose,
# please see benchmarks/benchmark_prefix_caching.py

# Common prefix.
prefix = (
    "You are an expert school principal, skilled in effectively managing "
    "faculty and staff. Draft 10-15 questions for a potential first grade "
    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
    "community, joyful discovery, and life-long learning. The candidate is "
    "coming in for a first-round panel interview for a 8th grade Math "
    "teaching role. They have 5 years of previous teaching experience "
    "as an assistant teacher at a co-ed, public school with experience "
    "in middle school math teaching. Based on these information, fulfill "
    "the following paragraph: ")

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

generating_prompts = [prefix + prompt for prompt in prompts]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0)

# Create an LLM without prefix caching as a baseline.
regular_llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.4)

print("Results without `enable_prefix_caching`")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = regular_llm.generate(generating_prompts, sampling_params)

regular_generated_texts = []
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    regular_generated_texts.append(generated_text)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("-" * 80)

# Destroy the LLM object and free up the GPU memory.
del regular_llm
cleanup_dist_env_and_memory()

# Create an LLM with prefix caching enabled.
prefix_cached_llm = LLM(model="facebook/opt-125m",
                        enable_prefix_caching=True,
                        gpu_memory_utilization=0.4)

# Warmup so that the shared prompt's KV cache is computed.
prefix_cached_llm.generate(generating_prompts[0], sampling_params)

# Generate with prefix caching.
outputs = prefix_cached_llm.generate(generating_prompts, sampling_params)

print("Results with `enable_prefix_caching`")

cached_generated_texts = []
# Print the outputs. You should see the same outputs as before.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    cached_generated_texts.append(generated_text)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("-" * 80)

# Compare the results and display the speedup
generated_same = all([
    regular_generated_texts[i] == cached_generated_texts[i]
    for i in range(len(prompts))
])
print(f"Generated answers are the same: {generated_same}")
