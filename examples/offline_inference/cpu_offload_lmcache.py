# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of cpu offloading
with LMCache.

Note that `pip install lmcache` is needed to run this example.
Learn more about LMCache in https://github.com/LMCache/LMCache.
"""
import os
import time

from lmcache.experimental.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# LMCache-related environment variables
# Use experimental features in LMCache
os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
# LMCache is set to use 256 tokens per chunk
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
# Enable local CPU backend in LMCache
os.environ["LMCACHE_LOCAL_CPU"] = "True"
# Set local CPU memory limit to 5.0 GB
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"

# This example script runs two requests with a shared prefix.
shared_prompt = "Hello, how are you?" * 1000
first_prompt = [
    shared_prompt + "Hello, my name is",
]
second_prompt = [
    shared_prompt + "Tell me a very long story",
]

sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

ktc = KVTransferConfig.from_cli(
    '{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}')
# Set GPU memory utilization to 0.8 for an A40 GPU with 40GB
# memory. Reduce the value if your GPU has less memory.
# Note that LMCache is not compatible with chunked prefill for now.
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2",
          kv_transfer_config=ktc,
          max_model_len=8000,
          enable_chunked_prefill=False,
          gpu_memory_utilization=0.8)

outputs = llm.generate(first_prompt, sampling_params)
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")
print("First request done.")

time.sleep(1)

outputs = llm.generate(second_prompt, sampling_params)
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")
print("Second request done.")

# Clean up lmcache backend
LMCacheEngineBuilder.destroy(ENGINE_NAME)
