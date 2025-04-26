# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of cpu offloading
with LMCache in vLLM v1.

Note that lmcache needs to be installed to run this example.
Learn more about LMCache in https://github.com/LMCache/LMCache.
"""
import os

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
    '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}')
# Set GPU memory utilization to 0.8 for an A40 GPU with 40GB
# memory. Reduce the value if your GPU has less memory.
# Note that LMCache is not compatible with chunked prefill for now.
llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct",
          kv_transfer_config=ktc,
          max_model_len=8000,
          gpu_memory_utilization=0.8)

# Should be able to see logs like the following:
# `LMCache INFO: Storing KV cache for 6006 out of 6006 tokens for request 0`
# This indicates that the KV cache has been stored in LMCache.
outputs = llm.generate(first_prompt, sampling_params)
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")

# Clean up lmcache backend
LMCacheEngineBuilder.destroy(ENGINE_NAME)
