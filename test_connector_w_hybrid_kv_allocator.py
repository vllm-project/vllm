# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

# Set token chunk size to 256
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
# Enable CPU memory backend
os.environ["LMCACHE_LOCAL_CPU"] = "True"
# Set CPU memory limit to 5GB
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "20.0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["LMCACHE_USE_LAYERWISE"] = "True"


from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Configure KV cache transfer to use LMCache
ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",
)

# Initialize LLM with LMCache configuration
# Adjust gpu_memory_utilization based on your GPU memory
llm = LLM(
    model="google/gemma-3-4b-it",
    kv_transfer_config=ktc,
    max_model_len=75000,
    gpu_memory_utilization=0.28,
    # gpu_memory_utilization=0.4,
    # gpu_memory_utilization=0.8,
    max_num_seqs=16,
    enforce_eager=True,
)

# Define sampling parameters
sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

# Run inference
outputs = llm.generate("hi" * 70000 + "\nhow are you?", sampling_params)
generated_text = outputs[0].outputs[0].text
print(f"Generated text: {generated_text!r}")

# This requires loading KV cache and will success
outputs = llm.generate("hi" * 10000 + "\nTell me a story.", sampling_params)
generated_text = outputs[0].outputs[0].text
print(f"Generated text: {generated_text!r}")

# flush out prefix cache in GPU
outputs = llm.generate("1" + "hi" * 70000 + "\nhow are you?", sampling_params)
generated_text = outputs[0].outputs[0].text
print(f"Generated text: {generated_text!r}")

print("YIFAN: finish request 2")

# This requires loading KV cache
# but this request cannot be executed as vLLM cannot allocate for long prefix
# stored by LMCache
outputs = llm.generate("hi" * 70000 + "\nTell me a story.", sampling_params)
generated_text = outputs[0].outputs[0].text
print(f"Generated text: {generated_text!r}")

print("YIFAN: finished")
