#!/usr/bin/env python3
"""
vLLM OpenAI API Server with CPU KV Cache Offloading
Production service launcher for 52k context using RAM
"""

import sys
import runpy
from vllm.config import KVTransferConfig

# Configure CPU offloading for KV cache - inject before import
kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "num_cpu_blocks": 50000,  # ~1.5GB CPU capacity per layer
        "block_size": 16,
    },
)

# Build command line arguments
sys.argv = [
    "vllm.entrypoints.openai.api_server",
    "--model", "/home/richardbrown/local_models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
    "--dtype", "auto",
    "--max-model-len", "52000",  # 32% increase with CPU offload!
    "--gpu-memory-utilization", "0.88",
    "--enforce-eager",
    "--max-num-seqs", "16",
    "--tensor-parallel-size", "1",
    "--enable-prefix-caching",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--tool-call-parser", "qwen3_coder",
    "--enable-auto-tool-choice",
]

# Monkey-patch the KVTransferConfig before starting the server
import vllm.engine.arg_utils
original_create_engine_config = vllm.engine.arg_utils.EngineArgs.create_engine_config

def patched_create_engine_config(self, *args, **kwargs):
    # Inject our kv_transfer_config
    if self.kv_transfer_config is None:
        self.kv_transfer_config = kv_transfer_config
    return original_create_engine_config(self, *args, **kwargs)

vllm.engine.arg_utils.EngineArgs.create_engine_config = patched_create_engine_config

# Now run the API server as a module
if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
