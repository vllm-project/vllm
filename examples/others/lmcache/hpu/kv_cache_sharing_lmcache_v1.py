# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of remote KV cache sharing
with LMCache.
We will launch 2 vllm instances, and launch an additional LMCache server.
KV cache is transferred in the following manner:
(1) vLLM instance 1 -> LMCache server (KV cache store).
(2) LMCache server -> vLLM instance 2 (KV cache reuse/retrieve).
Note that lmcache needs to be installed to run this example.
Learn more about LMCache in https://github.com/LMCache/LMCache.
"""

import argparse
import os
import subprocess
import time
from multiprocessing import Event, Process

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# LMCache-related environment variables
# LMCache is set to use 256 tokens per chunk
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
# Disable local CPU backend in LMCache
os.environ["LMCACHE_LOCAL_CPU"] = "False"
# Set local CPU memory buffer limit to 5.0 GB
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"
# Set the remote URL for LMCache server

# Set the serializer/deserializer between vllm and LMCache server
# `naive` indicates using raw bytes of the tensor without any compression
os.environ["LMCACHE_REMOTE_SERDE"] = "naive"
# GAUDI-NIC

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# prompts = [
#    "Hello, how are you?" * 1000,
# ]
prompts = [
    "San Francisco is a",
]


def run_store(store_done, prompts, tp_size):
    # We use GPU 0 for KV cache store process.
    os.environ["RANK"] = "0"
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    ktc = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_producer")
    # Set GPU memory utilization to 0.8 for an A40 GPU with 40GB
    # memory. Reduce the value if your GPU has less memory.
    llm = LLM(
        model=MODEL,
        kv_transfer_config=ktc,
        max_model_len=8000,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=tp_size,
        enforce_eager=True,
    )

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Producer Generated text: {generated_text!r}")
    print("KV cache store is finished.")
    store_done.set()

    # Clean up lmcache backend
    LMCacheEngineBuilder.destroy(ENGINE_NAME)


def run_retrieve(store_done, prompts, tp_size, timeout=1):
    # We use GPU 1 for KV cache retrieve process.
    decoder_rank = "1"
    os.environ["RANK"] = decoder_rank

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=20)
    ktc = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_consumer")
    # Set GPU memory utilization to 0.8 for an A40 GPU with 40GB
    # of memory. Reduce the value if your GPU has less memory.
    llm = LLM(
        model=MODEL,
        kv_transfer_config=ktc,
        max_model_len=8000,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=tp_size,
        enforce_eager=True,
    )

    print("Waiting for KV cache store to finish...")
    store_done.wait()
    time.sleep(timeout)

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Consumer Generated text: {generated_text!r}")

    # Clean up lmcache backend
    LMCacheEngineBuilder.destroy(ENGINE_NAME)


def run_lmcache_server(port):
    os.environ["LMCACHE_REMOTE_URL"] = f"lm://localhost:{port}"
    server_proc = subprocess.Popen(
        ["python", "-m", "lmcache.v1.server", "localhost", str(port)]
    )
    return server_proc


def run_redis_server(port):
    os.environ["LMCACHE_REMOTE_URL"] = f"redis://localhost:{port}"
    redis_server_path = "/usr/bin/redis-server"  # Update this to the correct path

    try:
        # Start the Redis server
        process = subprocess.Popen(
            [redis_server_path, "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Redis server started successfully!")
        print(f"Process ID: {process.pid}")
    except FileNotFoundError:
        print(
            "Error: Redis server executable not found. \
              Please check the path."
        )
    except Exception as e:
        print(f"An error occurred: {e}")
    return process


def main():
    args = parse_args()
    print(args)

    store_done = Event()
    store_process = Process(target=run_store, args=(store_done, prompts, args.tp_size))
    retrieve_process = Process(
        target=run_retrieve, args=(store_done, prompts, args.tp_size)
    )
    if args.remote_server == "lm":
        remote_server_process = run_lmcache_server(args.lm_port)
    elif args.remote_server == "redis":
        remote_server_process = run_redis_server(args.redis_port)
    else:
        print("Not supported lmcache server type")
        exit()
    print("kvshare store start")
    # Start KV cache store process
    store_process.start()

    print("kvshare retrieve start")
    # Start KV cache retrieve process
    retrieve_process.start()
    print("kvshare retrieve done")
    store_process.join()
    retrieve_process.join()
    # Clean up the processes
    retrieve_process.terminate()
    remote_server_process.terminate()
    remote_server_process.wait()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--remote_server",
        type=str,
        default="lm",
        help="remote lmcache server type. 'lm' or 'redis'",
    )
    parser.add_argument("--lm_port", type=int, default=8100, help="lm server port")
    parser.add_argument(
        "--redis_port", type=int, default=6379, help="redis server port"
    )
    parser.add_argument("--tp_size", type=int, default=1, help="tensor parallel size")

    return parser.parse_args()


if __name__ == "__main__":
    main()
