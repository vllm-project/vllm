# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of disaggregated prefilling
We will launch 2 vllm instances (GPU 0 for prefill and GPU 1 for decode),
and launch an additional LMCache server.
KV cache is transferred in the following manner: 
VLLM prefill node -> LMCache server -> VLLM decode node.

Learn more about Ray Data in https://docs.ray.io/en/latest/data/data.html
"""
import os
import subprocess
import time
from multiprocessing import Event, Process

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def run_prefill(prefill_done, prompts, port):
    # We use GPU 0 for prefill node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # LMCache-related environment variables
    # Use experimental features in LMCache
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    # Disable local CPU backend in LMCache
    os.environ["LMCACHE_LOCAL_CPU"] = "False"
    # Set local CPU memory buffer limit to 5.0 GB
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"
    # Set the remote URL for LMCache server
    os.environ["LMCACHE_REMTOTE_URL"] = f"lm://localhost:{port}"
    # Set the serializer/deserializer between vllm and LMCache server
    # `naive` indicates using raw bytes of the tensor without any compression
    os.environ["LMCACHE_REMOTE_SERDE"] = "naive"

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"LMCacheConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
    )
    # Set GPU memory utilization to 0.8 for an A40 GPU with 40GB
    # memory. Reduce the value if your GPU has less memory.
    llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct",
              kv_transfer_config=ktc,
              max_model_len=8000,
              gpu_memory_utilization=0.8)

    llm.generate(prompts, sampling_params)
    print("Prefill node is finished.")
    prefill_done.set()

    # To keep the prefill node running in case the decode node is not done
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Script stopped by user.")


def run_decode(prefill_done, prompts, port):
    # We use GPU 1 for decode node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # LMCache-related environment variables
    # Use experimental features in LMCache
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    # Disable local CPU backend in LMCache
    os.environ["LMCACHE_LOCAL_CPU"] = "False"
    # Set local CPU memory buffer limit to 5.0 GB
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"
    # Set the remote URL for LMCache server
    os.environ["LMCACHE_REMTOTE_URL"] = f"lm://localhost:{port}"
    # Set the serializer/deserializer between vllm and LMCache server
    # `naive` indicates using raw bytes of the tensor without any compression
    os.environ["LMCACHE_REMOTE_SERDE"] = "naive"

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"LMCacheConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
    )
    # Set GPU memory utilization to 0.8 for an A40 GPU with 40GB
    # of memory. Reduce the value if your GPU has less memory.
    llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct",
              kv_transfer_config=ktc,
              max_model_len=8000,
              gpu_memory_utilization=0.8)

    # Wait for the producer to start the pipe
    print("Waiting for prefill node to finish...")
    prefill_done.wait()

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def run_lmcache_server(port):
    subprocess.run([
        "python", "-m", "lmcache.experimental.server", "localhost",
        str(port)
    ])


if __name__ == "__main__":

    prompts = [
        "Hello, how are you?" * 1000,
    ]
    port = 8100

    prefill_done = Event()
    prefill_process = Process(target=run_prefill,
                              args=(prefill_done, prompts, port))
    decode_process = Process(target=run_decode,
                             args=(prefill_done, prompts, port))
    lmcache_server_process = Process(target=run_lmcache_server, args=(port, ))

    # Start LMCache server process
    lmcache_server_process.start()

    # Start prefill node
    prefill_process.start()

    # Start decode node
    decode_process.start()

    # Terminate the prefill node and server node when
    # decode is finished
    decode_process.join()
    prefill_process.terminate()
    lmcache_server_process.terminate()
