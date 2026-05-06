# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file demonstrates the example usage of cpu offloading
with LMCache in vLLM v1 or v0.

Usage:

    Specify vLLM version

    -v v0 : Use LMCacheConnector
            model = mistralai/Mistral-7B-Instruct-v0.2
            (Includes enable_chunked_prefill = True)

    -v v1 : Use LMCacheConnectorV1 (default)
            model = meta-llama/Meta-Llama-3.1-8B-Instruct
            (Without enable_chunked_prefill)

Note that `lmcache` is needed to run this example.
Requirements:
https://docs.lmcache.ai/getting_started/installation.html#prerequisites
Learn more about LMCache environment setup, please refer to:
https://docs.lmcache.ai/getting_started/installation.html
"""

import argparse
import contextlib
import os
import time
from dataclasses import asdict

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs


def setup_environment_variables():
    # LMCache-related environment variables
    # Use experimental features in LMCache
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    # Enable local CPU backend in LMCache
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    # Set local CPU memory limit to 5.0 GB
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )
    # Set GPU memory utilization to 0.8 for an A40 GPU with 40GB
    # memory. Reduce the value if your GPU has less memory.
    # Note: LMCache supports chunked prefill (see vLLM#14505, LMCache#392).
    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=8000,
        gpu_memory_utilization=0.8,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: list[str],
    sampling_params: SamplingParams,
    req_str: str,
):
    # Should be able to see logs like the following:
    # `LMCache INFO: Storing KV cache for 6006 out of 6006 tokens for request 0`
    # This indicates that the KV cache has been stored in LMCache.
    start = time.time()
    outputs = llm.generate(prompt, sampling_params)
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--version",
        choices=["v0", "v1"],
        default="v1",
        help="Specify vLLM version (default: v1)",
    )
    return parser.parse_args()


def main():
    lmcache_connector = "LMCacheConnectorV1"
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    setup_environment_variables()
    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        # This example script runs two requests with a shared prefix.
        # Define the shared prompt and specific prompts
        shared_prompt = "Hello, how are you?" * 1000
        first_prompt = [
            shared_prompt + "Hello, my name is",
        ]
        second_prompt = [
            shared_prompt + "Tell me a very long story",
        ]

        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

        # Print the first output
        print_output(llm, first_prompt, sampling_params, "first")

        time.sleep(1)

        # print the second output
        print_output(llm, second_prompt, sampling_params, "second")


if __name__ == "__main__":
    main()
