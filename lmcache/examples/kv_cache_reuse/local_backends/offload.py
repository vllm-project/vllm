# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import contextlib
import os
import time

# Third Party
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def setup_environment_variables(vllm_version: str, use_disk: bool = False):
    # LMCache-related environment variables

    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    if use_disk:
        # Disable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "False"

        # Set the maximum size of the local CPU buffer size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

        # Enable local disk backend in LMCache
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"

        # Set the maximum size of the local disk size to 10GB
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
    else:
        # Enable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "True"

        # Set the maximum size of the local CPU size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

    if vllm_version == "v0":
        os.environ["VLLM_USE_V1"] = "0"


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str, vllm_version: str):
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )
    # Set GPU memory utilization to 0.8 for an A40 GPU with 40GB
    # memory. Reduce the value if your GPU has less memory.
    # Note: LMCache supports chunked prefill (see vLLM#14505, LMCache#392).
    #
    # Pass kwargs directly to LLM() instead of routing through
    # EngineArgs + asdict(). asdict() emits None for every unset EngineArgs
    # field, and vLLM >= 0.20 / pydantic v2 rejects None for
    # CompilationConfig fields like cudagraph_capture_sizes (list) and
    # pass_config.fuse_minimax_qk_norm (bool). See issue #3438.
    llm_kwargs = {
        "model": model,
        "kv_transfer_config": ktc,
        "max_model_len": 8000,
        "gpu_memory_utilization": 0.8,
    }
    if vllm_version == "v0":
        llm_kwargs["enable_chunked_prefill"] = True  # Only in v0
    llm = LLM(**llm_kwargs)
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
        help=(
            "Specify vLLM version (default: v1). "
            "v0 requires vLLM <= 0.10.x; vLLM 0.11.0+ removed the V0 engine."
        ),
    )
    parser.add_argument(
        "-d",
        "--use-disk",
        action="store_true",
        help="Specify whether to use disk as backend (default: False)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.version == "v0":
        lmcache_connector = "LMCacheConnector"
        model = "mistralai/Mistral-7B-Instruct-v0.2"
    else:
        lmcache_connector = "LMCacheConnectorV1"
        model = "mistralai/Mistral-7B-Instruct-v0.2"

    setup_environment_variables(args.version, args.use_disk)

    with build_llm_with_lmcache(lmcache_connector, model, args.version) as llm:
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
