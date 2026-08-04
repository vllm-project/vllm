# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import asdict
import argparse
import contextlib
import json
import os
import time

# Third Party
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def setup_environment_variables(raw_block_path: str, use_uring: bool = False) -> None:
    """Set up LMCache-related environment variables for the Rust raw block backend.

    Configures environment variables for LMCache including chunk size, storage
    plugins, and Rust raw block backend specific settings.

    Args:
        raw_block_path: Path to the raw block device for storage.
        use_uring: Whether to enable io_uring path

    Returns:
        None
    """
    # LMCache-related environment variables

    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Disable local CPU backend in LMCache
    os.environ["LMCACHE_LOCAL_CPU"] = "False"

    # Set the maximum size of the local disk size to 5GB
    os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "5"

    os.environ["LMCACHE_STORAGE_PLUGINS"] = "raw_block"

    # Raw block specific extra config
    os.environ["LMCACHE_EXTRA_CONFIG"] = json.dumps(
        {
            "storage_plugin.raw_block.module_path": "lmcache.v1.storage_backend.plugins.rust_raw_block_backend",  # noqa: E501
            "storage_plugin.raw_block.class_name": "RustRawBlockBackend",
            "rust_raw_block.device_path": raw_block_path,
            "rust_raw_block.use_odirect": True,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
            "rust_raw_block.use_uring": use_uring,
        }
    )


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    """Build a vLLM LLM instance with LMCache integration.

    Creates a context manager that builds a vLLM LLM instance configured with
    LMCache for KV cache management. The LLM is yielded and cleaned up on exit.

    Args:
        lmcache_connector: The LMCache connector name to use
        model: The model name.
    """
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )
    # Set GPU memory utilization to 0.5 for an A100 GPU with 40GB
    # memory. Update it accordingly for different GPU.
    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=8000,
        gpu_memory_utilization=0.5,
    )
    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        # Clean up the LMCache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: list[str],
    sampling_params: SamplingParams,
    req_str: str,
) -> None:
    """Generate text using the LLM and print the output with timing information.

    Args:
        llm: The vLLM LLM instance to use for generation.
        prompt: The input prompt(s) as a list of strings.
        sampling_params: Sampling parameters for generation.
        req_str: A string identifier for the request.

    Returns:
        None
    """
    start = time.time()
    outputs = llm.generate(prompt, sampling_params)
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - disk_path: Path to the raw block device for storage.
            - use_uring: Whether to enable io_uring path.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disk_path",
        type=str,
    )
    parser.add_argument(
        "--use_uring",
        action="store_true",
        help="Enable io_uring path (requires Linux kernel >= 5.1)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the Rust backend offload example.

    Sets up environment variables, builds an LLM with LMCache integration,
    and runs two requests with a shared prefix to demonstrate KV cache reuse.

    Returns:
        None
    """
    args = parse_args()

    connector = "LMCacheConnectorV1"
    model = "Qwen/Qwen3-8B"

    setup_environment_variables(args.disk_path, args.use_uring)

    with build_llm_with_lmcache(connector, model) as llm:
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
