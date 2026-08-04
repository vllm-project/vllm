# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import asdict
import argparse
import contextlib
import os
import time

# Third Party
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def setup_environment_variables(
    use_disk: bool = False,
    blend_special_str: str = " # # ",
    enable_sparse: bool = False,
):
    # LMCache-related environment variables

    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Blending related config
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"

    if enable_sparse:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        os.environ["LMCACHE_EXTRA_CONFIG"] = '{"enable_sparse": true}'

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


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=32648,
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
        enforce_eager=True,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: list[int],
    sampling_params: SamplingParams,
    req_str: str,
):
    start = time.time()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt}, sampling_params=sampling_params
    )
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--use-disk",
        action="store_true",
        help="Specify whether to use disk as backend (default: False)",
    )

    parser.add_argument(
        "-b",
        "--blend-special-str",
        default="# #",
        help="Specify the special separators to separate chunks (default: '# #')",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )

    parser.add_argument(
        "--enable-sparse",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    lmcache_connector = "LMCacheConnectorV1"
    model = args.model

    setup_environment_variables(
        args.use_disk, args.blend_special_str, args.enable_sparse
    )

    tokenizer = AutoTokenizer.from_pretrained(model)

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        # Define the shared prompt and specific prompts
        warmup_prompt = tokenizer.encode("Nice to meet you" * 500)[1:]
        sys_prompt = [1, 733, 16289, 28793] + tokenizer.encode(
            "You are a very helpful assistant. "
            "Please answer the question with instructions."
        )
        chunk1_prompt = tokenizer.encode("Hello, how are you?" * 500)[1:]
        chunk2_prompt = tokenizer.encode("Hello, what's up?" * 500)[1:]
        chunk3_prompt = tokenizer.encode("Hi, what are you up to?" * 500)[1:]
        blend_special_str = tokenizer.encode(os.getenv("LMCACHE_BLEND_SPECIAL_STR"))[1:]

        first_prompt = (
            sys_prompt
            + blend_special_str
            + chunk1_prompt
            + blend_special_str
            + chunk2_prompt
            + blend_special_str
            + chunk3_prompt
            + blend_special_str
            + tokenizer.encode("Hello, my name is")[1:]
            + [733, 28748, 16289, 28793]
        )

        second_prompt = (
            sys_prompt
            + blend_special_str
            + chunk2_prompt
            + blend_special_str
            + chunk1_prompt
            + blend_special_str
            + chunk3_prompt
            + blend_special_str
            + tokenizer.encode("Hello, how are you?")[1:]
            + [733, 28748, 16289, 28793]
        )

        third_prompt = (
            sys_prompt
            + blend_special_str
            + chunk2_prompt
            + blend_special_str
            + chunk1_prompt
            + blend_special_str
            + chunk3_prompt
            + blend_special_str
            + tokenizer.encode("Hello, what's up?")[1:]
            + [733, 28748, 16289, 28793]
        )

        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

        print_output(llm, warmup_prompt, sampling_params, "warmup")

        # Print the first output
        print_output(llm, first_prompt, sampling_params, "first")

        time.sleep(1)

        # print the second output
        print_output(
            llm, second_prompt, sampling_params, "second (warming up blend code path)"
        )

        time.sleep(1)

        # print the third output
        print_output(llm, third_prompt, sampling_params, "third")


if __name__ == "__main__":
    main()
