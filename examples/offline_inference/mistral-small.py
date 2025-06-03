# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa
import argparse

from vllm import LLM
from vllm.sampling_params import SamplingParams

# This script is an offline demo for running Mistral-Small-3.1
#
# If you want to run a server/client setup, please follow this code:
#
# - Server:
#
# ```bash
# # Mistral format
# vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
#   --tokenizer-mode mistral --config-format mistral --load-format mistral \
#   --limit-mm-per-prompt '{"image":4}' --max-model-len 16384
#
# # HF format
# vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
#   --limit-mm-per-prompt '{"image":4}' --max-model-len 16384
# ```
#
# - Client:
#
# ```bash
# curl --location 'http://<your-node-url>:8000/v1/chat/completions' \
# --header 'Content-Type: application/json' \
# --header 'Authorization: Bearer token' \
# --data '{
#     "model": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
#     "messages": [
#       {
#         "role": "user",
#         "content": [
#             {"type" : "text", "text": "Describe this image in detail please."},
#             {"type": "image_url", "image_url": {"url": "https://s3.amazonaws.com/cms.ipressroom.com/338/files/201808/5b894ee1a138352221103195_A680%7Ejogging-edit/A680%7Ejogging-edit_hero.jpg"}},
#             {"type" : "text", "text": "and this one as well. Answer in French."},
#             {"type": "image_url", "image_url": {"url": "https://www.wolframcloud.com/obj/resourcesystem/images/a0e/a0ee3983-46c6-4c92-b85d-059044639928/6af8cfb971db031b.png"}}
#         ]
#       }
#     ]
#   }'
# ```
#
# Usage:
#     python demo.py simple
#     python demo.py advanced

# Lower max_model_len and/or max_num_seqs on low-VRAM GPUs.
# These scripts have been tested on 2x L40 GPUs


def run_simple_demo(args: argparse.Namespace):
    model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    sampling_params = SamplingParams(max_tokens=8192)

    llm = LLM(
        model=model_name,
        tokenizer_mode="mistral" if args.format == "mistral" else "auto",
        config_format="mistral" if args.format == "mistral" else "auto",
        load_format="mistral" if args.format == "mistral" else "auto",
        limit_mm_per_prompt={"image": 1},
        max_model_len=4096,
        max_num_seqs=2,
        tensor_parallel_size=2,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    prompt = "Describe this image in one sentence."
    image_url = "https://picsum.photos/id/237/200/300"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]
    outputs = llm.chat(messages, sampling_params=sampling_params)
    print("-" * 50)
    print(outputs[0].outputs[0].text)
    print("-" * 50)


def run_advanced_demo(args: argparse.Namespace):
    model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    max_img_per_msg = 3
    max_tokens_per_img = 4096

    sampling_params = SamplingParams(max_tokens=8192, temperature=0.7)
    llm = LLM(
        model=model_name,
        tokenizer_mode="mistral" if args.format == "mistral" else "auto",
        config_format="mistral" if args.format == "mistral" else "auto",
        load_format="mistral" if args.format == "mistral" else "auto",
        limit_mm_per_prompt={"image": max_img_per_msg},
        max_model_len=max_img_per_msg * max_tokens_per_img,
        tensor_parallel_size=2,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    prompt = "Describe the following image."

    url_1 = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/yosemite.png"
    url_2 = "https://picsum.photos/seed/picsum/200/300"
    url_3 = "https://picsum.photos/id/32/512/512"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url_1}},
                {"type": "image_url", "image_url": {"url": url_2}},
            ],
        },
        {
            "role": "assistant",
            "content": "The images show nature.",
        },
        {
            "role": "user",
            "content": "More details please and answer only in French!.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url_3}},
            ],
        },
    ]

    outputs = llm.chat(messages=messages, sampling_params=sampling_params)
    print("-" * 50)
    print(outputs[0].outputs[0].text)
    print("-" * 50)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a demo in simple or advanced mode."
    )

    parser.add_argument(
        "mode",
        choices=["simple", "advanced"],
        help="Specify the demo mode: 'simple' or 'advanced'",
    )

    parser.add_argument(
        "--format",
        choices=["mistral", "hf"],
        default="mistral",
        help="Specify the format of the model to load.",
    )

    parser.add_argument(
        "--disable-mm-preprocessor-cache",
        action="store_true",
        help="If True, disables caching of multi-modal preprocessor/mapper.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "simple":
        print("Running simple demo...")
        run_simple_demo(args)
    elif args.mode == "advanced":
        print("Running advanced demo...")
        run_advanced_demo(args)


if __name__ == "__main__":
    main()
