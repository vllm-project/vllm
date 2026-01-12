# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for multimodal embedding.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""

import argparse
from dataclasses import asdict

from vllm import LLM, EngineArgs
from vllm.multimodal.utils import fetch_image

image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/cat_snow.jpg"
text = "A cat standing in the snow."
multi_modal_data = {"image": fetch_image(image_url)}


def print_embeddings(embeds):
    embeds_trimmed = (str(embeds[:4])[:-1] + ", ...]") if len(embeds) > 4 else embeds
    print(f"Embeddings: {embeds_trimmed} (size={len(embeds)})")


def run_qwen3_vl():
    engine_args = EngineArgs(
        model="Qwen/Qwen3-VL-Embedding-2B",
        runner="pooling",
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
    )
    default_instruction = "Represent the user's input."
    image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    text_prompt = f"<|im_start|>system\n{default_instruction}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    image_prompt = f"<|im_start|>system\n{default_instruction}<|im_end|>\n<|im_start|>user\n{image_placeholder}<|im_end|>\n<|im_start|>assistant\n"
    image_text_prompt = f"<|im_start|>system\n{default_instruction}<|im_end|>\n<|im_start|>user\n{image_placeholder}{text}<|im_end|>\n<|im_start|>assistant\n"

    llm = LLM(**asdict(engine_args))

    print("Text embedding output:")
    outputs = llm.embed(text_prompt, use_tqdm=False)
    print_embeddings(outputs[0].outputs.embedding)

    print("Image embedding output:")
    outputs = llm.embed(
        {
            "prompt": image_prompt,
            "multi_modal_data": multi_modal_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)

    print("Image+Text embedding output:")
    outputs = llm.embed(
        {
            "prompt": image_text_prompt,
            "multi_modal_data": multi_modal_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)


model_example_map = {
    "qwen3_vl": run_qwen3_vl,
}


def parse_args():
    parser = argparse.ArgumentParser(
        "Script to run a specified VLM through vLLM offline api."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=model_example_map.keys(),
        required=True,
        help="The name of the embedding model.",
    )
    return parser.parse_args()


def main(args):
    model_example_map[args.model]()


if __name__ == "__main__":
    args = parse_args()
    main(args)
