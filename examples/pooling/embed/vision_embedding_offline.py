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
from pathlib import Path

from PIL.Image import Image

from vllm import LLM, EngineArgs
from vllm.multimodal.utils import fetch_image
from vllm.utils.print_utils import print_embeddings

ROOT_DIR = Path(__file__).parent.parent.parent
EMBED_TEMPLATE_DIR = ROOT_DIR / "pooling/embed/template/"

image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/cat_snow.jpg"
text = "A cat standing in the snow."
multi_modal_data = {"image": fetch_image(image_url)}


def run_clip():
    engine_args = EngineArgs(
        model="openai/clip-vit-base-patch32",
        runner="pooling",
        limit_mm_per_prompt={"image": 1},
    )

    llm = LLM(**asdict(engine_args))

    print("Text embedding output:")
    outputs = llm.embed(text, use_tqdm=False)
    print_embeddings(outputs[0].outputs.embedding)

    print("Image embedding output:")
    prompt = ""  # For image input, make sure that the prompt text is empty
    outputs = llm.embed(
        {
            "prompt": prompt,
            "multi_modal_data": multi_modal_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)


def run_e5_v():
    engine_args = EngineArgs(
        model="royokong/e5-v",
        runner="pooling",
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
    )

    llm = LLM(**asdict(engine_args))

    llama3_template = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"  # noqa: E501

    print("Text embedding output:")
    prompt_text = llama3_template.format(
        f"{text}\nSummary above sentence in one word: "
    )
    outputs = llm.embed(prompt_text, use_tqdm=False)
    print_embeddings(outputs[0].outputs.embedding)

    print("Image embedding output:")
    prompt_image = llama3_template.format("<image>\nSummary above image in one word: ")
    outputs = llm.embed(
        {
            "prompt": prompt_image,
            "multi_modal_data": multi_modal_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)


def run_qwen3_vl():
    try:
        from qwen_vl_utils import smart_resize
    except ModuleNotFoundError:
        print(
            "WARNING: `qwen-vl-utils` not installed, input images will not "
            "be automatically resized. This can cause different results "
            "comparing with HF repo's example. "
            "You can enable this functionality by `pip install qwen-vl-utils`."
        )
        smart_resize = None

    if smart_resize is not None:

        def post_process_image(image: Image) -> Image:
            width, height = image.size
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=32,
            )
            return image.resize((resized_width, resized_height))

        multi_modal_data["image"] = post_process_image(multi_modal_data["image"])

    engine_args = EngineArgs(
        model="Qwen/Qwen3-VL-Embedding-2B",
        runner="pooling",
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"do_resize": False} if smart_resize is not None else None,
    )
    default_instruction = "Represent the user's input."
    image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    prompt_text = f"<|im_start|>system\n{default_instruction}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    prompt_image = f"<|im_start|>system\n{default_instruction}<|im_end|>\n<|im_start|>user\n{image_placeholder}<|im_end|>\n<|im_start|>assistant\n"
    prompt_image_text = f"<|im_start|>system\n{default_instruction}<|im_end|>\n<|im_start|>user\n{image_placeholder}{text}<|im_end|>\n<|im_start|>assistant\n"

    llm = LLM(**asdict(engine_args))

    print("Text embedding output:")
    outputs = llm.embed(prompt_text, use_tqdm=False)
    print_embeddings(outputs[0].outputs.embedding)

    print("Image embedding output:")
    outputs = llm.embed(
        {
            "prompt": prompt_image,
            "multi_modal_data": multi_modal_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)

    print("Image+Text embedding output:")
    outputs = llm.embed(
        {
            "prompt": prompt_image_text,
            "multi_modal_data": multi_modal_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)


def run_siglip():
    engine_args = EngineArgs(
        model="google/siglip-base-patch16-224",
        runner="pooling",
        limit_mm_per_prompt={"image": 1},
    )

    llm = LLM(**asdict(engine_args))

    print("Text embedding output:")
    outputs = llm.embed(text, use_tqdm=False)
    print_embeddings(outputs[0].outputs.embedding)

    print("Image embedding output:")
    prompt = ""  # For image input, make sure that the prompt text is empty
    outputs = llm.embed(
        {
            "prompt": prompt,
            "multi_modal_data": multi_modal_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)


def run_vlm2vec_phi3v():
    engine_args = EngineArgs(
        model="TIGER-Lab/VLM2Vec-Full",
        runner="pooling",
        max_model_len=4096,
        trust_remote_code=True,
        mm_processor_kwargs={"num_crops": 4},
        limit_mm_per_prompt={"image": 1},
    )

    llm = LLM(**asdict(engine_args))
    image_token = "<|image_1|>"

    print("Text embedding output:")
    prompt_text = f"Find me an everyday image that matches the given caption: {text}"
    outputs = llm.embed(prompt_text, use_tqdm=False)
    print_embeddings(outputs[0].outputs.embedding)

    print("Image embedding output:")
    prompt_image = f"{image_token} Find a day-to-day image that looks similar to the provided image."  # noqa: E501
    outputs = llm.embed(
        {
            "prompt": prompt_image,
            "multi_modal_data": multi_modal_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)

    print("Image+Text embedding output:")
    prompt_image_text = (
        f"{image_token} Represent the given image with the following question: {text}"  # noqa: E501
    )
    outputs = llm.embed(
        {
            "prompt": prompt_image_text,
            "multi_modal_data": multi_modal_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)


def run_vlm2vec_qwen2vl():
    # vLLM does not support LoRA adapters on multi-modal encoder,
    # so we merge the weights first
    from huggingface_hub.constants import HF_HUB_CACHE
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from vllm.entrypoints.chat_utils import load_chat_template

    model_id = "TIGER-Lab/VLM2Vec-Qwen2VL-2B"

    base_model = AutoModelForImageTextToText.from_pretrained(model_id)
    lora_model = PeftModel.from_pretrained(
        base_model,
        model_id,
        config=PeftConfig.from_pretrained(model_id),
    )
    model = lora_model.merge_and_unload().to(dtype=base_model.dtype)
    model._hf_peft_config_loaded = False  # Needed to save the merged model

    processor = AutoProcessor.from_pretrained(
        model_id,
        # `min_pixels` and `max_pixels` are deprecated for
        # transformers `preprocessor_config.json`
        size={"shortest_edge": 3136, "longest_edge": 12845056},
    )
    processor.chat_template = load_chat_template(
        # The original chat template is not correct
        EMBED_TEMPLATE_DIR / "vlm2vec_qwen2vl.jinja",
    )

    merged_path = str(
        Path(HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--") + "-vllm")
    )
    print(f"Saving merged model to {merged_path}...")
    print(
        "NOTE: This directory is not tracked by `huggingface_hub` "
        "so you have to delete this manually if you don't want it anymore."
    )
    model.save_pretrained(merged_path)
    processor.save_pretrained(merged_path)
    print("Done!")

    engine_args = EngineArgs(
        model=merged_path,
        runner="pooling",
        max_model_len=4096,
        mm_processor_kwargs={
            "min_pixels": 3136,
            "max_pixels": 12845056,
        },
        limit_mm_per_prompt={"image": 1},
    )

    llm = LLM(**asdict(engine_args))
    image_token = "<|image_pad|>"

    print("Text embedding output:")
    prompt_text = f"Find me an everyday image that matches the given caption: {text}"
    outputs = llm.embed(prompt_text, use_tqdm=False)
    print_embeddings(outputs[0].outputs.embedding)

    print("Image embedding output:")
    prompt_image = f"{image_token} Find a day-to-day image that looks similar to the provided image."  # noqa: E501
    outputs = llm.embed(
        {
            "prompt": prompt_image,
            "multi_modal_data": multi_modal_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)

    print("Image+Text embedding output:")
    prompt_image_text = (
        f"{image_token} Represent the given image with the following question: {text}"  # noqa: E501
    )
    outputs = llm.embed(
        {
            "prompt": prompt_image_text,
            "multi_modal_data": multi_modal_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)


model_example_map = {
    "clip": run_clip,
    "e5_v": run_e5_v,
    "qwen3_vl": run_qwen3_vl,
    "siglip": run_siglip,
    "vlm2vec_phi3v": run_vlm2vec_phi3v,
    "vlm2vec_qwen2vl": run_vlm2vec_qwen2vl,
}


def parse_args():
    parser = argparse.ArgumentParser(
        "Script to run a specified VLM through vLLM offline api."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="vlm2vec_phi3v",
        choices=model_example_map.keys(),
        help="The name of the embedding model.",
    )
    return parser.parse_args()


def main(args):
    model_example_map[args.model]()


if __name__ == "__main__":
    args = parse_args()
    main(args)
