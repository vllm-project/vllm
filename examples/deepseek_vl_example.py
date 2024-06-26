import argparse
import os
import subprocess

import torch
from PIL import Image

from vllm import LLM
from vllm.multimodal.image import ImageFeatureData, ImagePixelData
from vllm.model_executor.models.deepseek_vl import VLMImageProcessor

# The assets are located at `s3://air-example-data-2/vllm_opensource_llava/`.
# You can use `.buildkite/download-images.sh` to download them
from vllm import SamplingParams

sample_params = SamplingParams(temperature=0, max_tokens=1024)

model = "deepseek-ai/deepseek-vl-7b-chat"


def run_deepseek_vl_pixel_values(*, disable_image_processor: bool = False):
    llm = LLM(
        model=model,
        image_input_type="pixel_values",
        image_token_id=100015,
        image_input_shape="1,3,1024,1024",
        image_feature_size=576,
        disable_image_processor=False,
        gpu_memory_utilization=0.9,
        max_model_len=3072,
        enforce_eager=True,
    )

    prompt = f"You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n User: {'<image_placeholder>'*576} Describe the content of this image.\nAssistant:"

    if disable_image_processor:
        image = get_image_features()
    else:
        image = Image.open("images/stop_sign.jpg")

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": ImagePixelData(image),
        },
        sampling_params=sample_params,
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def run_deepseek_vl_image_features():
    llm = LLM(
        model=model,
        image_input_type="image_features",
        image_token_id=100015,
        image_input_shape="1,3,1024,1024",
        image_feature_size=576,
        gpu_memory_utilization=0.9,
        max_model_len=3072,
        enforce_eager=True,
    )
    prompt = f"You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n User: {'<image_placeholder>'*576} Describe the content of this image.\nAssistant:"

    image: torch.Tensor = get_image_features()

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": ImageFeatureData(image),
        },
        sampling_params=sample_params,
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def get_image_features():
    image_feature = VLMImageProcessor(1024)(Image.open("images/stop_sign.jpg"))[
        "pixel_values"
    ]
    torch.save(image_feature, "images/deepseek_vl_stop_sign.pt")
    return torch.load("images/deepseek_vl_stop_sign.pt")


def main(args):
    if args.type == "pixel_values":
        run_deepseek_vl_pixel_values()
    else:
        run_deepseek_vl_image_features()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo on deepseek-vl")
    parser.add_argument(
        "--type",
        type=str,
        choices=["pixel_values", "image_features"],
        default="pixel_values",
        help="image input type",
    )
    args = parser.parse_args()
    # Download from s3
    s3_bucket_path = "s3://air-example-data-2/vllm_opensource_llava/"
    local_directory = "images"

    # Make sure the local directory exists or create it
    os.makedirs(local_directory, exist_ok=True)

    # Use AWS CLI to sync the directory, assume anonymous access
    subprocess.check_call(
        [
            "aws",
            "s3",
            "sync",
            s3_bucket_path,
            local_directory,
            "--no-sign-request",
        ]
    )
    main(args)
