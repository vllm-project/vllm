import argparse
import os
import subprocess

import torch

from vllm import LLM
from vllm.sequence import MultiModalData

def run_idefics2_pixel_values():
    llm = LLM(
        model="HuggingFaceM4/idefics2-8b",
        image_input_type="pixel_values",
        image_token_id=32000,
        image_input_shape="1,3,980,980",
        image_feature_size=576,
        dtype='float16',
    )

    prompt = "<image>" * 64 + (
        "\nUSER: What is the content of this image?\nASSISTANT:")

    # This should be provided by another online or offline component.
    images = torch.load("images/bird_pixel_values.pt")
    print(images.shape)
    outputs = llm.generate(prompt,
                           multi_modal_data=MultiModalData(
                               type=MultiModalData.Type.IMAGE, data=images))

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

def run_llava_pixel_values():
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        image_input_type="pixel_values",
        image_token_id=32000,
        image_input_shape="1,3,336,336",
        image_feature_size=576,
    )

    prompt = "<image>" * 576 + (
        "\nUSER: What is the content of this image?\nASSISTANT:")

    # This should be provided by another online or offline component.
    images = torch.load("images/stop_sign_pixel_values.pt")

    outputs = llm.generate(prompt,
                           multi_modal_data=MultiModalData(
                               type=MultiModalData.Type.IMAGE, data=images))
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def run_llava_image_features():
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        image_input_type="image_features",
        image_token_id=32000,
        image_input_shape="1,576,1024",
        image_feature_size=576,
    )

    prompt = "<image>" * 576 + (
        "\nUSER: What is the content of this image?\nASSISTANT:")

    # This should be provided by another online or offline component.
    images = torch.load("images/stop_sign_image_features.pt")

    outputs = llm.generate(prompt,
                           multi_modal_data=MultiModalData(
                               type=MultiModalData.Type.IMAGE, data=images))
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        
def main(args):
    if args.model == "llava" and args.type == "pixel_values":
        run_llava_pixel_values()
    elif args.model == "llava" and args.type == "image_features":
        run_llava_image_features()
    elif args.model == "idefics2" and args.type == "pixel_values":
        run_idefics2_pixel_values()
    elif args.model == "idefics2" and args.type == "image_features":
        print("Idefics2 does not support image_features")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo on Llava")
    
    parser.add_argument("--model",
                        type=str,
                        choices=["llava", "idefics2"],
                        default="llava",
                        help="model type")

    parser.add_argument("--type",
                        type=str,
                        choices=["pixel_values", "image_features"],
                        default="pixel_values",
                        help="image input type")
                        
    args = parser.parse_args()
    # Download from s3
    s3_bucket_path = "s3://vllm-public-assets/vision_model_images/"
    local_directory = "images"

    # Make sure the local directory exists or create it
    os.makedirs(local_directory, exist_ok=True)

    # Use AWS CLI to sync the directory, assume anonymous access
    subprocess.check_call([
        "aws",
        "s3",
        "sync",
        s3_bucket_path,
        local_directory,
        "--no-sign-request",
    ])

    main(args)
    