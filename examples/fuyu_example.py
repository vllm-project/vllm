import os
import subprocess

import torch

from vllm import LLM
from vllm.sequence import MultiModalData

# The assets are located at `s3://air-example-data-2/vllm_opensource_llava/`.


def run_fuyu_pixel_values():
    llm = LLM(
        model="adept/fuyu-8b",
        image_input_type="pixel_values",
        image_token_id=71011,
    )

    # load and create image prompt
    images = torch.load("images/cherry_blossom_pixel_values.pt")

    _, _, H, W = images.shape
    nrow = H // 30 + 1 if H % 30 else H // 30
    ncol = W // 30 + 1 if W % 30 else W // 30

    image_prompt = ("|SPEAKER|" * ncol + "|NEWLINE|") * nrow
    prompt = image_prompt + "Generate a coco-style caption.\n"

    # inference
    outputs = llm.generate(
        prompt,
        multi_modal_data=MultiModalData(MultiModalData.Type.IMAGE, images),
    )

    # Print the outputs.
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")


if __name__ == "__main__":
    # Download from s3
    s3_bucket_path = "s3://air-example-data-2/vllm_opensource_llava/"
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
    run_fuyu_pixel_values()
