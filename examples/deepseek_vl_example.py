import os
import subprocess

import torch
from PIL import Image

# The assets are located at `s3://air-example-data-2/vllm_opensource_llava/`.
# You can use `.buildkite/download-images.sh` to download them
from vllm import LLM, SamplingParams

sample_params = SamplingParams(temperature=0, max_tokens=1024)
model = "deepseek-ai/deepseek-vl-7b-chat"
model = "deepseek-ai/deepseek-vl-1.3b-chat"
prompt = "You are a helpful language and vision assistant." \
    "You are able to understand the visual content that the user provides," \
    "and assist the user with a variety of tasks using natural language.\n" \
    "User: <image_placeholder> Describe the content of this image.\nAssistant:"


def run_deepseek_vl():
    llm = LLM(model=model,
              max_model_len=3072,
              enforce_eager=True,
              dtype=torch.bfloat16)

    image = Image.open("images/stop_sign.jpg")

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        },
        sampling_params=sample_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def main():
    run_deepseek_vl()


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
    main()
