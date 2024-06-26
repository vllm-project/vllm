import math
import os
import subprocess

from PIL import Image

from vllm import LLM, SamplingParams
from vllm.multimodal.image import ImagePixelData


def run_fuyu_pixel_values():
    llm = LLM(
        model="adept/fuyu-8b",
        max_model_len=4096,
        image_input_type="pixel_values",
        image_token_id=71011,
        image_input_shape="1,3,1080,1920",
        image_feature_size=2304,
    )

    # load and create image prompt
    image = Image.open("images/stop_sign.jpg")
    W, H = image.size

    nrow = math.ceil(min(H, 1080) / 30)
    ncol = math.ceil(min(W, 1920) / 30)

    # single-image prompt
    prompt = "<image>\nWhat is the content of this image?\n"
    prompt = prompt.replace("<image>",
                            ("|SPEAKER|" * ncol + "|NEWLINE|") * nrow)

    sampling_params = SamplingParams(temperature=0, max_tokens=64)

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": ImagePixelData(image),
        },
        sampling_params=sampling_params)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


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
