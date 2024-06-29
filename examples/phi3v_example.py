import os
import subprocess

from PIL import Image

from vllm import LLM, SamplingParams
from vllm.multimodal.image import ImagePixelData


def run_phi3v():
    model_path = "microsoft/Phi-3-vision-128k-instruct"

    # Note: The model has 128k context length by default which may cause OOM
    # In this example, we override max_model_len to 2048.
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        image_input_type="pixel_values",
        image_token_id=32044,
        image_input_shape="1,3,1008,1344",
        image_feature_size=1921,
        max_model_len=2048,
    )

    image = Image.open("images/cherry_blossom.jpg")

    # single-image prompt
    prompt = "<|user|>\n<|image_1|>\nWhat is the season?<|end|>\n<|assistant|>\n"  # noqa: E501
    prompt = prompt.replace("<|image_1|>", "<|image|>" * 1921 + "<s>")

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
    run_phi3v()
