import os
import subprocess

from PIL import Image

from vllm import LLM, SamplingParams
from vllm.multimodal.image import ImagePixelData


def run_phi3v():
    model_path = "microsoft/Phi-3-vision-128k-instruct"
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=4096,
        image_input_type="pixel_values",
        image_token_id=32044,
        image_input_shape="1,3,1008,1344",
        image_feature_size=1024,
        disable_image_processor=False,
    )

    image = Image.open("images/stop_sign.jpg")
    user_prompt = "<|user|>\n"
    assistant_prompt = "<|assistant|>\n"
    suffix = "<|end|>\n"

    # single-image prompt
    prompt = "What is shown in this image?"
    prompt = user_prompt+"<|image|>"*1921+f"<s>\n{prompt}{suffix}{assistant_prompt}"

    sampling_params = SamplingParams(temperature=0, max_tokens=64)

    outputs = llm.generate({
        "prompt": prompt,
        "sampling_params": sampling_params,
        "multi_modal_data": ImagePixelData(image),
    })
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
