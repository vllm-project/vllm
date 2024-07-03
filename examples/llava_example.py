import os
import subprocess

from PIL import Image

from vllm import LLM

# The assets are located at `s3://air-example-data-2/vllm_opensource_llava/`.
# You can use `.buildkite/download-images.sh` to download them


def run_llava():
    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    image = Image.open("images/stop_sign.jpg")

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        },
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def main():
    run_llava()


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
