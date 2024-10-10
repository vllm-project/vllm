import argparse
import requests
from io import BytesIO
from PIL import Image, ImageFile

from vllm import LLM
from vllm.sampling_params import SamplingParams


ImageFile.LOAD_TRUNCATED_IMAGES = True


def download_image(url: str):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open the image from the response content
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        return image
    else:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")


def vllm_generate():
    inputs = [
        {
            "prompt": "Describe this image.",
            "multi_modal_data": {"image": download_image("https://picsum.photos/id/9/1080/720")}
        },
        {
            "prompt": "Describe what you see in this image.",
            "multi_modal_data": {"image": download_image("https://picsum.photos/id/23/1080/720")}
        },
    ]

    outputs = llm.generate(
        inputs, 
        sampling_params=sampling_params
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def vllm_chat():
    url_1 = "https://picsum.photos/id/9/1080/720"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url_1
                    }
                },
            ],
        },
        {
            "role": "assistant",
            "content": "The image shows some objects.",
        },
        {
            "role": "user",
            "content": "What objects do you exactly see in the image?",
        },
    ]

    outputs = llm.chat(
        messages,
        sampling_params=sampling_params
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vllm example for Molmo-7B-O-0924, Molmo-7B-D-0924, Molmo-72B-0924"
    )
    parser.add_argument("--model_path", type=str, default="allenai/Molmo-7B-D-0924")
    args = parser.parse_args()
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        max_tokens=768,
        temperature=0,
    )
    vllm_generate()
    vllm_chat()