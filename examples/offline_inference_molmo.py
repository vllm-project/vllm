import argparse
import numpy as np
import requests
from io import BytesIO
import base64
from PIL import Image, ImageFile, ImageOps
import torch
from typing import Optional

from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm_molmo.molmo import MolmoForCausalLM


ImageFile.LOAD_TRUNCATED_IMAGES = True


def download_image_to_numpy(url):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open the image from the response content
        image = Image.open(BytesIO(response.content)).convert("RGB")

        image = ImageOps.exif_transpose(image)
        
        # Convert the image to a NumPy array
        image_array = np.array(image).astype(np.uint8)
        
        return image_array
    else:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")


def vllm_generate():
    inputs = [
        {
            "prompt": "Describe this image.",
            "multi_modal_data": {"image": download_image_to_numpy("https://picsum.photos/id/9/1080/720")}
        },
        {
            "prompt": "Describe what you see in this image.",
            "multi_modal_data": {"image": download_image_to_numpy("https://picsum.photos/id/23/1080/720")}
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

        # Error: Invalid message role {{ message['role'] }} at index {{ loop.index }}
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def set_tf_memory_growth():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            # Set memory growth for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth set for GPUs")
        except RuntimeError as e:
            print(e)


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
    set_tf_memory_growth()
    vllm_generate()
    vllm_chat()