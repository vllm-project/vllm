# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An example showing how to use vLLM to serve multimodal models
and run online serving with OpenAI client.

Launch the vLLM server with the following command:

(single image inference with Llava)
vllm serve llava-hf/llava-1.5-7b-hf

(multi-image inference with Phi-3.5-vision-instruct)
vllm serve microsoft/Phi-3.5-vision-instruct --runner generate \
    --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt.image 2

(audio inference with Ultravox)
vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b \
    --max-model-len 4096 --trust-remote-code

run the script with
python openai_chat_completion_client_for_multimodal.py --chat-type audio
"""

import base64
import os

import requests
from openai import OpenAI
from utils import get_first_model

from vllm.utils.argparse_utils import FlexibleArgumentParser

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

headers = {"User-Agent": "vLLM Example Client"}


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url, headers=headers) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return result


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local file content to base64 format."""

    with open(file_path, "rb") as file:
        file_content = file.read()
        result = base64.b64encode(file_content).decode("utf-8")

    return result


# Text-only inference
def run_text_only(model: str, max_completion_tokens: int) -> None:
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "What's the capital of France?"}],
        model=model,
        max_completion_tokens=max_completion_tokens,
    )

    result = chat_completion.choices[0].message.content
    print("Chat completion output:\n", result)


# Single-image input inference
def run_single_image(model: str, max_completion_tokens: int) -> None:
    ## Use image url in the payload
    image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    image_file = "/path/to/image.jpg"  # local file
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=max_completion_tokens,
    )

    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output from image url:\n", result)

    ## Use local image url in the payload
    # Launch the API server/engine with the --allowed-local-media-path argument.
    if os.path.exists(image_file):
        chat_completion_from_local_image_url = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"file://{image_file}"},
                        },
                    ],
                }
            ],
            model=model,
            max_completion_tokens=max_completion_tokens,
        )
        result = chat_completion_from_local_image_url.choices[0].message.content
        print("Chat completion output from local image file:\n", result)
    else:
        print(f"Local image file not found at {image_file}, skipping local file test.")

    ## Use base64 encoded image in the payload
    image_base64 = encode_base64_content_from_url(image_url)
    chat_completion_from_base64 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=max_completion_tokens,
    )

    result = chat_completion_from_base64.choices[0].message.content
    print("Chat completion output from base64 encoded image:", result)

    ## Use base64 encoded local image in the payload
    if os.path.exists(image_file):
        local_image_base64 = encode_base64_content_from_file(image_file)
        chat_completion_from_local_image_base64 = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{local_image_base64}"
                            },
                        },
                    ],
                }
            ],
            model=model,
            max_completion_tokens=max_completion_tokens,
        )

        result = chat_completion_from_local_image_base64.choices[0].message.content
        print("Chat completion output from base64 encoded local image:", result)
    else:
        print(f"Local image file not found at {image_file}, skipping local file test.")


# Multi-image input inference
def run_multi_image(model: str, max_completion_tokens: int) -> None:
    image_url_duck = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg"
    image_url_lion = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/lion.jpg"
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What are the animals in these images?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url_duck},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url_lion},
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=max_completion_tokens,
    )

    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output:\n", result)


# Video input inference
def run_video(model: str, max_completion_tokens: int) -> None:
    video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4"
    video_base64 = encode_base64_content_from_url(video_url)

    ## Use video url in the payload
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this video?"},
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=max_completion_tokens,
    )

    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output from video url:\n", result)

    ## Use base64 encoded video in the payload
    chat_completion_from_base64 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this video?"},
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=max_completion_tokens,
    )

    result = chat_completion_from_base64.choices[0].message.content
    print("Chat completion output from base64 encoded video:\n", result)


# Audio input inference
def run_audio(model: str, max_completion_tokens: int) -> None:
    from vllm.assets.audio import AudioAsset

    audio_url = AudioAsset("winning_call").url
    audio_base64 = encode_base64_content_from_url(audio_url)

    # OpenAI-compatible schema (`input_audio`)
    chat_completion_from_base64 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this audio?"},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            # Any format supported by librosa is supported
                            "data": audio_base64,
                            "format": "wav",
                        },
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=max_completion_tokens,
    )

    result = chat_completion_from_base64.choices[0].message.content
    print("Chat completion output from input audio:\n", result)

    # HTTP URL
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this audio?"},
                    {
                        "type": "audio_url",
                        "audio_url": {
                            # Any format supported by librosa is supported
                            "url": audio_url
                        },
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=max_completion_tokens,
    )

    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output from audio url:\n", result)

    # base64 URL
    chat_completion_from_base64 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this audio?"},
                    {
                        "type": "audio_url",
                        "audio_url": {
                            # Any format supported by librosa is supported
                            "url": f"data:audio/ogg;base64,{audio_base64}"
                        },
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=max_completion_tokens,
    )

    result = chat_completion_from_base64.choices[0].message.content
    print("Chat completion output from base64 encoded audio:\n", result)


def run_multi_audio(model: str, max_completion_tokens: int) -> None:
    from vllm.assets.audio import AudioAsset

    # Two different audios to showcase batched inference.
    audio_url = AudioAsset("winning_call").url
    audio_base64 = encode_base64_content_from_url(audio_url)
    audio_url2 = AudioAsset("azacinto_foscolo").url
    audio_base64_2 = encode_base64_content_from_url(audio_url2)

    # OpenAI-compatible schema (`input_audio`)
    chat_completion_from_base64 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Are these two audios the same?"},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": "wav",
                        },
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64_2,
                            "format": "wav",
                        },
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=max_completion_tokens,
    )

    result = chat_completion_from_base64.choices[0].message.content
    print("Chat completion output from input audio:\n", result)


example_function_map = {
    "text-only": run_text_only,
    "single-image": run_single_image,
    "multi-image": run_multi_image,
    "multi-audio": run_multi_audio,
    "video": run_video,
    "audio": run_audio,
}


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using OpenAI client for online serving with "
        "multimodal language models served with vLLM."
    )
    parser.add_argument(
        "--chat-type",
        "-c",
        type=str,
        default="single-image",
        choices=list(example_function_map.keys()),
        help="Conversation type with multimodal data.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        "-n",
        type=int,
        default=128,
        help="Maximum number of tokens to generate for each completion.",
    )
    return parser.parse_args()


def main(args) -> None:
    chat_type = args.chat_type
    model = get_first_model(client)
    example_function_map[chat_type](model, args.max_completion_tokens)


if __name__ == "__main__":
    args = parse_args()
    main(args)
