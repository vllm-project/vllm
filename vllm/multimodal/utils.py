# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os
from io import BytesIO
from typing import Union

import requests
from PIL import Image

from vllm.envs import VLLM_IMAGE_FETCH_TIMEOUT
from vllm.utils import make_async
from vllm.multimodal.image import ImagePixelData


def encode_image_base64(image: Image.Image) -> str:
    """encode image to base64 format."""
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def load_image_from_base64(image: Union[bytes, str]) -> Image.Image:
    """load image from base64 format."""
    return Image.open(BytesIO(base64.b64decode(image)))


def fetch_image(image_url: str) -> Image.Image:
    """load image from url, local path or openai GPT4V."""
    
    # Avoid circular import
    from vllm import __version__ as VLLM_VERSION

    headers = {"User-Agent": f"vLLM/{VLLM_VERSION}"}
    if image_url.startswith('http'):
        response = requests.get(image_url,
                                headers=headers,
                                timeout=VLLM_IMAGE_FETCH_TIMEOUT)
        response.raise_for_status()

        # Open the image using PIL
        img = Image.open(BytesIO(response.content))
    elif image_url.startswith('data:image'):
        img = load_image_from_base64(image_url.split(',')[1])
    else:
        raise ValueError("Invalid image url: A valid image url must start with either 'data:image' or 'http'.")

    return img

async_fetch_image = make_async(fetch_image) # type: ignore

async def async_get_and_parse_image(image_url: str) -> ImagePixelData:
    with await async_fetch_image(image_url) as image:
        return ImagePixelData(image)
