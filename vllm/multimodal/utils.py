import base64
from io import BytesIO
from typing import Optional, Union
from urllib.parse import urlparse

import aiohttp
import requests
from PIL import Image

from vllm.envs import VLLM_IMAGE_FETCH_TIMEOUT
from vllm.multimodal.base import MultiModalDataDict
from vllm.version import __version__ as VLLM_VERSION


def _validate_remote_url(url: str, *, name: str):
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ["http", "https"]:
        raise ValueError(f"Invalid '{name}': A valid '{name}' "
                         "must have scheme 'http' or 'https'.")


def _get_request_headers():
    return {"User-Agent": f"vLLM/{VLLM_VERSION}"}


def _load_image_from_bytes(b: bytes):
    image = Image.open(BytesIO(b))
    image.load()
    return image


def _load_image_from_data_url(image_url: str):
    # Only split once and assume the second part is the base64 encoded image
    _, image_base64 = image_url.split(",", 1)
    return load_image_from_base64(image_base64)


def fetch_image(image_url: str) -> Image.Image:
    """Load PIL image from a url or base64 encoded openai GPT4V format"""
    if image_url.startswith('http'):
        _validate_remote_url(image_url, name="image_url")

        headers = _get_request_headers()

        with requests.get(url=image_url, headers=headers) as response:
            response.raise_for_status()
            image_raw = response.content
        image = _load_image_from_bytes(image_raw)

    elif image_url.startswith('data:image'):
        image = _load_image_from_data_url(image_url)
    else:
        raise ValueError("Invalid 'image_url': A valid 'image_url' must start "
                         "with either 'data:image' or 'http'.")

    return image


class ImageFetchAiohttp:
    aiohttp_client: Optional[aiohttp.ClientSession] = None

    @classmethod
    def get_aiohttp_client(cls) -> aiohttp.ClientSession:
        if cls.aiohttp_client is None:
            timeout = aiohttp.ClientTimeout(total=VLLM_IMAGE_FETCH_TIMEOUT)
            connector = aiohttp.TCPConnector()
            cls.aiohttp_client = aiohttp.ClientSession(timeout=timeout,
                                                       connector=connector)

        return cls.aiohttp_client

    @classmethod
    async def fetch_image(cls, image_url: str) -> Image.Image:
        """Load PIL image from a url or base64 encoded openai GPT4V format"""

        if image_url.startswith('http'):
            _validate_remote_url(image_url, name="image_url")

            client = cls.get_aiohttp_client()
            headers = _get_request_headers()

            async with client.get(url=image_url, headers=headers) as response:
                response.raise_for_status()
                image_raw = await response.read()
            image = _load_image_from_bytes(image_raw)

        elif image_url.startswith('data:image'):
            image = _load_image_from_data_url(image_url)
        else:
            raise ValueError(
                "Invalid 'image_url': A valid 'image_url' must start "
                "with either 'data:image' or 'http'.")

        return image


async def async_get_and_parse_image(image_url: str) -> MultiModalDataDict:
    image = await ImageFetchAiohttp.fetch_image(image_url)
    return {"image": image}


def encode_image_base64(image: Image.Image, format: str = 'JPEG') -> str:
    """Encode a pillow image to base64 format."""

    buffered = BytesIO()
    if format == 'JPEG':
        image = image.convert('RGB')
    image.save(buffered, format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def load_image_from_base64(image: Union[bytes, str]) -> Image.Image:
    """Load image from base64 format."""
    return _load_image_from_bytes(base64.b64decode(image))


def rescale_image_size(image: Image.Image, size_factor: float) -> Image.Image:
    """Rescale the dimensions of an image by a constant factor."""
    new_width = int(image.width * size_factor)
    new_height = int(image.height * size_factor)
    return image.resize((new_width, new_height))
