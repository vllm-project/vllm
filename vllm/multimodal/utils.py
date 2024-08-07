import base64
from io import BytesIO
from typing import Union

from PIL import Image

from vllm.connections import global_http_connection
from vllm.envs import VLLM_IMAGE_FETCH_TIMEOUT
from vllm.multimodal.base import MultiModalDataDict


def _load_image_from_bytes(b: bytes):
    image = Image.open(BytesIO(b))
    image.load()
    return image


def _load_image_from_data_url(image_url: str):
    # Only split once and assume the second part is the base64 encoded image
    _, image_base64 = image_url.split(",", 1)
    return load_image_from_base64(image_base64)


def fetch_image(image_url: str, *, image_mode: str = "RGB") -> Image.Image:
    """
    Load a PIL image from a HTTP or base64 data URL.

    By default, the image is converted into RGB format.
    """
    if image_url.startswith('http'):
        image_raw = global_http_connection.get_bytes(
            image_url, timeout=VLLM_IMAGE_FETCH_TIMEOUT)
        image = _load_image_from_bytes(image_raw)

    elif image_url.startswith('data:image'):
        image = _load_image_from_data_url(image_url)
    else:
        raise ValueError("Invalid 'image_url': A valid 'image_url' must start "
                         "with either 'data:image' or 'http'.")

    return image.convert(image_mode)


async def async_fetch_image(image_url: str,
                            *,
                            image_mode: str = "RGB") -> Image.Image:
    """
    Asynchronously load a PIL image from a HTTP or base64 data URL.

    By default, the image is converted into RGB format.
    """
    if image_url.startswith('http'):
        image_raw = await global_http_connection.async_get_bytes(
            image_url, timeout=VLLM_IMAGE_FETCH_TIMEOUT)
        image = _load_image_from_bytes(image_raw)

    elif image_url.startswith('data:image'):
        image = _load_image_from_data_url(image_url)
    else:
        raise ValueError("Invalid 'image_url': A valid 'image_url' must start "
                         "with either 'data:image' or 'http'.")

    return image.convert(image_mode)


async def async_get_and_parse_image(image_url: str) -> MultiModalDataDict:
    image = await async_fetch_image(image_url)
    return {"image": image}


def encode_image_base64(
    image: Image.Image,
    *,
    image_mode: str = "RGB",
    format: str = "JPEG",
) -> str:
    """
    Encode a pillow image to base64 format.

    By default, the image is converted into RGB format before being encoded.
    """
    buffered = BytesIO()
    image = image.convert(image_mode)
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
