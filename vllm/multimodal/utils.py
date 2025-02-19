from functools import lru_cache
from pathlib import Path
from typing import Optional, TypeVar, Union
from urllib.parse import ParseResult, urlparse

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

import vllm.envs as envs
from vllm.connections import HTTPConnection, global_http_connection
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer

from .audio import AudioMediaIO
from .base import MediaIO
from .image import ImageMediaIO
from .inputs import PlaceholderRange
from .video import VideoMediaIO

logger = init_logger(__name__)

cached_get_tokenizer = lru_cache(get_tokenizer)

_M = TypeVar("_M")


class MediaConnector:

    def __init__(
        self,
        connection: HTTPConnection = global_http_connection,
        *,
        allowed_local_media_path: str = "",
    ) -> None:
        super().__init__()

        self.connection = connection

        if allowed_local_media_path:
            allowed_local_media_path_ = Path(allowed_local_media_path)

            if not allowed_local_media_path_.exists():
                raise ValueError(
                    "Invalid `--allowed-local-media-path`: The path "
                    f"{allowed_local_media_path_} does not exist.")
            if not allowed_local_media_path_.is_dir():
                raise ValueError(
                    "Invalid `--allowed-local-media-path`: The path "
                    f"{allowed_local_media_path_} must be a directory.")
        else:
            allowed_local_media_path_ = None

        self.allowed_local_media_path = allowed_local_media_path_

    def _load_data_url(
        self,
        url_spec: ParseResult,
        media_io: MediaIO[_M],
    ) -> _M:
        data_spec, data = url_spec.path.split(",", 1)
        media_type, data_type = data_spec.split(";", 1)

        if data_type != "base64":
            msg = "Only base64 data URLs are supported for now."
            raise NotImplementedError(msg)

        return media_io.load_base64(media_type, data)

    def _load_file_url(
        self,
        url_spec: ParseResult,
        media_io: MediaIO[_M],
    ) -> _M:
        allowed_local_media_path = self.allowed_local_media_path
        if allowed_local_media_path is None:
            raise RuntimeError("Cannot load local files without "
                               "`--allowed-local-media-path`.")

        filepath = Path(url_spec.path)
        if allowed_local_media_path not in filepath.resolve().parents:
            raise ValueError(
                f"The file path {filepath} must be a subpath "
                f"of `--allowed-local-media-path` {allowed_local_media_path}.")

        return media_io.load_file(filepath)

    def load_from_url(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: Optional[int] = None,
    ) -> _M:
        url_spec = urlparse(url)

        if url_spec.scheme.startswith("http"):
            connection = self.connection
            data = connection.get_bytes(url, timeout=fetch_timeout)

            return media_io.load_bytes(data)

        if url_spec.scheme == "data":
            return self._load_data_url(url_spec, media_io)

        if url_spec.scheme == "file":
            return self._load_file_url(url_spec, media_io)

        msg = "The URL must be either a HTTP, data or file URL."
        raise ValueError(msg)

    async def load_from_url_async(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: Optional[int] = None,
    ) -> _M:
        url_spec = urlparse(url)
        if url_spec.scheme.startswith("http"):
            try:
                import requests
                image = Image.open(requests.get(url, stream=True).raw)
                return image
            except:
                connection = self.connection
                data = await connection.async_get_bytes(url, timeout=fetch_timeout)

                return media_io.load_bytes(data)

        if url_spec.scheme == "data":
            return self._load_data_url(url_spec, media_io)

        if url_spec.scheme == "file":
            return self._load_file_url(url_spec, media_io)
    
        import os
        if url_spec.scheme == "" and os.path.exists(url):
            image = Image.open(url).convert('RGB')
            return image

        msg = "The URL must be either a HTTP, data, file URL or a exist path."
        raise ValueError(msg)

    def fetch_audio(
        self,
        audio_url: str,
    ) -> tuple[np.ndarray, Union[int, float]]:
        """
        Load audio from a URL.
        """
        audio_io = AudioMediaIO()

        return self.load_from_url(
            audio_url,
            audio_io,
            fetch_timeout=envs.VLLM_AUDIO_FETCH_TIMEOUT,
        )

    async def fetch_audio_async(
        self,
        audio_url: str,
    ) -> tuple[np.ndarray, Union[int, float]]:
        """
        Asynchronously fetch audio from a URL.
        """
        audio_io = AudioMediaIO()

        return await self.load_from_url_async(
            audio_url,
            audio_io,
            fetch_timeout=envs.VLLM_AUDIO_FETCH_TIMEOUT,
        )

    def fetch_image(
        self,
        image_url: str,
        *,
        image_mode: str = "RGB",
    ) -> Image.Image:
        """
        Load a PIL image from a HTTP or base64 data URL.

        By default, the image is converted into RGB format.
        """
        image_io = ImageMediaIO(image_mode=image_mode)

        return self.load_from_url(
            image_url,
            image_io,
            fetch_timeout=envs.VLLM_IMAGE_FETCH_TIMEOUT,
        )

    async def fetch_image_async(
        self,
        image_url: str,
        *,
        image_mode: str = "RGB",
    ) -> Image.Image:
        """
        Asynchronously load a PIL image from a HTTP or base64 data URL.

        By default, the image is converted into RGB format.
        """
        image_io = ImageMediaIO(image_mode=image_mode)

        return await self.load_from_url_async(
            image_url,
            image_io,
            fetch_timeout=envs.VLLM_IMAGE_FETCH_TIMEOUT,
        )

    def fetch_video(
        self,
        video_url: str,
        *,
        image_mode: str = "RGB",
        num_frames: int = 32,
    ) -> npt.NDArray:
        """
        Load video from a HTTP or base64 data URL.
        """
        image_io = ImageMediaIO(image_mode=image_mode)
        video_io = VideoMediaIO(image_io, num_frames=num_frames)

        return self.load_from_url(
            video_url,
            video_io,
            fetch_timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
        )

    async def fetch_video_async(
        self,
        video_url: str,
        *,
        image_mode: str = "RGB",
        num_frames: int = 32,
    ) -> npt.NDArray:
        """
        Asynchronously load video from a HTTP or base64 data URL.

        By default, the image is converted into RGB format.
        """
        image_io = ImageMediaIO(image_mode=image_mode)
        video_io = VideoMediaIO(image_io, num_frames=num_frames)

        return await self.load_from_url_async(
            video_url,
            video_io,
            fetch_timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
        )


global_media_connector = MediaConnector()
"""The global :class:`MediaConnector` instance used by vLLM."""

fetch_audio = global_media_connector.fetch_audio
fetch_image = global_media_connector.fetch_image
fetch_video = global_media_connector.fetch_video


def encode_audio_base64(
    audio: np.ndarray,
    sampling_rate: int,
) -> str:
    """Encode audio as base64."""
    audio_io = AudioMediaIO()
    return audio_io.encode_base64((audio, sampling_rate))


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
    image_io = ImageMediaIO(image_mode=image_mode)
    return image_io.encode_base64(image, image_format=format)


def encode_video_base64(frames: npt.NDArray) -> str:
    image_io = ImageMediaIO()
    video_io = VideoMediaIO(image_io)
    return video_io.encode_base64(frames)


def resolve_visual_encoder_outputs(
    encoder_outputs: Union[torch.Tensor, list[torch.Tensor]],
    feature_sample_layers: Optional[list[int]],
    post_layer_norm: Optional[torch.nn.LayerNorm],
    max_possible_layers: int,
) -> torch.Tensor:
    """Given the outputs a visual encoder module that may correspond to the
    output of the last layer, or a list of hidden states to be stacked,
    handle post normalization and resolve it into a single output tensor.

    Args:
        encoder_outputs: Output of encoder's last layer or all hidden states.
        feature_sample_layers: Optional layer indices to grab from the encoder
            outputs; if provided, encoder outputs must be a list.
        post_layer_norm: Post norm to apply to the output of the encoder.
        max_possible_layers: Total layers in the fully loaded visual encoder.

    """
    if feature_sample_layers is None:
        if post_layer_norm is not None:
            return post_layer_norm(encoder_outputs)
        return encoder_outputs

    # Get the hidden states corresponding to the layer indices.
    # Negative values are relative to the full visual encoder,
    # so offset them depending on how many layers were loaded.
    # NOTE: this assumes that encoder_outputs contains a list
    # of hidden states in the same order as the encoder layers
    # that produced them.
    offset = max_possible_layers - len(encoder_outputs)
    hs_pool = [
        encoder_outputs[layer_idx]
        if layer_idx >= 0 else encoder_outputs[layer_idx + offset]
        for layer_idx in feature_sample_layers
    ]

    # Apply post-norm on the final hidden state if we are using it
    uses_last_layer = feature_sample_layers[-1] in (len(hs_pool) - 1, -1)
    if post_layer_norm is not None and uses_last_layer:
        hs_pool[-1] = post_layer_norm(encoder_outputs)
    return torch.cat(hs_pool, dim=-1)


# Utilities for input processors
_T = TypeVar("_T", str, int)


def repeat_and_pad_token(
    token: _T,
    *,
    repeat_count: int = 1,
    pad_token_left: Optional[_T] = None,
    pad_token_right: Optional[_T] = None,
) -> list[_T]:
    replacement = [token] * repeat_count
    if pad_token_left is not None:
        replacement = [pad_token_left] + replacement
    if pad_token_right is not None:
        replacement = replacement + [pad_token_right]

    return replacement


def repeat_and_pad_placeholder_tokens(
    tokenizer: AnyTokenizer,
    prompt: Optional[str],
    prompt_token_ids: list[int],
    *,
    placeholder_token_id: int,
    repeat_count: Union[int, list[int]],
    pad_token_left: Optional[int] = None,
    pad_token_right: Optional[int] = None,
) -> tuple[Optional[str], list[int], list[PlaceholderRange]]:
    if isinstance(repeat_count, int):
        repeat_count = [repeat_count]

    if prompt is None:
        new_prompt = None
    else:
        placeholder_token_str = tokenizer.decode(placeholder_token_id)
        pad_token_str_left = (None if pad_token_left is None else
                              tokenizer.decode(pad_token_left))
        pad_token_str_right = (None if pad_token_right is None else
                               tokenizer.decode(pad_token_right))

        placeholder_token_count = prompt.count(placeholder_token_str)
        # This is an arbitrary number to distinguish between the two cases
        if placeholder_token_count > 16:
            logger.warning(
                "Please follow the prompt format that is "
                "documented on HuggingFace which does not involve "
                "repeating %s tokens.", placeholder_token_str)
        if placeholder_token_count < len(repeat_count):
            logger.warning(
                "The number of multi-modal placeholder tokens in the prompt "
                "is less than the number of multi-modal inputs. Extra "
                "placeholder tokens will be treated as plain text")
            repeat_count = repeat_count[:placeholder_token_count]

        prompt_parts = prompt.split(placeholder_token_str,
                                    maxsplit=len(repeat_count))
        new_prompt = ""
        for i, repeat_count_item in enumerate(repeat_count):
            replacement_str = "".join(
                repeat_and_pad_token(
                    placeholder_token_str,
                    repeat_count=repeat_count_item,
                    pad_token_left=pad_token_str_left,
                    pad_token_right=pad_token_str_right,
                ))
            # The image tokens are removed to be consistent with HuggingFace
            new_prompt += prompt_parts[i] + replacement_str
        new_prompt += prompt_parts[-1]

    new_token_ids = list[int]()
    placeholder_ranges = list[PlaceholderRange]()
    placeholder_token_idx = 0
    for i, token in enumerate(prompt_token_ids):
        if token == placeholder_token_id:
            curr_repeat_count = repeat_count[placeholder_token_idx]
            replacement_ids = repeat_and_pad_token(
                placeholder_token_id,
                repeat_count=curr_repeat_count,
                pad_token_left=pad_token_left,
                pad_token_right=pad_token_right,
            )
            offset = len(new_token_ids)
            if pad_token_left is not None:
                offset += 1
            placeholder_ranges.append({
                "offset": offset,
                "length": curr_repeat_count,
            })
            new_token_ids.extend(replacement_ids)
            placeholder_token_idx += 1

            # No need to further scan the list since we replaced all tokens
            if placeholder_token_idx >= len(repeat_count):
                new_token_ids.extend(prompt_token_ids[i + 1:])
                break
        else:
            new_token_ids.append(token)

    return new_prompt, new_token_ids, placeholder_ranges


def consecutive_placeholder_ranges(
        num_items: int,
        item_size: int,
        initial_offset: int = 0) -> list[PlaceholderRange]:
    """Returns a list of consecutive PlaceholderRanges of a fixed size"""

    return [
        PlaceholderRange(offset=initial_offset + i * item_size,
                         length=item_size) for i in range(num_items)
    ]