import base64
import os
from functools import lru_cache
from io import BytesIO
from typing import Any, List, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

import vllm.envs as envs
from vllm.connections import global_http_connection
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer

from .inputs import MultiModalDataDict, PlaceholderRange

logger = init_logger(__name__)

cached_get_tokenizer = lru_cache(get_tokenizer)


def _load_image_from_bytes(b: bytes) -> Image.Image:
    image = Image.open(BytesIO(b))
    image.load()
    return image


def _is_subpath(image_path: str, allowed_local_media_path: str) -> bool:
    # Get the common path
    common_path = os.path.commonpath([
        os.path.abspath(image_path),
        os.path.abspath(allowed_local_media_path)
    ])
    # Check if the common path is the same as allowed_local_media_path
    return common_path == os.path.abspath(allowed_local_media_path)


def _load_image_from_file(image_url: str,
                          allowed_local_media_path: str) -> Image.Image:
    if not allowed_local_media_path:
        raise ValueError("Invalid 'image_url': Cannot load local files without"
                         "'--allowed-local-media-path'.")
    if allowed_local_media_path:
        if not os.path.exists(allowed_local_media_path):
            raise ValueError(
                "Invalid '--allowed-local-media-path': "
                f"The path {allowed_local_media_path} does not exist.")
        if not os.path.isdir(allowed_local_media_path):
            raise ValueError(
                "Invalid '--allowed-local-media-path': "
                f"The path {allowed_local_media_path} must be a directory.")

    # Only split once and assume the second part is the image path
    _, image_path = image_url.split("file://", 1)
    if not _is_subpath(image_path, allowed_local_media_path):
        raise ValueError(
            f"Invalid 'image_url': The file path {image_path} must"
            " be a subpath of '--allowed-local-media-path'"
            f" '{allowed_local_media_path}'.")

    image = Image.open(image_path)
    image.load()
    return image


def _load_image_from_data_url(image_url: str) -> Image.Image:
    # Only split once and assume the second part is the base64 encoded image
    _, image_base64 = image_url.split(",", 1)
    return load_image_from_base64(image_base64)


def fetch_image(image_url: str,
                *,
                image_mode: str = "RGB",
                allowed_local_media_path: str = "") -> Image.Image:
    """
    Load a PIL image from a HTTP or base64 data URL.

    By default, the image is converted into RGB format.
    """
    if image_url.startswith('http'):
        image_raw = global_http_connection.get_bytes(
            image_url,
            timeout=envs.VLLM_IMAGE_FETCH_TIMEOUT,
        )
        image = _load_image_from_bytes(image_raw)

    elif image_url.startswith('data:image'):
        image = _load_image_from_data_url(image_url)
    elif image_url.startswith('file://'):
        image = _load_image_from_file(image_url, allowed_local_media_path)
    else:
        raise ValueError("Invalid 'image_url': A valid 'image_url' must start "
                         "with either 'data:image', 'file://' or 'http'.")

    return image.convert(image_mode)


async def async_fetch_image(image_url: str,
                            *,
                            image_mode: str = "RGB",
                            allowed_local_media_path: str = "") -> Image.Image:
    """
    Asynchronously load a PIL image from a HTTP or base64 data URL.

    By default, the image is converted into RGB format.
    """
    if image_url.startswith('http'):
        image_raw = await global_http_connection.async_get_bytes(
            image_url,
            timeout=envs.VLLM_IMAGE_FETCH_TIMEOUT,
        )
        image = _load_image_from_bytes(image_raw)

    elif image_url.startswith('data:image'):
        image = _load_image_from_data_url(image_url)
    elif image_url.startswith('file://'):
        image = _load_image_from_file(image_url, allowed_local_media_path)
    else:
        raise ValueError("Invalid 'image_url': A valid 'image_url' must start "
                         "with either 'data:image', 'file://' or 'http'.")

    return image.convert(image_mode)


def _load_video_frames_from_bytes(b: bytes):
    frame = Image.open(BytesIO(b))
    return np.array(frame)


def load_video_frames_from_base64(frame: Union[bytes, str]):
    """Load frame from base64 format."""
    return _load_video_frames_from_bytes(base64.b64decode(frame))


def _load_video_from_bytes(b: bytes, num_frames: int = 32):
    _, decord = try_import_video_packages()

    video_path = BytesIO(b)
    vr = decord.VideoReader(video_path, num_threads=1)
    total_frame_num = len(vr)

    if total_frame_num > num_frames:
        uniform_sampled_frames = np.linspace(0,
                                             total_frame_num - 1,
                                             num_frames,
                                             dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    else:
        frame_idx = [i for i in range(0, total_frame_num)]
    frames = vr.get_batch(frame_idx).asnumpy()

    return frames


def _load_video_from_data_url(video_url: str):
    # Only split once and assume the second part is the base64 encoded image
    frames_base64 = video_url.split(",")[1:]
    return np.stack([
        load_video_frames_from_base64(frame_base64)
        for frame_base64 in frames_base64
    ])


def fetch_video(video_url: str, *, num_frames: int = 32) -> npt.NDArray:
    """
    Load video from a HTTP or base64 data URL.
    """
    if video_url.startswith('http') or video_url.startswith('https'):
        video_raw = global_http_connection.get_bytes(
            video_url,
            timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
        )
        video = _load_video_from_bytes(video_raw, num_frames)
    elif video_url.startswith('data:video'):
        video = _load_video_from_data_url(video_url)
    else:
        raise ValueError("Invalid 'video_url': A valid 'video_url' must start "
                         "with either 'data:video' or 'http'.")
    return video


async def async_fetch_video(video_url: str,
                            *,
                            num_frames: int = 32) -> npt.NDArray:
    """
    Asynchronously load video from a HTTP or base64 data URL.

    By default, the image is converted into RGB format.
    """
    if video_url.startswith('http') or video_url.startswith('https'):
        video_raw = await global_http_connection.async_get_bytes(
            video_url,
            timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
        )
        video = _load_video_from_bytes(video_raw, num_frames)
    elif video_url.startswith('data:video'):
        video = _load_video_from_data_url(video_url)
    else:
        raise ValueError("Invalid 'video_url': A valid 'video_url' must start "
                         "with either 'data:video' or 'http'.")
    return video


def try_import_audio_packages() -> Tuple[Any, Any]:
    try:
        import librosa
        import soundfile
    except ImportError as exc:
        raise ImportError(
            "Please install vllm[audio] for audio support.") from exc
    return librosa, soundfile


def fetch_audio(audio_url: str) -> Tuple[np.ndarray, Union[int, float]]:
    """
    Load audio from a URL.
    """
    librosa, _ = try_import_audio_packages()

    if audio_url.startswith("http"):
        audio_bytes = global_http_connection.get_bytes(
            audio_url,
            timeout=envs.VLLM_AUDIO_FETCH_TIMEOUT,
        )
    elif audio_url.startswith("data:audio"):
        _, audio_base64 = audio_url.split(",", 1)
        audio_bytes = base64.b64decode(audio_base64)
    else:
        raise ValueError("Invalid 'audio_url': A valid 'audio_url' must start "
                         "with either 'data:audio' or 'http'.")

    return librosa.load(BytesIO(audio_bytes), sr=None)


async def async_fetch_audio(
        audio_url: str) -> Tuple[np.ndarray, Union[int, float]]:
    """
    Asynchronously fetch audio from a URL.
    """
    librosa, _ = try_import_audio_packages()

    if audio_url.startswith("http"):
        audio_bytes = await global_http_connection.async_get_bytes(
            audio_url,
            timeout=envs.VLLM_AUDIO_FETCH_TIMEOUT,
        )
    elif audio_url.startswith("data:audio"):
        _, audio_base64 = audio_url.split(",", 1)
        audio_bytes = base64.b64decode(audio_base64)
    else:
        raise ValueError("Invalid 'audio_url': A valid 'audio_url' must start "
                         "with either 'data:audio' or 'http'.")

    return librosa.load(BytesIO(audio_bytes), sr=None)


def get_and_parse_audio(audio_url: str) -> MultiModalDataDict:
    audio, sr = fetch_audio(audio_url)
    return {"audio": (audio, sr)}


def get_and_parse_image(
        image_url: str,
        *,
        allowed_local_media_path: str = "") -> MultiModalDataDict:
    image = fetch_image(image_url,
                        allowed_local_media_path=allowed_local_media_path)
    return {"image": image}


def get_and_parse_video(video_url: str) -> MultiModalDataDict:
    video = fetch_video(video_url)
    return {"video": video}


async def async_get_and_parse_audio(audio_url: str) -> MultiModalDataDict:
    audio, sr = await async_fetch_audio(audio_url)
    return {"audio": (audio, sr)}


async def async_get_and_parse_image(
        image_url: str,
        *,
        allowed_local_media_path: str = "") -> MultiModalDataDict:
    image = await async_fetch_image(
        image_url, allowed_local_media_path=allowed_local_media_path)
    return {"image": image}


async def async_get_and_parse_video(video_url: str) -> MultiModalDataDict:
    video = await async_fetch_video(video_url)
    return {"video": video}


def encode_audio_base64(
    audio: np.ndarray,
    sampling_rate: int,
) -> str:
    """Encode audio as base64."""
    _, soundfile = try_import_audio_packages()

    buffered = BytesIO()
    soundfile.write(buffered, audio, sampling_rate, format="WAV")

    return base64.b64encode(buffered.getvalue()).decode('utf-8')


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


def rescale_image_size(image: Image.Image,
                       size_factor: float,
                       transpose: int = -1) -> Image.Image:
    """Rescale the dimensions of an image by a constant factor."""
    new_width = int(image.width * size_factor)
    new_height = int(image.height * size_factor)
    image = image.resize((new_width, new_height))
    if transpose >= 0:
        image = image.transpose(Image.Transpose(transpose))
    return image


def try_import_video_packages() -> Any:
    try:
        import cv2
        import decord
    except ImportError as exc:
        raise ImportError(
            "Please install vllm[video] for video support.") from exc
    return cv2, decord


def resize_video(frames: npt.NDArray, size: Tuple[int, int]) -> npt.NDArray:
    cv2, _ = try_import_video_packages()

    num_frames, _, _, channels = frames.shape
    new_height, new_width = size
    resized_frames = np.empty((num_frames, new_height, new_width, channels),
                              dtype=frames.dtype)
    for i, frame in enumerate(frames):
        resized_frame = cv2.resize(frame, (new_width, new_height))
        resized_frames[i] = resized_frame
    return resized_frames


def rescale_video_size(frames: npt.NDArray, size_factor: float) -> npt.NDArray:
    _, height, width, _ = frames.shape
    new_height = int(height * size_factor)
    new_width = int(width * size_factor)

    return resize_video(frames, (new_height, new_width))


def sample_frames_from_video(frames: npt.NDArray,
                             num_frames: int) -> npt.NDArray:
    total_frames = frames.shape[0]
    if num_frames == -1:
        return frames
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        sampled_frames = frames[frame_indices, ...]
        return sampled_frames


def encode_video_base64(frames: npt.NDArray):
    base64_frames = []
    frames_list = [frames[i] for i in range(frames.shape[0])]
    for frame in frames_list:
        img_base64 = encode_image_base64(Image.fromarray(frame))
        base64_frames.append(img_base64)
    return ",".join(base64_frames)


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
) -> List[_T]:
    replacement = [token] * repeat_count
    if pad_token_left is not None:
        replacement = [pad_token_left] + replacement
    if pad_token_right is not None:
        replacement = replacement + [pad_token_right]

    return replacement


def repeat_and_pad_placeholder_tokens(
    tokenizer: AnyTokenizer,
    prompt: Optional[str],
    prompt_token_ids: List[int],
    *,
    placeholder_token_id: int,
    repeat_count: Union[int, List[int]],
    pad_token_left: Optional[int] = None,
    pad_token_right: Optional[int] = None,
) -> Tuple[Optional[str], List[int], List[PlaceholderRange]]:
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

    new_token_ids: List[int] = []
    placeholder_ranges: List[PlaceholderRange] = []
    placeholder_token_idx = 0
    for i, token in enumerate(prompt_token_ids):
        if token == placeholder_token_id:
            replacement_ids = repeat_and_pad_token(
                placeholder_token_id,
                repeat_count=repeat_count[placeholder_token_idx],
                pad_token_left=pad_token_left,
                pad_token_right=pad_token_right,
            )
            placeholder_ranges.append({
                "offset": len(new_token_ids),
                "length": len(replacement_ids)
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


def consecutive_placeholder_ranges(num_items: int,
                                   item_size: int) -> List[PlaceholderRange]:
    """Returns a list of consecutive PlaceholderRanges of a fixed size"""

    return [
        PlaceholderRange(offset=i * item_size, length=item_size)
        for i in range(num_items)
    ]
