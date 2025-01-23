from functools import lru_cache
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Optional, TypeVar, Union
from urllib.parse import ParseResult, urlparse

import numpy as np
import numpy.typing as npt
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

if TYPE_CHECKING:
    from .hasher import MultiModalHashDict
    from .inputs import MultiModalKwargs, MultiModalPlaceholderDict


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
            connection = self.connection
            data = await connection.async_get_bytes(url, timeout=fetch_timeout)

            return media_io.load_bytes(data)

        if url_spec.scheme == "data":
            return self._load_data_url(url_spec, media_io)

        if url_spec.scheme == "file":
            return self._load_file_url(url_spec, media_io)

        msg = "The URL must be either a HTTP, data or file URL."
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


def merge_and_sort_multimodal_metadata(
    mm_positions: "MultiModalPlaceholderDict",
    mm_hashes: Optional["MultiModalHashDict"],
) -> tuple[list[str], list[PlaceholderRange], Optional[list[str]]]:
    """Given a MultiModalPlaceholderDict, merge all PlaceholderRange
    objects from all available modalities into a single list of 
    PlaceholderRange, sorted by their offset (starting index in the input 
    sequence) in the ascending order.

    Optionally if a MultiModalHashDict is given, same operation will be 
    applied to the object and the sorted list of hashes will be returned.

    Raises:
        ValueError: If the input prompt has interleaved placeholders from
            different modalities (e.g, "<image><audio><image> Describe the 
            content.")
    
    Returns:
        list[str]: Sorted list of involved modalities.
        list[PlaceholderRange]: Sorted list of all PlaceholdeRanges from 
            mm_positions.
        Optional[list[str]]: Sorted list of all hashes from mm_hashes if 
            given, None otherwise.
    """

    modalities = list(mm_positions.keys())

    assert len(modalities) > 0, "No modalities found in the mm_positions."

    # For single modality, placeholder ranges and hashes are already sorted
    # so we can return the list directly.
    if len(modalities) == 1:
        if mm_hashes is None:
            return modalities, list(mm_positions[modalities[0]]), None
        else:
            return modalities, list(mm_positions[modalities[0]]), list(
                mm_hashes[modalities[0]])

    placeholder_lists_with_modality = [(modality, mm_positions[modality])
                                       for modality in modalities]

    if mm_hashes is None:
        sorted_placeholder_lists = sorted(placeholder_lists_with_modality,
                                          key=lambda x: x[1][0]['offset'])
        sorted_hash_lists = None
    else:
        hashes_lists = [
            mm_hashes[modality] for modality in modalities
            if modality in mm_hashes
        ]
        sorted_pairs = sorted(zip(placeholder_lists_with_modality,
                                  hashes_lists),
                              key=lambda x: x[0][1][0]['offset'])
        sorted_placeholder_tuple, sorted_hash_tuple = zip(*sorted_pairs)
        sorted_placeholder_lists = list(sorted_placeholder_tuple)
        sorted_hash_lists = list(sorted_hash_tuple)

    sorted_modalities = [modality for modality, _ in sorted_placeholder_lists]

    # Flatten sorted list of lists to a single list and verify there is no
    # interleaving of placeholders from different modalities.
    merged_placeholders: list[PlaceholderRange] = []
    for modality, placeholder_list in sorted_placeholder_lists:
        if merged_placeholders and placeholder_list[0][
                'offset'] < merged_placeholders[-1]['offset']:
            raise ValueError(
                "Interleaved mixed-modality inference is currently not "
                "supported.")
        merged_placeholders.extend(placeholder_list)

    if sorted_hash_lists is not None:
        merged_hashes = []
        for hash_list in sorted_hash_lists:
            merged_hashes.extend(hash_list)
    else:
        merged_hashes = None

    return sorted_modalities, merged_placeholders, merged_hashes


def group_mm_inputs_by_modality(
        mm_inputs: list["MultiModalKwargs"]) -> list[list["MultiModalKwargs"]]:
    """Group consecutive MultiModalKwargs from mm_inputs with the same modality 
    together into the same list for batching purpose. For MultiModalKwargs with 
    multiple modalities, put them into their own list.

    Args:
        mm_inputs: List of MultiModalKwargs.

    Returns:
        list[list[MultiModalKwargs]]: List of list of MultiModalKwargs, each 
        inner list contains consecutive MultiModalKwargs with same modality, or
        one with multimodal modalities.
    """
    if not mm_inputs:
        return []

    def modality_group_func(mm_input: "MultiModalKwargs") -> Union[str, int]:
        # If the input has multiple modalities, return a id as the unique key
        # for the mm_input input.
        if len(mm_input.modalities) > 1:
            return id(mm_input)

        elif len(mm_input.modalities) == 1:
            return list(mm_input.modalities)[0]

        # FIXME(Isotr0py): Modality of mm_input from legacy pipeline is empty,
        # this is used to make InternVL with legacy pipeline still work with v1.
        else:
            return ""

    return [
        list(group) for _, group in groupby(mm_inputs, key=modality_group_func)
    ]
