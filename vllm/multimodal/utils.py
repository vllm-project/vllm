# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mimetypes
import warnings
from collections import defaultdict
from collections.abc import Generator, Sequence
from itertools import groupby
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from vllm.utils.import_utils import LazyLoader

from .hasher import MultiModalHasher
from .inputs import (
    BatchedTensorInputs,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalPlaceholderDict,
    MultiModalSharedField,
)
from .media import AudioMediaIO, ImageMediaIO, MediaConnector, VideoMediaIO

if TYPE_CHECKING:
    import torch.types
else:
    torch = LazyLoader("torch", globals(), "torch")


def __getattr__(name: str):
    if name == "MEDIA_CONNECTOR_REGISTRY":
        from .media import MEDIA_CONNECTOR_REGISTRY

        warnings.warn(
            "`vllm.multimodal.utils.MEDIA_CONNECTOR_REGISTRY` "
            "has been moved to `vllm.multimodal.media.MEDIA_CONNECTOR_REGISTRY`. "
            "The old name will be removed in v0.17.",
            DeprecationWarning,
            stacklevel=2,
        )

        return MEDIA_CONNECTOR_REGISTRY

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def encode_audio_base64(
    audio: np.ndarray,
    sampling_rate: int,
    *,
    format: str = "WAV",
) -> str:
    """Encode audio as base64."""
    audio_io = AudioMediaIO()
    return audio_io.encode_base64((audio, sampling_rate), audio_format=format)


def encode_audio_url(
    audio: np.ndarray,
    sampling_rate: int,
    *,
    format: str = "WAV",
) -> str:
    """Encode audio as a data URL."""
    audio_b64 = encode_audio_base64(audio, sampling_rate, format=format)
    mimetype = mimetypes.types_map.get("." + format.lower(), "audio")
    return f"data:{mimetype};base64,{audio_b64}"


def encode_image_base64(
    image: Image.Image,
    *,
    image_mode: str = "RGB",
    format: str = "PNG",
) -> str:
    """
    Encode a pillow image to base64 format.

    By default, the image is converted into RGB format before being encoded.
    """
    image_io = ImageMediaIO(image_mode=image_mode)
    return image_io.encode_base64(image, image_format=format)


def encode_image_url(
    image: Image.Image,
    *,
    image_mode: str = "RGB",
    format: str = "PNG",
) -> str:
    """
    Encode a pillow image as a data URL.

    By default, the image is converted into RGB format before being encoded.
    """
    image_b64 = encode_image_base64(image, image_mode=image_mode, format=format)
    mimetype = mimetypes.types_map.get("." + format.lower(), "image")
    return f"data:{mimetype};base64,{image_b64}"


def encode_video_base64(
    frames: npt.NDArray,
    *,
    format: str = "JPEG",
) -> str:
    image_io = ImageMediaIO()
    video_io = VideoMediaIO(image_io)
    return video_io.encode_base64(frames, video_format=format)


def encode_video_url(
    frames: npt.NDArray,
    *,
    format: str = "JPEG",
) -> str:
    video_b64 = encode_video_base64(frames, format=format)

    if format.lower() == "jpeg":
        mimetype = "video/jpeg"
    else:
        mimetype = mimetypes.types_map.get("." + format.lower(), "video")

    return f"data:{mimetype};base64,{video_b64}"


def allocate_gpu_mm_processors(
    mm_processor_device: str,
    mm_processor_count: int,
    *,
    available_device_count: int,
    engine_device_count: int,
) -> list[str]:
    """
    Allocate each processor to a GPU that is not being used by EngineCore,
    if possible.

    Returns:
        The device to allocate for each multi-modal processor.
    """
    device_type, *rest = mm_processor_device.rsplit(":", 1)
    if len(rest) == 0:
        # Try to run each processor on a different GPU, preferably those
        # that are not used by vLLM engine
        if available_device_count > engine_device_count:
            remaining_count = available_device_count - engine_device_count
            processor_gpu_idxs = [
                engine_device_count + server_idx % remaining_count
                for server_idx in range(mm_processor_count)
            ]
        else:
            processor_gpu_idxs = [
                server_idx % available_device_count
                for server_idx in range(mm_processor_count)
            ]
    else:
        # Already targeted a specific GPU
        (device_idx,) = map(int, rest)
        processor_gpu_idxs = [device_idx] * mm_processor_count

    return [f"{device_type}:{gpu_idx}" for gpu_idx in processor_gpu_idxs]


def argsort_mm_positions(
    mm_positions: MultiModalPlaceholderDict,
) -> list[tuple[str, int]]:
    """
    Given a `MultiModalPlaceholderDict`, output a sequence of keys to
    sort the dictionary by `offset` (starting index in the input sequence)
    in ascending order.

    Returns:
        A list of `(modality, idx)`, which can be used to access an item
        by `mm_positions[modality][idx]`.
    """
    flat_items = (
        (modality, idx, item)
        for modality, items in mm_positions.items()
        for idx, item in enumerate(items)
    )

    sorted_flat_items = sorted(flat_items, key=lambda x: x[2].offset)

    return [(modality, idx) for modality, idx, _ in sorted_flat_items]


def _get_group_hash(elem: MultiModalFieldElem):
    if not isinstance(elem.field, MultiModalSharedField):
        return None

    return MultiModalHasher.hash_kwargs(data=elem.data)


def _batch_mm_items(
    items: Sequence[MultiModalKwargsItem],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
):
    elems = defaultdict[str, list[MultiModalFieldElem]](list)
    for item in items:
        for key, elem in item.items():
            elems[key].append(elem)

    return {
        key: elems[0].field.reduce_data(
            elems,
            device=device,
            pin_memory=pin_memory,
        )
        for key, elems in elems.items()
    }


def group_and_batch_mm_items(
    items: Sequence[MultiModalKwargsItem],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
) -> Generator[tuple[int, BatchedTensorInputs]]:
    """
    Group consecutive items (possibly from different requests) into batches.

    Items must be split across groups if any of the following occurs,
    as the batch would otherwise be invalid:
    - They have different fields (e.g. mixed image and embedding inputs).
    - They have different values in `MultiModalSharedField`.

    Args:
        items: List of `MultiModalKwargsItem`.
        device: The device to place the grouped tensors on.
        pin_memory: Whether to pin memory for faster host-to-device transfer.

    Yields:
        A tuple `(num_items, grouped_kwargs)`, where:
        - `kwargs` is a dictionary of keyword arguments to pass to the model;
        - `num_items` is the corresponding number of items.
    """
    group_ids = [
        tuple(
            (key, _get_group_hash(elem))
            for key, elem in sorted(item.items(), key=lambda kv: kv[0])
        )
        for item in items
    ]
    group_sizes = [sum(1 for _ in group) for _, group in groupby(group_ids)]

    start_idx = 0
    for group_size in group_sizes:
        group_data = _batch_mm_items(
            items[start_idx : start_idx + group_size],
            device=device,
            pin_memory=pin_memory,
        )

        yield group_size, group_data

        start_idx += group_size

    assert start_idx == len(items)


def group_mm_kwargs_by_modality(
    mm_kwargs: list[tuple[str, MultiModalKwargsItem]],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
) -> Generator[tuple[str, int, BatchedTensorInputs], None, None]:
    """
    Group consecutive items (possibly from different requests) into batches.

    Items must be split across groups if any of the following occurs,
    as the batch would otherwise be invalid:
    - They have different fields (e.g. mixed image and embedding inputs).
    - They have different values in `MultiModalSharedField`.

    To simplify the implementation of `embed_multimodal`, we add another
    restriction that the items in a batch must belong to the same modality.

    Args:
        mm_kwargs: List of `(modality, item)`.
        device: The device to place the grouped tensors on.
        pin_memory: Whether to pin memory for faster host-to-device transfer.

    Yields:
        A tuple `(modality, num_items, grouped_kwargs)`, where:
        - `modality` is the modality of the batch;
        - `kwargs` is a dictionary of keyword arguments to pass to the model;
        - `num_items` is the corresponding number of items.
    """
    for modality, group in groupby(mm_kwargs, key=lambda x: x[0]):
        items_lst = [item for _, item in group]

        for num_items, mm_kwargs_batch in group_and_batch_mm_items(
            items_lst,
            device=device,
            pin_memory=pin_memory,
        ):
            yield modality, num_items, mm_kwargs_batch


def fetch_audio(
    audio_url: str,
    audio_io_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, int | float]:
    """
    Args:
        audio_url: URL of the audio file to fetch.
        audio_io_kwargs: Additional kwargs passed to handle audio IO.

    Warning:
        This method has direct access to local files and is only intended
        to be called by user code. Never call this from the online server!
    """
    media_io_kwargs = None if not audio_io_kwargs else {"audio": audio_io_kwargs}
    media_connector = MediaConnector(
        media_io_kwargs=media_io_kwargs,
        allowed_local_media_path="/",
    )
    return media_connector.fetch_audio(audio_url)


def fetch_image(
    image_url: str,
    image_io_kwargs: dict[str, Any] | None = None,
) -> Image.Image:
    """
    Args:
        image_url: URL of the image file to fetch.
        image_io_kwargs: Additional kwargs passed to handle image IO.

    Warning:
        This method has direct access to local files and is only intended
        to be called by user code. Never call this from the online server!
    """
    media_io_kwargs = None if not image_io_kwargs else {"image": image_io_kwargs}
    media_connector = MediaConnector(
        media_io_kwargs=media_io_kwargs,
        allowed_local_media_path="/",
    )
    return media_connector.fetch_image(image_url)


def fetch_video(
    video_url: str,
    video_io_kwargs: dict[str, Any] | None = None,
) -> tuple[npt.NDArray, dict[str, Any]]:
    """
    Args:
        video_url: URL of the video file to fetch.
        video_io_kwargs: Additional kwargs passed to handle video IO.

    Warning:
        This method has direct access to local files and is only intended
        to be called by user code. Never call this from the online server!
    """
    media_io_kwargs = None if not video_io_kwargs else {"video": video_io_kwargs}
    media_connector = MediaConnector(
        media_io_kwargs=media_io_kwargs,
        allowed_local_media_path="/",
    )
    return media_connector.fetch_video(video_url)
