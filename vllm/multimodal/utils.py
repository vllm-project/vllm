# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import atexit
import itertools
import math
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeVar, Union
from urllib.parse import ParseResult, urlparse
from urllib.request import url2pathname

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image, UnidentifiedImageError
from typing_extensions import deprecated

import vllm.envs as envs
from vllm.connections import HTTPConnection, global_http_connection
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather)

from .audio import AudioMediaIO
from .base import MediaIO
from .image import ImageEmbeddingMediaIO, ImageMediaIO
from .video import VideoMediaIO

_M = TypeVar("_M")

if TYPE_CHECKING:
    from .inputs import (BatchedTensorInputs, MultiModalKwargs,
                         MultiModalKwargsItem, MultiModalKwargsItems,
                         MultiModalPlaceholderDict)
else:
    BatchedTensorInputs = Any
    MultiModalKwargs = Any
    MultiModalKwargsItem = Any
    MultiModalKwargsItems = Any
    MultiModalPlaceholderDict = Any

global_thread_pool = ThreadPoolExecutor(
    max_workers=envs.VLLM_MEDIA_LOADING_THREAD_COUNT)
atexit.register(global_thread_pool.shutdown)


class MediaConnector:

    def __init__(
        self,
        media_io_kwargs: Optional[dict[str, dict[str, Any]]] = None,
        connection: HTTPConnection = global_http_connection,
        *,
        allowed_local_media_path: str = "",
    ) -> None:
        """
        Args:
            media_io_kwargs: Additional args passed to process media 
                             inputs, keyed by modalities. For example, 
                             to set num_frames for video, set 
                             `--media-io-kwargs '{"video":{"num_frames":40}}'`
            connection: HTTP connection client to download media contents.
            allowed_local_media_path: A local directory to load media files
                                      from.
        """
        super().__init__()

        self.media_io_kwargs: dict[str, dict[
            str, Any]] = media_io_kwargs if media_io_kwargs else {}
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

        filepath = Path(url2pathname(url_spec.path))
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
        loop = asyncio.get_running_loop()

        if url_spec.scheme.startswith("http"):
            connection = self.connection
            data = await connection.async_get_bytes(url, timeout=fetch_timeout)
            future = loop.run_in_executor(global_thread_pool,
                                          media_io.load_bytes, data)
            return await future

        if url_spec.scheme == "data":
            future = loop.run_in_executor(global_thread_pool,
                                          self._load_data_url, url_spec,
                                          media_io)
            return await future

        if url_spec.scheme == "file":
            future = loop.run_in_executor(global_thread_pool,
                                          self._load_file_url, url_spec,
                                          media_io)
            return await future
        msg = "The URL must be either a HTTP, data or file URL."
        raise ValueError(msg)

    def fetch_audio(
        self,
        audio_url: str,
    ) -> tuple[np.ndarray, Union[int, float]]:
        """
        Load audio from a URL.
        """
        audio_io = AudioMediaIO(**self.media_io_kwargs.get("audio", {}))

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
        audio_io = AudioMediaIO(**self.media_io_kwargs.get("audio", {}))

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
        Load a PIL image from an HTTP or base64 data URL.

        By default, the image is converted into RGB format.
        """
        image_io = ImageMediaIO(image_mode=image_mode,
                                **self.media_io_kwargs.get("image", {}))

        try:
            return self.load_from_url(
                image_url,
                image_io,
                fetch_timeout=envs.VLLM_IMAGE_FETCH_TIMEOUT,
            )
        except UnidentifiedImageError as e:
            # convert to ValueError to be properly caught upstream
            raise ValueError(str(e)) from e

    async def fetch_image_async(
        self,
        image_url: str,
        *,
        image_mode: str = "RGB",
    ) -> Image.Image:
        """
        Asynchronously load a PIL image from an HTTP or base64 data URL.

        By default, the image is converted into RGB format.
        """
        image_io = ImageMediaIO(image_mode=image_mode,
                                **self.media_io_kwargs.get("image", {}))

        try:
            return await self.load_from_url_async(
                image_url,
                image_io,
                fetch_timeout=envs.VLLM_IMAGE_FETCH_TIMEOUT,
            )
        except UnidentifiedImageError as e:
            # convert to ValueError to be properly caught upstream
            raise ValueError(str(e)) from e

    def fetch_video(
        self,
        video_url: str,
        *,
        image_mode: str = "RGB",
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """
        Load video from an HTTP or base64 data URL.
        """
        image_io = ImageMediaIO(image_mode=image_mode,
                                **self.media_io_kwargs.get("image", {}))
        video_io = VideoMediaIO(image_io,
                                **self.media_io_kwargs.get("video", {}))

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
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """
        Asynchronously load video from an HTTP or base64 data URL.

        By default, the image is converted into RGB format.
        """
        image_io = ImageMediaIO(image_mode=image_mode,
                                **self.media_io_kwargs.get("image", {}))
        video_io = VideoMediaIO(image_io,
                                **self.media_io_kwargs.get("video", {}))

        return await self.load_from_url_async(
            video_url,
            video_io,
            fetch_timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
        )

    def fetch_image_embedding(
        self,
        data: str,
    ) -> torch.Tensor:
        """
        Load image embedding from a URL.
        """
        image_embedding_io = ImageEmbeddingMediaIO()

        return image_embedding_io.load_base64("", data)


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


def argsort_mm_positions(
        mm_positions: MultiModalPlaceholderDict) -> list[tuple[str, int]]:
    """
    Given a `MultiModalPlaceholderDict`, output a sequence of keys to
    sort the dictionary by `offset` (starting index in the input sequence)
    in ascending order.

    Returns:
        A list of `(modality, idx)`, which can be used to access an item
        by `mm_positions[modality][idx]`.
    """
    flat_items = ((modality, idx, item)
                  for modality, items in mm_positions.items()
                  for idx, item in enumerate(items))

    sorted_flat_items = sorted(flat_items, key=lambda x: x[2].offset)

    return [(modality, idx) for modality, idx, _ in sorted_flat_items]


# Temporary back-compatibility for plugins that define model runner
@deprecated("`group_mm_inputs_by_modality` is superseded by "
            "`group_mm_kwargs_by_modality` and will be removed in v0.13. "
            "Please use `group_mm_kwargs_by_modality` instead.")
def group_mm_inputs_by_modality(
    mm_inputs: list[MultiModalKwargsItems]
) -> list[list[MultiModalKwargsItems]]:
    if not mm_inputs:
        return []

    def modality_group_func(
            mm_input: MultiModalKwargsItems) -> Union[str, int]:
        # If the input has multiple modalities, return an id as the unique key
        # for the mm_input input.
        if len(mm_input) > 1:
            return id(mm_input)

        elif len(mm_input) == 1:
            return next(iter(mm_input.keys()))

        raise AssertionError("This line should be unreachable.")

    return [
        list(group) for _, group in groupby(mm_inputs, key=modality_group_func)
    ]


def group_mm_kwargs_by_modality(
    mm_kwargs: list[MultiModalKwargsItem],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
) -> Iterable[tuple[str, int, BatchedTensorInputs]]:
    """Group consecutive `MultiModalKwargsItem`s from `mm_kwargs` with the same
    modality together into the same `MultiModalKwargs` instance.

    Args:
        mm_kwargs: List of `MultiModalKwargsItem`.
        device: The device to place the grouped tensors on.
        pin_memory: Whether to pin memory for faster host-to-device transfer.

    Yields:
        A tuple `(modality, num_items, grouped_kwargs)`.
    """
    from vllm.multimodal.inputs import MultiModalKwargs, MultiModalKwargsItems

    for modality, items in groupby(mm_kwargs, key=lambda item: item.modality):
        items_lst = list(items)

        # mm_kwargs_group = MultiModalKwargsItems.from_items(items_lst) \
        #    .get_data(pin_memory=pin_memory)

        # if device is not None:
        #     mm_kwargs_group = json_map_leaves(
        #         lambda x: x.to(device=device),
        #         mm_kwargs_group,
        #     )

        # TODO: Once V0 is removed, we can use the merging logic above
        # to avoid creating an extra batch dimension (except for fields
        # that are meant to be stacked anyway).
        # We will also need to update each model to remove `flatten_bn`.
        mm_kwargs_group = MultiModalKwargs.as_kwargs(
            MultiModalKwargs.batch(
                [
                    MultiModalKwargsItems.from_seq([item]).get_data()
                    for item in items_lst
                ],
                pin_memory=pin_memory,
            ),
            device=device,
        )

        yield modality, len(items_lst), mm_kwargs_group


def run_dp_sharded_vision_model(image_input: torch.Tensor,
                                vision_model: torch.nn.Module) -> torch.Tensor:
    """Run a vision model with data parallelism (DP) sharding. The function 
    will shard the input image tensor on the first dimension and run the vision
    model

    Args:
        image_input (torch.Tensor): Image input tensor.
        vision_model (torch.nn.Module): Vision model.
    Returns:
        torch.Tensor: Output image embeddings
    """

    num_chunks = image_input.shape[0]
    mp_world_size = get_tensor_model_parallel_world_size()
    num_chunks_per_rank = (num_chunks + mp_world_size - 1) // mp_world_size
    num_padded_chunks = num_chunks_per_rank * mp_world_size - num_chunks
    pad = (0, ) * (2 * (image_input.dim() - 1)) + (0, num_padded_chunks)
    image_input_padded = torch.nn.functional.pad(image_input, pad)
    rank = get_tensor_model_parallel_rank()
    image_input_per_rank = image_input_padded[rank *
                                              num_chunks_per_rank:(rank + 1) *
                                              num_chunks_per_rank, ...]

    vision_embeddings = vision_model(image_input_per_rank)
    # Ensure tensor is contiguous before all_gather
    vision_embeddings = vision_embeddings.contiguous()
    vision_embeddings = tensor_model_parallel_all_gather(vision_embeddings,
                                                         dim=0)
    vision_embeddings = vision_embeddings[:num_chunks, ...]
    return vision_embeddings


def get_load_balance_assignment(
    sizes: list[int],
    num_gpus: int = 2,
) -> tuple[list[int], list[int], list[int]]:
    """
    Generate load balancing assignment and metadata 
    for distributing data across GPUs.
    The load is determined by the total image sizes,
    not the number of images.
    
    Args:
        sizes: The size of each image
        num_gpus: Number of GPUs to balance across
    
    Returns:
        shuffle_indices: 
            Indices to reorder data for balanced loading
        gpu_sample_counts: 
            Number of samples assigned to each GPU
        grouped_sizes_per_gpu: 
            Total size assigned to each GPU
    
    Example:
        ```
        sizes = [1000, 100, 200, 50]
        num_gpus=2
        ```

    """

    n_samples = len(sizes)

    # Handle edge cases
    if n_samples == 0:
        return [], [0] * num_gpus, [0] * num_gpus

    # Use greedy algorithm - balance by total size, not sample count
    gpu_assignments = [list[int]() for _ in range(num_gpus)]
    gpu_loads = [0] * num_gpus  # This tracks total SIZE, not sample count

    # Sort indices by size (largest first for better load balancing)
    # sizes = [1000, 100, 200, 50]
    # large_to_small_indices = [0, 2, 1, 3]
    large_to_small_indices = sorted(range(n_samples),
                                    key=lambda i: sizes[i],
                                    reverse=True)

    for idx in large_to_small_indices:
        # Find GPU with minimum current load (by total size)
        min_gpu = min(range(num_gpus), key=lambda i: gpu_loads[i])
        gpu_assignments[min_gpu].append(idx)
        gpu_loads[min_gpu] += sizes[idx]

    # Create shuffle indices and counts
    shuffle_indices = list[int]()
    gpu_sample_counts = list[int]()
    for gpu_id in range(num_gpus):
        # GPU_0 = [1000] = [0]
        # GPU_1 = [200, 100, 50] = [2, 1, 3]
        # shuffle_indices = [0, 2, 1, 3]
        shuffle_indices.extend(gpu_assignments[gpu_id])
        # GPU_0 = [1]
        # GPU_1 = [3]
        # gpu_sample_counts = [1, 3]
        gpu_sample_counts.append(len(gpu_assignments[gpu_id]))

    return (shuffle_indices, gpu_sample_counts, gpu_loads)


def run_dp_sharded_mrope_vision_model(
    vision_model: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw_list: list[list[int]],
    *,
    rope_type: Literal["rope_3d", "rope_2d"],
) -> tuple[torch.Tensor, ...]:
    """Run a vision model with data parallelism (DP) sharding. 
    The function will shard the input image tensor on the 
    first dimension and run the vision model.
    This function is used to run the vision model with mrope.
    
    Args:
        vision_model (torch.nn.Module): Vision model.
        pixel_values (torch.Tensor): Image/Video input tensor.
        grid_thw_list: List of grid dimensions for each image
        rope_type: Type of rope used in the vision model.
                   Different rope types have different dimension to do ViT.
                   "rope_3d" for 3D rope (e.g., Qwen2.5-VL)
                   "rope_2d" for 2D rope (e.g., Kimi-VL)
    Returns:
        torch.Tensor: Output image embeddings

    Example:
        ```
        vision_model.out_hidden_size = 64
        vision_model.spatial_merge_size = 2
        pixel_values.shape = (1350, channel)
        grid_thw_list = [[1, 10, 100], [1, 10, 10], [1, 10, 20], [1, 50]]
        tp_size=2
        ```

    """
    tp_size = get_tensor_model_parallel_world_size()

    # GPU_0 tp_rank_local = 0
    # GPU_1 tp_rank_local = 1
    tp_rank_local = get_tensor_model_parallel_rank()

    # patches_per_image = [1000, 100, 200, 50]
    patches_per_image = [math.prod(grid_thw) for grid_thw in grid_thw_list]
    # patches_per_image = [0, 1000, 1100, 1300, 1350]
    cum_patches_per_image = [0, *itertools.accumulate(patches_per_image)]

    # Get load balancing assignment with all metadata
    # image_to_tp_rank = [0, 2, 1, 3]
    # gpu_sample_counts = [1, 3]
    # grouped_pixel_values_len = [1000, 350]
    (image_to_tp_rank, gpu_sample_counts,
     grouped_pixel_values_len) = get_load_balance_assignment(
         patches_per_image, tp_size)

    # cu_gpu_sample_counts = [0, 1, 4]
    cum_gpu_sample_counts = [0, *itertools.accumulate(gpu_sample_counts)]

    # GPU_0 image_idxs_local = [0]
    # GPU_1 image_idxs_local = [2, 1, 3]
    image_idxs_local = image_to_tp_rank[cum_gpu_sample_counts[tp_rank_local]:
                                        cum_gpu_sample_counts[tp_rank_local +
                                                              1]]

    # Get the pixel values for the local images based on the image_idxs_local
    if len(image_idxs_local) > 0:
        pixel_values_local = torch.cat([
            pixel_values[cum_patches_per_image[i]:cum_patches_per_image[i + 1]]
            for i in image_idxs_local
        ])
    else:
        # Handle case where this rank has no images
        pixel_values_local = torch.empty((0, pixel_values.shape[1]),
                                         device=pixel_values.device,
                                         dtype=pixel_values.dtype)
    # embed_dim_reduction_factor = 2 * 2
    if rope_type == "rope_2d":
        embed_dim_reduction_factor = (vision_model.merge_kernel_size[0] *
                                      vision_model.merge_kernel_size[1])
    else:
        embed_dim_reduction_factor = (vision_model.spatial_merge_size *
                                      vision_model.spatial_merge_size)

    # Find the max length across all ranks
    # The output embedding of every DP rank has to be
    # padded to this length for tensor_model_parallel_all_gather
    # to work
    max_len_per_rank = max(
        grouped_pixel_values_len) // embed_dim_reduction_factor
    local_grid_thw_list = [grid_thw_list[i] for i in image_idxs_local]

    # Run the vision model on the local pixel_values_local
    if rope_type == "rope_2d":
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(
                pixel_values_local, torch.tensor(local_grid_thw_list))
            if isinstance(image_embeds_local, list):
                image_embeds_local = torch.cat(image_embeds_local, dim=0)
        else:
            out_dim = getattr(vision_model.config, "hidden_size", None)
            image_embeds_local = torch.empty(
                (0, embed_dim_reduction_factor, out_dim),
                device=pixel_values.device,
                dtype=pixel_values.dtype)
    else:
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(pixel_values_local,
                                              local_grid_thw_list)
        else:
            # Handle empty case
            image_embeds_local = torch.empty((0, vision_model.out_hidden_size),
                                             device=pixel_values.device,
                                             dtype=pixel_values.dtype)

    # Pad the output based on max_len_per_rank
    # for tensor_model_parallel_all_gather to work
    current_len = image_embeds_local.shape[0]
    if current_len < max_len_per_rank:
        padding_size = max_len_per_rank - current_len
        if rope_type == "rope_2d":
            padding = torch.empty((padding_size, image_embeds_local.shape[1],
                                   image_embeds_local.shape[2]),
                                  dtype=image_embeds_local.dtype,
                                  device=image_embeds_local.device)
        else:
            padding = torch.empty((padding_size, image_embeds_local.shape[1]),
                                  dtype=image_embeds_local.dtype,
                                  device=image_embeds_local.device)
        image_embeds_local_padded = torch.cat([image_embeds_local, padding],
                                              dim=0)
    else:
        image_embeds_local_padded = image_embeds_local

    # Do all_gather to collect embeddings from all ranks
    gathered_embeds = tensor_model_parallel_all_gather(
        image_embeds_local_padded, dim=0)

    # Remove padding and reconstruct per-rank embeddings
    rank_embeddings = list[torch.Tensor]()
    for rank in range(tp_size):
        start_idx = rank * max_len_per_rank
        end_idx = start_idx + (grouped_pixel_values_len[rank] //
                               embed_dim_reduction_factor)
        rank_embeddings.append(gathered_embeds[start_idx:end_idx])

    patches_per_output_image = [(patch_size // embed_dim_reduction_factor)
                                for patch_size in patches_per_image]

    # Reconstruct embeddings in the original order
    original_order_embeddings = [None] * len(grid_thw_list)
    current_idx = 0
    for rank in range(tp_size):
        count = gpu_sample_counts[rank]
        if count > 0:
            # Get images assigned to this rank in shuffled order
            # GPU_0 = image_idxs_local  [0]
            # GPU_1 = image_idxs_local  [2, 1, 3]
            rank_images = image_to_tp_rank[current_idx:current_idx + count]

            rank_embed = rank_embeddings[rank]
            # Split rank embeddings back to individual images
            embed_start = 0
            for img_idx in rank_images:
                img_patches = patches_per_output_image[img_idx]
                original_order_embeddings[img_idx] = rank_embed[
                    embed_start:embed_start + img_patches]
                embed_start += img_patches
            current_idx += count
    out_embeddings = tuple(embed for embed in original_order_embeddings
                           if embed is not None)
    assert len(out_embeddings) == len(
        original_order_embeddings), "Found unassigned embeddings"
    return out_embeddings


def fetch_audio(
    audio_url: str,
    audio_io_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, Union[int, float]]:
    """
    Args:
        audio_url: URL of the audio file to fetch.
        audio_io_kwargs: Additional kwargs passed to handle audio IO.
    """
    media_io_kwargs = None if not audio_io_kwargs else {
        "audio": audio_io_kwargs
    }
    media_connector = MediaConnector(media_io_kwargs=media_io_kwargs)
    return media_connector.fetch_audio(audio_url)


def fetch_image(
    image_url: str,
    image_io_kwargs: Optional[dict[str, Any]] = None,
) -> Image.Image:
    """
    Args:
        image_url: URL of the image file to fetch.
        image_io_kwargs: Additional kwargs passed to handle image IO.
    """
    media_io_kwargs = None if not image_io_kwargs else {
        "image": image_io_kwargs
    }
    media_connector = MediaConnector(media_io_kwargs=media_io_kwargs)
    return media_connector.fetch_image(image_url)


def fetch_video(
    video_url: str,
    video_io_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[npt.NDArray, dict[str, Any]]:
    """
    Args:
        video_url: URL of the video file to fetch.
        video_io_kwargs: Additional kwargs passed to handle video IO.
    """
    media_io_kwargs = None if not video_io_kwargs else {
        "video": video_io_kwargs
    }
    media_connector = MediaConnector(media_io_kwargs=media_io_kwargs)
    return media_connector.fetch_video(video_url)
