from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Optional

import cv2
import numpy as np
import numpy.typing as npt

from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.transformers_utils.processor import get_video_processor
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import is_list_of

from .base import MultiModalData
from .image import ImagePlugin
from .inputs import MultiModalKwargs, VideoItem

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

cached_get_video_processor = lru_cache(get_video_processor)
cached_get_tokenizer = lru_cache(get_tokenizer)


class VideoPlugin(ImagePlugin):
    """Plugin for video data."""

    def get_data_key(self) -> str:
        return "video"

    def _get_hf_video_processor(
        self,
        model_config: "ModelConfig",
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if mm_processor_kwargs is None:
            mm_processor_kwargs = {}
        return cached_get_video_processor(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code,
            **mm_processor_kwargs)

    def _default_input_mapper(
        self,
        ctx: InputContext,
        data: MultiModalData[VideoItem],
        **mm_processor_kwargs,
    ) -> MultiModalKwargs:
        model_config = ctx.model_config

        if isinstance(data, list) and len(data) == 1:
            data = data[0]  # type: ignore

        if isinstance(data, np.ndarray) or is_list_of(data, np.ndarray):
            video_processor = self._get_hf_video_processor(
                model_config,
                mm_processor_kwargs,
            )
            if video_processor is None:
                raise RuntimeError("No HuggingFace processor is available "
                                   "to process the video object")
            try:
                # NOTE: Similar to image; it may be a good idea to filter and
                # pass mm_processor_kwargs here too, but for now we don't to
                # avoid extra complexity if the initializer and preprocess
                # signatures of the processor don't align
                batch_data = video_processor(data, return_tensors="pt").data
            except Exception:
                logger.error("Failed to process video (%s)", data)
                raise

            return MultiModalKwargs(batch_data)

        raise TypeError(f"Invalid video type: {type(data)}")

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        return 4096


def resize_video(frames: npt.NDArray, size: tuple[int, int]) -> npt.NDArray:
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

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_frames = frames[frame_indices, ...]
    return sampled_frames
