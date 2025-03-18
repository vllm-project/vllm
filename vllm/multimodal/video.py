# SPDX-License-Identifier: Apache-2.0

import base64
import importlib.util
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import numpy.typing as npt
from PIL import Image

from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.transformers_utils.processor import cached_get_video_processor
from vllm.utils import is_list_of

from .base import MediaIO, ModalityData
from .image import ImageMediaIO, ImagePlugin
from .inputs import MultiModalKwargs, VideoItem

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)


class VideoPlugin(ImagePlugin):
    """Plugin for video data."""

    def get_data_key(self) -> str:
        return "video"

    def _get_hf_video_processor(
        self,
        model_config: "ModelConfig",
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
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
        data: ModalityData[VideoItem],
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
    # lazy import cv2 to avoid bothering users who only use text models
    import cv2
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


class VideoLoader:

    @classmethod
    def load_bytes(self, data: bytes, num_frames: int = -1) -> npt.NDArray:
        raise NotImplementedError


class OpenCVVideoBackend(VideoLoader):

    def get_cv2_video_api(self):
        import cv2.videoio_registry as vr

        api_pref = None
        for backend in vr.getStreamBufferedBackends():
            if not vr.hasBackend(backend):
                continue
            if not vr.isBackendBuiltIn(backend):
                _, abi, api = vr.getStreamBufferedBackendPluginVersion(backend)
                if (abi < 1 or (abi == 1 and api < 2)):
                    continue
            api_pref = backend
            break
        return api_pref

    @classmethod
    def load_bytes(cls, data: bytes, num_frames: int = -1) -> npt.NDArray:
        import cv2

        backend = cls().get_cv2_video_api()
        cap = cv2.VideoCapture(BytesIO(data), backend, [])
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        frames = []
        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        full_read = num_frames == -1 or total_frames_num < num_frames
        if full_read:
            frame_idx = list(range(0, total_frames_num))
        else:
            uniform_sampled_frames = np.linspace(0,
                                                 total_frames_num - 1,
                                                 num_frames,
                                                 dtype=int)
            frame_idx = uniform_sampled_frames.tolist()

        for i in frame_idx:
            if not full_read:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        return np.stack(frames)


class DecordVideoBackend(VideoLoader):

    @classmethod
    def load_bytes(cls, data: bytes, num_frames: int = -1) -> npt.NDArray:
        import decord

        vr = decord.VideoReader(BytesIO(data), num_threads=1)
        total_frame_num = len(vr)

        if num_frames == -1:
            return vr.get_batch(list(range(total_frame_num))).asnumpy()

        if total_frame_num > num_frames:
            uniform_sampled_frames = np.linspace(0,
                                                 total_frame_num - 1,
                                                 num_frames,
                                                 dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
        else:
            frame_idx = list(range(0, total_frame_num))

        return vr.get_batch(frame_idx).asnumpy()


class VideoMediaIO(MediaIO[npt.NDArray]):

    def __init__(
        self,
        image_io: ImageMediaIO,
        *,
        num_frames: int = 32,
    ) -> None:
        super().__init__()

        self.image_io = image_io
        self.num_frames = num_frames

        decord_available = importlib.util.find_spec("decord") is not None
        self.video_loader = (DecordVideoBackend
                             if decord_available else OpenCVVideoBackend)

    def load_bytes(self, data: bytes) -> npt.NDArray:
        return self.video_loader.load_bytes(data, self.num_frames)

    def load_base64(self, media_type: str, data: str) -> npt.NDArray:
        if media_type.lower() == "video/jpeg":
            load_frame = partial(
                self.image_io.load_base64,
                "image/jpeg",
            )

            return np.stack([
                np.array(load_frame(frame_data))
                for frame_data in data.split(",")
            ])

        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> npt.NDArray:
        with filepath.open("rb") as f:
            data = f.read()

        return self.load_bytes(data)

    def encode_base64(
        self,
        media: npt.NDArray,
        *,
        video_format: str = "JPEG",
    ) -> str:
        video = media

        if video_format == "JPEG":
            encode_frame = partial(
                self.image_io.encode_base64,
                image_format=video_format,
            )

            return ",".join(
                encode_frame(Image.fromarray(frame)) for frame in video)

        msg = "Only JPEG format is supported for now."
        raise NotImplementedError(msg)
