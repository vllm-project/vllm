from typing import Dict, Tuple, Type, Union

import torch
import numpy as np
from vllm.config import ModelConfig, WhisperConfig
from vllm.logger import init_logger
from vllm.sequence import SequenceData
from vllm.transformers_utils.whisper_processor import cached_get_whisper_processor

from .base import MultiModalData, MultiModalPlugin

logger = init_logger(__name__)


def _get_dummy_seq_data(seq_len: int,
                        whisper_config: WhisperConfig) -> SequenceData:
    token_ids = [0, 0, 0]

    return SequenceData(token_ids)


def _get_dummy_values(whisper_config: WhisperConfig) -> torch.Tensor:
    values_dtype = torch.float16

    return torch.zeros((30 * whisper_config), dtype=values_dtype)


def get_dummy_audio_data(
    seq_len: int,
    model_config: ModelConfig,
    whisper_config: WhisperConfig,
) -> Tuple[SequenceData, MultiModalData]:
    """Standard dummy data factory for image data (to be used in
    :meth:`vlm.multimodal.MultiModalRegistry.register_dummy_data`)."""
    seq_data = _get_dummy_seq_data(seq_len, whisper_config)
    values = _get_dummy_values(whisper_config)


    fake_mm_data: MultiModalData
    if config_input_type == ImageInputType.PIXEL_VALUES:
        fake_mm_data = ImagePixelData(values)
    elif config_input_type == ImageInputType.IMAGE_FEATURES:
        fake_mm_data = ImageFeatureData(values)
    else:
        raise NotImplementedError

    return seq_data, fake_mm_data


class AudioData(MultiModalData):
    """
    The pixel data of an image. Can be one of:

    - :class:``PIL.Image``: An image object. Requires that a HuggingFace
      processor is available to the model.
    - :class:``torch.Tensor``: The raw pixel data which is passed to the model
      without additional pre-processing.
    """

    def __init__(self, audio: Union[np.array, torch.Tensor]) -> None:

        self.audio = audio

    def __repr__(self) -> str:
        return str(self.audio)


class AudioPlugin(MultiModalPlugin[AudioData]):

    def get_data_type(self) -> Type[AudioData]:
        return AudioData

    def _get_hf_whisper_processor(self, model_config: ModelConfig,
                                whisper_config: WhisperConfig):
        if whisper_config is None or whisper_config.whisper_processor is None:
            return None

        return cached_get_whisper_processor(
            whisper_config.whisper_processor,
            revision=whisper_config.whisper_processor_revision,
        )

    def _default_input_processor(
            self, data: AudioData, model_config: ModelConfig,
            whisper_config: WhisperConfig) -> Dict[str, torch.Tensor]:
        audio = data.audio

        processor = self._get_hf_whisper_processor(model_config, whisper_config)
        if processor is None:
            raise RuntimeError("No HuggingFace processor is available"
                                "to process the audio object")
        try:
            return processor(image, return_tensors="pt").to(model_config.dtype)
        except Exception:
            logger.error("Failed to process audio (%s)", image)
            raise
