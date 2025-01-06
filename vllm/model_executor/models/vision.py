from abc import ABC, abstractmethod
from typing import Final, Generic, Optional, Protocol, TypeVar

from transformers import PretrainedConfig

from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        InputProcessingContext,
                                        ProcessingCache)

_C = TypeVar("_C", bound=PretrainedConfig)


class VisionEncoderInfo(ABC, Generic[_C]):

    def __init__(self, vision_config: _C) -> None:
        super().__init__()

        self.vision_config = vision_config

    @abstractmethod
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_max_image_tokens(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_image_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_patch_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_patch_grid_length(self) -> int:
        raise NotImplementedError


def vision_encoder_info(vision_config: PretrainedConfig) -> VisionEncoderInfo:
    # Avoid circular imports
    from .clip import CLIPEncoderInfo, CLIPVisionConfig
    from .pixtral import PixtralHFEncoderInfo, PixtralVisionConfig
    from .siglip import SiglipEncoderInfo, SiglipVisionConfig

    if isinstance(vision_config, CLIPVisionConfig):
        return CLIPEncoderInfo(vision_config)
    if isinstance(vision_config, PixtralVisionConfig):
        return PixtralHFEncoderInfo(vision_config)
    if isinstance(vision_config, SiglipVisionConfig):
        return SiglipEncoderInfo(vision_config)

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


class VisionLanguageConfig(Protocol):
    vision_config: Final[PretrainedConfig]


class BaseVisionLanguageMultiModalProcessor(BaseMultiModalProcessor):

    def __init__(self,
                 ctx: InputProcessingContext,
                 *,
                 cache: Optional[ProcessingCache] = None,
                 enable_sanity_checks: bool = True) -> None:
        super().__init__(ctx,
                         cache=cache,
                         enable_sanity_checks=enable_sanity_checks)

        vision_config = self._get_hf_config().vision_config
        self._vision_encoder_info = vision_encoder_info(vision_config)

    @abstractmethod
    def _get_hf_config(self) -> VisionLanguageConfig:
        raise NotImplementedError
