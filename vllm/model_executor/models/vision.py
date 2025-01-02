from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from transformers import PretrainedConfig

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
    def get_num_patches(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_image_size(self) -> int:
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
