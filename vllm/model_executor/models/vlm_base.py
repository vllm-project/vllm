from torch import nn

from vllm.config import VisionLanguageConfig


class VLMBase(nn.Module):
    """Base class for all vision-language models."""

    def __init__(self, vision_language_config: VisionLanguageConfig) -> None:
        super().__init__()

        self.vision_language_config = vision_language_config
