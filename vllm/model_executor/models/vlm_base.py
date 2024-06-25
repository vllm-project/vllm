from torch import nn

from vllm.config import VisionLanguageConfig


class VisionLanguageModelBase(nn.Module):
    """Base class for all vision language models (VLMs)."""

    def __init__(self, vision_language_config: VisionLanguageConfig) -> None:
        super().__init__()

        self.vision_language_config = vision_language_config
