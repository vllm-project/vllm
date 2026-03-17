# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PretrainedConfig

from vllm.multimodal.processing import PromptUpdateDetails
from vllm.tokenizers import TokenizerLike

from .internvl import (
    IMG_CONTEXT,
    IMG_END,
    IMG_START,
    BaseInternVLProcessor,
)


class Eagle2_5_VLProcessor(BaseInternVLProcessor):
    """
    Custom processor for Eagle2.5-VL model.
    Extends BaseInternVLProcessor with Eagle-specific token handling.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> None:
        # Skip super().__init__() to avoid config manipulation
        # Directly initialize all required attributes
        self.config = config
        self.tokenizer = tokenizer

        # Image size with force_image_size override
        image_size: int = config.vision_config.image_size
        if hasattr(config, "force_image_size") and config.force_image_size:
            image_size = config.force_image_size

        patch_size: int = config.vision_config.patch_size
        downsample_ratio: float = getattr(config, "downsample_ratio", 0.5)

        # Compute num_image_token
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (downsample_ratio**2)
        )
        self.image_size = image_size

        # Dynamic patch settings with defaults
        self.min_dynamic_patch = (
            min_dynamic_patch
            if min_dynamic_patch is not None
            else getattr(config, "min_dynamic_patch", 1)
        )
        self.max_dynamic_patch = (
            max_dynamic_patch
            if max_dynamic_patch is not None
            else getattr(config, "max_dynamic_patch", 12)
        )
        self.dynamic_image_size = (
            dynamic_image_size
            if dynamic_image_size is not None
            else getattr(config, "dynamic_image_size", True)
        )
        self.use_thumbnail: bool = getattr(config, "use_thumbnail", True)

    @property
    def image_token_id(self) -> int:
        """Get the image token ID from config or tokenizer."""
        if hasattr(self.config, "image_token_index"):
            return self.config.image_token_index
        # Fallback to tokenizer vocab - use <IMG_CONTEXT> (ID: 151667)
        vocab = self.tokenizer.get_vocab()
        if IMG_CONTEXT in vocab:
            return vocab[IMG_CONTEXT]
        raise ValueError(f"Cannot find image token '{IMG_CONTEXT}' in vocabulary")

    def get_image_repl(
        self,
        feature_size: int,
        num_patches: int | None,
    ) -> PromptUpdateDetails[str]:
        """Get image replacement string for prompt."""
        repl_features = IMG_CONTEXT * feature_size
        repl_full = IMG_START + repl_features + IMG_END

        return PromptUpdateDetails.select_text(repl_full, IMG_CONTEXT)
