# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Kimi-K3 multimodal preprocessing."""

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions, ImageDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageProcessorItems, ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.transformers_utils.processors.kimi_k3 import KimiK3Processor

logger = init_logger(__name__)


def navit_resize_image(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None,
):
    # Apply the patch limits.
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    new_w = min(new_w, patch_limit_on_one_side * patch_size)
    new_h = min(new_h, patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size

    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        # Calculate new dimensions after padding and patching
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor

        assert token_height * merge_kernel_size <= patch_limit_on_one_side, (
            f"token_height {token_height} * merge_kernel_size {merge_kernel_size} > "
            f"patch_limit_on_one_side {patch_limit_on_one_side}"
        )
        assert token_width * merge_kernel_size <= patch_limit_on_one_side, (
            f"token_width {token_width} * merge_kernel_size {merge_kernel_size} > "
            f"patch_limit_on_one_side {patch_limit_on_one_side}"
        )

        num_tokens = token_height * token_width
    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "sampled_nframes": 1,
    }


class KimiK3ProcessingInfo(BaseProcessingInfo):
    """Processing information for the image-only Kimi-K3 model.

    K3 uses the standard ``image`` modality (unlike K2.5's unified
    ``vision_chunk``), so it builds its own ``KimiK3Processor`` wrapper around
    the checkpoint's image processor and resolves the ``<|media_pad|>`` token
    id the same way K2.5 does.
    """

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)

        self.hf_config = hf_config = self.get_hf_config()

        tokenizer = self.get_tokenizer()
        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
        )

        # Resolve token ID from the tokenizer because transformers v5
        # may remap token IDs vs config.json.
        config_token_id = hf_config.media_placeholder_token_id
        resolved_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        is_valid_resolved = isinstance(resolved_token_id, int) and (
            unk_token_id is None or resolved_token_id != unk_token_id
        )
        if is_valid_resolved and resolved_token_id != config_token_id:
            logger.warning_once(
                "Kimi-K3 config.media_placeholder_token_id (%d) disagrees "
                "with tokenizer mapping for <|media_pad|> (%d). "
                "Using tokenizer value.",
                config_token_id,
                resolved_token_id,
            )
            media_token_id = resolved_token_id
            # Patch config so downstream code also sees the correct ID.
            hf_config.media_placeholder_token_id = resolved_token_id
        else:
            media_token_id = config_token_id

        self.media_token_id = media_token_id
        self.media_token = tokenizer.decode(media_token_id)

        self.image_processor = image_processor
        self.hf_processor = KimiK3Processor(
            tokenizer=tokenizer,
            image_processor=image_processor,
        )
        self.media_tokens_calculator = image_processor.media_tokens_calculator

    def get_hf_processor(self, **kwargs: object) -> KimiK3Processor:
        return self.hf_processor

    def get_hf_config(self) -> KimiK3Config:
        return self.ctx.get_hf_config(KimiK3Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # None means unlimited
        return {"image": None}

    @classmethod
    def get_max_image_size(
        cls,
        patch_size: int,
        merge_kernel_size: int,
        in_patch_limit: int,
        patch_limit_on_one_side: int,
        fixed_output_tokens: int | None,
    ) -> ImageSize:
        max_side = patch_limit_on_one_side * patch_size
        best_score = (-1, -1)
        best_size = (max_side, max_side)

        for width_patches in range(patch_limit_on_one_side + 1):
            width = min((width_patches + 1) * patch_size - 1, max_side)
            for height_patches in range(width_patches, patch_limit_on_one_side + 1):
                height = min((height_patches + 1) * patch_size - 1, max_side)
                resize_config = navit_resize_image(
                    width,
                    height,
                    patch_size,
                    merge_kernel_size,
                    in_patch_limit,
                    patch_limit_on_one_side,
                    fixed_output_tokens,
                )
                padded_width = resize_config["new_width"] + resize_config["pad_width"]
                padded_height = (
                    resize_config["new_height"] + resize_config["pad_height"]
                )
                num_patches = padded_width // patch_size * (padded_height // patch_size)
                score = (resize_config["num_tokens"], num_patches)
                if score > best_score:
                    best_score = score
                    best_size = (width, height)
        return ImageSize(width=best_size[0], height=best_size[1])


class KimiK3DummyInputsBuilder(BaseDummyInputsBuilder[KimiK3ProcessingInfo]):
    """Builds image-based dummy inputs for K3 profiling.

    The dummy text is made of ``<|kimi_image_placeholder|>`` tokens — exactly
    the placeholder that K3's ``_get_prompt_updates`` expands — and the dummy
    mm data is a plain list of PIL images under the ``image`` key.
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return self.info.get_hf_config().image_placeholder * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        media_proc_cfg = self.info.image_processor.media_proc_cfg
        max_size = self.info.get_max_image_size(
            media_proc_cfg["patch_size"],
            media_proc_cfg["merge_kernel_size"],
            media_proc_cfg["in_patch_limit"],
            media_proc_cfg["patch_limit_on_one_side"],
            media_proc_cfg["fixed_output_tokens"],
        )
        num_images = mm_counts.get("image", 0)
        image_overrides = cast(
            ImageDummyOptions | None,
            mm_options.get("image") if mm_options else None,
        )
        return {
            "image": self._get_dummy_images(
                width=max_size.width,
                height=max_size.height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class KimiK3MultiModalProcessor(BaseMultiModalProcessor[KimiK3ProcessingInfo]):
    """Image-only multi-modal processor for Kimi-K3."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Override so the base always routes through the text+mm path
        # (`KimiK3Processor.__call__`). Otherwise the mm-only fast path calls
        # the checkpoint image processor directly with bare PIL images, but it
        # requires `{"type": "image", "image": PIL}` media dicts that only our
        # wrapper builds.
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Slice the flattened patch tensor back into per-image items.

        ``pixel_values`` holds all patches from every image concatenated; each
        image's patch count is ``prod(grid_thws[i])``. ``grid_thws`` is one
        ``[N_t, N_h, N_w]`` row per image.
        """
        grid_thws = hf_inputs.get("grid_thws", torch.empty((0, 3)))
        grid_sizes = grid_thws.prod(-1)

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", grid_sizes),
            grid_thws=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Expand each K3 image placeholder into a resolution-aware update.

        K3's prompt carries a single ``<|kimi_image_placeholder|>`` token per
        image. This replaces that token with
        ``<|media_begin|>image {w}x{h}<|media_content|>{pads}<|media_end|>``,
        embedding the per-image resolution in the prompt and marking only the
        ``<|media_pad|>`` positions as embedding slots (the number of pads is
        the feature size returned by ``media_tokens_calculator``).
        """
        media_token_id = self.info.media_token_id
        media_token = self.info.media_token
        image_placeholder = self.info.get_hf_config().image_placeholder

        def get_replacement(item_idx: int) -> PromptUpdateDetails[str]:
            images = mm_items.get_items("image", ImageProcessorItems)
            image = images.get(item_idx)
            if image is None:
                raise ValueError(f"Missing image data at index {item_idx}")

            # The checkpoint image processor works on media dicts, so wrap the
            # PIL image before asking it for the token count.
            num_media_token = self.info.media_tokens_calculator(
                {"type": "image", "image": image}
            )
            pads = media_token * num_media_token

            # NOTE: `width`/`height` are the ORIGINAL upload dimensions, not the
            # post-preprocess (smart-resized) ones. `image` comes from the
            # untouched parsed `mm_items`; the checkpoint image processor
            # (`KimiK3VisionProcessor.preprocess`) only produces new tensors via
            # `image.resize(...)` and never mutates the stored PIL. This matches
            # the reference HF processor (`KimiK3Processor.preprocess_medias`),
            # which also builds the prompt from the original `img.size`. The
            # resize is reflected only in the pad count above.
            width, height = images.get_image_size(item_idx)
            full = (
                f"<|media_begin|>image {width}x{height}<|media_content|>"
                f"{pads}<|media_end|>"
            )

            return PromptUpdateDetails.select_token_id(full, media_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=image_placeholder,
                replacement=get_replacement,
            ),
        ]
