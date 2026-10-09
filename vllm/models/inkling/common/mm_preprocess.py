# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling multimodal preprocessing."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import regex as re
import torch
from PIL import Image
from transformers import InklingProcessor
from transformers.feature_extraction_utils import BatchFeature

from vllm.config.multimodal import (
    MultiModalDummyOptions,
)
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

from ..configs import InklingMMConfig

# Block-start markers (<|content_image|>, <|content_audio_input|>), kept verbatim
IMAGE_MARKER_ID = 200005
AUDIO_MARKER_ID = 200020
# Per-patch / per-frame placeholders (<|unused_200054|>, <|unused_200053|>)
IMAGE_TOKEN_ID = 200054
AUDIO_TOKEN_ID = 200053

# Long-edge upscale applied before patchifying (upstream defaults to no rescale)
DEFAULT_RESCALE_IMAGE_FRAC = 2.0

# Maximum audio tokens accepted per clip. At the dMel rate of 20 tokens/s
# (50 ms hop) this is ~10 minutes of audio. It bounds the persistent per-request
# buffers and the encoder/memory budget; longer clips are rejected up front.
MAX_AUDIO_TOKENS = 12_000


class InklingMultiModalDataParser(MultiModalDataParser):
    def _parse_audio_data(self, data: Any) -> Any:
        if isinstance(data, (np.ndarray, torch.Tensor)) and data.ndim == 2:
            raise ValueError(
                "Inkling raw 2-D audio has an ambiguous channel layout. "
                "Provide encoded audio or a list of mono waveforms."
            )
        return super()._parse_audio_data(data)


def _rescale_image(
    image: Image.Image, frac: float | None, max_upscaled_long_edge: int | None
) -> Image.Image:
    """Scale the long edge by `frac` with PIL LANCZOS, as the reference does."""
    image = image.convert("RGB")
    if frac is None:
        return image
    long_edge = max(image.size)
    target_long_edge = long_edge * frac
    if max_upscaled_long_edge is not None:
        target_long_edge = min(target_long_edge, max(max_upscaled_long_edge, long_edge))
    ratio = target_long_edge / long_edge
    if ratio == 1.0:
        return image
    size = tuple(max(1, math.floor(d * ratio + 0.5)) for d in image.size)
    return image.resize(size, resample=Image.Resampling.LANCZOS)


def inkling_vision_enabled(config: InklingMMConfig) -> bool:
    return getattr(config.vision_config, "decoder_dmodel", None) is not None


def inkling_audio_enabled(config: InklingMMConfig) -> bool:
    return getattr(config.audio_config, "decoder_dmodel", None) is not None


class InklingProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> InklingMMConfig:
        return self.ctx.get_hf_config(InklingMMConfig)

    def get_hf_processor(self, **kwargs: object) -> InklingProcessor:
        return self.ctx.get_hf_processor(InklingProcessor, **kwargs)

    def get_data_parser(self) -> MultiModalDataParser:
        # The feature extractor requires audio at its sampling rate
        feature_extractor = self.get_hf_processor().feature_extractor
        return InklingMultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        config = self.get_hf_config()
        limits: dict[str, int | None] = {}
        if inkling_vision_enabled(config):
            limits["image"] = None
        if inkling_audio_enabled(config):
            limits["audio"] = None
        return limits

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        # Let vLLM profile dummy inputs to determine the max token counts; the
        # image patch count is data-dependent, and the dummy audio is sized to
        # MAX_AUDIO_TOKENS so audio is profiled/budgeted at its allowed maximum.
        return None


class InklingDummyInputsBuilder(BaseDummyInputsBuilder[InklingProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # One placeholder per media item; the processor expands each into N
        # copies once feature row counts are known.
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        # Use spellings the renderer would emit; tokenization is bypassed in
        # _apply_hf_processor_main (we build input_ids directly), so the exact
        # text only needs to be a stable per-item marker.
        return ("<|content_image|>" * num_images) + (
            "<|content_audio_input|>" * num_audios
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        config = self.info.get_hf_config()
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)

        mm_data: dict[str, Any] = {}
        if num_images:
            patch_size = getattr(config.vision_config, "patch_size", 40)
            # A square image ~4 patches wide so the dummy emits several patches.
            side = patch_size * 4
            mm_data["image"] = self._get_dummy_images(
                width=side,
                height=side,
                num_images=num_images,
                overrides=mm_options.get("image"),
            )
        if num_audios:
            # Size the dummy at the maximum allowed audio so memory/encoder
            # budgeting reflects the largest clip we accept (MAX_AUDIO_TOKENS).
            feature_extractor = self.info.get_hf_processor().feature_extractor
            audio_len = MAX_AUDIO_TOKENS * feature_extractor.hop_length
            mm_data["audio"] = self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=mm_options.get("audio"),
            )
        return mm_data


class InklingMultiModalProcessor(BaseMultiModalProcessor[InklingProcessingInfo]):
    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data, hf_kwargs, passthrough_data = self._get_hf_mm_inputs(
            mm_items, hf_kwargs
        )

        prompt_text = self.dummy_inputs.get_dummy_text(mm_items.get_all_counts())

        processor = self.info.get_hf_processor(**hf_kwargs)
        tokenizer = self.info.get_tokenizer()

        images = mm_data.get("images") or []
        audios = mm_data.get("audio") or []
        if not isinstance(images, list):
            images = list(cast(Iterable[Any], images))
        if not isinstance(audios, list):
            audios = list(cast(Iterable[Any], audios))

        prompt_ids = self._tokenize_with_placeholders(
            prompt_text, tokenizer, len(images), len(audios)
        )

        data: dict[str, Any] = {"input_ids": [prompt_ids]}

        if images:
            image_processor = processor.image_processor
            frac = image_processor.rescale_image_frac or DEFAULT_RESCALE_IMAGE_FRAC
            max_long_edge = image_processor.rescale_image_max_upscaled_long_edge
            # Resize with PIL: torchvision LANCZOS does not match the reference
            images = [_rescale_image(img, frac, max_long_edge) for img in images]
            img_feat = image_processor(
                images, rescale_image_frac=None, return_tensors="pt"
            )
            data["pixel_values"] = img_feat["pixel_values"].to(torch.bfloat16)
            data["num_patches"] = img_feat["num_patches"].to(torch.int64)

        if audios:
            feature_extractor = processor.feature_extractor
            aud_feat = feature_extractor(
                audios,
                sampling_rate=feature_extractor.sampling_rate,
                return_tensors="pt",
            )
            audio_input_ids = processor._extract_dmel_bins(aud_feat["input_features"])
            num_audio_tokens = aud_feat["input_features_mask"].sum(-1)
            for i, n in enumerate(num_audio_tokens.tolist()):
                if n > MAX_AUDIO_TOKENS:
                    raise ValueError(
                        f"Audio clip {i} produces {n} tokens, exceeding the "
                        f"maximum of {MAX_AUDIO_TOKENS} (~10 min at 20 tokens/s). "
                        "Provide a shorter clip."
                    )
            data["audio_input_ids"] = torch.cat(
                [ids[:n] for ids, n in zip(audio_input_ids, num_audio_tokens)]
            )
            data["num_audio_tokens"] = num_audio_tokens.to(torch.int64)

        processed_data = BatchFeature(data=data, tensor_type=None)
        return self._finalize_hf_mm_data(
            mm_data, hf_kwargs, passthrough_data, processed_data
        )

    def _tokenize_with_placeholders(
        self,
        prompt: str,
        tokenizer: Any,
        num_images: int,
        num_audios: int,
    ) -> list[int]:
        """Tokenize `prompt`, emitting the block-start marker id per media item.

        Each marker (kept verbatim) is later expanded by ``_get_prompt_updates``
        into ``<marker> + <placeholder> * N``.
        """
        image_marker = "<|content_image|>"
        audio_marker = "<|content_audio_input|>"

        pattern = f"({re.escape(image_marker)}|{re.escape(audio_marker)})"
        chunks = re.split(pattern, prompt)

        ids: list[int] = []
        seen_img = seen_aud = 0
        for chunk in chunks:
            if chunk == image_marker:
                ids.append(IMAGE_MARKER_ID)
                seen_img += 1
            elif chunk == audio_marker:
                ids.append(AUDIO_MARKER_ID)
                seen_aud += 1
            elif chunk:
                ids.extend(tokenizer.encode(chunk, add_special_tokens=False))

        # Reconcile against the declared media counts only when media is
        # present. With no media items, emit the markers verbatim; the
        # marker<->item correspondence is enforced later by
        # ``_get_prompt_updates`` once the media features are available.
        if num_images or num_audios:
            # Fail clearly on a placeholder/media-count mismatch instead of
            # crashing with an IndexError deep in the per-item replacement logic.
            if num_images and seen_img != num_images:
                raise ValueError(
                    f"Prompt contains {seen_img} image placeholder(s), but only "
                    f"{num_images} image(s) were provided."
                )
            if num_audios and seen_aud != num_audios:
                raise ValueError(
                    f"Prompt contains {seen_aud} audio placeholder(s), but only "
                    f"{num_audios} audio input(s) were provided."
                )
        return ids

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_patches = hf_inputs.get("num_patches", torch.empty(0, dtype=torch.int64))
        num_audio_tokens = hf_inputs.get(
            "num_audio_tokens", torch.empty(0, dtype=torch.int64)
        )
        return dict(
            # Ragged per-image patches, grouped by num_patches.
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", num_patches),
            num_patches=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            # Ragged per-audio frames, grouped by num_audio_tokens.
            audio_input_ids=MultiModalFieldConfig.flat_from_sizes(
                "audio", num_audio_tokens
            ),
            num_audio_tokens=MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: Any,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        out_mm_data = out_mm_kwargs.get_data()
        num_patches: Any = out_mm_data.get("num_patches")
        num_audio_tokens: Any = out_mm_data.get("num_audio_tokens")

        # Keep the block-start marker and append N placeholder tokens after it;
        # only the placeholder positions are flagged as embeddings (is_embed), so
        # the marker stays a normal text token while the tower features scatter
        # into the placeholders.
        def image_replacement(item_idx: int) -> PromptUpdateDetails:
            n = int(num_patches[item_idx])
            return PromptUpdateDetails.select_token_id(
                [IMAGE_MARKER_ID] + [IMAGE_TOKEN_ID] * n, IMAGE_TOKEN_ID
            )

        def audio_replacement(item_idx: int) -> PromptUpdateDetails:
            n = int(num_audio_tokens[item_idx])
            return PromptUpdateDetails.select_token_id(
                [AUDIO_MARKER_ID] + [AUDIO_TOKEN_ID] * n, AUDIO_TOKEN_ID
            )

        updates: list[PromptUpdate] = []
        if num_patches is not None and len(num_patches) > 0:
            updates.append(
                PromptReplacement(
                    modality="image",
                    target=[IMAGE_MARKER_ID],
                    replacement=image_replacement,
                )
            )
        if num_audio_tokens is not None and len(num_audio_tokens) > 0:
            updates.append(
                PromptReplacement(
                    modality="audio",
                    target=[AUDIO_MARKER_ID],
                    replacement=audio_replacement,
                )
            )
        return updates
