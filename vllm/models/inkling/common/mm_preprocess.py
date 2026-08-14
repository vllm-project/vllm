# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling multimodal preprocessing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import regex as re
import torch
from transformers.feature_extraction_utils import BatchFeature

from vllm.config.multimodal import (
    AudioDummyOptions,
    BaseDummyOptions,
    ImageDummyOptions,
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
from vllm.transformers_utils.processors.inkling import (
    AUDIO_MARKER_ID,
    AUDIO_TOKEN_ID,
    IMAGE_MARKER_ID,
    IMAGE_TOKEN_ID,
    InklingAudioFeatureExtractor,
    InklingImageProcessor,
    InklingProcessor,
)

from ..configs import InklingMMConfig

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


def inkling_vision_enabled(config: InklingMMConfig) -> bool:
    return getattr(config.vision_config, "decoder_dmodel", None) is not None


def inkling_audio_enabled(config: InklingMMConfig) -> bool:
    return getattr(config.audio_config, "decoder_dmodel", None) is not None


class InklingProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> InklingMMConfig:
        return self.ctx.get_hf_config(InklingMMConfig)

    def get_hf_processor(self, **kwargs: object) -> InklingProcessor:
        config = self.get_hf_config()
        vision_config = config.vision_config
        audio_config = config.audio_config

        image_processor = InklingImageProcessor(
            patch_size=getattr(vision_config, "patch_size", None) or 40,
        )

        if inkling_audio_enabled(config):
            audio_params = {
                "n_mels": audio_config.n_mel_bins,
                "num_dmel_bins": audio_config.mel_vocab_size,
                "dmel_min_value": audio_config.dmel_min_value,
                "dmel_max_value": audio_config.dmel_max_value,
            }
        else:
            audio_params = {}
        audio_extractor = InklingAudioFeatureExtractor(params=audio_params)

        return InklingProcessor(
            image_processor=image_processor,
            audio_feature_extractor=audio_extractor,
            tokenizer=self.get_tokenizer(),
        )

    def get_data_parser(self) -> MultiModalDataParser:
        # Audio inputs must be resampled to the dMel feature extractor's rate
        # before `process_audios` (see InklingAudioFeatureExtractor._decode_one).
        # Without a target_sr the default parser raises on any audio input.
        extractor = self.get_hf_processor().audio_feature_extractor
        return InklingMultiModalDataParser(
            target_sr=extractor.params.sample_rate,
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
        # _call_hf_processor (we build input_ids directly), so the exact text
        # only needs to be a stable per-item marker.
        return ("<|content_image|>" * num_images) + (
            "<|content_audio_input|>" * num_audios
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        config = self.info.get_hf_config()
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)

        mm_data: dict[str, Any] = {}
        if num_images:
            patch_size = getattr(config.vision_config, "patch_size", 40)
            # A square image ~4 patches wide so the dummy emits several patches.
            side = patch_size * 4
            image_overrides = mm_options.get("image")
            mm_data["image"] = self._get_dummy_images(
                width=side,
                height=side,
                num_images=num_images,
                overrides=cast(ImageDummyOptions | None, image_overrides),
            )
        if num_audios:
            # Size the dummy at the maximum allowed audio so memory/encoder
            # budgeting reflects the largest clip we accept (MAX_AUDIO_TOKENS).
            params = self.info.get_hf_processor().audio_feature_extractor.params
            hop = int(round(params.audio_token_duration_s * params.sample_rate))
            audio_len = MAX_AUDIO_TOKENS * hop
            audio_overrides = mm_options.get("audio")
            mm_data["audio"] = self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=cast(AudioDummyOptions | None, audio_overrides),
            )
        return mm_data


class InklingMultiModalProcessor(BaseMultiModalProcessor[InklingProcessingInfo]):
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Inkling is not a standard HF processor (no fused text+mm call), so we run
        # the vendored extractors ourselves and tokenize the text separately.
        # The MM placeholders in `prompt` are expanded later by the prompt
        # updates, so here we emit ONE placeholder id per media item.
        processor = self.info.get_hf_processor(**mm_kwargs)
        tokenizer = self.info.get_tokenizer()

        images = mm_data.get("images") or []
        audios = mm_data.get("audios") or []
        if not isinstance(images, list):
            images = list(cast(Iterable[Any], images))
        if not isinstance(audios, list):
            audios = list(cast(Iterable[Any], audios))

        prompt_ids = self._tokenize_with_placeholders(
            prompt, tokenizer, len(images), len(audios)
        )

        data: dict[str, Any] = {"input_ids": [prompt_ids]}

        if images:
            img_feat = processor.process_images(images)
            data["pixel_values"] = img_feat["vision_patches_bthwc"]
            data["num_patches"] = torch.tensor(
                img_feat["num_patches"], dtype=torch.int64
            )

        if audios:
            aud_feat = processor.process_audios(audios)
            per_clip = aud_feat["dmel_bins"]
            num_audio_tokens = aud_feat["num_audio_tokens"]
            for i, n in enumerate(num_audio_tokens):
                if int(n) > MAX_AUDIO_TOKENS:
                    raise ValueError(
                        f"Audio clip {i} produces {int(n)} tokens, exceeding the "
                        f"maximum of {MAX_AUDIO_TOKENS} (~10 min at 20 tokens/s). "
                        "Provide a shorter clip."
                    )
            if per_clip:
                input_audio_features = torch.cat(
                    [torch.as_tensor(c) for c in per_clip], dim=0
                )
            else:
                input_audio_features = torch.empty(0)
            data["input_audio_features"] = input_audio_features
            data["num_audio_tokens"] = torch.tensor(num_audio_tokens, dtype=torch.int64)

        return BatchFeature(data=data, tensor_type=None)

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
        # present. With no media items -- e.g. the base text-only tokenization
        # probe (``_apply_hf_processor_text_only``), which calls this via
        # ``_call_hf_processor`` with empty ``mm_data`` -- emit the markers
        # verbatim; the marker<->item correspondence is enforced later by
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
            num_patches=MultiModalFieldConfig.batched("image"),
            # Ragged per-audio frames, grouped by num_audio_tokens.
            input_audio_features=MultiModalFieldConfig.flat_from_sizes(
                "audio", num_audio_tokens
            ),
            num_audio_tokens=MultiModalFieldConfig.batched("audio"),
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
