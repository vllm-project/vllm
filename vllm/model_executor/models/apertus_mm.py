# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2025 The Swiss AI Initiative.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate the architectural differences made by
# the Swiss AI Initiative that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multimodal Apertus model compatible with HuggingFace weights."""

from collections.abc import Iterable, Mapping, Sequence
import os

import torch
from torch import nn
from transformers import ApertusConfig

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict, MultiModalInput, mm_input
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    ImageProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdate,
    TimingContext,
)
from vllm.logger import init_logger

from .apertus import ApertusForCausalLM
from .apertus_utils import ApertusAudioTokenizer, ApertusImageTokenizer
from .interfaces import MultiModalEmbeddings, SupportsMultiModal

logger = init_logger(__name__)


class ApertusProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> ApertusConfig:
        return self.ctx.get_hf_config(ApertusConfig)

    def get_data_parser(self) -> MultiModalDataParser:
        # Apertus audio tokenizer expects mono waveform at 24kHz.
        return MultiModalDataParser(
            target_sr=ApertusAudioTokenizer.DEFAULT_TARGET_SAMPLING_RATE,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        del mm_counts
        # Avoid huge dummy-input estimation by using Apertus' known
        # tokenizer ceilings.
        ds = ApertusImageTokenizer.EMU35_DS_FACTOR
        max_px = ApertusImageTokenizer.DEFAULT_MAX_PIXELS
        base_image_tokens = (max_px // (ds * ds)) + 512

        # WavTokenizer40 emits ~40 codes/sec. Apertus wraps them with
        # <|audio_start|> and <|audio_end|>.
        audio_tokens_per_second = 40
        max_audio_seconds = 300
        base_audio_tokens = (
            audio_tokens_per_second * max_audio_seconds
        ) + 4  # bos, boa, <|audio_start|> and <|audio_end|>

        max_tokens = {
            "image": min(base_image_tokens, seq_len),
            "audio": min(base_audio_tokens, seq_len),
        }

        return max_tokens


class ApertusDummyInputsBuilder(BaseDummyInputsBuilder[ApertusProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        return (
            ApertusImageTokenizer.DEFAULT_IMAGE_PLACEHOLDER * num_images +
            ApertusAudioTokenizer.DEFAULT_AUDIO_PLACEHOLDER * num_audios)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        image_overrides = mm_options.get("image")
        audio_overrides = mm_options.get("audio")
        max_side = int(ApertusImageTokenizer.DEFAULT_MAX_PIXELS**0.5)

        audio_tokens_per_second = 40
        max_audio_seconds = 300
        audio_token_budget = (audio_tokens_per_second * max_audio_seconds) + 4
        audio_seconds = max(1,
                            (audio_token_budget - 4) // audio_tokens_per_second)
        audio_length = (
            audio_seconds * ApertusAudioTokenizer.DEFAULT_TARGET_SAMPLING_RATE)

        return {
            "image":
            self._get_dummy_images(
                width=max_side,
                height=max_side,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "audio":
            self._get_dummy_audios(
                length=audio_length,
                num_audios=num_audios,
                overrides=audio_overrides,
            ),
        }


class ApertusMultiModalProcessor(
        BaseMultiModalProcessor[ApertusProcessingInfo]):

    def __init__(
        self,
        info: ApertusProcessingInfo,
        dummy_inputs: BaseDummyInputsBuilder[ApertusProcessingInfo],
        *,
        cache: object | None = None,
    ) -> None:
        super().__init__(info, dummy_inputs, cache=cache)
        self.image_tokenizer = ApertusImageTokenizer()
        self.audio_tokenizer = ApertusAudioTokenizer()

    def _get_mm_fields_config(
        self,
        hf_inputs: object,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        return []

    @staticmethod
    def _find_placeholders(prompt: str, aliases: Sequence[str]) -> list[str]:
        placeholders: list[tuple[int, str]] = []
        for alias in aliases:
            start = 0
            while True:
                idx = prompt.find(alias, start)
                if idx < 0:
                    break
                placeholders.append((idx, alias))
                start = idx + len(alias)

        placeholders.sort(key=lambda item: item[0])
        return [placeholder for _, placeholder in placeholders]

    def _validate_supported_inputs(self, mm_items: MultiModalDataItems) -> None:
        supported_modalities = {"image", "audio"}
        unsupported = [
            modality for modality in mm_items
            if modality not in supported_modalities
        ]
        if unsupported:
            raise ValueError(
                "Apertus multimodal preprocessing currently supports only "
                f"{sorted(supported_modalities)} inputs. "
                f"Unsupported modalities: {unsupported}")

    def _tokenize_text(
        self,
        text: str,
        tokenization_kwargs: Mapping[str, object],
    ) -> list[int]:
        tokenizer = self.info.get_tokenizer()
        token_ids = tokenizer.encode(text, **dict(tokenization_kwargs))
        return list(token_ids)

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalInput:
        self._validate_supported_inputs(inputs.mm_data_items)

        tokenizer = self.info.get_tokenizer()
        prompt_text = (
            inputs.prompt if isinstance(inputs.prompt, str) else
            tokenizer.decode(inputs.prompt))
        merged_mm_processor_kwargs = dict(self.info.ctx.get_merged_mm_kwargs(
            inputs.hf_processor_mm_kwargs))
        if "model_name_or_path" not in merged_mm_processor_kwargs:
            merged_mm_processor_kwargs["model_name_or_path"] = self.info.ctx.model_config.model

        model_path = merged_mm_processor_kwargs.get("model_name_or_path")
        if model_path and isinstance(model_path, str):
            model_path_abs = os.path.abspath(model_path)
            if os.path.isdir(model_path_abs):
                if os.path.exists(os.path.join(model_path_abs, "emu35_vison_tokenizer.safetensors")):
                    if "apertus_vq_hub" not in merged_mm_processor_kwargs:
                        merged_mm_processor_kwargs["apertus_vq_hub"] = model_path_abs
                if os.path.exists(os.path.join(model_path_abs, "wavtokenizer_large_unify_600_24k.safetensors")):
                    if "apertus_audio_tokenizer_path" not in merged_mm_processor_kwargs:
                        merged_mm_processor_kwargs["apertus_audio_tokenizer_path"] = model_path_abs

        num_images = inputs.mm_data_items.get_count("image", strict=False)
        num_audios = inputs.mm_data_items.get_count("audio", strict=False)
        image_aliases = self.image_tokenizer.placeholder_aliases(
            tokenizer,
            merged_mm_processor_kwargs,
        )
        image_placeholders = self._find_placeholders(prompt_text, image_aliases)
        audio_aliases = self.audio_tokenizer.placeholder_aliases(
            merged_mm_processor_kwargs)
        audio_placeholders = self._find_placeholders(prompt_text, audio_aliases)

        if image_placeholders and num_images == 0:
            # Apertus behavior: remove surplus image placeholders when no image is
            # provided for that position.
            image_placeholder_removals = self._bind_and_group_updates(
                [
                    PromptReplacement(
                        modality="image",
                        target=lambda item_idx: image_placeholders[item_idx],
                        replacement=lambda item_idx: "",
                    )
                ],
                {"image": len(image_placeholders)},
            )
            logger.info(
                "[Apertus MM] removing %d unresolved image placeholder(s) "
                "because received 0 image input(s)",
                len(image_placeholders),
            )
            prompt_text, _ = self._apply_text_matches(
                prompt_text,
                image_placeholder_removals,
            )
            image_placeholders = []

        if num_images == 0 and num_audios == 0:
            if audio_placeholders:
                raise ValueError(
                    "Apertus audio placeholder/input mismatch: found "
                    f"{len(audio_placeholders)} placeholder(s) in the prompt "
                    f"using aliases {audio_aliases}, but received 0 audio input(s)."
                )

            with timing_ctx.record("tokenize"):
                prompt_token_ids = self._tokenize_text(
                    prompt_text,
                    inputs.tokenization_kwargs,
                )
            return mm_input(
                prompt_token_ids=prompt_token_ids,
                mm_kwargs=MultiModalKwargsItems({}),
                mm_hashes={},
                mm_placeholders={},
                prompt=prompt_text,
            )

        mm_counts: dict[str, int] = {}
        prompt_replacements: list[PromptReplacement] = []

        if num_images > 0:
            with timing_ctx.record("encode_apertus_images"):
                image_items = inputs.mm_data_items.get_items(
                    "image", ImageProcessorItems)
                images = image_items.get_all()
                image_prompts = self.image_tokenizer.encode_images(
                    images,
                    tokenizer=tokenizer,
                    mm_processor_kwargs=merged_mm_processor_kwargs,
                )

            if len(image_placeholders) < len(image_prompts):
                raise ValueError(
                    "Apertus image placeholder/input mismatch: found "
                    f"{len(image_placeholders)} placeholder(s) in the prompt "
                    f"using aliases {image_aliases}, but received "
                    f"{len(image_prompts)} image input(s). "
                    "Received more images than placeholders; refusing to "
                    "silently drop or reorder images.")
            if len(image_placeholders) > len(image_prompts):
                logger.info(
                    "[Apertus MM] prompt has %d image placeholder(s) but only %d "
                    "image input(s); extra placeholder(s) will be replaced by \"\"",
                    len(image_placeholders),
                    len(image_prompts),
                )
            mm_counts["image"] = len(image_placeholders)
            prompt_replacements.append(
                PromptReplacement(
                    modality="image",
                    target=lambda item_idx: image_placeholders[item_idx],
                    replacement=lambda item_idx: (image_prompts[item_idx] if
                                                  item_idx < len(image_prompts)
                                                  else ""),
                ))

        if num_audios > 0:
            with timing_ctx.record("encode_apertus_audios"):
                audio_items = inputs.mm_data_items.get_items(
                    "audio", AudioProcessorItems)
                audios = audio_items.get_all()
                audio_prompts = self.audio_tokenizer.encode_audios(
                    audios,
                    tokenizer=tokenizer,
                    mm_processor_kwargs=merged_mm_processor_kwargs,
                )

            if len(audio_placeholders) != len(audio_prompts):
                raise ValueError(
                    "Apertus audio placeholder/input mismatch: found "
                    f"{len(audio_placeholders)} placeholder(s) in the prompt "
                    f"using aliases {audio_aliases}, but received "
                    f"{len(audio_prompts)} audio input(s).")
            mm_counts["audio"] = len(audio_prompts)
            prompt_replacements.append(
                PromptReplacement(
                    modality="audio",
                    target=lambda item_idx: audio_placeholders[item_idx],
                    replacement=lambda item_idx: audio_prompts[item_idx],
                ))
        elif audio_placeholders:
            raise ValueError(
                "Apertus audio placeholder/input mismatch: found "
                f"{len(audio_placeholders)} placeholder(s) in the prompt using "
                f"aliases {audio_aliases}, but received 0 audio input(s).")

        prompt_updates = self._bind_and_group_updates(prompt_replacements,
                                                      mm_counts)

        with timing_ctx.record("apply_prompt_updates"):
            merged_prompt, match_result = self._apply_text_matches(
                prompt_text, prompt_updates)

        if not all(
                update_idx is not None
                for update_idxs in match_result.values()
                for update_idx in update_idxs):
            raise RuntimeError(
                "Failed to replace all Apertus multimodal placeholders.")

        with timing_ctx.record("tokenize"):
            prompt_token_ids = self._tokenize_text(
                merged_prompt,
                inputs.tokenization_kwargs,
            )

        return mm_input(
            prompt_token_ids=prompt_token_ids,
            mm_kwargs=MultiModalKwargsItems({}),
            mm_hashes={},
            mm_placeholders={},
            prompt=merged_prompt,
        )


@MULTIMODAL_REGISTRY.register_processor(
    ApertusMultiModalProcessor,
    info=ApertusProcessingInfo,
    dummy_inputs=ApertusDummyInputsBuilder,
)
class ApertusForConditionalGeneration(
        ApertusForCausalLM,
        SupportsMultiModal,
):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.image_tokenizer = ApertusImageTokenizer()
        self.audio_tokenizer = ApertusAudioTokenizer()
        self.vision_tower = None
        self.audio_tower = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_keys = super().load_weights(weights)

        model_path = getattr(self.config, "_name_or_path", "")
        mm_processor_kwargs = {"model_name_or_path": model_path}

        if model_path and os.path.isdir(model_path):
            model_path_abs = os.path.abspath(model_path)
            mm_processor_kwargs["apertus_vq_hub"] = model_path_abs
            mm_processor_kwargs["apertus_audio_tokenizer_path"] = model_path_abs

        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda")
        device_str = str(device)
        mm_processor_kwargs["apertus_vision_tokenizer_device"] = device_str
        mm_processor_kwargs["apertus_audio_tokenizer_device"] = device_str

        logger.info("[Apertus MM] Loading vision tokenizer onto device %s in load_weights...", device_str)
        self.vision_tower = self.image_tokenizer.load_vision_tokenizer(mm_processor_kwargs)

        logger.info("[Apertus MM] Loading audio tokenizer onto device %s in load_weights...", device_str)
        self.audio_tower = self.audio_tokenizer.get_audio_tokenizer(mm_processor_kwargs)

        return loaded_keys


    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return ApertusImageTokenizer.DEFAULT_IMAGE_PLACEHOLDER
        if modality == "audio":
            return ApertusAudioTokenizer.DEFAULT_AUDIO_PLACEHOLDER

        raise ValueError(f"Unsupported modality: {modality}")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        if kwargs:
            raise ValueError(
                "Apertus multimodal inputs are serialized to token IDs during "
                "preprocessing and should not reach the model as multimodal "
                f"kwargs. Got keys: {sorted(kwargs)}")
        return []

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            raise ValueError(
                "Apertus does not merge multimodal embeddings in the model. "
                "Multimodal inputs must be serialized to token IDs by the "
                "processor.")
        return super().embed_input_ids(input_ids)
