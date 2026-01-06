# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Qwen2.5-Omni model (thinker part)."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Annotated, Any, Literal

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniConfig,
    Qwen2_5OmniThinkerConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoder,
)
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import (
    Qwen2_5OmniProcessor,
)
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionTransformer,
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLProcessingInfo,
    Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoInputs,
    Qwen2_5_VLVideoPixelInputs,
)
from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioProcessingInfo,
    _get_feat_extract_output_lengths,
)
from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalDataParser
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    ImageItem,
    ModalityData,
    MultiModalDataDict,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
    split_list_into_ranges,
)
from .vision import get_llm_pos_ids_for_vision

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None

logger = init_logger(__name__)


class Qwen2_5OmniAudioFeatureInputs(TensorSchema):
    """
    Dimensions:
        - na: Number of audios
        - nmb: Number of mel bins
        - msl: Maximum sequence length
        - tsl: Total sequence length
    """

    type: Literal["audio_features"]
    input_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nmb", "tsl", dynamic_dims={"tsl"}),
    ]

    audio_feature_lengths: Annotated[torch.Tensor, TensorShape("na")]

    feature_attention_mask: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("na", "msl", dynamic_dims={"msl"}),
    ]


def create_qwen2_5_omni_thinker_field_factory(
    spatial_merge_size: int,
) -> Callable[[Mapping[str, torch.Tensor]], Mapping[str, MultiModalFieldConfig]]:
    def _qwen2_5_omni_thinker_field_config(hf_inputs: Mapping[str, torch.Tensor]):
        audio_feature_lengths = hf_inputs.get(
            "audio_feature_lengths", torch.empty((0,))
        )

        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_pixel_grid_sizes = image_grid_thw.prod(-1)
        image_embed_grid_sizes = (
            image_pixel_grid_sizes // spatial_merge_size // spatial_merge_size
        )

        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        video_grid_sizes = video_grid_thw.prod(-1)
        video_embed_grid_sizes = (
            video_grid_sizes // spatial_merge_size // spatial_merge_size
        )

        num_videos = len(video_grid_sizes)

        return dict(
            input_audio_features=MultiModalFieldConfig.flat_from_sizes(
                "audio", audio_feature_lengths, dim=1
            ),
            feature_attention_mask=MultiModalFieldConfig.batched("audio"),
            audio_feature_lengths=MultiModalFieldConfig.batched("audio"),
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_pixel_grid_sizes
            ),
            image_embeds=MultiModalFieldConfig.flat_from_sizes(
                "image", image_embed_grid_sizes
            ),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_sizes
            ),
            video_embeds=MultiModalFieldConfig.flat_from_sizes(
                "video", video_embed_grid_sizes
            ),
            video_grid_thw=MultiModalFieldConfig.batched("video"),
            second_per_grid_ts=MultiModalFieldConfig.batched("video"),
            use_audio_in_video=MultiModalFieldConfig.shared("video", num_videos),
        )

    return _qwen2_5_omni_thinker_field_config


class Qwen2_5OmniThinkerMultiModalDataParser(Qwen2VLMultiModalDataParser):
    def __init__(self, spatial_merge_size: int, *args, **kwargs):
        self._spatial_merge_size = spatial_merge_size
        super().__init__(self._spatial_merge_size, *args, **kwargs)

    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[ImageItem],
    ) -> ModalityDataItems[Any, Any]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"input_audio_features", "audio_feature_lengths"},
                fields_factory=create_qwen2_5_omni_thinker_field_factory(
                    self._spatial_merge_size
                ),
            )

        return super()._parse_audio_data(data)


class Qwen2_5OmniThinkerProcessingInfo(
    Qwen2AudioProcessingInfo, Qwen2_5_VLProcessingInfo
):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen2_5OmniConfig).thinker_config

    def get_hf_processor(self, **kwargs: object) -> Qwen2_5OmniProcessor:
        return self.ctx.get_hf_processor(
            Qwen2_5OmniProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_feature_extractor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None, "image": None, "video": None}


class Qwen2_5OmniThinkerDummyInputsBuilder(
    BaseDummyInputsBuilder[Qwen2_5OmniThinkerProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        hf_processor = self.info.get_hf_processor()

        audio_token: str = hf_processor.audio_token
        image_token: str = hf_processor.image_token
        video_token: str = hf_processor.video_token

        return (
            audio_token * num_audios
            + image_token * num_images
            + video_token * num_videos
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        feature_extractor = self.info.get_feature_extractor()

        target_audio_length = (
            min(
                feature_extractor.chunk_length,
                30,
            )
            * feature_extractor.sampling_rate
        )
        target_width, target_height = self.info.get_image_size_with_most_features()
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts
        )

        image_overrides = mm_options.get("image") if mm_options else None
        video_overrides = mm_options.get("video") if mm_options else None
        audio_overrides = mm_options.get("audio") if mm_options else None

        mm_data = {
            "audio": self._get_dummy_audios(
                length=target_audio_length,
                num_audios=num_audios,
                overrides=audio_overrides,
            ),
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "video": self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
                overrides=video_overrides,
            ),
        }

        return mm_data


class Qwen2_5OmniThinkerMultiModalProcessor(
    BaseMultiModalProcessor[Qwen2_5OmniThinkerProcessingInfo]
):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return Qwen2_5OmniThinkerMultiModalDataParser(
            spatial_merge_size=self.info.get_hf_config().vision_config.spatial_merge_size,
            target_sr=feature_extractor.sampling_rate,
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        # NOTE: WhisperFeatureExtractor cannot handle empty list of audios
        if audios:
            # NOTE: Qwen2.5-Omni processor accept "audio"
            mm_data["audio"] = audios
            mm_kwargs = dict(
                **mm_kwargs,
            )

        hf_inputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        input_features = hf_inputs.pop("input_features", None)
        feature_attention_mask = hf_inputs.get("feature_attention_mask", None)
        if "input_audio_features" not in hf_inputs and input_features is not None:
            if feature_attention_mask is not None:
                input_features = input_features.permute(0, 2, 1)[
                    feature_attention_mask.bool()
                ].permute(1, 0)
            hf_inputs["input_audio_features"] = input_features
        if (
            "audio_feature_lengths" not in hf_inputs
            and feature_attention_mask is not None
        ):
            hf_inputs["audio_feature_lengths"] = feature_attention_mask.sum(-1)

        video_second_per_grid = hf_inputs.get("video_second_per_grid", None)
        if video_second_per_grid is not None:
            hf_inputs["second_per_grid_ts"] = video_second_per_grid

        use_audio_in_video = mm_kwargs.get("use_audio_in_video", False)
        hf_inputs["use_audio_in_video"] = torch.tensor(use_audio_in_video)

        return hf_inputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return create_qwen2_5_omni_thinker_field_factory(
            self.info.get_hf_config().vision_config.spatial_merge_size
        )(hf_inputs)

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        """
        Qwen2.5-Omni reimplements this function to handle `use_audio_in_video`.
        """
        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)
        self._validate_mm_updates(mm_prompt_updates, mm_item_counts)

        if is_update_applied:
            mm_placeholders = self._find_mm_placeholders(
                prompt_ids,
                mm_prompt_updates,
            )
            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts,
            )
        else:
            prompt_ids, mm_placeholders = self._apply_prompt_updates(
                prompt_ids,
                mm_prompt_updates,
            )
            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts,
            )

        return prompt_ids, mm_placeholders

    @classmethod
    def omni_get_updates_use_audio_in_video(
        cls,
        thinker_config: PretrainedConfig,
        audio_len: int,
        video_grid_thw: list[int] | torch.Tensor,
        video_second_per_grid_t: float,
    ) -> list[int]:
        """Get video prompt updates when `use_audio_in_video` is True.

        In this case, audio and vision update ids will be split into
        chunks and interleaved (details in `_omni_get_input_positions_tensor`).

        <|video_bos|><|VIDEO|><|video_eos|> =>
        <|video_bos|><|audio_bos|>(... chunks ...)<|audio_eos|><|video_eos|>
        """

        audio_token_id = thinker_config.audio_token_index
        video_token_id = thinker_config.video_token_index
        audio_start_token_id = thinker_config.audio_start_token_id
        audio_end_token_id = thinker_config.audio_end_token_id
        seconds_per_chunk = thinker_config.seconds_per_chunk
        spatial_merge_size = thinker_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(
            thinker_config.vision_config, "tokens_per_second", 25
        )

        grid_t = video_grid_thw[0]
        grid_h = video_grid_thw[1]
        grid_w = video_grid_thw[2]
        t_ntoken_per_chunk = int(tokens_per_second * seconds_per_chunk)
        t_index = (
            torch.arange(grid_t) * video_second_per_grid_t * tokens_per_second
        ).long()
        t_index_split_chunk = split_list_into_ranges(t_index, t_ntoken_per_chunk)

        updates = [audio_start_token_id]
        added_audio_len = 0
        for t_chunk in t_index_split_chunk:
            vision_ntoken_per_chunk = (
                len(t_chunk) * grid_h * grid_w // (spatial_merge_size**2)
            )
            updates.extend([video_token_id] * vision_ntoken_per_chunk)

            audio_chunk_size = min(t_ntoken_per_chunk, audio_len - added_audio_len)
            updates.extend(audio_chunk_size * [audio_token_id])
            added_audio_len += audio_chunk_size
        if added_audio_len < audio_len:
            updates.extend((audio_len - added_audio_len) * [audio_token_id])
        updates.extend([audio_end_token_id])

        return updates

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        vocab = tokenizer.get_vocab()

        audio_token = processor.audio_token
        image_token = processor.image_token
        video_token = processor.video_token
        audio_token_id = vocab[audio_token]
        image_token_id = vocab[image_token]
        video_token_id = vocab[video_token]

        out_mm_data = out_mm_kwargs.get_data()
        audio_feature_lengths = out_mm_data.get("audio_feature_lengths")
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        if audio_feature_lengths is None and feature_attention_mask is None:
            audio_output_lengths = []
        elif audio_feature_lengths is not None:
            _, audio_output_lens = _get_feat_extract_output_lengths(
                audio_feature_lengths
            )
            audio_output_lengths = audio_output_lens.tolist()
        elif feature_attention_mask is not None:
            assert isinstance(feature_attention_mask, torch.Tensor)
            _, audio_output_lens = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            audio_output_lengths = audio_output_lens.tolist()

        # number of audios read from video.
        audio_in_video_item_idx = 0

        def get_replacement_qwen2_audio(item_idx: int):
            item_idx += audio_in_video_item_idx

            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio = audios.get(item_idx)
                raise ValueError(
                    f"The audio {audio} (len={len(audio)}) is too short "
                    "to be represented inside the model"
                )

            return [audio_token_id] * num_features

        def get_replacement_qwen2_vision(item_idx: int, modality: str):
            grid_thw = out_mm_data[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)
            merge_length = image_processor.merge_size**2

            token_id = image_token_id if modality == "image" else video_token_id
            return [token_id] * (int(grid_thw.prod()) // merge_length)

        use_audio_in_video = hf_processor_mm_kwargs.get("use_audio_in_video", False)
        thinker_config = self.info.get_hf_config()

        def get_replacement_qwen2_use_audio_in_video(item_idx: int):
            nonlocal audio_in_video_item_idx

            audio_num_features = audio_output_lengths[
                audio_in_video_item_idx + item_idx
            ]
            video_grid_thw = out_mm_data["video_grid_thw"][item_idx]

            audio_in_video_item_idx += 1

            second_per_grid_ts = hf_processor_mm_kwargs.get("second_per_grid_ts", None)
            if second_per_grid_ts:
                video_second_per_grid_t = second_per_grid_ts[item_idx]
            else:
                video_second_per_grid_t = 1.0

            return self.omni_get_updates_use_audio_in_video(
                thinker_config=thinker_config,
                audio_len=audio_num_features,
                video_grid_thw=video_grid_thw,
                video_second_per_grid_t=video_second_per_grid_t,
            )

        video_replacement_fn = (
            get_replacement_qwen2_use_audio_in_video
            if use_audio_in_video
            else partial(get_replacement_qwen2_vision, modality="video")
        )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_qwen2_audio,
            ),
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=partial(get_replacement_qwen2_vision, modality="image"),
            ),
            PromptReplacement(
                modality="video",
                target=video_token,
                replacement=video_replacement_fn,
            ),
        ]

    def _apply_hf_processor_main(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        enable_hf_prompt_update: bool,
    ) -> tuple[list[int], BatchFeature, bool]:
        """
        Qwen2.5-Omni reimplements this function to handle text only.
        """
        if isinstance(prompt, str):
            if enable_hf_prompt_update:
                return self._apply_hf_processor_text_mm(
                    prompt_text=prompt,
                    mm_items=mm_items,
                    hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                    tokenization_kwargs=tokenization_kwargs,
                )
            tokenizer = self.info.get_tokenizer()
            prompt_ids = tokenizer.encode(prompt)
        else:
            prompt_ids = self._apply_hf_processor_tokens_only(prompt)

        mm_processed_data = self._apply_hf_processor_mm_only(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

        return prompt_ids, mm_processed_data, False

    def _apply_hf_processor_mm_only(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """
        Qwen2.5-Omni reimplements this function to handle `use_audio_in_video`.
        """
        mm_counts = mm_items.get_all_counts()

        use_audio_in_video = hf_processor_mm_kwargs.get("use_audio_in_video", False)
        if use_audio_in_video and "video" in mm_counts:
            assert "audio" in mm_counts
            mm_counts["audio"] -= mm_counts["video"]

        _, mm_processed_data, _ = self._apply_hf_processor_text_mm(
            prompt_text=self.dummy_inputs.get_dummy_text(mm_counts),
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

        return mm_processed_data


class Qwen2_5OmniConditionalGenerationMixin:
    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Qwen2_5OmniAudioFeatureInputs | None:
        input_audio_features = kwargs.pop("input_audio_features", None)
        audio_feature_lengths = kwargs.pop("audio_feature_lengths", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        if input_audio_features is None:
            return None

        return Qwen2_5OmniAudioFeatureInputs(
            type="audio_features",
            input_features=input_audio_features,
            audio_feature_lengths=audio_feature_lengths,
            feature_attention_mask=feature_attention_mask,
        )

    def _parse_and_validate_image_input(
        self,
        **kwargs: dict[str, Any],
    ) -> Qwen2_5_VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return Qwen2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(
        self,
        **kwargs: dict[str, Any],
    ) -> Qwen2_5_VLVideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        if video_embeds is not None:
            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError(
                    "Incorrect type of video embeddings. "
                    f"Got type: {type(video_embeds)}"
                )
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
            )

    def _process_audio_input(
        self,
        audio_input: Qwen2_5OmniAudioFeatureInputs,
        audio_hashes: list[str] | None = None,
        cached_audio_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]

        audio_feat_lengths, audio_output_lengths = (
            self.audio_tower._get_feat_extract_output_lengths(audio_feature_lengths)
        )

        audio_outputs = self.audio_tower(
            input_features.to(self.audio_tower.dtype),
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_feat_lengths,
        )
        return audio_outputs.last_hidden_state.split(audio_output_lengths.tolist())

    def _process_image_input(
        self, image_input: Qwen2_5_VLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"].type(self.visual.dtype)

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        with set_forward_context(None, self.vllm_config):
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())

    def _process_video_input(
        self,
        video_input: Qwen2_5_VLVideoInputs,
        video_hashes: list[str] = None,
        cached_video_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if video_input["type"] == "video_embeds":
            return video_input["video_embeds"].type(self.visual.dtype)

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
        with set_forward_context(None, self.vllm_config):
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)
        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return video_embeds.split(sizes.tolist())


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5OmniThinkerMultiModalProcessor,
    info=Qwen2_5OmniThinkerProcessingInfo,
    dummy_inputs=Qwen2_5OmniThinkerDummyInputsBuilder,
)
class Qwen2_5OmniThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsLoRA,
    SupportsMRoPE,
    Qwen2_5OmniConditionalGenerationMixin,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "thinker.lm_head.": "language_model.lm_head.",
            "thinker.model.": "language_model.model.",
            "thinker.": "",
        }
    )
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "attn.qkv": [
            "attn.q",
            "attn.k",
            "attn.v",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|IMAGE|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|VIDEO|><|vision_end|>"
        if modality.startswith("audio"):
            return f"Audio {i}: <|audio_bos|><|AUDIO|><|audio_eos|>"

        raise ValueError("Only image, video or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        thinker_config: Qwen2_5OmniThinkerConfig = (
            vllm_config.model_config.hf_config.thinker_config
        )
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = thinker_config
        self.multimodal_config = multimodal_config

        # force "use_flash_attention_2=True" to audio tower to align
        # the results.
        if flash_attn is not None:
            audio_config = thinker_config.audio_config
            audio_config._attn_implementation_autoset = True
            audio_config._attn_implementation = "flash_attention_2"
        else:
            logger.warning(
                "flash_attn is not available, the model may not yield the "
                "exactly same result as the transformers implementation "
                "in the audio tower part."
            )

        if multimodal_config.get_limit_per_prompt("audio"):
            self.audio_tower = Qwen2_5OmniAudioEncoder(thinker_config.audio_config)
        else:
            self.audio_tower = None

        if multimodal_config.get_limit_per_prompt(
            "image"
        ) or multimodal_config.get_limit_per_prompt("video"):
            self.visual = Qwen2_5_VisionTransformer(
                vision_config=thinker_config.vision_config,
                norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                multimodal_config=multimodal_config,
            )
        else:
            self.visual = None

        self.quant_config = quant_config
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            hf_config=thinker_config.text_config,
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
            if (
                input_key in ("input_audio_features")
                and "audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(
                    **kwargs
                )
        return mm_input_by_modality

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        """
        Example:

            (V_i are vision position ids, A_i are audio position ids)

            |V_1 ...    V_n|A_1 ...   A_n|V_n+1 ... V_2n|A_n+1 ... A_2n|...
            |vision chunk 1|audio chunk 1|vision chunk 2|audio chunk 2 |...
        """
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {
                "image_grid_thw",
                "video_grid_thw",
                "second_per_grid_ts",
                "audio_feature_lengths",
                "use_audio_in_video",
            },
        )
        image_grid_thw = kwargs.get("image_grid_thw", [])
        video_grid_thw = kwargs.get("video_grid_thw", [])
        second_per_grid_ts = kwargs.get("second_per_grid_ts", [])
        audio_feature_lengths = kwargs.get("audio_feature_lengths", [])
        use_audio_in_video = any(kwargs.get("use_audio_in_video", []))

        image_grid_thw = (torch.stack if image_grid_thw else torch.tensor)(
            image_grid_thw
        )
        video_grid_thw = (torch.stack if video_grid_thw else torch.tensor)(
            video_grid_thw
        )

        # TODO(fyabc): refactor and share more code with
        #  _vl_get_input_positions_tensor.

        thinker_config = self.config
        audio_token_id = thinker_config.audio_token_index
        image_token_id = thinker_config.image_token_index
        video_token_id = thinker_config.video_token_index
        audio_start_token_id = thinker_config.audio_start_token_id
        audio_end_token_id = thinker_config.audio_end_token_id
        vision_start_token_id = thinker_config.vision_start_token_id
        vision_end_token_id = thinker_config.vision_end_token_id
        seconds_per_chunk = thinker_config.seconds_per_chunk
        spatial_merge_size = thinker_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(
            thinker_config.vision_config, "tokens_per_second", 25
        )

        src_item = input_tokens
        audio_seqlens = audio_feature_lengths
        if not second_per_grid_ts:
            second_per_grid_ts = [1] * video_grid_thw.shape[0]
        audio_idx = 0
        video_idx = 0
        image_idx = 0
        new_src_item: list[int] = []
        llm_pos_ids_list: list[torch.Tensor] = []

        idx = 0
        while idx < len(src_item):
            new_src_item_len = len(new_src_item)
            start_idx = (
                llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            )
            if src_item[idx] not in [audio_token_id, video_token_id, image_token_id]:
                if use_audio_in_video and idx > 0:
                    if (
                        src_item[idx] == vision_end_token_id
                        and src_item[idx - 1] == audio_end_token_id
                    ):
                        # processing the <|audio_eos|> before <|vision_eos|>
                        start_idx -= 1
                    elif (
                        src_item[idx] == audio_start_token_id
                        and src_item[idx - 1] == vision_start_token_id
                    ):
                        # processing the <|audio_bos|> after <|vision_eos|>
                        start_idx -= 1
                new_src_item.append(src_item[idx])
                llm_pos_ids = torch.tensor([start_idx], dtype=torch.long).expand(3, -1)
                llm_pos_ids_list.append(llm_pos_ids)
            elif src_item[idx] == audio_token_id:
                assert audio_seqlens is not None
                audio_seqlen = audio_seqlens[audio_idx]
                place_num = ((audio_seqlen - 1) // 2 + 1 - 2) // 2 + 1
                new_src_item.extend([audio_token_id] * place_num)
                llm_pos_ids = torch.arange(place_num).expand(3, -1) + start_idx
                llm_pos_ids_list.append(llm_pos_ids)
                audio_idx += 1
            elif src_item[idx] == image_token_id:
                grid_t = image_grid_thw[image_idx][0]
                grid_hs = image_grid_thw[:, 1]
                grid_ws = image_grid_thw[:, 2]
                t_index = (torch.arange(grid_t) * 1 * tokens_per_second).long()
                llm_pos_ids = get_llm_pos_ids_for_vision(
                    start_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                )
                llm_pos_ids_list.append(llm_pos_ids)
                vision_seqlen = image_grid_thw[image_idx].prod() // (
                    spatial_merge_size**2
                )
                new_src_item.extend([image_token_id] * vision_seqlen)
                image_idx += 1
            elif src_item[idx] == video_token_id and not use_audio_in_video:
                grid_t = video_grid_thw[video_idx][0]
                grid_hs = video_grid_thw[:, 1]
                grid_ws = video_grid_thw[:, 2]
                t_index = (
                    torch.arange(grid_t)
                    * second_per_grid_ts[video_idx]
                    * tokens_per_second
                ).long()
                llm_pos_ids = get_llm_pos_ids_for_vision(
                    start_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                )
                llm_pos_ids_list.append(llm_pos_ids)
                vision_seqlen = video_grid_thw[video_idx].prod() // (
                    spatial_merge_size**2
                )
                new_src_item.extend([video_token_id] * vision_seqlen)
                video_idx += 1
            else:
                # read audio from video
                assert audio_seqlens is not None
                audio_seqlen = audio_seqlens[audio_idx]
                vision_seqlen = video_grid_thw[video_idx].prod() // (
                    spatial_merge_size**2
                )
                grid_t = video_grid_thw[video_idx][0]
                grid_h = video_grid_thw[video_idx][1]
                grid_w = video_grid_thw[video_idx][2]
                grid_hs = video_grid_thw[:, 1]
                grid_ws = video_grid_thw[:, 2]
                t_ntoken_per_chunk = int(tokens_per_second * seconds_per_chunk)
                t_index = (
                    torch.arange(grid_t)
                    * second_per_grid_ts[video_idx]
                    * tokens_per_second
                ).long()
                t_index_split_chunk = split_list_into_ranges(
                    t_index, t_ntoken_per_chunk
                )
                place_num = (((audio_seqlen - 1) // 2 + 1 - 2) // 2 + 1) + 2
                pure_audio_len = place_num - 2
                added_audio_len = 0
                audio_llm_pos_ids_list: list[torch.Tensor] = []
                for t_chunk in t_index_split_chunk:
                    vision_ntoken_per_chunk = (
                        len(t_chunk) * grid_h * grid_w // (spatial_merge_size**2)
                    )
                    new_src_item.extend([video_token_id] * vision_ntoken_per_chunk)
                    vision_llm_pos_ids_list = get_llm_pos_ids_for_vision(
                        start_idx,
                        video_idx,
                        spatial_merge_size,
                        t_chunk,
                        grid_hs,
                        grid_ws,
                    ).split(1, dim=1)
                    llm_pos_ids_list.extend(vision_llm_pos_ids_list)
                    new_src_item.extend(
                        min(t_ntoken_per_chunk, pure_audio_len - added_audio_len)
                        * [audio_token_id]
                    )
                    audio_start_idx = (
                        start_idx
                        if len(audio_llm_pos_ids_list) == 0
                        else audio_llm_pos_ids_list[-1][0].item() + 1
                    )
                    if min(t_ntoken_per_chunk, pure_audio_len - added_audio_len) > 0:
                        audio_llm_pos_ids_list = (
                            torch.arange(
                                min(
                                    t_ntoken_per_chunk, pure_audio_len - added_audio_len
                                )
                            ).expand(3, -1)
                            + audio_start_idx
                        ).split(1, dim=1)
                    else:
                        audio_llm_pos_ids_list = []
                    added_audio_len += min(
                        t_ntoken_per_chunk, pure_audio_len - added_audio_len
                    )
                    llm_pos_ids_list.extend(audio_llm_pos_ids_list)
                if added_audio_len < pure_audio_len:
                    new_src_item.extend(
                        (pure_audio_len - added_audio_len) * [audio_token_id]
                    )
                    audio_llm_pos_ids_list = (
                        torch.arange(pure_audio_len - added_audio_len).expand(3, -1)
                        + llm_pos_ids_list[-1].max()
                        + 1
                    ).split(1, dim=1)
                    llm_pos_ids_list.extend(audio_llm_pos_ids_list)
                audio_idx += 1
                video_idx += 1
            # move to the next token
            idx += len(new_src_item) - new_src_item_len

        llm_positions = torch.cat(llm_pos_ids_list, dim=1)
        mrope_position_delta = (
            torch.cat(llm_pos_ids_list, dim=1).max() + 1 - len(src_item)
        )

        return llm_positions, mrope_position_delta

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor corresponding to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += tuple(video_embeddings)
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                multimodal_embeddings += tuple(audio_embeddings)
        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        # This is to satisfy the type checker for each overload
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["talker.", "token2wav."]
        if self.audio_tower is None:
            skip_prefixes.extend(["audio_tower."])
        if self.visual is None:
            skip_prefixes.extend(["visual."])

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )
        loaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        return loaded_weights

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="merger.",
            tower_model=["visual.", "audio_tower."],
        )
