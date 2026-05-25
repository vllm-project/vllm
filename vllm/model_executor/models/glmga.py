# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers import BatchFeature
from transformers.video_utils import VideoMetadata

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

from .glm4_1v import (
    Glm4vDummyInputsBuilder,
    Glm4vForConditionalGeneration,
    Glm4vMultiModalProcessor,
    Glm4vProcessingInfo,
    _to_video_metadata,
)


class GlmGaProcessingInfo(Glm4vProcessingInfo):
    pass


class GlmGaDummyInputsBuilder(Glm4vDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        hf_config = self.info.get_hf_config()
        hf_processor = self.info.get_hf_processor()
        tokenizer = self.info.get_tokenizer()

        image_token_ids = [
            hf_config.image_start_token_id,
            hf_processor.image_token_id,
            hf_config.image_end_token_id,
        ]
        image_token = tokenizer.decode(image_token_ids)
        video_token_ids = [
            hf_config.video_start_token_id,
            hf_processor.video_token_id,
            hf_config.video_end_token_id,
        ]
        video_token = tokenizer.decode(video_token_ids)

        return image_token * num_images + video_token * num_videos


class GlmGaMultiModalProcessor(Glm4vMultiModalProcessor):
    @staticmethod
    def _get_direct_path_inputs(
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        prepared_data = dict(mm_data)
        prepared_kwargs = dict(mm_kwargs)

        videos = prepared_data.get("videos")
        if not (isinstance(videos, list) and len(videos) > 0):
            return prepared_data, prepared_kwargs

        hf_videos = []
        hf_video_metadata = []
        for item in videos:
            if isinstance(item, tuple) and len(item) == 2:
                video_array, metadata = item
                hf_videos.append(video_array)
                if isinstance(metadata, VideoMetadata):
                    hf_video_metadata.append(metadata)
                elif isinstance(metadata, Mapping):
                    hf_video_metadata.append(_to_video_metadata(metadata))
                    if "do_sample_frames" in metadata:
                        prepared_kwargs["do_sample_frames"] = metadata[
                            "do_sample_frames"
                        ]
                elif metadata is not None:
                    raise TypeError(
                        "Video metadata must be a mapping or VideoMetadata, "
                        f"got {type(metadata)}"
                    )
            else:
                hf_videos.append(item)

        prepared_data["videos"] = hf_videos
        if hf_video_metadata:
            prepared_data["video_metadata"] = hf_video_metadata
            prepared_kwargs["return_metadata"] = True

        return prepared_data, prepared_kwargs

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        prepared_data, prepared_kwargs = self._get_direct_path_inputs(
            mm_data, mm_kwargs
        )
        return super(Glm4vMultiModalProcessor, self)._call_hf_processor(
            prompt=prompt,
            mm_data=prepared_data,
            mm_kwargs=prepared_kwargs,
            tok_kwargs=tok_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        config = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        if hf_inputs.get("video_metadata"):
            config["video_metadata"] = MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            )
        return config

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        merge_length = image_processor.merge_size**2

        def get_image_replacement(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            placeholder = [
                hf_config.image_start_token_id,
                *([hf_processor.image_token_id] * num_tokens),
                hf_config.image_end_token_id,
            ]
            return PromptUpdateDetails.select_token_id(
                placeholder,
                embed_token_id=hf_processor.image_token_id,
            )

        def get_video_replacement(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            metadata_elem = out_item.get("video_metadata")
            if metadata_elem is not None:
                metadata = metadata_elem.data
                if not isinstance(metadata, VideoMetadata):
                    assert isinstance(metadata, Mapping)
                    metadata = _to_video_metadata(metadata)
            else:
                _, raw_metadata = mm_items["video"][item_idx]
                assert isinstance(raw_metadata, Mapping)
                metadata = _to_video_metadata(raw_metadata)

            placeholder = self.info._construct_video_placeholder_glm46v(
                metadata, grid_thw
            )
            return PromptUpdateDetails.select_token_id(
                placeholder,
                embed_token_id=hf_processor.image_token_id,
            )

        return [
            PromptReplacement(
                modality="image",
                target="<|begin_of_image|><|image|><|end_of_image|>",
                replacement=get_image_replacement,
            ),
            PromptReplacement(
                modality="video",
                target="<|begin_of_video|><|video|><|end_of_video|>",
                replacement=get_video_replacement,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    GlmGaMultiModalProcessor,
    info=GlmGaProcessingInfo,
    dummy_inputs=GlmGaDummyInputsBuilder,
)
class GlmGAForConditionalGeneration(Glm4vForConditionalGeneration):
    pass
