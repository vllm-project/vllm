# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
"""Inference-only Qwen3VL model compatible with HuggingFace weights."""
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
import torch
from transformers import BatchFeature
from transformers.models.qwen2_vl import Qwen2VLImageProcessorFast
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.qwen3_vl import (Qwen3VLProcessor,
                                          Qwen3VLVideoProcessor)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.video_utils import VideoMetadata

from vllm.model_executor.models.qwen2_vl import Qwen2VLProcessingInfo
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargsItem,
                                    MultiModalKwargsItems, VideoItem)
from vllm.multimodal.parse import (ImageSize, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        PromptReplacement, PromptUpdate,
                                        PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.utils import is_list_of


class Qwen3VLProcessingInfo(Qwen2VLProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3VLConfig)

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_tokenizer(self):
        return self.ctx.tokenizer

    def get_image_processor(self,
                            **kwargs: object) -> Qwen2VLImageProcessorFast:
        return self.get_hf_processor(**kwargs).image_processor

    def get_video_processor(self, **kwargs: object) -> Qwen3VLVideoProcessor:
        return self.get_hf_processor(**kwargs).video_processor

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 2,
        do_resize: bool = True,
        image_processor: Optional[Qwen2VLImageProcessorFast],
    ) -> tuple[ImageSize, int]:
        if image_processor is None:
            image_processor = self.get_image_processor()

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        if do_resize:
            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=image_processor.size["shortest_edge"],
                max_pixels=image_processor.size["longest_edge"],
            )
            preprocessed_size = ImageSize(width=resized_width,
                                          height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width,
                                          height=image_height)

        padded_num_frames = num_frames + num_frames % temporal_patch_size

        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return preprocessed_size, num_vision_tokens

    def _calculate_timestamps(self, indices: list[int] | torch.Tensor,
                              video_fps: float, merge_size: int):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            # don't update metadata's frames_indices directly
            indices = indices + [indices[-1]
                                 ] * (merge_size - len(indices) % merge_size)
        timestamps = [idx / video_fps for idx in indices]
        timestamps = [(timestamps[i] + timestamps[i + merge_size - 1]) / 2
                      for i in range(0, len(timestamps), merge_size)]
        return timestamps

    def _get_video_second_idx(
            self,
            metadata: dict[str, Any],
            out_item: MultiModalKwargsItem,
            do_sample_frames: Optional[bool] = None,
            sampled_fps: Optional[float] = None) -> list[int]:
        video_processor = self.get_video_processor()
        merge_size = video_processor.merge_size
        indices = metadata["frames_indices"]

        # metadata["fps"] refers to the true fps of the input video.
        video_fps = metadata["fps"]
        if do_sample_frames is None:
            do_sample_frames = metadata.get("do_sample_frames", False)

        # If video frames are sampled in HF processor (instead of vLLM
        # video loader), we need to re-calculate the indices from original
        # metadata.
        if do_sample_frames:
            # here video_fps is the fps of the sampled video, and
            # metadata["fps"] refers to the fps of the original video.
            video_fps = sampled_fps if sampled_fps else video_processor.fps
            total_num_frames = metadata["total_num_frames"]
            num_frames = int(total_num_frames / metadata["fps"] * video_fps)
            num_frames = min(
                min(max(num_frames, video_processor.min_frames),
                    video_processor.max_frames), total_num_frames)
            indices = np.linspace(0, total_num_frames - 1,
                                  num_frames).round().astype(int).tolist()
        timestamps = self._calculate_timestamps(indices, video_fps, merge_size)
        return timestamps


class Qwen3VLDummyInputsBuilder(BaseDummyInputsBuilder[Qwen3VLProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        video_token = "<|vision_start|><|video_pad|><|vision_end|>"

        return image_token * num_images + video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = (
            self.info.get_image_size_with_most_features())
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts)
        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
            "video":
            self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
            ),
        }

    def _get_dummy_videos(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        num_videos: int,
    ) -> list[VideoItem]:
        num_frames = max(num_frames, 2)
        video = np.full((num_frames, width, height, 3), 255, dtype=np.uint8)
        video_items = []
        for i in range(num_videos):
            video_metadata = {
                "fps": 2.0,
                "duration": num_frames / 2.0,
                "total_num_frames": num_frames,
                "frames_indices": [i for i in range(num_frames)],
                "video_backend": "opencv",
                "do_sample_frames": False,
            }
            video_item = (video.copy(), video_metadata)
            video_items.append(video_item)
        return video_items


class Qwen3VLMultiModalProcessor(BaseMultiModalProcessor[Qwen3VLProcessingInfo]
                                 ):

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(video_needs_metadata=True)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        processor = self.info.get_hf_processor(**mm_kwargs)

        # Separate video processing from image processing. Because the videos
        # are processed into serval image patches
        if ("videos" in mm_data and isinstance(mm_data["videos"], list)
                and len(mm_data["videos"]) > 0):
            video_grid_thw_lst = []
            pixel_values_videos_lst = []

            for item_idx, item in enumerate(mm_data.pop("videos", [])):
                video_array, metadata = item

                # NOTE: @JJJYmmm new attr metadata.frames_indices indicates
                # the sampled frames indices of pre-sampled videos, which is
                # used to calculate the timestamps. Make sure that
                # do_sample_frames in mm_kwargs is false for presampled videos.

                # NOTE: a copy of is created to update do_sample_frames,
                # otherwise mm_hash for the object will be incorrect.
                video_mm_kwargs = dict(**mm_kwargs)
                if "do_sample_frames" not in video_mm_kwargs:
                    # qwen_vl_utils already has "do_sample_frames" in
                    # mm_kwargs, don't overwrite it.
                    video_mm_kwargs["do_sample_frames"] = metadata.get(
                        "do_sample_frames", False)

                metadata = VideoMetadata(**{
                    k: metadata[k]
                    for k in metadata if k != "do_sample_frames"
                })

                video_mm_data = dict()
                video_mm_data["videos"] = [[video_array]]
                video_mm_data["video_metadata"] = [[metadata]]

                video_outputs = super()._call_hf_processor(
                    prompt="<|vision_start|><|video_pad|><|vision_end|>",
                    mm_data=video_mm_data,
                    mm_kwargs=video_mm_kwargs,
                    tok_kwargs=tok_kwargs,
                )
                input_ids = video_outputs.pop("input_ids")
                video_placeholder = processor.tokenizer.batch_decode(
                    input_ids)[0]
                prompt = prompt.replace(
                    "<|vision_start|><|video_pad|><|vision_end|>",
                    video_placeholder,
                    1,
                )

                video_grid_thw_lst.append(video_outputs["video_grid_thw"])
                pixel_values_videos_lst.append(
                    video_outputs["pixel_values_videos"])
            video_outputs = dict(
                pixel_values_videos=torch.cat(pixel_values_videos_lst),
                video_grid_thw=torch.cat(video_grid_thw_lst),
            )
        else:
            video_outputs = dict()

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        combined_outputs = dict(
            processed_outputs,
            **video_outputs,
        )
        return BatchFeature(combined_outputs)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_grid_sizes = image_grid_thw.prod(-1)

        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        video_grid_sizes = video_grid_thw.prod(-1)

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes),
            image_embeds=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_sizes),
            video_embeds=MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_sizes),
            video_grid_thw=MultiModalFieldConfig.batched("video"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        hf_config = self.info.get_hf_config()

        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        vision_end_token_id = hf_config.vision_end_token_id

        merge_length = image_processor.merge_size**2

        def get_image_replacement_qwen3vl(item_idx: int):
            print(out_mm_kwargs)
            grid_thw = out_mm_kwargs["image_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [hf_processor.image_token_id] * num_tokens

        def get_video_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            video, metadata = mm_items["video"][item_idx]
            do_sample_frames = hf_processor_mm_kwargs.get("do_sample_frames")
            sampled_fps = hf_processor_mm_kwargs.get("fps")
            if is_list_of(sampled_fps, float):
                sampled_fps = sampled_fps[item_idx]
            timestamps = self.info._get_video_second_idx(
                metadata, out_item, do_sample_frames, sampled_fps)

            assert len(timestamps) == grid_thw[0], (
                f"The timestamps length({len(timestamps)}) should be equal "
                f"video length ({grid_thw[0]}).")

            frames_idx_token = [
                tokenizer.encode(f"<{curr_time:.1f} seconds>",
                                 add_special_tokens=False)
                for curr_time in timestamps
            ]
            num_tokens_per_frame = int(grid_thw[1:].prod()) // merge_length
            placeholder = []
            for frame_idx in frames_idx_token:
                placeholder.extend(frame_idx)
                placeholder.extend([vision_start_token_id] +
                                   [video_token_id] * num_tokens_per_frame +
                                   [vision_end_token_id])
            return PromptUpdateDetails.select_token_id(placeholder,
                                                       video_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=hf_processor.image_token,
                replacement=get_image_replacement_qwen3vl,
            ),

            # NOTE: We match string on purpose since searching sequence of
            # token ids takes more time.
            PromptReplacement(
                modality="video",
                target="<|vision_start|><|video_pad|><|vision_end|>",
                replacement=get_video_replacement_qwen3vl,
            ),
        ]
