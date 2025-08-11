# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, Optional, Union

import torch
import torch.nn as nn
from transformers import (BatchFeature, LlavaNextVideoConfig,
                          LlavaNextVideoProcessor)

from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.models.clip import CLIPVisionModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import (ImageSize, MultiModalDataItems,
                                   VideoEmbeddingItems, VideoProcessorItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .llava import init_vision_tower_for_llava
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, WeightsMapper,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)
from .vision import get_vision_encoder_info


class LlavaNextVideoPixelInputs(TensorSchema):
    """    
    Dimensions:
        - bs: Batch size
        - nv: Number of videos
        - nf: Number of frames
        - nc: Number of channels (3)
        - h: Height of each frame
        - w: Width of each frame

    Note that `num_frames` may be different for each batch, in which case
    the data is passed as a list instead of a batched tensor.

    Note that it only supports one video input for one batch.
    """
    type: Literal["pixel_values_videos"] = "pixel_values_videos"

    data: Annotated[Union[torch.Tensor, list[torch.Tensor]],
                    TensorShape("bs", "nv", "nf", 3, "h", "w")]


class LlavaNextVideoProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(LlavaNextVideoConfig)

    def get_vision_encoder_info(self):
        return get_vision_encoder_info(self.get_hf_config())

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(LlavaNextVideoProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"video": 1}

    def get_image_size_with_most_features(self) -> ImageSize:
        vision_encoder_info = self.get_vision_encoder_info()
        width = height = vision_encoder_info.get_image_size()
        return ImageSize(width=width, height=height)

    def _get_num_frame_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        spatial_pool_stride = hf_config.spatial_pool_stride

        vision_encoder_info = self.get_vision_encoder_info()
        patch_grid_length = vision_encoder_info.get_patch_grid_length()
        pooled_grid_length = math.ceil(patch_grid_length / spatial_pool_stride)

        return pooled_grid_length * pooled_grid_length

    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
    ) -> int:
        num_frame_tokens = self._get_num_frame_tokens(
            image_width=image_width,
            image_height=image_height,
        )

        return num_frame_tokens * num_frames

    def _get_max_video_frames(self, max_tokens: int) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        num_frames = 0

        while True:
            next_num_frames = num_frames + 1
            next_max_tokens = self.get_num_video_tokens(
                image_width=target_width,
                image_height=target_height,
                num_frames=next_num_frames,
            )

            if next_max_tokens > max_tokens:
                break

            num_frames = next_num_frames

        return num_frames

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_videos = mm_counts.get("video", 0)

        max_total_frames = self._get_max_video_frames(seq_len)

        return max(max_total_frames // max(max_videos, 1), 1)


class LlavaNextVideoDummyInputsBuilder(
        BaseDummyInputsBuilder[LlavaNextVideoProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_videos = mm_counts.get("video", 0)

        processor = self.info.get_hf_processor()
        video_token = processor.video_token

        return video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        target_num_frames = \
            self.info.get_num_frames_with_most_features(seq_len, mm_counts)

        return {
            "video":
            self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
            )
        }


class LlavaNextVideoMultiModalProcessor(
        BaseMultiModalProcessor[LlavaNextVideoProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values_videos=MultiModalFieldConfig.batched("video"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        video_token_id = hf_config.video_token_index

        def get_replacement(item_idx: int):
            videos = mm_items.get_items(
                "video", (VideoEmbeddingItems, VideoProcessorItems))

            if isinstance(videos, VideoEmbeddingItems):
                num_video_tokens = videos.get_feature_size(item_idx)
            else:
                image_size = videos.get_frame_size(item_idx)
                num_video_tokens = self.info.get_num_video_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    num_frames=videos.get_num_frames(item_idx),
                )

            return [video_token_id] * num_video_tokens

        return [
            PromptReplacement(
                modality="video",
                target=[video_token_id],
                replacement=get_replacement,
            ),
        ]


# adopted from transformers modeling_llava_next_video.py
class LlavaNextVideoPooler(nn.Module):

    def __init__(self, config: LlavaNextVideoConfig):
        super().__init__()

        mode = config.spatial_pool_mode
        stride = config.spatial_pool_stride
        image_size = config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.image_size = image_size // patch_size**2

        if mode == "average":
            self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)
        elif mode == "max":
            self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        else:
            # TODO: Support Conv2d pooling layer, need to load weights
            raise ValueError(
                f"Unknown pooling mode: {mode}. Expected [`average`, `max`]")

    def forward(self, image_features: torch.Tensor):
        ori_width = int(
            math.sqrt(image_features.shape[1] * self.image_size //
                      self.image_size))
        ori_height = int(ori_width * self.image_size // self.image_size)

        batch_size, _, dim = image_features.shape
        image_features_spatial = image_features \
            .view(batch_size, ori_height, ori_height, dim) \
            .permute(0, 3, 1, 2)
        image_features_spatial = self.pool(image_features_spatial)

        return image_features_spatial.flatten(2).transpose(1, 2).contiguous()


class LlavaNextMultiModalProjector(nn.Module):

    def __init__(self, vision_hidden_size: int, text_hidden_size: int,
                 projector_hidden_act: str, multimodal_projector_bias: bool):
        super().__init__()

        self.linear_1 = nn.Linear(vision_hidden_size,
                                  text_hidden_size,
                                  bias=multimodal_projector_bias)
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = nn.Linear(text_hidden_size,
                                  text_hidden_size,
                                  bias=multimodal_projector_bias)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    LlavaNextVideoMultiModalProcessor,
    info=LlavaNextVideoProcessingInfo,
    dummy_inputs=LlavaNextVideoDummyInputsBuilder,
)
class LlavaNextVideoForConditionalGeneration(nn.Module, SupportsMultiModal,
                                             SupportsPP):

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "model.image_newline": "image_newline",
            "lm_head.": "language_model.lm_head.",
        })

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<image>"
        if modality.startswith("video"):
            return "<video>"

        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # Initialize the vision tower only up to the required feature layer
        self.vision_tower = init_vision_tower_for_llava(
            config,
            quant_config,
            require_post_norm=False,
            prefix=maybe_prefix(prefix, "vision_tower"))
        self.vision_resampler = LlavaNextVideoPooler(config)
        self.multi_modal_projector = LlavaNextMultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act,
            multimodal_projector_bias=config.multimodal_projector_bias)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.model.make_empty_intermediate_tensors)

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> Optional[LlavaNextVideoPixelInputs]:
        """
        A legal video input should have the following dimensions:
        {
            "pixel_values_videos" : 
                list[b, Tensor(nb_frames, nb_channels, height, width)]
        }
        """
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)

        if pixel_values_videos is None:
            return None

        expected_h = expected_w = self.config.vision_config.image_size
        return LlavaNextVideoPixelInputs(type="pixel_values_videos",
                                         data=pixel_values_videos,
                                         resolve_bindings={
                                             "h": expected_h,
                                             "w": expected_w,
                                         })

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _video_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel, SiglipVisionModel],
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        image_features = vision_tower(pixel_values)
        image_features = self._select_image_features(
            image_features,
            strategy=self.config.vision_feature_select_strategy,
        )
        image_features = self.vision_resampler(image_features)
        image_features = self.multi_modal_projector(image_features)
        return image_features

    def _process_video_pixels(self, inputs: LlavaNextVideoPixelInputs):
        assert self.vision_tower is not None

        video_pixels = inputs["data"]

        if isinstance(video_pixels, torch.Tensor):
            # TODO: support multiple videos per input
            b, num_videos, num_frames, c, h, w = video_pixels.shape
            assert (num_videos == 1)
            stacked_pixels = video_pixels.view(b * num_videos * num_frames, c,
                                               h, w)
            stacked_embeddings = self._video_pixels_to_features(
                self.vision_tower, stacked_pixels)
            embeds = stacked_embeddings.view(b, num_frames,
                                             *stacked_embeddings.shape[1:])

        elif is_list_of(video_pixels, torch.Tensor):
            frames_per_videos = [v.shape[0] for v in video_pixels]
            stacked_pixels = torch.cat(video_pixels, dim=0)
            stacked_embeddings = self._video_pixels_to_features(
                self.vision_tower, stacked_pixels)
            embeds = torch.split(stacked_embeddings, frames_per_videos, dim=0)
        else:
            raise ValueError(
                f"Unsupported type of video input {type(video_pixels)}")

        return [e.flatten(0, 1) for e in embeds]

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        video_input = self._parse_and_validate_video_input(**kwargs)
        if video_input is None:
            return []
        vision_embeddings = self._process_video_pixels(video_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.video_token_index)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for LlaVA-NeXT-Video.
        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values_videos: Pixels in each frames for each input videos.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            # This model doesn't support images for now
            ignore_unexpected_prefixes=["image_newline"],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
