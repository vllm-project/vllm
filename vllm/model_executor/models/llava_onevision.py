# SPDX-License-Identifier: Apache-2.0

import math
from functools import cached_property
from typing import (Final, Iterable, List, Literal, Mapping, Optional,
                    Protocol, Set, Tuple, TypedDict, Union)

import torch
import torch.nn as nn
from transformers import (BatchFeature, LlavaOnevisionConfig,
                          LlavaOnevisionProcessor)
from transformers.models.llava_onevision.modeling_llava_onevision import (
    get_anyres_image_grid_shape, unpad_image)
from typing_extensions import NotRequired

from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs,
                                    NestedTensors)
from vllm.multimodal.parse import (ImageSize, MultiModalDataItems,
                                   VideoEmbeddingItems, VideoProcessorItems)
from vllm.multimodal.processing import PromptReplacement
from vllm.multimodal.profiling import ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of

from .clip import CLIPVisionModel
from .interfaces import SupportsMultiModal, SupportsPP
from .llava import LlavaDummyInputsBuilder, init_vision_tower_for_llava
from .llava_next import (BaseLlavaNextMultiModalProcessor, LlavaNextLikeConfig,
                         LlavaNextProcessingInfo)
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, flatten_bn, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)

# For profile run
_MAX_FRAMES_PER_VIDEO = 16


class LlavaOnevisionVideoPixelInputs(TypedDict):
    type: Literal["pixel_values_videos"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """
    Shape: `(batch_size, num_videos, num_frames, num_channels, height, width)`

    Note that `num_videos` may be different for each batch, and 'num_frames'
    may be different for each video, in which case the data is passed as a
    list instead of a batched tensor.
    """


class LlavaOnevisionImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """
    Shape:
    `(batch_size * num_images, 1 + num_patches, num_channels, height, width)`

    Note that `num_patches` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """

    image_sizes: NotRequired[torch.Tensor]
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(height, width)` format.
    """


class LlavaOnevisionImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


LlavaOnevisionImageInputs = Union[LlavaOnevisionImagePixelInputs,
                                  LlavaOnevisionImageEmbeddingInputs]

LlavaOnevisionMultiInputs = Union[LlavaOnevisionImageInputs,
                                  LlavaOnevisionVideoPixelInputs]


class LlavaOnevisionLikeConfig(LlavaNextLikeConfig, Protocol):
    video_token_index: Final[int]


class LlavaOnevisionProcessingInfo(LlavaNextProcessingInfo):

    def get_hf_config(self) -> LlavaOnevisionLikeConfig:
        return self.ctx.get_hf_config(LlavaOnevisionConfig)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(LlavaOnevisionProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {
            "image": self.get_max_image_tokens(),
            "video": self.get_max_video_tokens(seq_len),
        }

    # Based on: https://github.com/huggingface/text-generation-inference/blob/v3.0.1/server/text_generation_server/models/vlm_causal_lm.py#L86
    # with additional logic afterwards taken from LlavaOnevisionProcessor
    def _get_num_unpadded_features(
        self,
        *,
        original_height: int,
        original_width: int,
        npatches: int,
        num_patch_height: int,
        num_patch_width: int,
    ) -> tuple[int, int]:
        current_height = npatches * num_patch_height
        current_width = npatches * num_patch_width

        aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height

        if aspect_ratio > current_aspect_ratio:
            new_height = (original_height * current_width) // original_width
            padding = (current_height - new_height) // 2
            current_height = current_height - (2 * padding)
        else:
            new_width = (original_width * current_height) // original_height
            padding = (current_width - new_width) // 2
            current_width = current_width - (2 * padding)

        unpadded_features = current_height * current_width
        newline_features = current_height

        ratio = math.sqrt(current_height * current_width / (9 * npatches**2))
        if ratio > 1.1:
            height_factor = int(current_height // ratio)
            width_factor = int(current_width // ratio)
            unpadded_features = height_factor * width_factor
            newline_features = height_factor

        return (unpadded_features, newline_features)

    def get_image_size_with_most_features(self) -> ImageSize:
        # NOTE: This hardcoded value is found via processor tests
        return ImageSize(width=1153, height=944)

    def _get_num_frame_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        spatial_pool_stride = getattr(hf_config, "spatial_pool_stride", 2)

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

        return num_frame_tokens * num_frames + 1  # Newline token

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

    def get_num_frames_with_most_features(self, seq_len: int) -> int:
        mm_config = self.ctx.get_mm_config()
        max_images = mm_config.limit_per_prompt.get("image", 1)
        max_videos = mm_config.limit_per_prompt.get("video", 1)

        max_image_tokens = self.get_max_image_tokens() * max_images
        max_total_frames = self._get_max_video_frames(seq_len -
                                                      max_image_tokens)
        max_frames_per_video = min(max_total_frames // max(max_videos, 1),
                                   _MAX_FRAMES_PER_VIDEO)

        return max(max_frames_per_video, 1)

    def get_max_video_tokens(self, seq_len: int) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_video_tokens(
            image_width=target_width,
            image_height=target_height,
            num_frames=self.get_num_frames_with_most_features(seq_len),
        )


class LlavaOnevisionDummyInputsBuilder(
        LlavaDummyInputsBuilder[LlavaOnevisionProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token
        video_token = processor.video_token

        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        target_num_frames = \
            self.info.get_num_frames_with_most_features(seq_len)

        mm_data = {
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
            )
        }

        return ProcessorInputs(
            prompt_text=image_token * num_images + video_token * num_videos,
            mm_data=mm_data,
        )


class LlavaOnevisionMultiModalProcessor(
        BaseLlavaNextMultiModalProcessor[LlavaOnevisionProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_sizes=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
            pixel_values_videos=MultiModalFieldConfig.batched("video"),
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        videos = mm_data.pop("videos", [])
        assert isinstance(videos, list)

        if not videos:
            return super()._call_hf_processor(
                prompt=prompt,
                mm_data=mm_data,
                mm_kwargs=mm_kwargs,
            )

        # LLaVA-OneVision processor doesn't support multiple videos
        # with different sizes when converting back to tensors
        # So, we process each component separately
        # NOTE: No prompt replacement is applied in this case
        processor = self.info.get_hf_processor()
        image_token = processor.image_token
        video_token = processor.video_token

        text_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data={},
            mm_kwargs=mm_kwargs,
        )

        images = mm_data.pop("images", [])
        assert isinstance(images, list)
        if images:
            processor_outputs = super()._call_hf_processor(
                prompt=image_token * len(images),
                mm_data={"images": images},
                mm_kwargs=mm_kwargs,
            )
            image_outputs = {
                k: v
                for k, v in processor_outputs.items()
                if k in ("pixel_values", "image_sizes")
            }
        else:
            image_outputs = {}

        pixel_values_videos = []
        for video in videos:
            item_outputs = super()._call_hf_processor(
                prompt=video_token,
                mm_data={"videos": video},
                mm_kwargs=mm_kwargs,
            )

            pixel_values_videos.append(item_outputs["pixel_values_videos"][0])

        video_outputs = {"pixel_values_videos": pixel_values_videos}

        combined_outputs = dict(
            text_outputs,
            **image_outputs,
            **video_outputs,
        )
        return BatchFeature(combined_outputs)

    def _hf_processor_applies_repl(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> bool:
        base_result = super()._hf_processor_applies_repl(
            prompt_text=prompt_text,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )

        return base_result and mm_items.get_count("video", strict=False) == 0

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        image_repls = super()._get_prompt_replacements(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            out_mm_kwargs=out_mm_kwargs,
        )

        hf_config = self.info.get_hf_config()
        video_token_id = hf_config.video_token_index

        def get_video_replacement(item_idx: int):
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

        return image_repls + [
            PromptReplacement(
                modality="video",
                target=[video_token_id],
                replacement=get_video_replacement,
            ),
        ]


class LlavaOnevisionMultiModalProjector(nn.Module):

    def __init__(self, config: LlavaOnevisionConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.vision_config.hidden_size,
                                  config.text_config.hidden_size,
                                  bias=config.multimodal_projector_bias)
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = nn.Linear(config.text_config.hidden_size,
                                  config.text_config.hidden_size,
                                  bias=config.multimodal_projector_bias)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    LlavaOnevisionMultiModalProcessor,
    info=LlavaOnevisionProcessingInfo,
    dummy_inputs=LlavaOnevisionDummyInputsBuilder)
class LlavaOnevisionForConditionalGeneration(nn.Module, SupportsMultiModal,
                                             SupportsPP):

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
        self.multi_modal_projector = LlavaOnevisionMultiModalProjector(config)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.image_newline = nn.Parameter(
            torch.empty(config.text_config.hidden_size))

        self.make_empty_intermediate_tensors = (
            self.language_model.model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _validate_image_sizes(self, data: torch.Tensor) -> torch.Tensor:
        expected_dims = (2, )

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    f"The expected shape of image sizes per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _validate_image_pixel_values(
        self, data: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape[1:])

            if actual_dims != expected_dims:
                expected_expr = ("num_patches", *map(str, expected_dims))
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaOnevisionImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if not isinstance(image_sizes, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image sizes. "
                                 f"Got type: {type(image_sizes)}")

            return LlavaOnevisionImagePixelInputs(
                type="pixel_values",
                data=self._validate_image_pixel_values(
                    flatten_bn(pixel_values)),
                image_sizes=self._validate_image_sizes(
                    flatten_bn(image_sizes, concat=True)),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeds. "
                                 f"Got type: {type(image_embeds)}")

            return LlavaOnevisionImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        raise AssertionError("This line should be unreachable.")

    def _validate_video_pixel_values(
        self, data: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape[2:])

            if actual_dims != expected_dims:
                expected_expr = ("num_frames", *map(str, expected_dims))
                raise ValueError(
                    "The expected shape of pixel values in each video frame "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_video_input(
            self,
            **kwargs: object) -> Optional[LlavaOnevisionVideoPixelInputs]:
        """
        A legal video input should have the following dimensions:
        {
            "pixel_values_videos" : 
                List[b, Tensor(nb_frames, nb_channels, height, width)]
        }
        """
        pixel_values = kwargs.pop("pixel_values_videos", None)

        if pixel_values is None:
            return None

        if not (is_list_of(pixel_values,
                           (torch.Tensor))  # different shape videos 
                or isinstance(pixel_values,
                              torch.Tensor)):  # same shape videos
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        return LlavaOnevisionVideoPixelInputs(
            type="pixel_values_videos",
            data=pixel_values,
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs)
            if input_key in ("pixel_values_videos",
                             "video_embeds") and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(
                    **kwargs)

        return modalities

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _image_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel, SiglipVisionModel],
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        image_features = vision_tower(pixel_values)
        return self._select_image_features(
            image_features,
            strategy=self.config.vision_feature_select_strategy,
        )

    # Based on: https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py
    def _merge_image_patch_embeddings(self,
                                      image_size: torch.Tensor,
                                      patch_embeddings: torch.Tensor,
                                      *,
                                      image_newline=None,
                                      vision_aspect_ratio="anyres_max_9",
                                      strategy: str) -> torch.Tensor:
        if strategy == "flat":
            return patch_embeddings.flatten(0, 1)

        if strategy.startswith("spatial"):
            height = width = self.config.vision_config.image_size \
                // self.config.vision_config.patch_size

            base_patch_embeds = patch_embeddings[0]
            if height * width != base_patch_embeds.shape[0]:
                raise ValueError(
                    "The number of patches is not consistent with the "
                    "image size.")

            if patch_embeddings.shape[0] > 1:
                other_patch_embeds = patch_embeddings[1:]

                # Move to CPU to avoid floating-point errors
                orig_height, orig_width = image_size.tolist()

                # image_aspect_ratio == "anyres"
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    (orig_height, orig_width),
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                num_patches = num_patch_height * num_patch_width

                # Image patches might be padded for batch processing
                other_patch_embeds = other_patch_embeds[:num_patches] \
                    .view(num_patch_height, num_patch_width, height, width, -1)

                if "unpad" in strategy:
                    other_patch_embeds = other_patch_embeds \
                        .permute(4, 0, 2, 1, 3).contiguous() \
                        .flatten(1, 2).flatten(2, 3)
                    other_patch_embeds = unpad_image(other_patch_embeds,
                                                     (orig_height, orig_width))
                    max_num_patches = int(
                        vision_aspect_ratio.removeprefix("anyres_max_"))
                    channels, curr_height, curr_width = other_patch_embeds.shape
                    ratio = math.sqrt(curr_height * curr_width /
                                      (max_num_patches * height**2))
                    if ratio > 1.1:
                        other_patch_embeds = other_patch_embeds[None]
                        other_patch_embeds = nn.functional.interpolate(
                            other_patch_embeds, [
                                int(curr_height // ratio),
                                int(curr_width // ratio)
                            ],
                            mode="bilinear")[0]
                    if image_newline is not None:
                        other_patch_embeds = torch.cat(
                            (
                                other_patch_embeds,
                                image_newline[:, None, None] \
                                .expand(*other_patch_embeds.shape[:-1], 1) \
                                .to(other_patch_embeds.device),
                            ),
                        dim=-1)
                    other_patch_embeds = other_patch_embeds \
                        .flatten(1, 2).transpose(0, 1)
                else:
                    other_patch_embeds = other_patch_embeds \
                        .permute(0, 2, 1, 3, 4).contiguous() \
                        .flatten(0, 3)

                merged_patch_embeddings = torch.cat(
                    (base_patch_embeds, other_patch_embeds), dim=0)
            else:
                if "unpad" in strategy:
                    merged_patch_embeddings = torch.cat(
                        (base_patch_embeds,
                         self.image_newline[None] \
                            .to(base_patch_embeds.device)
                    ), dim=0)
                else:
                    merged_patch_embeddings = base_patch_embeds

            return merged_patch_embeddings

        raise ValueError(f"Unexpected patch merge strategy: {strategy}")

    def _process_image_pixels(
        self,
        inputs: LlavaOnevisionImagePixelInputs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        assert self.vision_tower is not None

        pixel_values = inputs["data"]

        if isinstance(pixel_values, torch.Tensor):
            b, num_patches, c, h, w = pixel_values.shape
            stacked_pixel_values = pixel_values.view(b * num_patches, c, h, w)
            stacked_image_features = self._image_pixels_to_features(
                self.vision_tower, stacked_pixel_values)
            stacked_patch_embeddings = self.multi_modal_projector(
                stacked_image_features)

            return stacked_patch_embeddings.view(
                b, num_patches, *stacked_patch_embeddings.shape[1:])

        num_patches_per_batch = [v.shape[0] for v in pixel_values]
        stacked_pixel_values = torch.cat(pixel_values)
        stacked_image_features = self._image_pixels_to_features(
            self.vision_tower, stacked_pixel_values)

        return [
            self.multi_modal_projector(image_features) for image_features in
            torch.split(stacked_image_features, num_patches_per_batch)
        ]

    def _process_image_input(
        self,
        image_input: LlavaOnevisionImageInputs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if image_input["type"] == "image_embeds":
            return [image_input["data"]]

        patch_embeddings = self._process_image_pixels(image_input)

        image_sizes = image_input.get("image_sizes")
        if image_sizes is None:
            batch_size = len(image_input["data"])
            vision_config = self.config.vision_config
            default_height = default_width = vision_config.image_size
            image_sizes = torch.as_tensor([[default_height, default_width]
                                           for _ in range(batch_size)])

        return [
            self._merge_image_patch_embeddings(
                image_sizes[i],
                patch_features_batch,
                image_newline=self.image_newline,
                strategy="spatial_unpad")
            for i, patch_features_batch in enumerate(patch_embeddings)
        ]

    def _add_image_newline(
        self,
        video_features: torch.Tensor,
        videos: int = 1,
        frames: int = 1,
        strategy: str = "one_token",
    ) -> torch.Tensor:
        if strategy == "one_token":
            video_features = video_features.reshape(
                videos, frames * video_features.shape[1], -1)
            image_newline = self.image_newline[None, None, :].repeat(
                videos, 1, 1).to(video_features.device)
            video_features = torch.cat((video_features, image_newline), dim=1)
            return video_features
        raise ValueError(f"Unexpected video newline strategy: {strategy}")

    def _video_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel, SiglipVisionModel],
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        video_features = vision_tower(pixel_values)
        video_features = self._select_image_features(
            video_features,
            strategy=self.config.vision_feature_select_strategy,
        )
        video_features = self.multi_modal_projector(video_features)
        video_features = self.apply_pooling(video_features)
        return video_features

    def _process_video_pixels(self, inputs: LlavaOnevisionVideoPixelInputs):
        assert self.vision_tower is not None

        video_pixels = inputs["data"]

        if isinstance(video_pixels, torch.Tensor):
            b, num_videos, frames, c, h, w = video_pixels.shape
            pixel_values = video_pixels.view(b * num_videos * frames, c, h, w)
            stacked_embeddings = self._video_pixels_to_features(
                self.vision_tower, pixel_values)
            stacked_embeddings = self._add_image_newline(stacked_embeddings,
                                                         videos=b * num_videos,
                                                         frames=frames,
                                                         strategy="one_token")
            return stacked_embeddings
        elif is_list_of(video_pixels, torch.Tensor):
            stacked_embeddings = []
            for video_pixel in video_pixels:
                num_videos, frames, c, h, w = video_pixel.shape
                pixel_values = video_pixel.view(num_videos * frames, c, h, w)
                embeddings = self._video_pixels_to_features(
                    self.vision_tower, pixel_values)
                embeddings = self._add_image_newline(embeddings,
                                                     videos=num_videos,
                                                     frames=frames,
                                                     strategy="one_token")
                stacked_embeddings.append(embeddings)
            return stacked_embeddings
        else:
            raise ValueError(
                f"Unsupported type of video input {type(video_pixels)}")

    def apply_pooling(self, image_features, stride=2):
        vision_config = self.config.vision_config
        height = width = vision_config.image_size // vision_config.patch_size
        batch_frames, _, dim = image_features.shape
        image_features = image_features.view(batch_frames, height, width, -1)
        image_features = image_features.permute(0, 3, 1, 2)

        # TODO support other pooling types config
        height, width = image_features.shape[2:]
        scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
        image_feature = nn.functional.interpolate(image_features,
                                                  size=scaled_shape,
                                                  mode='bilinear')
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(batch_frames, -1, dim)
        return image_feature

    def get_multimodal_embeddings(
            self, **kwargs) -> Optional[tuple[torch.Tensor, ...]]:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += tuple(vision_embeddings)
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_video_pixels(video_input)
                multimodal_embeddings += tuple(video_embeddings)

        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.image_token_index, self.config.video_token_index])
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input: Optional[NestedTensors] = None,
        video_input: Optional[NestedTensors] = None,
    ) -> torch.Tensor:

        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=self.config.image_token_index,
            )

        if video_input is not None:
            video_embeds = self._process_video_pixels(video_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds,
                placeholder_token_id=self.config.video_token_index,
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for LlaVA-Onevision.
        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values_videos: Pixels in each frames for each input videos.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            if image_input is None and video_input is None:
                inputs_embeds = None
            else:
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    video_input=video_input)
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

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
