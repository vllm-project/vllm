# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Final, Literal, Protocol, TypeAlias

import torch
import torch.nn as nn
from transformers import BatchFeature, LlavaOnevisionConfig, LlavaOnevisionProcessor
from transformers.models.llava_onevision.modeling_llava_onevision import (
    get_anyres_image_grid_shape,
    unpad_image,
)

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.activation import get_act_fn
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageSize,
    MultiModalDataItems,
    VideoEmbeddingItems,
    VideoProcessorItems,
)
from vllm.multimodal.processing import PromptReplacement, PromptUpdate
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .clip import CLIPVisionModel
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .llava import LlavaDummyInputsBuilder, init_vision_tower_for_llava
from .llava_next import (
    BaseLlavaNextMultiModalProcessor,
    LlavaNextLikeConfig,
    LlavaNextProcessingInfo,
)
from .siglip import SiglipVisionModel
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

# For profile run
_MAX_FRAMES_PER_VIDEO = 16


class LlavaOnevisionVideoPixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of videos
        - f: Number of frames
        - c: Number of channels (3)
        - h: Height
        - w: Width

        Note that `f` may be different for each batch, and 'num_frames'
        may be different for each video, in which case the data is passed as a
        list instead of a batched tensor.
    """

    type: Literal["pixel_values_videos"] = "pixel_values_videos"

    pixel_values_videos: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "f", 3, "h", "w", dynamic_dims={"f"}),
    ]


class LlavaOnevisionImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - np: Number of patches (1 + num_patches)
        - c: Number of channels (3)
        - h: Height
        - w: Width

        Note that `num_patches` may be different per batch and image,
        in which case the data is passed as a list instead of a batched tensor.
    """

    type: Literal["pixel_values"] = "pixel_values"

    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "np", 3, "h", "w", dynamic_dims={"np"}),
    ]

    image_sizes: Annotated[torch.Tensor | None, TensorShape("bn", 2)]


class LlavaOnevisionImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - ifs: Image feature size
        - hs: Hidden size (must match language model backbone)
    """

    type: Literal["image_embeds"] = "image_embeds"

    data: Annotated[
        torch.Tensor,
        TensorShape("bn", "ifs", "hs"),
    ]


LlavaOnevisionImageInputs: TypeAlias = (
    LlavaOnevisionImagePixelInputs | LlavaOnevisionImageEmbeddingInputs
)

LlavaOnevisionMultiInputs: TypeAlias = (
    LlavaOnevisionImageInputs | LlavaOnevisionVideoPixelInputs
)


class LlavaOnevisionLikeConfig(LlavaNextLikeConfig, Protocol):
    video_token_index: Final[int]


class LlavaOnevisionProcessingInfo(LlavaNextProcessingInfo):
    def get_hf_config(self) -> LlavaOnevisionLikeConfig:
        return self.ctx.get_hf_config(LlavaOnevisionConfig)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(LlavaOnevisionProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

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
            new_height = int(
                round(original_height * (current_width / original_width), 7)
            )
            padding = (current_height - new_height) // 2
            current_height = current_height - (2 * padding)
        else:
            new_width = int(
                round(original_width * (current_height / original_height), 7)
            )
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

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_videos = mm_counts.get("video", 0)

        max_total_frames = self._get_max_video_frames(seq_len)
        max_frames_per_video = min(
            max_total_frames // max(max_videos, 1), _MAX_FRAMES_PER_VIDEO
        )

        return max(max_frames_per_video, 1)

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_video_tokens(
            image_width=target_width,
            image_height=target_height,
            num_frames=self.get_num_frames_with_most_features(seq_len, mm_counts),
        )


class LlavaOnevisionDummyInputsBuilder(
    LlavaDummyInputsBuilder[LlavaOnevisionProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token
        video_token = processor.video_token

        return image_token * num_images + video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts
        )

        image_overrides = mm_options.get("image") if mm_options else None
        video_overrides = mm_options.get("video") if mm_options else None

        return {
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


class LlavaOnevisionMultiModalProcessor(
    BaseLlavaNextMultiModalProcessor[LlavaOnevisionProcessingInfo]
):
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
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        videos = mm_data.pop("videos", [])
        assert isinstance(videos, list)

        if not videos:
            return super()._call_hf_processor(
                prompt=prompt,
                mm_data=mm_data,
                mm_kwargs=mm_kwargs,
                tok_kwargs=tok_kwargs,
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
            tok_kwargs=tok_kwargs,
        )

        images = mm_data.pop("images", [])
        assert isinstance(images, list)
        if images:
            processor_outputs = super()._call_hf_processor(
                prompt=image_token * len(images),
                mm_data={"images": images},
                mm_kwargs=mm_kwargs,
                tok_kwargs=tok_kwargs,
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
                tok_kwargs=tok_kwargs,
            )

            pixel_values_videos.append(item_outputs["pixel_values_videos"][0])

        video_outputs = {"pixel_values_videos": pixel_values_videos}

        combined_outputs = dict(
            text_outputs,
            **image_outputs,
            **video_outputs,
        )
        return BatchFeature(combined_outputs)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        base_result = super()._hf_processor_applies_updates(
            prompt_text=prompt_text,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

        return base_result and mm_items.get_count("video", strict=False) == 0

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_repls = super()._get_prompt_updates(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            out_mm_kwargs=out_mm_kwargs,
        )

        hf_config = self.info.get_hf_config()
        video_token_id = hf_config.video_token_index

        def get_video_replacement(item_idx: int):
            videos = mm_items.get_items(
                "video", (VideoEmbeddingItems, VideoProcessorItems)
            )

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
            *image_repls,
            PromptReplacement(
                modality="video",
                target=[video_token_id],
                replacement=get_video_replacement,
            ),
        ]


class LlavaOnevisionMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaOnevisionConfig):
        super().__init__()

        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    LlavaOnevisionMultiModalProcessor,
    info=LlavaOnevisionProcessingInfo,
    dummy_inputs=LlavaOnevisionDummyInputsBuilder,
)
class LlavaOnevisionForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "model.image_newline": "image_newline",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
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

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            # Initialize the vision tower only up to the required feature layer
            self.vision_tower = init_vision_tower_for_llava(
                config,
                quant_config=quant_config,
                multimodal_config=multimodal_config,
                require_post_norm=False,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.image_newline = nn.Parameter(
                torch.empty(config.text_config.hidden_size)
            )
            self.multi_modal_projector = LlavaOnevisionMultiModalProjector(config)

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> LlavaOnevisionImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return LlavaOnevisionImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                resolve_bindings={
                    "h": self.config.vision_config.image_size,
                    "w": self.config.vision_config.image_size,
                },
            )

        if image_embeds is not None:
            return LlavaOnevisionImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        raise AssertionError("This line should be unreachable.")

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> LlavaOnevisionVideoPixelInputs | None:
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

        return LlavaOnevisionVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            resolve_bindings={
                "h": self.config.vision_config.image_size,
                "w": self.config.vision_config.image_size,
            },
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

        return mm_input_by_modality

    def _image_pixels_to_features(
        self,
        vision_tower: CLIPVisionModel | SiglipVisionModel,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        return vision_tower(
            pixel_values,
            feature_select_strategy=self.config.vision_feature_select_strategy,
        )

    # Based on: https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py
    def _merge_image_patch_embeddings(
        self,
        image_size: torch.Tensor,
        patch_embeddings: torch.Tensor,
        *,
        image_newline=None,
        vision_aspect_ratio="anyres_max_9",
        strategy: str,
    ) -> torch.Tensor:
        if strategy == "flat":
            return patch_embeddings.flatten(0, 1)

        if strategy.startswith("spatial"):
            height = width = (
                self.config.vision_config.image_size
                // self.config.vision_config.patch_size
            )

            base_patch_embeds = patch_embeddings[0]
            if height * width != base_patch_embeds.shape[0]:
                raise ValueError(
                    "The number of patches is not consistent with the image size."
                )

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
                other_patch_embeds = other_patch_embeds[:num_patches].view(
                    num_patch_height, num_patch_width, height, width, -1
                )

                if "unpad" in strategy:
                    other_patch_embeds = (
                        other_patch_embeds.permute(4, 0, 2, 1, 3)
                        .contiguous()
                        .flatten(1, 2)
                        .flatten(2, 3)
                    )
                    other_patch_embeds = unpad_image(
                        other_patch_embeds, (orig_height, orig_width)
                    )
                    max_num_patches = int(
                        vision_aspect_ratio.removeprefix("anyres_max_")
                    )
                    channels, curr_height, curr_width = other_patch_embeds.shape
                    ratio = math.sqrt(
                        curr_height * curr_width / (max_num_patches * height**2)
                    )
                    if ratio > 1.1:
                        other_patch_embeds = other_patch_embeds[None]
                        other_patch_embeds = nn.functional.interpolate(
                            other_patch_embeds,
                            [int(curr_height // ratio), int(curr_width // ratio)],
                            mode="bilinear",
                        )[0]
                    if image_newline is not None:
                        other_patch_embeds = torch.cat(
                            (
                                other_patch_embeds,
                                image_newline[:, None, None]
                                .expand(*other_patch_embeds.shape[:-1], 1)
                                .to(other_patch_embeds.device),
                            ),
                            dim=-1,
                        )
                    other_patch_embeds = other_patch_embeds.flatten(1, 2).transpose(
                        0, 1
                    )
                else:
                    other_patch_embeds = (
                        other_patch_embeds.permute(0, 2, 1, 3, 4)
                        .contiguous()
                        .flatten(0, 3)
                    )

                merged_patch_embeddings = torch.cat(
                    (base_patch_embeds, other_patch_embeds), dim=0
                )
            else:
                if "unpad" in strategy:
                    merged_patch_embeddings = torch.cat(
                        (
                            base_patch_embeds,
                            self.image_newline[None].to(base_patch_embeds.device),
                        ),
                        dim=0,
                    )
                else:
                    merged_patch_embeddings = base_patch_embeds

            return merged_patch_embeddings

        raise ValueError(f"Unexpected patch merge strategy: {strategy}")

    def _process_image_pixels(
        self,
        inputs: LlavaOnevisionImagePixelInputs,
    ) -> torch.Tensor | list[torch.Tensor]:
        pixel_values = inputs["pixel_values"]

        if isinstance(pixel_values, torch.Tensor):
            b, num_patches, c, h, w = pixel_values.shape
            stacked_pixel_values = pixel_values.view(b * num_patches, c, h, w)
            stacked_image_features = self._image_pixels_to_features(
                self.vision_tower, stacked_pixel_values
            )
            stacked_patch_embeddings = self.multi_modal_projector(
                stacked_image_features
            )

            return stacked_patch_embeddings.view(
                b, num_patches, *stacked_patch_embeddings.shape[1:]
            )

        num_patches_per_batch = [v.shape[0] for v in pixel_values]
        stacked_pixel_values = torch.cat(pixel_values)
        stacked_image_features = self._image_pixels_to_features(
            self.vision_tower, stacked_pixel_values
        )

        return [
            self.multi_modal_projector(image_features)
            for image_features in torch.split(
                stacked_image_features, num_patches_per_batch
            )
        ]

    def _process_image_input(
        self,
        image_input: LlavaOnevisionImageInputs,
    ) -> torch.Tensor | list[torch.Tensor]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        patch_embeddings = self._process_image_pixels(image_input)

        image_sizes = image_input.get("image_sizes")
        if image_sizes is None:
            batch_size = len(image_input["pixel_values"])
            vision_config = self.config.vision_config
            default_height = default_width = vision_config.image_size
            image_sizes = torch.as_tensor(
                [[default_height, default_width] for _ in range(batch_size)]
            )

        return [
            self._merge_image_patch_embeddings(
                image_sizes[i],
                patch_features_batch,
                image_newline=self.image_newline,
                strategy="spatial_unpad",
            )
            for i, patch_features_batch in enumerate(patch_embeddings)
        ]

    def _video_pixels_to_features(
        self,
        vision_tower: CLIPVisionModel | SiglipVisionModel,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        video_features = vision_tower(
            pixel_values,
            feature_select_strategy=self.config.vision_feature_select_strategy,
        )
        video_features = self.multi_modal_projector(video_features)
        video_features = self.apply_pooling(video_features)
        return video_features

    def _process_video_pixels(self, inputs: LlavaOnevisionVideoPixelInputs):
        video_pixels = inputs["pixel_values_videos"]

        if isinstance(video_pixels, torch.Tensor):
            total_videos, frames, c, h, w = video_pixels.shape
            video_pixels_flat = video_pixels.view(total_videos * frames, c, h, w)

            embeddings_flat = self._video_pixels_to_features(
                self.vision_tower, video_pixels_flat
            )

            embeddings_flat = embeddings_flat.reshape(
                total_videos, frames * embeddings_flat.shape[1], -1
            )

            image_newline = self.image_newline[None, None, :].expand(
                total_videos, -1, -1
            )
            return torch.cat((embeddings_flat, image_newline), dim=1)

        frames_per_video = [len(video) for video in video_pixels]
        video_pixels_flat = torch.cat(video_pixels)

        embeddings_flat = self._video_pixels_to_features(
            self.vision_tower, video_pixels_flat
        )

        image_newline = self.image_newline[None, None, :]

        return [
            torch.cat(
                (
                    embeds.reshape(1, num_frame * embeddings_flat.shape[1], -1),
                    image_newline,
                ),
                dim=1,
            )
            for num_frame, embeds in zip(
                frames_per_video,
                torch.split(embeddings_flat, frames_per_video),
            )
        ]

    def apply_pooling(self, image_features: torch.Tensor, stride: int = 2):
        vision_config = self.config.vision_config
        height = width = vision_config.image_size // vision_config.patch_size
        batch_frames, _, dim = image_features.shape
        image_features = image_features.view(batch_frames, height, width, -1)
        image_features = image_features.permute(0, 3, 1, 2)

        # TODO support other pooling types config
        height, width = image_features.shape[2:]
        scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
        image_feature = nn.functional.interpolate(
            image_features, size=scaled_shape, mode="bilinear"
        )
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(batch_frames, -1, dim)
        return image_feature

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []
            return None

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
                video_embeddings = self._process_video_pixels(multimodal_input)
                multimodal_embeddings += tuple(video_embeddings)

        return multimodal_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for LlaVA-Onevision.
        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values_videos: Pixels in each frames for each input videos.
        """
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
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
