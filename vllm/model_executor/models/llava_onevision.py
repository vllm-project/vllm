# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, Protocol, TypeAlias, TypedDict, cast

import torch
import torch.nn as nn
from transformers import BatchFeature, LlavaOnevisionConfig, LlavaOnevisionProcessor
from transformers.models.llava_onevision.modeling_llava_onevision import (
    get_anyres_image_grid_shape,
    unpad_image,
)

from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import get_act_fn
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
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
from vllm.v1.worker.encoder_cudagraph_defs import ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG

from .clip import CLIPVisionModel
from .interfaces import (
    MultiModalEmbeddings,
    SupportsEncoderCudaGraph,
    SupportsMultiModal,
    SupportsPP,
)
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
from .vision import get_num_selected_vision_tokens, get_vision_encoder_info

# For profile run
_MAX_FRAMES_PER_VIDEO = 16


class LlavaOnevisionVideoPixelInputs(TensorSchema):
    """Dimensions:
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
    """Dimensions:
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
    """Dimensions:
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
    video_token_index: int


class LlavaOnevisionInputsByModality(TypedDict, total=False):
    image: LlavaOnevisionImageInputs | None
    video: LlavaOnevisionVideoPixelInputs | None


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
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        target_width, target_height = self.info.get_image_size_with_most_features()
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts
        )

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
            "video": self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=mm_counts.get("video", 0),
                overrides=mm_options.get("video"),
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
            image_sizes=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            image_embeds=MultiModalFieldConfig.batched("image"),
            pixel_values_videos=MultiModalFieldConfig.batched("video"),
        )

    def _get_hf_mm_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.dummy_inputs.get_dummy_text(mm_counts)

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        hf_data, hf_kwargs, passthrough_data = self._get_hf_mm_inputs(
            mm_items, hf_kwargs
        )

        if not hf_data:
            return self._finalize_hf_mm_data(hf_data, hf_kwargs, passthrough_data)

        prompt_text = hf_data.pop("text")
        assert isinstance(prompt_text, str)

        videos = hf_data.pop("videos", [])
        assert isinstance(videos, list)

        if not videos:
            processed_data = self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**hf_kwargs),
                dict(text=prompt_text, **hf_data),
                hf_kwargs,
            )
            return self._finalize_hf_mm_data(
                hf_data, hf_kwargs, passthrough_data, processed_data
            )

        # LLaVA-OneVision processor doesn't support multiple videos
        # with different sizes when converting back to tensors
        # So, we process each component separately
        # NOTE: No prompt replacement is applied in this case
        processor = self.info.get_hf_processor()
        image_token = processor.image_token
        video_token = processor.video_token

        images = hf_data.pop("images", [])
        assert isinstance(images, list)
        if images:
            processor_outputs = self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**hf_kwargs),
                dict(text=image_token * len(images), **{"images": images}),
                hf_kwargs,
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
            item_outputs = self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**hf_kwargs),
                dict(text=video_token, **{"videos": video}),
                hf_kwargs,
            )

            pixel_values_videos.append(item_outputs["pixel_values_videos"][0])

        video_outputs = {"pixel_values_videos": pixel_values_videos}

        combined_outputs = dict(
            input_ids=prompt_text,
            **image_outputs,
            **video_outputs,
        )
        processed_data = BatchFeature(combined_outputs)
        return self._finalize_hf_mm_data(
            hf_data, hf_kwargs, passthrough_data, processed_data
        )

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
                assert isinstance(videos, VideoProcessorItems)
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
class LlavaOnevisionForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsEncoderCudaGraph
):
    supports_encoder_cudagraph = True
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

    def _get_image_output_tokens(self, image_size: torch.Tensor) -> int:
        orig_height, orig_width = [int(v) for v in image_size.tolist()]
        grid_h, grid_w = get_anyres_image_grid_shape(
            (orig_height, orig_width),
            self.config.image_grid_pinpoints,
            self.config.vision_config.image_size,
        )
        patch_grid = (
            self.config.vision_config.image_size // self.config.vision_config.patch_size
        )
        current_height = patch_grid * grid_h
        current_width = patch_grid * grid_w

        aspect_ratio = orig_width / orig_height
        current_aspect_ratio = current_width / current_height
        if aspect_ratio > current_aspect_ratio:
            new_height = int(round(orig_height * current_width / orig_width, 7))
            current_height -= 2 * ((current_height - new_height) // 2)
        else:
            new_width = int(round(orig_width * current_height / orig_height, 7))
            current_width -= 2 * ((current_width - new_width) // 2)

        ratio = math.sqrt(current_height * current_width / (9 * patch_grid**2))
        if ratio > 1.1:
            current_height = int(current_height // ratio)
            current_width = int(current_width // ratio)

        return patch_grid**2 + current_height * current_width + current_height

    def _get_video_output_tokens(self, num_frames: int) -> int:
        vision_config = self.config.vision_config
        patch_grid = vision_config.image_size // vision_config.patch_size
        stride = getattr(self.config, "spatial_pool_stride", 2)
        pooled_grid = math.ceil(patch_grid / stride)
        return num_frames * pooled_grid**2 + 1

    def get_input_modality(self, mm_kwargs: dict[str, object]) -> str:
        return "video" if "pixel_values_videos" in mm_kwargs else "image"

    def _encoder_tokens_per_tile(self) -> int:
        encoder_info = get_vision_encoder_info(self.config)
        tile_size = encoder_info.get_image_size()
        tokens_per_tile = encoder_info.get_num_image_tokens(
            image_width=tile_size,
            image_height=tile_size,
        )
        return get_num_selected_vision_tokens(
            tokens_per_tile,
            self.config.vision_feature_select_strategy,
        )

    def _get_pixel_items(
        self, mm_kwargs: dict[str, Any], modality: str
    ) -> list[torch.Tensor]:
        key = "pixel_values" if modality == "image" else "pixel_values_videos"
        pixel_values = mm_kwargs[key]
        if isinstance(pixel_values, torch.Tensor):
            return list(pixel_values.unbind(0))
        return list(pixel_values)

    def _get_image_tile_counts(
        self,
        pixel_items: list[torch.Tensor],
        image_sizes: torch.Tensor | None,
    ) -> list[int]:
        if image_sizes is None:
            return [int(item.shape[0]) for item in pixel_items]

        counts: list[int] = []
        for item, image_size in zip(pixel_items, image_sizes):
            orig_height, orig_width = [int(v) for v in image_size.tolist()]
            grid_h, grid_w = get_anyres_image_grid_shape(
                (orig_height, orig_width),
                self.config.image_grid_pinpoints,
                self.config.vision_config.image_size,
            )
            count = 1 + grid_h * grid_w
            if count > item.shape[0]:
                raise ValueError(
                    f"Image needs {count} vision tiles, but only "
                    f"{item.shape[0]} were provided."
                )
            counts.append(count)
        return counts

    def _encoder_cudagraph_token_budgets(self) -> list[int]:
        vllm_config = getattr(self, "vllm_config", None)
        if vllm_config is None:
            return []

        comp_config = vllm_config.compilation_config
        user_budgets = list(comp_config.encoder_cudagraph_token_budgets or [])
        if user_budgets:
            return sorted(user_budgets)

        min_budget, max_budget = self.get_encoder_cudagraph_budget_range(vllm_config)
        user_max_vision_items = getattr(
            comp_config, "encoder_cudagraph_max_vision_items_per_batch", 0
        )
        effective_min = (
            max(min_budget, user_max_vision_items)
            if user_max_vision_items > 0
            else min_budget
        )
        budgets: list[int] = []
        budget = effective_min
        while budget <= max_budget:
            budgets.append(budget)
            budget *= 2
        if not budgets or budgets[-1] < max_budget:
            budgets.append(max_budget)
        return budgets

    def _encoder_padding_axis_values(self) -> tuple[int, ...]:
        budgets = self._encoder_cudagraph_token_budgets()
        if not budgets:
            return (0,)

        tokens_per_tile = self._encoder_tokens_per_tile()
        max_padding = 0
        previous_budget = 0
        for budget in budgets:
            bucket_tiles = (budget + tokens_per_tile - 1) // tokens_per_tile
            min_tiles = previous_budget // tokens_per_tile + 1
            max_padding = max(max_padding, bucket_tiles - min_tiles)
            previous_budget = budget
        return tuple(range(max_padding + 1))

    def _encoder_padding_axis(self, total_tiles: int) -> int:
        budgets = self._encoder_cudagraph_token_budgets()
        if not budgets:
            return 0
        tokens_per_tile = self._encoder_tokens_per_tile()
        raw_tokens = total_tiles * tokens_per_tile
        budget = next((b for b in budgets if b >= raw_tokens), None)
        if budget is None:
            return 0
        bucket_tiles = (budget + tokens_per_tile - 1) // tokens_per_tile
        return max(bucket_tiles - total_tiles, 0)

    def get_encoder_cudagraph_config(self):
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

        return EncoderCudaGraphConfig(
            modalities=["image", "video"],
            buffer_keys=["pixel_values"],
            out_hidden_size=self.config.text_config.hidden_size,
            max_frames_per_video=_MAX_FRAMES_PER_VIDEO,
            capture_axes=(self._encoder_padding_axis_values(),)
            if hasattr(self, "vllm_config")
            else (),
        )

    def get_encoder_cudagraph_budget_range(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int]:
        min_budget = self._encoder_tokens_per_tile()
        max_budget = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.model_config.max_model_len,
        )
        return min_budget, max(min_budget, max_budget)

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]):
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderItemSpec

        modality = self.get_input_modality(mm_kwargs)
        pixel_items = self._get_pixel_items(mm_kwargs, modality)
        tokens_per_tile = self._encoder_tokens_per_tile()

        if modality == "video":
            return [
                EncoderItemSpec(
                    input_size=int(item.shape[0]),
                    output_tokens=self._get_video_output_tokens(int(item.shape[0])),
                    path_output_tokens={
                        "default": int(item.shape[0]) * tokens_per_tile
                    },
                )
                for item in pixel_items
            ]

        image_sizes = mm_kwargs.get("image_sizes")
        tile_counts = self._get_image_tile_counts(pixel_items, image_sizes)
        if image_sizes is None:
            default_size = self.config.vision_config.image_size
            default_image_size = torch.tensor([default_size, default_size])
            multi_tile_tokens = self._get_image_output_tokens(default_image_size)
            output_tokens = [
                tokens_per_tile + 1 if tile_count == 1 else multi_tile_tokens
                for tile_count in tile_counts
            ]
        else:
            output_tokens = [
                self._get_image_output_tokens(image_size) for image_size in image_sizes
            ]

        return [
            EncoderItemSpec(
                input_size=tile_count,
                output_tokens=num_output_tokens,
                path_output_tokens={"default": tile_count * tokens_per_tile},
            )
            for tile_count, num_output_tokens in zip(tile_counts, output_tokens)
        ]

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        modality = self.get_input_modality(mm_kwargs)
        pixel_items = self._get_pixel_items(mm_kwargs, modality)

        if modality == "video":
            selected_videos = [pixel_items[i] for i in indices]
            selected: dict[str, Any] = {"pixel_values_videos": selected_videos}
            if hasattr(self, "vllm_config"):
                total_tiles = sum(int(item.shape[0]) for item in selected_videos)
                selected[ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG] = (
                    self._encoder_padding_axis(total_tiles),
                )
            return selected

        image_sizes = mm_kwargs.get("image_sizes")
        tile_counts = self._get_image_tile_counts(pixel_items, image_sizes)
        selected_pixels = [pixel_items[i][: tile_counts[i]] for i in indices]
        selected_sizes = None if image_sizes is None else image_sizes[indices]
        selected = {
            "pixel_values": selected_pixels,
            "image_sizes": selected_sizes,
        }
        if hasattr(self, "vllm_config"):
            total_tiles = sum(int(item.shape[0]) for item in selected_pixels)
            selected[ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG] = (
                self._encoder_padding_axis(total_tiles),
            )
        return selected

    def get_max_frames_per_video(self) -> int:
        return _MAX_FRAMES_PER_VIDEO

    def _flatten_encoder_pixels(self, mm_kwargs: dict[str, Any]) -> torch.Tensor:
        modality = self.get_input_modality(mm_kwargs)
        pixel_items = self._get_pixel_items(mm_kwargs, modality)
        if pixel_items:
            return torch.cat(pixel_items, dim=0)

        image_size = self.config.vision_config.image_size
        return torch.empty(
            (0, 3, image_size, image_size),
            device=self.image_newline.device,
            dtype=self.image_newline.dtype,
        )

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Any, ...] | None = None,
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphCaptureInputs

        del max_batch_size, max_frames_per_batch
        assert path == "default"
        tokens_per_tile = self._encoder_tokens_per_tile()
        bucket_tiles = max(
            (token_budget + tokens_per_tile - 1) // tokens_per_tile,
            1,
        )
        padding_tiles = int(axis_keys[0]) if axis_keys else 0
        num_tiles = max(bucket_tiles - padding_tiles, 1)
        vision_config = self.config.vision_config
        pixel_values = torch.randn(
            num_tiles,
            getattr(vision_config, "num_channels", 3),
            vision_config.image_size,
            vision_config.image_size,
            device=device,
            dtype=dtype,
        )
        return EncoderCudaGraphCaptureInputs(values={"pixel_values": pixel_values})

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphReplayBuffers

        del max_batch_size, max_frames_per_batch
        assert path == "default"
        return EncoderCudaGraphReplayBuffers(
            values={"pixel_values": self._flatten_encoder_pixels(mm_kwargs)}
        )

    def encoder_cudagraph_forward(
        self,
        inputs: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        assert path == "default"
        image_features = self._image_pixels_to_features(
            cast(CLIPVisionModel | SiglipVisionModel, self.vision_tower),
            inputs["pixel_values"],
        )
        projected = self.multi_modal_projector(image_features)
        return projected.flatten(0, 1)

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        assert path == "default"
        return self.encoder_cudagraph_forward(
            {"pixel_values": self._flatten_encoder_pixels(mm_kwargs)}
        )

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        assert batch_mm_kwargs is not None
        modality = self.get_input_modality(batch_mm_kwargs)
        pixel_items = self._get_pixel_items(batch_mm_kwargs, modality)
        item_sizes = [int(item.shape[0]) for item in pixel_items]
        tokens_per_tile = self._encoder_tokens_per_tile()
        total_tiles = sum(item_sizes)

        output = outputs["default"]
        hidden_size = output.shape[-1]
        tile_features = output[: total_tiles * tokens_per_tile].reshape(
            total_tiles, tokens_per_tile, hidden_size
        )

        if modality == "video":
            pooled_features = self.apply_pooling(tile_features)
            cursor = 0
            for orig_idx, num_frames in zip(indices, item_sizes):
                video_features = pooled_features[cursor : cursor + num_frames].reshape(
                    -1, hidden_size
                )
                cursor += num_frames
                merged = torch.cat((video_features, self.image_newline[None]), dim=0)
                if merged.shape[0] != per_item_out_tokens[orig_idx]:
                    raise ValueError("Unexpected LLaVA-OneVision video output length")
                dest[orig_idx] = merged.clone() if clone else merged
            return

        image_sizes = batch_mm_kwargs.get("image_sizes")
        if image_sizes is None:
            image_size = self.config.vision_config.image_size
            image_sizes = torch.tensor(
                [[image_size, image_size]] * len(pixel_items), dtype=torch.long
            )

        cursor = 0
        for orig_idx, num_tiles, image_size in zip(indices, item_sizes, image_sizes):
            patch_features = tile_features[cursor : cursor + num_tiles]
            cursor += num_tiles
            merged = self._merge_image_patch_embeddings(
                image_size,
                patch_features,
                image_newline=self.image_newline,
                strategy="spatial_unpad",
            )
            if merged.shape[0] != per_item_out_tokens[orig_idx]:
                raise ValueError("Unexpected LLaVA-OneVision image output length")
            dest[orig_idx] = merged.clone() if clone else merged

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.vllm_config = vllm_config
        self.multimodal_config = multimodal_config

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            # Initialize the vision tower only up to the required feature layer
            self.vision_tower = init_vision_tower_for_llava(
                config,
                quant_config=quant_config,
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
        """A legal video input should have the following dimensions:
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

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> LlavaOnevisionInputsByModality:
        mm_input_by_modality = LlavaOnevisionInputsByModality()

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
        if isinstance(image_input, LlavaOnevisionImageEmbeddingInputs):
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

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor corresponding to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            if modality == "image":
                image_input = mm_input_by_modality["image"]
                assert image_input is not None
                image_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                video_input = mm_input_by_modality["video"]
                assert video_input is not None
                video_embeddings = self._process_video_pixels(video_input)
                multimodal_embeddings += tuple(video_embeddings)

        return multimodal_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for LlaVA-Onevision.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
            intermediate_tensors: Intermediate tensors from prior forward pass.
            inputs_embeds: Optional tensor of input embeddings.
            **kwargs: Multimodal inputs for this batch, forwarded to the
                multimodal embedding path.

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
