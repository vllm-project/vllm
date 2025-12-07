# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# --------------------------------------------------------
# InternS1
# Copyright (c) 2025 Shanghai AI Lab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, TypeAlias

import regex as re
import torch
import torch.nn as nn
from transformers import BatchFeature, InternVLProcessor, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.models.got_ocr2.image_processing_got_ocr2_fast import (
    GotOcr2ImageProcessorFast,
)
from transformers.models.internvl.video_processing_internvl import (
    InternVLVideoProcessor,
)

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.interns1_vit import InternS1VisionModel
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import cached_video_processor_from_config
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)


class InternS1MultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2
        )
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2,
            config.text_config.hidden_size,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size
        )

    def forward(self, image_features):
        hidden_states = self.layer_norm(image_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class InternS1ImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bnp: Batch size * number of images * (1 + num_patches)
        - c: Number of channels (3)
        - h: Height
        - w: Width
        - bn: Batch size * number of images
    """

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[torch.Tensor, TensorShape("bnp", 3, "h", "w")]
    num_patches: Annotated[torch.Tensor, TensorShape("bn")]


class InternS1ImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - ni: Number of images
        - tifs: Total image feature size
        - hs: Hidden size (must match language model backbone)
    """

    type: Literal["image_embeds"] = "image_embeds"
    data: Annotated[torch.Tensor | list[torch.Tensor], TensorShape("ni", "tifs", "hs")]


InternS1ImageInputs: TypeAlias = InternS1ImagePixelInputs | InternS1ImageEmbeddingInputs


class InternS1VideoPixelInputs(TensorSchema):
    """
    Dimensions:
        - bnv: Batch size * number of videos * number of frames
        - bn: Batch size * number of images
        - c: Number of channels (3)
        - h: Height
        - w: Width
    """

    type: Literal["pixel_values_videos"] = "pixel_values_videos"
    pixel_values: Annotated[torch.Tensor, TensorShape("bnv", 3, "h", "w")]
    num_patches: Annotated[torch.Tensor, TensorShape("bn")]


class InternS1VideoEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - nv: Number of videos
        - tvfs: Total video feature size
        - hs: Hidden size (must match language model backbone)
    """

    type: Literal["video_embeds"] = "video_embeds"
    data: Annotated[torch.Tensor | list[torch.Tensor], TensorShape("nv", "tvfs", "hs")]


InternS1VideoInputs: TypeAlias = InternS1VideoPixelInputs | InternS1VideoEmbeddingInputs


def resolve_interns1_min_max_num(
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool,
    use_thumbnail: bool,
) -> tuple[int, int]:
    min_dynamic_patch = min_dynamic_patch if dynamic_image_size else 1
    max_dynamic_patch = max_dynamic_patch if dynamic_image_size else 1

    if use_thumbnail and max_dynamic_patch != 1:
        max_dynamic_patch += 1

    return min_dynamic_patch, max_dynamic_patch


def get_interns1_target_ratios(
    min_num: int,
    max_num: int,
) -> list[tuple[int, int]]:
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    return sorted(target_ratios, key=lambda x: x[0] * x[1])


class InternS1ProcessingInfo(BaseProcessingInfo):
    """ProcessingInfo for InternS1-style models."""

    def get_hf_processor(self, **kwargs: object) -> InternVLProcessor:
        hf_processor = self.ctx.get_hf_processor(InternVLProcessor, **kwargs)
        hf_processor.video_processor = cached_video_processor_from_config(
            self.ctx.renderer_config,
            processor_cls=InternVLVideoProcessor,
            size=hf_processor.image_processor.size,
            **kwargs,
        )
        return hf_processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: GotOcr2ImageProcessorFast | None = None,
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor().image_processor

        if not isinstance(processor, GotOcr2ImageProcessorFast):
            raise ValueError(
                f"GotOcr2ImageProcessorFast is expected but got {type(processor)}"
            )
        num_image_patches = processor.get_number_of_image_patches(
            image_height, image_width, images_kwargs=dict()
        )
        num_image_tokens = self.get_hf_processor().image_seq_length * num_image_patches
        return num_image_tokens

    def resolve_target_ratios(self, use_thumbnail: bool | None = None):
        image_processor = self.get_hf_processor().image_processor
        min_dynamic_patch = image_processor.min_patches
        max_dynamic_patch = image_processor.max_patches
        # HF format's InternVL processor uses `crop_to_patches` which is
        # equivalent to `use_thumbnail` in original format.
        use_thumbnail = image_processor.crop_to_patches
        dynamic_image_size = True
        min_num, max_num = resolve_interns1_min_max_num(
            min_dynamic_patch,
            max_dynamic_patch,
            dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )

        return get_interns1_target_ratios(min_num, max_num)

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()

        hf_config = self.ctx.get_hf_config()
        base_height, base_width = hf_config.vision_config.image_size
        target_ratios = self.resolve_target_ratios()

        largest_feature_size, largest_feature_pinpoint = 0, None
        for wr, hr in target_ratios:
            width, height = base_width * wr, base_height * hr

            feat_size = self.get_num_image_tokens(
                image_width=width,
                image_height=height,
                processor=processor.image_processor,
            )
            if feat_size > largest_feature_size:
                largest_feature_size = feat_size
                largest_feature_pinpoint = ImageSize(width=width, height=height)

        assert not (largest_feature_size == 0 or largest_feature_pinpoint is None), (
            "Cannot have a largest feature size of 0!"
        )

        return largest_feature_pinpoint

    def get_max_image_tokens(self) -> int:
        processor = self.get_hf_processor()
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            processor=processor.image_processor,
        )

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_images = mm_counts.get("image", 0)
        max_videos = mm_counts.get("video", 0)

        processor = self.get_hf_processor()

        max_image_tokens = self.get_max_image_tokens() * max_images
        max_total_frames = (seq_len - max_image_tokens) // processor.image_seq_length
        max_frames_per_video = max_total_frames // max(max_videos, 1)

        return max(max_frames_per_video, 1)


class InternS1DummyInputsBuilder(BaseDummyInputsBuilder[InternS1ProcessingInfo]):
    """DummyInputsBuilder for InternS1-style models."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        image_token = self.info.get_hf_processor().image_token
        video_token = self.info.get_hf_processor().video_token

        return image_token * num_images + video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        target_width, target_height = self.info.get_image_size_with_most_features()
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts
        )
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        config = self.info.get_hf_config()
        image_size_h, image_size_w = config.vision_config.image_size

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
                width=image_size_w,
                height=image_size_h,
                num_frames=target_num_frames,
                num_videos=num_videos,
                overrides=video_overrides,
            ),
        }


class InternS1MultiModalProcessor(BaseMultiModalProcessor[InternS1ProcessingInfo]):
    """Basic image-only MultiModalProcessor for InternS1-style models."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        videos = mm_data.pop("videos", [])
        images = mm_data.pop("images", [])
        assert isinstance(videos, list)
        assert isinstance(images, list)

        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        tokenizer = hf_processor.tokenizer
        video_token_id = tokenizer.encode(
            hf_processor.video_token, add_special_tokens=False
        )
        assert len(video_token_id) == 1
        video_token_id = video_token_id[0]

        prompt = re.sub(hf_processor.image_token, "<image_placeholder>", prompt)
        prompt = re.sub(hf_processor.video_token, "<video_placeholder>", prompt)

        image_outputs = {}
        if images:
            image_pixel_values = []
            for image in images:
                processed_outputs = super()._call_hf_processor(
                    prompt=hf_processor.image_token,
                    mm_data={"images": image},
                    mm_kwargs=mm_kwargs,
                    tok_kwargs=tok_kwargs,
                )
                image_pixel_values.append(processed_outputs.pop("pixel_values"))

                input_ids = processed_outputs.pop("input_ids")
                image_placeholder = tokenizer.batch_decode(input_ids)[0]
                prompt = prompt.replace("<image_placeholder>", image_placeholder, 1)

            num_patches = [len(item) for item in image_pixel_values]
            image_outputs = {
                "pixel_values": torch.concat(image_pixel_values),
                "image_num_patches": torch.tensor(num_patches),
                "image_token_id": torch.tensor(hf_processor.image_token_id),
            }

        video_outputs = {}
        if videos:
            video_pixel_values = []
            for video in videos:
                processed_outputs = super()._call_hf_processor(
                    prompt=hf_processor.video_token,
                    mm_data={"videos": video},
                    mm_kwargs=mm_kwargs,
                    tok_kwargs=tok_kwargs,
                )
                video_pixel_values.append(processed_outputs.pop("pixel_values"))

                input_ids = processed_outputs.pop("input_ids")
                input_ids[input_ids == hf_processor.image_token_id] = video_token_id

                video_placeholder = tokenizer.batch_decode(input_ids)[0]
                prompt = prompt.replace("<video_placeholder>", video_placeholder, 1)

            num_frames = [len(item) for item in video_pixel_values]
            video_outputs = {
                "pixel_values_videos": torch.concat(video_pixel_values),
                "video_num_patches": torch.tensor(num_frames),
                "video_token_id": torch.tensor(video_token_id),
            }

        prompt = re.sub("<image_placeholder>", hf_processor.image_token, prompt)
        prompt = re.sub("<video_placeholder>", hf_processor.video_token, prompt)
        text_outputs = tokenizer(prompt, **tok_kwargs, return_tensors="pt")

        return BatchFeature({**text_outputs, **image_outputs, **video_outputs})

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_num_patches = hf_inputs.get("image_num_patches", torch.empty(0))
        video_num_patches = hf_inputs.get("video_num_patches", torch.empty(0))
        num_images = len(image_num_patches)
        num_videos = len(video_num_patches)

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_num_patches
            ),
            image_num_patches=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
            image_token_id=MultiModalFieldConfig.shared("image", num_images),
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                "video", video_num_patches
            ),
            video_num_patches=MultiModalFieldConfig.batched("video"),
            video_token_id=MultiModalFieldConfig.shared("video", num_videos),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        img_context_token = hf_processor.image_token
        start_image_token = hf_processor.start_image_token
        end_image_token = hf_processor.end_image_token
        video_token = hf_processor.video_token

        out_mm_data = out_mm_kwargs.get_data()
        if "video_num_patches" in out_mm_data:
            video_num_patches = out_mm_data["video_num_patches"]
            assert isinstance(video_num_patches, torch.Tensor)
            video_num_patches = video_num_patches.tolist()
        else:
            video_num_patches = []

        if "image_num_patches" in out_mm_data:
            image_num_patches = out_mm_data["image_num_patches"]
            assert isinstance(image_num_patches, torch.Tensor)
            image_num_patches = image_num_patches.tolist()
        else:
            image_num_patches = []

        def get_replacement_interns1_image(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                feature_size = images.get_feature_size(item_idx)
            else:
                num_patches = image_num_patches[item_idx]
                feature_size = num_patches * hf_processor.image_seq_length

            repl_features = img_context_token * feature_size
            repl_full = start_image_token + repl_features + end_image_token
            return PromptUpdateDetails.select_text(repl_full, img_context_token)

        def get_replacement_interns1_video(item_idx: int):
            num_patches = video_num_patches[item_idx]
            repl_features = video_token * hf_processor.image_seq_length
            repl_features_with_sep = start_image_token + repl_features + end_image_token
            # num_patches is equal to num_frames
            repl_full = "\n".join(
                [f"Frame{i + 1}: {repl_features_with_sep}" for i in range(num_patches)]
            )

            return PromptUpdateDetails.select_text(repl_full, video_token)

        return [
            PromptReplacement(
                modality="image",
                target=img_context_token,
                replacement=get_replacement_interns1_image,
            ),
            PromptReplacement(
                modality="video",
                target=video_token,
                replacement=get_replacement_interns1_video,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    InternS1MultiModalProcessor,
    info=InternS1ProcessingInfo,
    dummy_inputs=InternS1DummyInputsBuilder,
)
class InternS1ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA
):
    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        # transformers InternVLProcessor uses <IMG_CONTEXT> as the separator
        # refer to https://github.com/huggingface/transformers/blob/f90de364c2484c7c325bbe05befdcf487bd75b63/src/transformers/models/internvl/processing_internvl.py#L116
        if modality.startswith("image"):
            return "<IMG_CONTEXT>"
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

        image_size = config.vision_config.image_size[0]
        patch_size = config.vision_config.patch_size[0]
        self.patch_size = patch_size
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio

        self.llm_arch_name = config.text_config.architectures[0]
        self.vision_tower = self._init_vision_model(
            config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "vision_tower"),
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.multi_modal_projector = self._init_mlp1(config)

        self.img_context_token_id = None
        self.video_context_token_id = None

        self.visual_token_mask = None
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ):
        num_hidden_layers = config.vision_config.num_hidden_layers
        return InternS1VisionModel(
            config.vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers,
            prefix=prefix,
        )

    def _init_mlp1(self, config: PretrainedConfig) -> nn.Module:
        return InternS1MultiModalProjector(config)

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vit_embeds = self.vision_tower(pixel_values=pixel_values)
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        vit_embeds = self.multi_modal_projector(vit_embeds)
        return vit_embeds

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> InternS1ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_num_patches = kwargs.pop("image_num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if image_embeds is not None:
            return InternS1ImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        image_token_id = kwargs["image_token_id"]
        if isinstance(image_token_id, torch.Tensor):
            image_token_id = image_token_id.flatten().unique().item()

        assert isinstance(image_token_id, int)
        self.img_context_token_id = image_token_id

        if pixel_values is not None:
            h, w = self.config.vision_config.image_size
            return InternS1ImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                num_patches=image_num_patches,
                resolve_bindings={
                    "h": h,
                    "w": w,
                },
            )

        raise AssertionError("This line should be unreachable.")

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> InternS1VideoInputs | None:
        pixel_values_flat_video = kwargs.pop("pixel_values_videos", None)
        video_num_patches = kwargs.pop("video_num_patches", None)
        video_embeds = kwargs.pop("video_embeds", None)

        if pixel_values_flat_video is None and video_embeds is None:
            return None

        if video_embeds is not None:
            return InternS1VideoEmbeddingInputs(
                type="video_embeds",
                data=video_embeds,
            )

        video_token_id = kwargs["video_token_id"]
        if isinstance(video_token_id, torch.Tensor):
            video_token_id = video_token_id.flatten().unique().item()

        assert isinstance(video_token_id, int)
        self.video_context_token_id = video_token_id

        if pixel_values_flat_video is not None:
            h, w = self.config.vision_config.image_size
            return InternS1VideoPixelInputs(
                type="pixel_values_videos",
                num_patches=video_num_patches,
                pixel_values=pixel_values_flat_video,
                resolve_bindings={
                    "h": h,
                    "w": w,
                },
            )

        raise AssertionError("This line should be unreachable.")

    def _process_vision_input(
        self,
        image_input: InternS1ImageInputs | InternS1VideoInputs,
    ) -> tuple[torch.Tensor, ...]:
        if (
            image_input["type"] == "image_embeds"
            or image_input["type"] == "video_embeds"
        ):
            return image_input["data"]

        assert self.vision_tower is not None

        image_embeds = self.extract_feature(image_input["pixel_values"])

        num_patches = image_input["num_patches"]

        # Only one image in the current batch
        if len(num_patches) == 1:
            return (image_embeds.view(-1, self.config.text_config.hidden_size),)

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1, self.config.text_config.hidden_size)
        image_feature_sizes = [
            num_patches * feature_size for num_patches in num_patches
        ]
        return image_embeds.split(image_feature_sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "images" not in modalities
            ):
                modalities["images"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_videos",) and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(**kwargs)

        return modalities

    def _set_visual_token_mask(self, input_ids: torch.Tensor) -> None:
        self.visual_token_mask = None

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor corresponding to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                image_embeddings = self._process_vision_input(image_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_vision_input(video_input)
                multimodal_embeddings += tuple(video_embeddings)

        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            self._set_visual_token_mask(input_ids)

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
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None

        forward_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
        }

        hidden_states = self.language_model.model(**forward_kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="multi_modal_projector",
            tower_model="vision_tower",
        )
