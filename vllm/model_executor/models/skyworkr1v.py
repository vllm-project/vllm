# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://huggingface.co/Skywork/Skywork-R1V-38B/blob/main/modeling_skywork_chat.py
# --------------------------------------------------------
# SkyworkR1V
# Copyright (c) 2025 Skywork
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, TypeAlias

import torch
import torch.nn as nn
from transformers import BatchFeature, PretrainedConfig

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.models.intern_vit import (
    InternVisionModel,
    InternVisionPatchModel,
)
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
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processors.internvl import (
    InternVLImageProcessor,
    InternVLProcessor,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix


class SkyworkR1VImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bnp: Batch size * number of images * (1 + num_patches)
        - c: Number of channels (3)
        - h: Height
        - w: Width
        - bn: Batch size * number of images
    """

    type: Literal["pixel_values"] = "pixel_values"

    pixel_values_flat: Annotated[
        torch.Tensor,
        TensorShape("bnp", 3, "h", "w"),
    ]

    num_patches: Annotated[
        torch.Tensor,
        TensorShape("bn"),
    ]


class SkyworkR1VImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - ni: Number of images
        - ifs: Image feature size
        - hs: Hidden size (must match the hidden size of language model
          backbone)
    """

    type: Literal["image_embeds"] = "image_embeds"

    data: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("ni", "ifs", "hs"),
    ]


SkyworkR1VImageInputs: TypeAlias = (
    SkyworkR1VImagePixelInputs | SkyworkR1VImageEmbeddingInputs
)


class SkyworkR1VProcessingInfo(BaseProcessingInfo):
    def get_image_processor(self, **kwargs):
        config = self.get_hf_config()
        vision_config = config.vision_config

        kwargs = self.ctx.get_merged_mm_kwargs(kwargs)
        kwargs.setdefault("image_size", vision_config.image_size)
        kwargs.setdefault("min_dynamic_patch", config.min_dynamic_patch)
        kwargs.setdefault("max_dynamic_patch", config.max_dynamic_patch)
        kwargs.setdefault("dynamic_image_size", config.dynamic_image_size)
        kwargs.setdefault("use_thumbnail", config.use_thumbnail)

        return InternVLImageProcessor(**kwargs)

    def get_hf_processor(self, **kwargs: object) -> InternVLProcessor:
        config = self.get_hf_config()
        vision_config = config.vision_config

        image_processor = self.get_image_processor(**kwargs)
        image_size = image_processor.image_size
        patch_size = vision_config.patch_size
        downsample_ratio = config.downsample_ratio
        image_seq_length = int((image_size // patch_size) ** 2 * (downsample_ratio**2))

        return InternVLProcessor(
            tokenizer=self.get_tokenizer(),
            image_processor=image_processor,
            image_seq_length=image_seq_length,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: InternVLProcessor,
    ) -> int:
        return processor.get_num_image_tokens(
            image_width=image_width,
            image_height=image_height,
        )

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()
        image_processor = processor.image_processor

        base_size = image_processor.image_size
        target_ratios = processor.resolve_target_ratios()

        largest_feature_size, largest_feature_pinpoint = 0, None
        for wr, hr in target_ratios:
            width, height = base_size * wr, base_size * hr

            feat_size = self.get_num_image_tokens(
                image_width=width,
                image_height=height,
                processor=processor,
            )
            if feat_size > largest_feature_size:
                largest_feature_size = feat_size
                largest_feature_pinpoint = ImageSize(width=width, height=height)

        if largest_feature_size == 0 or largest_feature_pinpoint is None:
            raise ValueError("Cannot have a largest feature size of 0!")

        return largest_feature_pinpoint


class SkyworkR1VDummyInputsBuilder(BaseDummyInputsBuilder[SkyworkR1VProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        return "<image>" * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        target_width, target_height = self.info.get_image_size_with_most_features()
        num_images = mm_counts.get("image", 0)

        image_overrides = mm_options.get("image")

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class SkyworkR1VMultiModalProcessor(BaseMultiModalProcessor[SkyworkR1VProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        image_token_id = hf_processor.image_token_id

        # Since there may be extra tokens in the feature placeholders,
        # we need to pass the image token ID to the model to select the
        # tokens to merge from the vision encoder outputs
        processed_outputs["image_token_id"] = torch.tensor(image_token_id)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_num_patches = hf_inputs.get("image_num_patches", torch.empty(0))
        num_images = len(image_num_patches)

        return dict(
            pixel_values_flat=MultiModalFieldConfig.flat_from_sizes(
                "image", image_num_patches
            ),
            image_num_patches=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
            image_token_id=MultiModalFieldConfig.shared("image", num_images),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        out_mm_data = out_mm_kwargs.get_data()
        if "image_num_patches" in out_mm_data:
            image_num_patches = out_mm_data["image_num_patches"]
            assert isinstance(image_num_patches, torch.Tensor)
            image_num_patches = image_num_patches.tolist()
        elif "image_embeds" in out_mm_data:
            # TODO: Use image size information in dictionary embedding inputs
            # to compute num_patches (similar to Qwen2-VL)
            image_num_patches = [None] * len(out_mm_data["image_embeds"])
        else:
            image_num_patches = []

        def get_replacement_skyworkr1v(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                feature_size = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                feature_size = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    processor=hf_processor,
                )

            num_patches = image_num_patches[item_idx]
            if num_patches is not None:
                assert isinstance(num_patches, int)

            return hf_processor.get_image_repl(num_patches, num_features=feature_size)

        return [
            PromptReplacement(
                modality="image",
                target="<image>",
                replacement=get_replacement_skyworkr1v,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    SkyworkR1VMultiModalProcessor,
    info=SkyworkR1VProcessingInfo,
    dummy_inputs=SkyworkR1VDummyInputsBuilder,
)
class SkyworkR1VChatModel(nn.Module, SupportsMultiModal, SupportsPP):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self._patch_quant_config(config, quant_config)

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        llm_arch_name = config.text_config.architectures[0]
        self.is_mono = llm_arch_name == "SkyworkLM2VEForCausalLM"

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_model = self._init_vision_model(
                config,
                quant_config=quant_config,
                is_mono=self.is_mono,
                prefix=maybe_prefix(prefix, "vision_model"),
            )
            self.mlp1 = self._init_mlp1(
                config, quant_config, prefix=maybe_prefix(prefix, "mlp1")
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.img_context_token_id = None
        self.visual_token_mask = None
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _patch_quant_config(
        self, config: PretrainedConfig, quant_config: QuantizationConfig
    ):
        # the awq models from OpenGVLab missing `modules_to_not_convert`
        # patch the quant_config to add `modules_to_not_convert` back
        if isinstance(quant_config, AWQConfig):
            text_config = config.text_config
            llm_quant_config = getattr(text_config, "quantization_config", None)
            if (not quant_config.modules_to_not_convert) and (
                llm_quant_config is not None
            ):
                quant_config.modules_to_not_convert.append("vision_model")

    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        *,
        is_mono: bool,
        prefix: str,
    ):
        if not is_mono:
            vision_feature_layer = config.select_layer
            if vision_feature_layer < 0:
                num_hidden_layers = (
                    config.vision_config.num_hidden_layers + vision_feature_layer + 1
                )
            else:
                num_hidden_layers = vision_feature_layer + 1

            return InternVisionModel(
                config.vision_config,
                quant_config=quant_config,
                num_hidden_layers_override=num_hidden_layers,
                prefix=prefix,
            )
        else:
            return InternVisionPatchModel(config.vision_config)

    def _init_mlp1(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ) -> nn.Module:
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        return nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            ReplicatedLinear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
                llm_hidden_size,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.1",
            ),
            nn.GELU(),
            ReplicatedLinear(
                llm_hidden_size,
                llm_hidden_size,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.3",
            ),
        )

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
        if self.ps_version == "v1":
            pass
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vit_embeds = self.vision_model(pixel_values=pixel_values)
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> SkyworkR1VImageInputs | None:
        pixel_values_flat = kwargs.pop("pixel_values_flat", None)
        image_num_patches = kwargs.pop("image_num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values_flat is None and image_embeds is None:
            return None

        if image_embeds is not None:
            return SkyworkR1VImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        image_token_id = kwargs["image_token_id"]
        if isinstance(image_token_id, torch.Tensor):
            image_token_id = image_token_id.flatten().unique().item()

        assert isinstance(image_token_id, int)
        self.img_context_token_id = image_token_id

        if pixel_values_flat is not None:
            return SkyworkR1VImagePixelInputs(
                type="pixel_values",
                pixel_values_flat=pixel_values_flat,
                num_patches=image_num_patches,
                resolve_bindings={
                    "h": self.config.vision_config.image_size,
                    "w": self.config.vision_config.image_size,
                },
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: SkyworkR1VImageInputs,
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        image_embeds = self.extract_feature(image_input["pixel_values_flat"])

        num_patches = image_input["num_patches"]

        # Only one image in the current batch
        if len(num_patches) == 1:
            return image_embeds.view(-1, self.config.text_config.hidden_size).unsqueeze(
                0
            )

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1, self.config.text_config.hidden_size)
        image_feature_sizes = [
            num_patches * feature_size for num_patches in num_patches
        ]
        return image_embeds.split(image_feature_sizes)

    def _set_visual_token_mask(self, input_ids: torch.Tensor) -> None:
        if self.is_mono:
            self.visual_token_mask = (input_ids == self.img_context_token_id).reshape(
                -1, 1
            )
        else:
            self.visual_token_mask = None

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
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
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        forward_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
        }

        # Only required if the model is mono-architecture
        if self.visual_token_mask is not None:
            forward_kwargs.update({"visual_token_mask": self.visual_token_mask})
            self.visual_token_mask = None

        hidden_states = self.language_model.model(**forward_kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = [
            "action_embed",
            "temporal_embed",
            "track_embed",
            "track_embed_decoder",
            "box_token",
            "cg_criterion",
            "cg_model",
            "loc_encoder",
            "loc_decoder",
            "sam",
            "temporal_token",
            "track_token",
        ]
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights)
