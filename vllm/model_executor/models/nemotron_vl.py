# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from abc import ABC
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, PretrainedConfig
from transformers.image_processing_utils_fast import BaseImageProcessorFast

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.models.internvl import (
    BaseInternVLDummyInputsBuilder, BaseInternVLMultiModalProcessor,
    BaseInternVLProcessingInfo, InternVLImageEmbeddingInputs,
    InternVLImageInputs, InternVLImagePixelInputs, InternVLProcessor)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.multimodal.processing import PromptUpdateDetails
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import (
    cached_image_processor_from_config)
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from .utils import (AutoWeightsLoader, flatten_bn, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)

IMG_START = '<img>'
IMG_END = '</img>'
IMG_CONTEXT = '<image>'


class NemotronVLProcessor(InternVLProcessor):

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: AnyTokenizer,
        image_processor: BaseImageProcessorFast,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> None:
        ABC.__init__(self)
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        image_size: int = config.force_image_size
        patch_size: int = config.patch_size

        if min_dynamic_patch is None:
            min_dynamic_patch = 1
        assert isinstance(min_dynamic_patch, int)

        if max_dynamic_patch is None:
            max_dynamic_patch = self.image_processor.max_num_tiles
        assert isinstance(max_dynamic_patch, int)

        if dynamic_image_size is None:
            dynamic_image_size = True
        assert isinstance(dynamic_image_size, bool)

        self.num_image_token = int(
            (image_size // patch_size)**2 * (config.downsample_ratio**2))
        self.image_size = image_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail: bool = self.image_processor.use_thumbnail

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[IMG_CONTEXT]

    def _preprocess_image(
        self,
        text: list[str],
        images: list[Image.Image],
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> tuple[list[str], dict[str, torch.Tensor]]:
        if len(images) == 0:
            image_inputs = {}
        else:
            pixel_values_lst = self._images_to_pixel_values_lst(
                images,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=max_dynamic_patch,
                dynamic_image_size=dynamic_image_size,
            )
            image_inputs: dict[str, NestedTensors] = {
                "pixel_values_flat":
                torch.cat(pixel_values_lst),
                "image_num_patches":
                torch.tensor([len(item) for item in pixel_values_lst]),
            }

            for pixel_values in pixel_values_lst:
                num_patches = pixel_values.shape[0]
                feature_size = num_patches * self.num_image_token
                image_repl = self.get_image_repl(feature_size, num_patches)
                NVL_IMAGE_CONTEXT = image_repl.full.replace(
                    "<image>", "<NVL_IMG_CONTEXT>")
                text = [
                    t.replace('<image>', NVL_IMAGE_CONTEXT, 1) for t in text
                ]
            text = [t.replace("<NVL_IMG_CONTEXT>", IMG_CONTEXT) for t in text]
        return text, image_inputs

    def get_image_repl(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> PromptUpdateDetails[str]:
        repl_features = IMG_CONTEXT * feature_size
        repl_full = IMG_START + repl_features + IMG_END

        return PromptUpdateDetails.select_text(repl_full, IMG_CONTEXT)


class NemotronVLProcessingInfo(BaseInternVLProcessingInfo):
    """Processing info for Nemotron VL models."""

    def get_hf_processor(self, **kwargs: object) -> NemotronVLProcessor:
        return self.ctx.init_processor(
            NemotronVLProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            image_processor=self.get_image_processor(),
            **kwargs,
        )

    def get_image_processor(self, **kwargs: object):
        return cached_image_processor_from_config(
            self.ctx.model_config,
            **kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    BaseInternVLMultiModalProcessor[NemotronVLProcessingInfo],
    info=NemotronVLProcessingInfo,
    dummy_inputs=BaseInternVLDummyInputsBuilder[NemotronVLProcessingInfo])
class LlamaNemotronVLChatModel(nn.Module, SupportsMultiModal, SupportsPP,
                               SupportsLoRA):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
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
            (image_size // patch_size)**2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        self.llm_arch_name = config.text_config.architectures[0]
        self.vision_model = self._init_vision_model(
            config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "vision_model"),
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.mlp1 = self._init_mlp1(config)

        self.img_context_token_id = None

        self.visual_token_mask = None
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _patch_quant_config(self, config: PretrainedConfig,
                            quant_config: QuantizationConfig):
        # the awq models from OpenGVLab missing `modules_to_not_convert`
        # patch the quant_config to add `modules_to_not_convert` back
        if isinstance(quant_config, AWQConfig):
            text_config = config.text_config
            llm_quant_config = getattr(text_config, "quantization_config",
                                       None)
            if (not quant_config.modules_to_not_convert) and \
                (llm_quant_config is not None):
                quant_config.modules_to_not_convert.append("vision_model")

    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        *,
        prefix: str,
    ):
        return AutoModel.from_config(config.vision_config,
                                     trust_remote_code=True)

    def _init_mlp1(self, config: PretrainedConfig) -> nn.Sequential:
        vit_hidden_size = config.vit_hidden_size
        vision_projection_hidden_size = config.projector_hidden_size
        llm_hidden_size = config.text_config.hidden_size

        return nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio)**2,
                         bias=True),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio)**2,
                      vision_projection_hidden_size,
                      bias=True),
            nn.GELU(),
            nn.Linear(vision_projection_hidden_size, llm_hidden_size),
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            pass
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1/blob/main/modeling.py#L177
        vit_embeds = self.vision_model(x=pixel_values).features
        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)

        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds,
                                        scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1,
                                        vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:

        #use force_image_size to get image_size
        h = w = self.config.force_image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f" per patch is {expected_expr}. "
                    f"You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[InternVLImageInputs]:
        pixel_values_flat = kwargs.pop("pixel_values_flat", None)
        image_num_patches = kwargs.pop("image_num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values_flat is None and image_embeds is None:
            return None

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return InternVLImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        image_token_id = kwargs["image_token_id"]
        assert isinstance(image_token_id, torch.Tensor)
        self.img_context_token_id = image_token_id.flatten().unique().item()

        if pixel_values_flat is not None:
            if not isinstance(pixel_values_flat, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values_flat)}")

            if not isinstance(image_num_patches, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image_num_patches. "
                                 f"Got type: {type(image_num_patches)}")

            pixel_values_flat = flatten_bn(pixel_values_flat, concat=True)
            image_num_patches = flatten_bn(image_num_patches, concat=True)

            return InternVLImagePixelInputs(
                type="pixel_values",
                pixel_values_flat=self._validate_pixel_values(
                    pixel_values_flat),
                num_patches=image_num_patches,
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: InternVLImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_model is not None

        image_embeds = self.extract_feature(image_input["pixel_values_flat"])

        num_patches = image_input["num_patches"]

        # Only one image in the current batch
        if len(num_patches) == 1:
            return (image_embeds.view(-1,
                                      self.config.text_config.hidden_size), )

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1,
                                         self.config.text_config.hidden_size)
        image_feature_sizes = [
            num_patches * feature_size for num_patches in num_patches
        ]
        return image_embeds.split(image_feature_sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values_flat",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs)

        return modalities

    def _set_visual_token_mask(self, input_ids: torch.Tensor) -> None:
        self.visual_token_mask = None

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:

        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += vision_embeddings

        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            context_token_ids = [self.img_context_token_id]
            assert len(context_token_ids) >= 1
            self._set_visual_token_mask(input_ids)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                context_token_ids,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> IntermediateTensors:

        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        forward_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
        }

        # Only required if the model is mono-architecture
        if self.visual_token_mask is not None:
            forward_kwargs.update(
                {"visual_token_mask": self.visual_token_mask})
            self.visual_token_mask = None

        hidden_states = self.language_model.model(**forward_kwargs)
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
        ## Ignore registered_buffers
        ## see https://huggingface.co/nvidia/C-RADIOv2-H/blob/main/input_conditioner.py#L28 # noqa: E501
        skip_substrs = ["norm_mean", "norm_std"]
        loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
        return loader.load_weights(weights)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="mlp1",
            tower_model="vision_model")
