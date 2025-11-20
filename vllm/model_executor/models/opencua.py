# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from Qwen2.5-VL implementation
# Copyright 2025 The vLLM team.
# Copyright 2025 XLANG Lab, The University of Hong Kong

"""Inference-only OpenCUA-7B model compatible with HuggingFace weights."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.models.qwen2_vl import (
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
)

from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargs,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .interfaces import (
    MultiModalEmbeddings,
)
from .qwen2_5_vl import (
    Qwen2_5_VisionTransformer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs,
)
from .qwen2_vl import (
    Qwen2VLDummyInputsBuilder,
    Qwen2VLMultiModalDataParser,
    Qwen2VLProcessingInfo,
    _create_qwen2vl_field_factory,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from .vision import (
    run_dp_sharded_mrope_vision_model,
)


class OpenCUAVisionTransformer(Qwen2_5_VisionTransformer):
    """Vision Transformer for OpenCUA with upstream flash attention enabled."""

    def __init__(
        self,
        vision_config: Any,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        attn_backend_override: AttentionBackendEnum | None = None,
    ) -> None:
        super().__init__(
            vision_config=vision_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=prefix,
            use_data_parallel=use_data_parallel,
            attn_backend_override=attn_backend_override,
        )


class OpenCUAProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(None)

    def get_hf_processor(self, **kwargs: object):
        """Load OpenCUA processor."""
        tokenizer = self.get_tokenizer()
        vision_config = self.ctx.get_hf_image_processor_config()
        return OpenCUAProcessor(
            vision_config=vision_config,
            tokenizer=tokenizer,
            **kwargs,
        )


class OpenCUAProcessor(Qwen2VLProcessor):
    def check_argument_for_proper_class(self, attribute_name: str, arg: object) -> None:
        if attribute_name == "tokenizer":
            return
        return super().check_argument_for_proper_class(attribute_name, arg)

    def __init__(
        self,
        vision_config: dict,
        tokenizer: AnyTokenizer,
        **kwargs,
    ):
        image_processor = Qwen2VLImageProcessor(**vision_config)
        video_processor = Qwen2VLVideoProcessor(**vision_config)
        chat_template = kwargs.pop("chat_template", None)

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )

        self.image_token = "<|media_placeholder|>"

    def __call__(
        self,
        text=None,
        images=None,
        return_tensors=None,
        **kwargs,
    ):
        if text is not None:
            if not isinstance(text, list):
                text = [text]
            text_inputs = self.tokenizer(text, **kwargs)
        else:
            text_inputs = {}

        image_inputs = {}
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            if len(images) > 0:
                image_inputs = self.image_processor(
                    images, return_tensors=return_tensors or "pt"
                )

        combined_inputs = {**text_inputs, **image_inputs}

        return BatchFeature(combined_inputs, tensor_type=return_tensors)


class OpenCUAMultiModalProcessor(BaseMultiModalProcessor[OpenCUAProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return Qwen2VLMultiModalDataParser(
            self.info.get_hf_config().vision_config.spatial_merge_size
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _create_qwen2vl_field_factory(
            self.info.get_hf_config().vision_config.spatial_merge_size
        )(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        hf_config = self.info.get_hf_config()

        image_token_str = getattr(hf_processor, "image_token", "<|media_placeholder|>")
        image_token_id = vocab.get(
            image_token_str,
            getattr(hf_config, "media_placeholder_token_id", 151664),
        )

        merge_length = image_processor.merge_size**2

        def get_replacement_opencua(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [image_token_id] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_opencua,
            )
        ]


class OpenCUADummyInputsBuilder(Qwen2VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        image_token = "<|media_placeholder|>"

        return image_token * num_images


@MULTIMODAL_REGISTRY.register_processor(
    OpenCUAMultiModalProcessor,
    info=OpenCUAProcessingInfo,
    dummy_inputs=OpenCUADummyInputsBuilder,
)
class OpenCUAForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    merge_by_field_config = True
    multimodal_cpu_fields = {"image_grid_thw"}

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            "vision_tower.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )

    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|media_placeholder|>"
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.config = config
        self.vllm_config = vllm_config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config

        if multimodal_config.get_limit_per_prompt("image"):
            attn_backend_override = (
                multimodal_config.mm_encoder_attn_backend
                if multimodal_config is not None
                else None
            )
            self.visual = OpenCUAVisionTransformer(
                vision_config=config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                use_data_parallel=self.use_data_parallel,
                attn_backend_override=attn_backend_override,
            )
        else:
            self.visual = None

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.language_model.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def _parse_and_validate_image_input(
        self, **kwargs: object
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

    def _process_image_input(
        self, image_input: Qwen2_5_VLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        if grid_thw is None:
            raise ValueError("image_grid_thw is required for image input")
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if self.visual is None:
            raise ValueError("Visual encoder is not initialized")

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"]
            with set_forward_context(None, self.vllm_config):
                if self.use_data_parallel:
                    return run_dp_sharded_mrope_vision_model(
                        self.visual, pixel_values, grid_thw_list, rope_type="rope_3d"
                    )
                else:
                    image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)

        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        image_embeds_split = image_embeds.split(sizes)
        return image_embeds_split

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds"):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
                break
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        if "image" in mm_input_by_modality:
            image_input = mm_input_by_modality["image"]
            image_embeddings = self._process_image_input(image_input)
            return tuple(image_embeddings)

        return []

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
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = []
        if self.visual is None:
            skip_prefixes.extend(["visual."])
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.merger.",
            tower_model="visual.",
        )
