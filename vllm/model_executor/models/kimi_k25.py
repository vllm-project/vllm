# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""
Kimi-K2.5 Model Implementation for vLLM.

Kimi-K2.5 extends Kimi-K2 with vision support

This module defines:
- KimiK25ProcessingInfo/KimiK25MultiModalProcessor: Processing logic
- KimiK25ForConditionalGeneration: Main model class
"""

import copy
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import torch
from torch import nn
from transformers import BatchFeature
from transformers.processing_utils import ProcessorMixin

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
    vision_tower_forward,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
    VisionChunk,
    VisionChunkImage,
    VisionChunkVideo,
)
from vllm.multimodal.parse import MultiModalDataItems, VisionChunkProcessorItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import KimiK25Config
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .utils import PPMissingLayer, is_pp_missing_parameter, maybe_prefix

logger = init_logger(__name__)


# Dummy input dimensions for profiling.
@dataclass
class MaxImageTokenMeta:
    width: int = 3000
    height: int = 3000


class KimiK25MediaPixelInputs(TensorSchema):
    """
    Media input schema for K2-VL model.

    Dimensions:
        - np: Number of patches (flattened from all media items)
        - ps: Patch size
        - nm: Number of media items
    """

    type: Literal["pixel_values"] = "pixel_values"

    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("np", 3, "ps", "ps"),
    ]

    grid_thws: Annotated[torch.Tensor, TensorShape("nm", 3)]


class MoonshotKimiVAutoProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, media_processor=None, tokenizer=None):
        super().__init__(tokenizer)
        self.media_processor = media_processor

    # We do not support str input for text here
    def __call__(
        self,
        vision_chunks: list[VisionChunk] | None = None,
        *,
        text: list[int],
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            vision_chunks: List of VisionChunk items to be processed.
                For image: VisionChunkImage with type='image', image=PIL.Image
                For video_chunk: VisionChunkVideo with type='video_chunk', video_chunk=list[PIL.Image]
            text: The token ids to be fed to a model (required).
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- list of token ids to be fed to a model.
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `vision_chunks` is not `None`.
            - **grid_thws** -- list of image 3D grid in LLM. Returned when `vision_chunks` is not `None`.
        """
        mm_inputs = {}
        if vision_chunks is not None:
            assert isinstance(vision_chunks, list)
            mm_inputs = self.media_processor.preprocess(vision_chunks)
        # XXX: _apply_hf_processor_text_mm will call tolist() on input_ids
        return BatchFeature(
            data={
                "input_ids": torch.tensor([text]),
                **mm_inputs,
            }
        )


class KimiK25ProcessingInfo(BaseProcessingInfo):
    """Processing information for Kimi-K2.5 model.

    Provides configuration and utilities for processing both
    images and video-chunks.
    """

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)
        self.hf_config = self.get_hf_config()
        self.media_token_id = self.hf_config.media_placeholder_token_id
        media_processor = cached_get_image_processor(
            self.ctx.model_config.model, trust_remote_code=True
        )
        self.media_processor = media_processor
        self.hf_processor = MoonshotKimiVAutoProcessor(
            media_processor=self.media_processor,
            tokenizer=self.get_tokenizer(),
        )
        self.media_tokens_calculator = self.media_processor.media_tokens_calculator

    def get_hf_processor(self):
        return self.hf_processor

    def get_hf_config(self):
        return self.ctx.get_hf_config(KimiK25Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # None means unlimited
        return {"vision_chunk": None}


class KimiK25DummyInputsBuilder(BaseDummyInputsBuilder[KimiK25ProcessingInfo]):
    """Builds dummy inputs for Kimi-K2.5 model profiling."""

    def __init__(self, info: KimiK25ProcessingInfo) -> None:
        super().__init__(info)
        self.media_token_id = self.info.media_token_id
        self.frame_per_chunk = self.info.media_processor.num_frames_per_chunk

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> list[int]:
        num_media = mm_counts.get("vision_chunk", 0)
        return [self.media_token_id] * num_media

    def get_dummy_mm_items(self):
        dummy_videos = self._get_dummy_images(
            height=MaxImageTokenMeta.height,
            width=MaxImageTokenMeta.width,
            num_images=self.frame_per_chunk,
        )

        video_chunk_dummy_item = VisionChunkVideo(
            type="video_chunk", video_chunk=dummy_videos
        )
        video_chunk_num_tokens = self.info.media_tokens_calculator(
            video_chunk_dummy_item
        )

        image_dummy_item = VisionChunkImage(
            type="image",
            image=self._get_dummy_images(
                height=MaxImageTokenMeta.height,
                width=MaxImageTokenMeta.width,
                num_images=1,
            )[0],
        )
        image_num_tokens = self.info.media_tokens_calculator(image_dummy_item)
        # return the larger one
        if video_chunk_num_tokens >= image_num_tokens:
            return [video_chunk_dummy_item]
        else:
            return [image_dummy_item]

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        # TODO: Support mm_options for vision_chunk to allow user configuration
        dummy_items = self.get_dummy_mm_items()
        return {"vision_chunk": dummy_items}


class KimiK25MultiModalProcessor(BaseMultiModalProcessor[KimiK25ProcessingInfo]):
    """Multi-modal processor for Kimi-K2.5.

    Handles both image and video-chunk modalities.
    """

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Indicates how to slice media input into multiple items.

        pixel_values: [N, 3, patch_size, patch_size], all patches collected from B medias
        grid_thws: [B,3], each item: [N_t, N_h ,N_w], indicates the grid size in time/height/width direction
                    for current item.

        by multiplying [N_t, N_h ,N_w], we get the number of patches for each media item, thus we can slice
        pixel_values by pixel_values[start:start + N_t*N_h*N_w] to get patches of one item.

        """
        grid_thws = hf_inputs.get("grid_thws", torch.empty((0, 3)))
        grid_sizes = grid_thws.prod(-1)

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "vision_chunk", grid_sizes
            ),
            grid_thws=MultiModalFieldConfig.batched("vision_chunk"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        media_token_id = hf_config.media_placeholder_token_id

        def get_replacement(item_idx: int):
            media = mm_items.get_items("vision_chunk", (VisionChunkProcessorItems,))
            num_media_token = self.info.media_tokens_calculator(media[item_idx])
            return [media_token_id] * num_media_token

        return [
            PromptReplacement(
                modality="vision_chunk",
                target=[media_token_id],
                replacement=get_replacement,
            ),
        ]

    def split_video_chunks(self, video):
        return self.info.media_processor.split_video_chunks(video)


@MULTIMODAL_REGISTRY.register_processor(
    KimiK25MultiModalProcessor,
    info=KimiK25ProcessingInfo,
    dummy_inputs=KimiK25DummyInputsBuilder,
)
class KimiK25ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """Kimi-K2.5 model for conditional generation.

    Supports both image and video-chunk modalities.
    Video-chunks are temporal segments (typically 4 frames) that are
    processed with temporal pooling.
    """

    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        # Kimi-K2.5 uses video_chunk for all media types
        if modality == "image":
            return "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
        elif modality == "video":
            # return a placeholder, to be replaced in the future.
            return "<|kimi_k25_video_placeholder|>"

        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiK25Config = model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        # Check for MoonViT config compatibility
        self.use_data_parallel = (
            model_config.multimodal_config.mm_encoder_tp_mode == "data"
        )
        self.hidden_size = config.text_config.hidden_size
        self.device = torch.cuda.current_device()
        # Build vision tower directly with KimiK25VisionConfig
        self.vision_tower = MoonViT3dPretrainedModel(
            config.vision_config,
            prefix=maybe_prefix(prefix, "vision_tower"),
        )
        self.vision_tower = self.vision_tower.to(
            device=self.device, dtype=model_config.dtype
        )

        self.mm_projector = KimiK25MultiModalProjector(
            config=config.vision_config,
            use_data_parallel=self.use_data_parallel,
            prefix=maybe_prefix(prefix, "mm_projector"),
        )
        self.mm_projector = self.mm_projector.to(
            device=self.device, dtype=model_config.dtype
        )

        self.quant_config = quant_config
        sub_vllm_config = copy.deepcopy(vllm_config)
        sub_vllm_config.model_config.hf_config = (
            sub_vllm_config.model_config.hf_config.text_config
        )
        self.language_model = DeepseekV2Model(
            vllm_config=sub_vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.text_config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config.vocab_size, scale=logit_scale)
        self.media_placeholder: int = self.config.media_placeholder_token_id

    def _parse_and_validate_media_input(
        self, **kwargs: object
    ) -> KimiK25MediaPixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        grid_thws = kwargs.pop("grid_thws", None)
        if pixel_values is None:
            return None

        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)

        if len(pixel_values.shape) == 5 or len(pixel_values.shape) == 3:
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1], *pixel_values.shape[2:]
            )

        # The batch dimension of pixel_values has been flattened into shape[0]
        target_dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(target_dtype)
        assert isinstance(grid_thws, torch.Tensor), (
            f"expect grid_thws to be a tensor, get {type(grid_thws)}"
        )
        # In some cases (e.g. with merger), grid_thws has an extra middle dimension
        grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])
        assert grid_thws.ndim == 2 and grid_thws.size(1) == 3, (
            f"unexpected shape for grid_thws: {grid_thws.shape}"
        )

        return KimiK25MediaPixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            grid_thws=grid_thws,
        )

    def _process_media_input(
        self, media_input: KimiK25MediaPixelInputs
    ) -> list[torch.Tensor]:
        # NOTE(moyan): This forward will automatically batch the forward pass internally
        media_features = vision_tower_forward(
            self.vision_tower,
            media_input["pixel_values"],
            media_input["grid_thws"],
            mm_projector=self.mm_projector,
            use_data_parallel=self.use_data_parallel,
        )
        return media_features

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        # Validate the multimodal input keyword arguments
        media_input = self._parse_and_validate_media_input(**kwargs)
        if media_input is None:
            return None

        # Run multimodal inputs through encoder and projector
        vision_embeddings = self._process_media_input(media_input)
        return vision_embeddings

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states, **kwargs)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        config = self.config.text_config
        if not getattr(config, "n_routed_experts", None):
            return []
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=config.n_routed_experts,
            num_redundant_experts=0,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        config = self.config.text_config
        _KEYS_TO_MODIFY_MAPPING = {
            "language_model.lm_head": "lm_head",
            "language_model.model": "language_model",
            # mm_projector -> mm_projector mapping
            # "mm_projector": "mm_projector",
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        if getattr(config, "kv_lora_rank", None) and getattr(
            config, "q_lora_rank", None
        ):
            stacked_params_mapping += [
                (".fused_qkv_a_proj", ".q_a_proj", 0),
                (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            ]
        expert_params_mapping = self.get_expert_mapping()

        params_dict = dict(self.named_parameters())

        for args in weights:
            name, loaded_weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)

            use_default_weight_loading = False
            if "vision" in name:
                if self.vision_tower is not None:
                    use_default_weight_loading = True
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    if ("mlp.experts." in name) and name not in params_dict:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id, **kwargs)
                    break
                else:
                    for _, (
                        param_name,
                        weight_name,
                        expert_id,
                        shard_id,
                    ) in enumerate(expert_params_mapping):
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)

                        if is_pp_missing_parameter(name, self):
                            continue

                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name,
                            expert_id=expert_id,
                            shard_id=shard_id,
                            **kwargs,
                        )
                        break
                    else:
                        use_default_weight_loading = True

            if use_default_weight_loading:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, **kwargs)


def get_spec_layer_idx_from_weight_name(
    config: KimiK25Config, weight_name: str
) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            # might start with language_model.model.layers
            if f"model.layers.{layer_idx + i}." in weight_name:
                return layer_idx + i
    return None
