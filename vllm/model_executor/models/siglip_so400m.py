# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import sys
import typing
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

import torch
from PIL import Image
from torch import nn
from transformers import (BatchFeature, SiglipImageProcessor, SiglipProcessor,
                          SiglipTextConfig, SiglipTokenizer,
                          SiglipVisionConfig)

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.pooler import DispatchPooler, Pooler
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        PromptIndexTargets, PromptInsertion,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder

from .idefics2_vision_model import (Idefics2VisionAttention,
                                    Idefics2VisionEmbeddings,
                                    Idefics2VisionMLP)
from .interfaces import SupportsMultiModal
from .siglip import SiglipEncoderInfo
from .utils import WeightsMapper

if typing.TYPE_CHECKING:
    from vllm.multimodal.processing import InputProcessingContext
    from vllm.multimodal.utils import PlaceholderFeaturesInfo


class SiglipSo400mProcessingInfo(SiglipEncoderInfo):

    def __init__(self, ctx: "InputProcessingContext") -> None:
        self.ctx = ctx
        self.model_id = ctx.model_config.model
        hf_config = ctx.get_hf_config()
        self.hf_config = hf_config
        self.vision_config = hf_config.vision_config

    def get_hf_processor(self, **kwargs: Any) -> SiglipProcessor:
        vision_config = self.hf_config
        if hasattr(vision_config, "vision_config"):
            vision_config = vision_config.vision_config
        image_processor = SiglipImageProcessor(size={
            "height":
            vision_config.image_size,
            "width":
            vision_config.image_size
        }, )
        tokenizer_path = self.ctx.model_config.tokenizer
        tokenizer = SiglipTokenizer.from_pretrained(tokenizer_path, **kwargs)
        return SiglipProcessor(image_processor=image_processor,
                               tokenizer=tokenizer)

    def get_tokenizer(self) -> SiglipTokenizer:
        return self.get_hf_processor().tokenizer

    def get_allowed_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": sys.maxsize}

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": sys.maxsize}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Optional[Mapping[str, int]]:
        return {
            "image": self.get_num_image_tokens(image_width=0, image_height=0)
        }


class SiglipSo400mDummyInputsBuilder(
        BaseDummyInputsBuilder[SiglipSo400mProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        image_size = self.info.get_image_size()
        return {
            "image":
            self._get_dummy_images(width=image_size,
                                   height=image_size,
                                   num_images=num_images)
        }


class SiglipSo400mMultiModalProcessor(
        BaseMultiModalProcessor[SiglipSo400mProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> "BatchFeature":
        # 1. Process text
        tokenizer: SiglipTokenizer = self.info.get_hf_processor(
            **mm_kwargs).tokenizer
        text_inputs = tokenizer(prompt,
                                padding=True,
                                return_tensors="pt",
                                **tok_kwargs)

        # 2. Process images
        images = mm_data.get("images")
        if images:
            image_inputs = self._process_image(images)
            text_inputs.update(image_inputs)
        return text_inputs

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, Any],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            patch_attention_mask=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        num_images = mm_items.get_all_counts().get("image", 0)
        if not num_images:
            return []
        return [
            PromptInsertion(modality="image",
                            target=PromptIndexTargets.start(),
                            insertion="")
        ]

    def _validate_mm_placeholders(
        self,
        mm_placeholders: Mapping[str, list["PlaceholderFeaturesInfo"]],
        mm_item_counts: Mapping[str, int],
    ) -> None:
        pass

    def _process_image(self, images: list[Image.Image]) -> MultiModalDataDict:
        # This method provides custom batching for the NaViT model to handle
        # variable-sized images. It dynamically pads images to a uniform size
        # and generates a patch_attention_mask to ignore the padded areas.
        processor: SiglipImageProcessor = self.info.get_hf_processor(
        ).image_processor
        patch_size = self.info.get_patch_size()
        processed_images = [
            processor(img, return_tensors="pt") for img in images
        ]
        max_height = max(p_img.pixel_values.shape[2]
                         for p_img in processed_images)
        max_width = max(p_img.pixel_values.shape[3]
                        for p_img in processed_images)
        max_padded_height = math.ceil(max_height / patch_size) * patch_size
        max_padded_width = math.ceil(max_width / patch_size) * patch_size
        batched_pixel_values = []
        batched_patch_masks = []
        for p_img in processed_images:
            pixel_values = p_img.pixel_values.squeeze(0)
            _, h, w = pixel_values.shape
            pad_right = max_padded_width - w
            pad_bottom = max_padded_height - h
            padded_pixel_values = nn.functional.pad(
                pixel_values, (0, pad_right, 0, pad_bottom))
            batched_pixel_values.append(padded_pixel_values)
            num_patches_h = h // patch_size
            num_patches_w = w // patch_size
            max_patches_h = max_padded_height // patch_size
            max_patches_w = max_padded_width // patch_size
            mask = torch.zeros((max_patches_h, max_patches_w),
                               dtype=torch.bool)
            mask[:num_patches_h, :num_patches_w] = True
            batched_patch_masks.append(mask.flatten())
        return BatchFeature({
            "pixel_values":
            torch.stack(batched_pixel_values),
            "patch_attention_mask":
            torch.stack(batched_patch_masks),
        })


class SiglipAttention(Idefics2VisionAttention):

    def __init__(self,
                 config,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__(config=config,
                         quant_config=quant_config,
                         prefix=prefix)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)

        attn_output = self.attn(query_states,
                                key_states,
                                value_states,
                                attn_bias=attention_mask)

        attn_output, _ = self.out_proj(attn_output)
        return attn_output


class SiglipEncoderLayer(nn.Module):

    def __init__(self,
                 config,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.self_attn")
        self.layer_norm2 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)
        self.mlp = Idefics2VisionMLP(config,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.mlp")

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SiglipNavitVisionEmbeddings(Idefics2VisionEmbeddings):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)
        self.position_embedding = nn.Embedding(self.num_positions,
                                               self.embed_dim)

    def forward(self, pixel_values: torch.FloatTensor,
                patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(1)
        if patch_attention_mask.dim() == 3:
            patch_attention_mask = patch_attention_mask.squeeze(1)
        return super().forward(pixel_values, patch_attention_mask)


class SiglipMultiheadAttentionPoolingHead(nn.Module):

    def __init__(self,
                 config: SiglipVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(config.hidden_size,
                                               config.num_attention_heads,
                                               batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.mlp = Idefics2VisionMLP(config=config,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.mlp")

    def forward(self,
                hidden_state: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1).to(hidden_state.device)
        if attention_mask is not None:
            key_padding_mask = ~attention_mask
        else:
            key_padding_mask = None
        pooled_output, _ = self.attention(probe,
                                          hidden_state,
                                          hidden_state,
                                          key_padding_mask=key_padding_mask)
        residual = pooled_output
        pooled_output = self.layernorm(pooled_output)
        mlp_output, _ = self.mlp(pooled_output)
        pooled_output = residual + mlp_output
        final_output = pooled_output[:, 0]
        return final_output


class SiglipVisionTower(nn.Module):

    def __init__(self,
                 config: SiglipVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "vision_model"):
        super().__init__()
        self.config = config
        self.embeddings = SiglipNavitVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config,
                                     quant_config,
                                     prefix=f"{prefix}.encoder")
        self.post_layernorm = nn.LayerNorm(config.hidden_size,
                                           eps=config.layer_norm_eps)
        self.head = SiglipMultiheadAttentionPoolingHead(
            config, quant_config, prefix=f"{prefix}.head")

    def forward(self, pixel_values: torch.Tensor,
                patch_attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixel_values, patch_attention_mask)
        for layer in self.encoder.layers:
            x = layer(x, attention_mask=None)
        x = self.post_layernorm(x)
        return x


class SiglipTextEmbeddings(nn.Module):

    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        self.token_embedding = VocabParallelEmbedding(config.vocab_size,
                                                      config.hidden_size)
        self.position_embedding = VocabParallelEmbedding(
            config.max_position_embeddings, config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        seq_length = input_ids.shape[1]
        max_position_embeddings = self.position_embedding.num_embeddings
        if seq_length > max_position_embeddings:
            input_ids = input_ids[:, :max_position_embeddings]
            seq_length = max_position_embeddings

        position_ids = self.position_ids[:, :seq_length].to(input_ids.device)
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        embeddings = token_embeds + position_embeds
        return embeddings


class SiglipEncoder(nn.Module):

    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.layers = nn.ModuleList([
            SiglipEncoderLayer(config,
                               quant_config,
                               prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])


class SiglipTextTower(nn.Module):

    def __init__(self,
                 config: SiglipTextConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "text_model"):
        super().__init__()
        self.config = config
        self.embeddings = SiglipTextEmbeddings(config)
        self.encoder = SiglipEncoder(config,
                                     quant_config,
                                     prefix=f"{prefix}.encoder")
        self.final_layer_norm = nn.LayerNorm(config.hidden_size,
                                             eps=config.layer_norm_eps)
        projection_size = getattr(config, 'projection_size',
                                  config.hidden_size)
        self.head = ColumnParallelLinear(config.hidden_size,
                                         projection_size,
                                         bias=True,
                                         gather_output=True,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.head")

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        x = self.embeddings(input_ids)

        actual_seq_len = x.shape[1]
        attention_mask = attention_mask[:, :actual_seq_len]

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=x.dtype)
        extended_attention_mask = (
            1.0 - extended_attention_mask) * torch.finfo(x.dtype).min

        for layer in self.encoder.layers:
            x = layer(x, attention_mask=extended_attention_mask)
        return x


@MULTIMODAL_REGISTRY.register_processor(
    SiglipSo400mMultiModalProcessor,
    info=SiglipSo400mProcessingInfo,
    dummy_inputs=SiglipSo400mDummyInputsBuilder,
)
class SiglipSo400mModel(nn.Module, SupportsMultiModal):
    is_pooling_model = True
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "text_model.": "text_tower.",
        "vision_model.": "vision_tower.",
    })

    def __init__(self,
                 vllm_config: VllmConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        siglip_s0400_vision_config = config.vision_config
        siglip_s0400_text_config = config.text_config
        quant_config = vllm_config.quant_config

        self.vision_tower = SiglipVisionTower(siglip_s0400_vision_config,
                                              quant_config)
        self.text_tower = SiglipTextTower(siglip_s0400_text_config,
                                          quant_config)
        self.pad_token_id = self.config.text_config.pad_token_id

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = DispatchPooler(
            {"encode": Pooler.for_encode(pooler_config)})

    def get_multimodal_embeddings(
            self, **kwargs: object) -> SiglipNavitVisionEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.text_tower.embeddings.token_embedding(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        **kwargs: object,
    ) -> torch.Tensor:
        pixel_values = kwargs.get("pixel_values")

        if pixel_values is not None:
            # Path A: Image Reasoning
            hidden_states = self.vision_tower(pixel_values,
                                              kwargs["patch_attention_mask"])
        elif input_ids is not None:
            # Path B: Textual Inference
            attention_mask = (input_ids != self.pad_token_id).long()
            hidden_states = self.text_tower(input_ids, attention_mask)
        # Engine preheating
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            raise ValueError(
                "'pixel_values','input_ids' or 'inputs_embeds' must be provided"
            )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        mapped_weights = []
        for name, loaded_weight in weights:
            # Replace using the rules defined in self.hf_to_vllm_mapper
            prefix_map = self.hf_to_vllm_mapper.orig_to_new_prefix.items()
            for old_prefix, new_prefix in prefix_map():
                if name.startswith(old_prefix):
                    name = name.replace(old_prefix, new_prefix, 1)
                    break
            mapped_weights.append((name, loaded_weight))
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in mapped_weights:
            is_qkv_part = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                target_name = name.replace(weight_name, param_name)
                # Prevent KeyError
                if target_name not in params_dict:
                    continue
                param = params_dict[target_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                is_qkv_part = True
                break
            if is_qkv_part:
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
