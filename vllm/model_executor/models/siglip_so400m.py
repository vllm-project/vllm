# vllm/model_executor/models/siglip_so400m.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

import torch
from PIL import Image
from torch import nn
from transformers import (PretrainedConfig, SiglipConfig, SiglipImageProcessor,
                          SiglipProcessor, SiglipTextConfig, SiglipTokenizer,
                          SiglipVisionConfig)

from vllm.attention import Attention
from vllm.config import VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalInputs, MultiModalKwargs,
                                    PlaceholderRange)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder

from .interfaces import SupportsMultiModal


class SiglipSo400mProcessingInfo(BaseProcessingInfo):

    def get_hf_processor(self, **kwargs):
        vision_config = self.get_hf_config()
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

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_image_size(self) -> int:
        hf_config = self.get_hf_config()
        if hasattr(hf_config, "vision_config"):
            return hf_config.vision_config.image_size
        elif hasattr(hf_config, "image_size"):
            return hf_config.image_size
        raise TypeError(
            f"Could not find image_size in config of type {type(hf_config)}")

    def get_patch_size(self) -> int:
        hf_config = self.get_hf_config()
        if hasattr(hf_config, "vision_config"):
            return hf_config.vision_config.patch_size
        elif hasattr(hf_config, "patch_size"):
            return hf_config.patch_size
        raise TypeError(
            f"Could not find patch_size in config of type {type(hf_config)}")


class SiglipSo400mDummyInputsBuilder(
        BaseDummyInputsBuilder[SiglipSo400mProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "a photo of a cat"

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        if num_images == 0:
            return {}
        image_size = self.info.get_image_size()
        return {
            "image":
            self._get_dummy_images(width=image_size,
                                   height=image_size,
                                   num_images=num_images)
        }


class SiglipSo400mMultiModalProcessor(
        BaseMultiModalProcessor[SiglipSo400mProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: "PretrainedConfig",
        hf_processor_mm_kwargs: Mapping[str, Any],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_ids=MultiModalFieldConfig.batched("text"),
            attention_mask=MultiModalFieldConfig.batched("text"),
            pixel_values=MultiModalFieldConfig.batched("image"),
            patch_attention_mask=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        return []

    def _process_image(self, images: list[Image.Image]) -> MultiModalDataDict:
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
        return {
            "pixel_values": torch.stack(batched_pixel_values),
            "patch_attention_mask": torch.stack(batched_patch_masks),
        }

    def apply(self,
              prompt: list[str],
              mm_data: MultiModalDataDict,
              hf_processor_mm_kwargs: Mapping[str, object],
              tokenization_kwargs: Optional[Mapping[str, object]] = None,
              return_mm_hashes: bool = False) -> MultiModalInputs:
        tokenizer: SiglipTokenizer = self.info.get_hf_processor().tokenizer
        text_inputs = tokenizer(prompt, padding=True, return_tensors="pt")
        images = mm_data.get("image", [])
        if images:
            image_inputs = self._process_image(images)
            text_inputs.update(image_inputs)
        num_images = len(images)
        mm_placeholders = {
            "image":
            [PlaceholderRange(offset=0, length=0) for _ in range(num_images)]
        }
        return MultiModalInputs(type="multimodal",
                                prompt="",
                                prompt_token_ids=[],
                                mm_kwargs=MultiModalKwargs.from_hf_inputs(
                                    text_inputs,
                                    self._get_mm_fields_config(None, {})),
                                mm_placeholders=mm_placeholders)


class SiglipMLP(nn.Module):

    def __init__(self,
                 config,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(config.hidden_size,
                                        config.intermediate_size,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.fc1")
        self.fc2 = RowParallelLinear(config.intermediate_size,
                                     config.hidden_size,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.fc2")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class SiglipAttention(nn.Module):

    def __init__(self,
                 config,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, tp_size)
        self.head_dim = self.embed_dim // self.num_heads
        self.q_proj = ColumnParallelLinear(self.embed_dim,
                                           self.embed_dim,
                                           bias=True,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.q_proj")
        self.k_proj = ColumnParallelLinear(self.embed_dim,
                                           self.embed_dim,
                                           bias=True,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.k_proj")
        self.v_proj = ColumnParallelLinear(self.embed_dim,
                                           self.embed_dim,
                                           bias=True,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.v_proj")
        self.out_proj = RowParallelLinear(self.embed_dim,
                                          self.embed_dim,
                                          bias=True,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.out_proj")
        scaling = self.head_dim**-0.5
        self.attn = Attention(self.num_heads_per_partition,
                              self.head_dim,
                              scaling,
                              prefix=prefix)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        out, _ = self.attn(q, k, v, attention_mask)
        attn_output, _ = self.out_proj(out)
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
        self.mlp = SiglipMLP(config,
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


class SiglipNavitVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_patches_per_side = config.image_size // config.patch_size
        self.num_positions = self.num_patches_per_side**2
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.position_embedding = nn.Embedding(self.num_positions,
                                               self.embed_dim)

    def forward(self, pixel_values: torch.FloatTensor,
                patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        device = embeddings.device
        max_nb_patches_h = max_im_h // self.patch_size
        max_nb_patches_w = max_im_w // self.patch_size
        boundaries = torch.arange(1 / self.num_patches_per_side,
                                  1.0,
                                  1 / self.num_patches_per_side,
                                  device=device)
        position_ids = torch.full(size=(batch_size,
                                        max_nb_patches_h * max_nb_patches_w),
                                  fill_value=0,
                                  device=device,
                                  dtype=torch.long)
        for batch_idx, p_attn_mask_flat in enumerate(patch_attention_mask):
            p_attn_mask = p_attn_mask_flat.view(max_nb_patches_h,
                                                max_nb_patches_w)
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0, :].sum()
            fractional_coords_h = torch.arange(0,
                                               1 - 1e-6,
                                               1 / nb_patches_h.item(),
                                               device=device)
            fractional_coords_w = torch.arange(0,
                                               1 - 1e-6,
                                               1 / nb_patches_w.item(),
                                               device=device)
            bucket_coords_h = torch.bucketize(fractional_coords_h,
                                              boundaries,
                                              right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w,
                                              boundaries,
                                              right=True)
            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side +
                       bucket_coords_w).flatten()
            position_ids[batch_idx, p_attn_mask_flat] = pos_ids
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


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
        self.mlp = SiglipMLP(config=config,
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
        pooled_output = self.mlp(pooled_output)
        pooled_output = residual + pooled_output
        return pooled_output[:, 0]


class SiglipVisionTower(nn.Module):

    def __init__(self,
                 config: SiglipVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.embeddings = SiglipNavitVisionEmbeddings(config)
        self.encoder_layers = nn.ModuleList([
            SiglipEncoderLayer(config,
                               quant_config,
                               prefix=f"vision_model.encoder.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.post_layernorm = nn.LayerNorm(config.hidden_size,
                                           eps=config.layer_norm_eps)
        self.head = SiglipMultiheadAttentionPoolingHead(
            config, quant_config, prefix="vision_model.head")

    def forward(self, pixel_values: torch.Tensor,
                patch_attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixel_values, patch_attention_mask)
        for layer in self.encoder_layers:
            x = layer(x, attention_mask=None)
        x = self.post_layernorm(x)
        pooled_output = self.head(x, attention_mask=patch_attention_mask)
        return pooled_output


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
        seq_length = input_ids.shape[1]
        position_ids = self.position_ids[:, :seq_length].to(input_ids.device)
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        return token_embeds + position_embeds


class SiglipTextTower(nn.Module):

    def __init__(self,
                 config: SiglipTextConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.embeddings = SiglipTextEmbeddings(config)
        self.encoder_layers = nn.ModuleList([
            SiglipEncoderLayer(config,
                               quant_config,
                               prefix=f"text_model.encoder.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size,
                                             eps=config.layer_norm_eps)
        self.head = ColumnParallelLinear(config.hidden_size,
                                         config.projection_size,
                                         bias=True,
                                         quant_config=quant_config,
                                         prefix="text_model.head")

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_ids)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=x.dtype)
        extended_attention_mask = (
            1.0 - extended_attention_mask) * torch.finfo(x.dtype).min
        for layer in self.encoder_layers:
            x = layer(x, attention_mask=extended_attention_mask)
        x = self.final_layer_norm(x)
        eos_indices = attention_mask.sum(dim=1) - 1
        pooled_output = x[torch.arange(x.shape[0], device=x.device),
                          eos_indices]
        pooled_output, _ = self.head(pooled_output)
        return pooled_output


@MULTIMODAL_REGISTRY.register_processor(
    SiglipSo400mMultiModalProcessor,
    info=SiglipSo400mProcessingInfo,
    dummy_inputs=SiglipSo400mDummyInputsBuilder,
)
class SiglipSo400mModel(nn.Module, SupportsMultiModal):
    is_pooling_model = True
    config_class = SiglipConfig

    def __init__(
        self,
        config: Optional[SiglipConfig] = None,
        vllm_config: Optional[VllmConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        if vllm_config:
            hf_config = vllm_config.model_config.hf_config
            if quant_config is None:
                quant_config = vllm_config.quant_config
        elif config:
            hf_config = config
        else:
            raise ValueError(
                "Either 'config' or 'vllm_config' must be provided.")

        if isinstance(hf_config, SiglipVisionConfig):
            vision_config = hf_config
            text_config_dict = vision_config.to_dict()

            # 最终修正: 在创建 TextConfig 时，直接把 projection_size 传进去
            text_config = SiglipTextConfig.from_dict(
                text_config_dict, projection_size=vision_config.hidden_size)

            hf_config = SiglipConfig(vision_config=vision_config,
                                     text_config=text_config)

        self.vision_tower = SiglipVisionTower(hf_config.vision_config,
                                              quant_config)
        self.text_tower = SiglipTextTower(hf_config.text_config, quant_config)

    # ... (forward 和 load_weights 方法保持不变) ...
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        patch_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        image_embeds = None
        if pixel_values is not None:
            assert patch_attention_mask is not None
            image_embeds = self.vision_tower(pixel_values,
                                             patch_attention_mask)
            image_embeds = image_embeds / torch.linalg.vector_norm(
                image_embeds, ord=2, dim=-1, keepdim=True)
        text_embeds = None
        if input_ids is not None:
            assert attention_mask is not None
            text_embeds = self.text_tower(input_ids, attention_mask)
            text_embeds = text_embeds / torch.linalg.vector_norm(
                text_embeds, ord=2, dim=-1, keepdim=True)
        return {"image_embeds": image_embeds, "text_embeds": text_embeds}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name.startswith("logit_"):
                continue
            if name.startswith("vision_model."):
                vllm_name = name.replace("vision_model.encoder.layers.",
                                         "vision_tower.encoder_layers.")
                vllm_name = vllm_name.replace("vision_model.", "vision_tower.")
                param = params_dict.get(vllm_name)
                if param is not None:
                    default_weight_loader(param, loaded_weight)
            if name.startswith("text_model."):
                vllm_name = name.replace("text_model.encoder.layers.",
                                         "text_tower.encoder_layers.")
                vllm_name = vllm_name.replace("text_model.", "text_tower.")
                param = params_dict.get(vllm_name)
                if param is not None:
                    default_weight_loader(param, loaded_weight)
