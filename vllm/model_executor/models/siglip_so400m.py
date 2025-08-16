# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import io
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

import torch
from PIL import Image
from torch import nn
from transformers import (PretrainedConfig, SiglipImageProcessor,
                          SiglipProcessor, SiglipTextConfig, SiglipTokenizer,
                          SiglipVisionConfig)

from vllm.config import VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.pooler import (AllPool, PoolerHead,
                                               PoolerIdentity, SimplePooler)
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
                                        PromptInsertion, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder

from .interfaces import SupportsMultiModal
from .siglip import SiglipEncoderInfo


class SiglipSo400mProcessingInfo(SiglipEncoderInfo):

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
        return [
            PromptInsertion(modality="image", num_tokens=0, item_idx=i)
            for i in range(mm_items.get_item_count("image"))
        ]

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

        images = mm_data.get("image")
        if images and not isinstance(images, list):
            images = [images]

        if images:
            image_inputs = self._process_image(images)
            text_inputs.update(image_inputs)

        mm_hashes = {}
        if return_mm_hashes and images:
            image_hashes = []
            for img in images:
                with io.BytesIO() as buf:
                    img.save(buf, format='PNG')
                    image_bytes = buf.getvalue()
                hasher = hashlib.sha256()
                hasher.update(image_bytes)
                image_hashes.append(hasher.hexdigest())
            mm_hashes["image"] = image_hashes

        num_images = len(images) if images else 0
        mm_placeholders = {
            "image":
            [PlaceholderRange(offset=0, length=0) for _ in range(num_images)]
        }

        final_prompt_token_ids = text_inputs["input_ids"][0].tolist()

        return MultiModalInputs(type="multimodal",
                                prompt="",
                                prompt_token_ids=final_prompt_token_ids,
                                mm_kwargs=MultiModalKwargs.from_hf_inputs(
                                    text_inputs,
                                    self._get_mm_fields_config(None, {})),
                                mm_placeholders=mm_placeholders,
                                mm_hashes=mm_hashes)


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
        self.scale = self.head_dim**-0.5

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        q_trans, _ = self.q_proj(hidden_states)
        k_trans, _ = self.k_proj(hidden_states)
        v_trans, _ = self.v_proj(hidden_states)

        # 1. Reshape Q, K, V for multi-head attention
        batch_size, seq_len, _ = q_trans.shape
        q_trans = q_trans.view(batch_size, seq_len,
                               self.num_heads_per_partition,
                               self.head_dim).transpose(1, 2)
        k_trans = k_trans.view(batch_size, seq_len,
                               self.num_heads_per_partition,
                               self.head_dim).transpose(1, 2)
        v_trans = v_trans.view(batch_size, seq_len,
                               self.num_heads_per_partition,
                               self.head_dim).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        attn_weights = torch.matmul(q_trans, k_trans.transpose(
            -2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_trans)

        # 3. Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim)
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
        self.position_embedding = VocabParallelEmbedding(
            self.num_positions, self.embed_dim)

    def forward(self, pixel_values: torch.FloatTensor,
                patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(1)

        if patch_attention_mask.dim() == 3:
            patch_attention_mask = patch_attention_mask.squeeze(1)

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
        self.encoder_layers = nn.ModuleList([
            SiglipEncoderLayer(config,
                               quant_config,
                               prefix=f"{prefix}.encoder.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.post_layernorm = nn.LayerNorm(config.hidden_size,
                                           eps=config.layer_norm_eps)
        self.head = SiglipMultiheadAttentionPoolingHead(
            config, quant_config, prefix=f"{prefix}.head")

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
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        seq_length = input_ids.shape[1]
        position_ids = self.position_ids[:, :seq_length].to(input_ids.device)
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        embeddings = token_embeds + position_embeds
        return embeddings


class SiglipTextTower(nn.Module):

    def __init__(self,
                 config: SiglipTextConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "text_model"):
        super().__init__()
        self.config = config
        self.embeddings = SiglipTextEmbeddings(config)
        self.encoder_layers = nn.ModuleList([
            SiglipEncoderLayer(config,
                               quant_config,
                               prefix=f"{prefix}.encoder.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size,
                                             eps=config.layer_norm_eps)
        projection_size = getattr(config, 'projection_size',
                                  config.hidden_size)
        self.head = ColumnParallelLinear(config.hidden_size,
                                         projection_size,
                                         bias=True,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.head")

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        x = self.embeddings(input_ids)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=x.dtype)
        extended_attention_mask = (
            1.0 - extended_attention_mask) * torch.finfo(x.dtype).min
        for layer in self.encoder_layers:
            x = layer(x, attention_mask=extended_attention_mask)
        x = self.final_layer_norm(x)
        eos_indices = torch.clamp(attention_mask.sum(dim=1) - 1, min=0)
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
    supported_tasks = ["encode", "embed"]

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
        self.pooler = SimplePooler(AllPool(), PoolerHead(PoolerIdentity()))
        self.pad_token_id = self.config.text_config.pad_token_id
        self.weight_cache: dict[str, torch.Tensor] = {}
        self.is_post_warmup = False

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        attention_mask = (input_ids != self.pad_token_id).long()
        text_embeds = self.text_tower(input_ids, attention_mask)
        text_embeds = text_embeds / torch.linalg.vector_norm(
            text_embeds, ord=2, dim=-1, keepdim=True)
        return text_embeds

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:

        if self.is_post_warmup and self.weight_cache:
            self.load_weights(self.weight_cache.items())
            self.weight_cache = {}

        if not self.is_post_warmup:
            self.is_post_warmup = True

        pixel_values = kwargs.get("pixel_values")
        patch_attention_mask = kwargs.get("patch_attention_mask")

        if pixel_values is not None:
            image_embeds = self.vision_tower(pixel_values,
                                             patch_attention_mask)
            image_embeds = image_embeds / torch.linalg.vector_norm(
                image_embeds, ord=2, dim=-1, keepdim=True)
            final_embedding = image_embeds
        else:
            final_embedding = inputs_embeds

        return final_embedding.unsqueeze(1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        hf_weights = dict(weights)
        if not self.weight_cache:
            self.weight_cache = hf_weights.copy()
        params_dict = dict(self.named_parameters())
        direct_mapping = [
            "vision_model.head.attention.in_proj_weight",
            "vision_model.head.attention.in_proj_bias",
            "vision_model.head.attention.out_proj.weight",
            "vision_model.head.attention.out_proj.bias",
            "vision_model.head.probe",
        ]
        for hf_name in direct_mapping:
            if hf_name in hf_weights:
                loaded_weight = hf_weights.pop(hf_name)
                vllm_name = hf_name.replace("vision_model.", "vision_tower.",
                                            1)
                param = params_dict.get(vllm_name)
                if param is not None:
                    param.data.copy_(loaded_weight)
                else:
                    print(
                        f"Warning: Special weight '{hf_name}' found but "
                        f"corresponding param '{vllm_name}' not in vLLM model."
                    )

        # Load the rest of the weights
        for name, loaded_weight in hf_weights.items():
            if name.startswith("logit_"):
                continue
            if name.startswith("vision_model."):
                vllm_name = name.replace("vision_model.", "vision_tower.", 1)
            elif name.startswith("text_model."):
                vllm_name = name.replace("text_model.", "text_tower.", 1)
                vllm_name = vllm_name.replace("encoder.layers",
                                              "encoder_layers")
            else:
                vllm_name = name
            param = params_dict.get(vllm_name)
            if param is not None:
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
