# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from transformers import SiglipVisionConfig

from vllm.attention import Attention
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.siglip import SiglipMLP
from vllm.model_executor.models.vision import VisionEncoderInfo


class SiglipSo400mEncoderInfo(VisionEncoderInfo[SiglipVisionConfig]):
    """Helper class to provide model-specific info for NaViT-based SigLIP."""

    def get_num_image_tokens(self, *, image_width: int,
                             image_height: int) -> int:
        patch_size = self.get_patch_size()
        return (image_height // patch_size) * (image_width // patch_size)

    def get_image_size(self) -> int:
        return self.vision_config.image_size

    def get_patch_size(self) -> int:
        return self.vision_config.patch_size


class SiglipNavitVisionEmbeddings(nn.Module):
    """NaViT-style Vision Embeddings layer."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_positions = self.num_patches_per_side**2
        self.position_embedding = nn.Embedding(self.num_positions,
                                               self.embed_dim)

    def forward(self,
                pixel_values: torch.FloatTensor,
                patch_attention_mask: torch.BoolTensor,
                tgt_sizes: Optional[torch.IntTensor] = None) -> torch.Tensor:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(target_dtype))
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

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            p_attn_mask = p_attn_mask.to(device)

            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0,
                                               1 - 1e-6,
                                               1 / nb_patches_h,
                                               device=device)
            fractional_coords_w = torch.arange(0,
                                               1 - 1e-6,
                                               1 / nb_patches_w,
                                               device=device)

            bucket_coords_h = torch.bucketize(fractional_coords_h,
                                              boundaries,
                                              right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w,
                                              boundaries,
                                              right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side +
                       bucket_coords_w).flatten()

            position_ids[batch_idx, p_attn_mask.view(-1)] = pos_ids

        embeddings += self.position_embedding(position_ids)
        return embeddings


class SiglipSo400mAttention(nn.Module):

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, tp_size)

        self.q_proj = ColumnParallelLinear(self.embed_dim,
                                           self.embed_dim,
                                           bias=True,
                                           quant_config=quant_config)
        self.k_proj = ColumnParallelLinear(self.embed_dim,
                                           self.embed_dim,
                                           bias=True,
                                           quant_config=quant_config)
        self.v_proj = ColumnParallelLinear(self.embed_dim,
                                           self.embed_dim,
                                           bias=True,
                                           quant_config=quant_config)
        self.out_proj = RowParallelLinear(self.embed_dim,
                                          self.embed_dim,
                                          bias=True,
                                          quant_config=quant_config)

        scaling = self.head_dim**-0.5
        self.attn = Attention(self.num_heads_per_partition, self.head_dim,
                              scaling)

    def forward(self,
                hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        out = self.attn(q, k, v)

        attn_output, _ = self.out_proj(out)
        return attn_output, None


class SiglipSo400mEncoderLayer(nn.Module):

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.self_attn = SiglipSo400mAttention(config,
                                               quant_config=quant_config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config, quant_config=quant_config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)

    def forward(self,
                hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        residual = hidden_states

        x = self.layer_norm1(hidden_states)
        x, _ = self.self_attn(x)
        x = x + residual

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual

        return x, None


class SiglipSo400mMultiheadAttentionPoolingHead(nn.Module):

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config=config, quant_config=quant_config)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state, _ = self.attention(probe, hidden_state, hidden_state)

        residual = hidden_state
        x = self.layernorm(hidden_state)
        x = self.mlp(x)
        x = x + residual

        return x[:, 0]


class SiglipSo400mVisionModel(nn.Module):
    _supports_multimodal = True
    config_class = SiglipVisionConfig

    def __init__(self,
                 config: SiglipVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.embeddings = SiglipNavitVisionEmbeddings(config)

        self.encoder_layers = nn.ModuleList([
            SiglipSo400mEncoderLayer(config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])

        self.post_layernorm = nn.LayerNorm(config.hidden_size,
                                           eps=config.layer_norm_eps)
        self.head = SiglipSo400mMultiheadAttentionPoolingHead(
            config, quant_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        x = self.embeddings(pixel_values, patch_attention_mask)

        for layer in self.encoder_layers:
            x, _ = layer(x)

        x = self.post_layernorm(x)
        pooled_output = self.head(x)
        return pooled_output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for hf_name, loaded_weight in weights:
            if not hf_name.startswith("vision_model."):
                continue
            vllm_name = hf_name[len("vision_model."):]
            vllm_name = vllm_name.replace("encoder.layers", "encoder_layers")

            if "self_attn." in vllm_name:
                vllm_name = vllm_name.replace("self_attn.", "self_attn.")

            if "head.attention.in_proj" in vllm_name:
                param = params_dict.get(vllm_name)
                if param is not None:
                    default_weight_loader(param, loaded_weight)
                continue

            param = params_dict.get(vllm_name)
            if param is not None:
                default_weight_loader(param, loaded_weight)
