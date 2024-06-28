"""Minimal implementation of CLIPVisionModel intended to be only used 
within a vision language model."""
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPAttention

from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal.image import ImageFeatureData, ImagePixelData
from vllm.sequence import SequenceData


def get_clip_patch_grid_length(*, image_size: int, patch_size: int) -> int:
    assert image_size % patch_size == 0
    return image_size // patch_size


def get_clip_num_patches(*, image_size: int, patch_size: int) -> int:
    grid_length = get_clip_patch_grid_length(image_size=image_size,
                                             patch_size=patch_size)
    return grid_length * grid_length


def get_clip_image_feature_size(hf_config: CLIPVisionConfig) -> int:
    return get_clip_num_patches(image_size=hf_config.image_size,
                                patch_size=hf_config.patch_size)


def dummy_seq_data_for_clip(
    hf_config: CLIPVisionConfig,
    seq_len: int,
    *,
    image_token_id: int,
    image_feature_size_override: Optional[int] = None,
):
    if image_feature_size_override is None:
        image_feature_size = get_clip_image_feature_size(hf_config)
    else:
        image_feature_size = image_feature_size_override

    token_ids = [image_token_id] * image_feature_size
    token_ids += [0] * (seq_len - image_feature_size)
    return SequenceData(token_ids)


def dummy_pixel_data_for_clip(
    hf_config: CLIPVisionConfig,
    *,
    image_width_override: Optional[int] = None,
    image_height_override: Optional[int] = None,
):
    width = height = hf_config.image_size
    if image_width_override is not None:
        width = image_width_override
    if image_height_override is not None:
        height = image_height_override

    image = Image.new("RGB", (width, height), color=0)
    return ImagePixelData(image)


def dummy_feature_data_for_clip(
    hf_config: CLIPVisionConfig,
    *,
    image_feature_size_override: Optional[int] = None,
):
    if image_feature_size_override is None:
        image_feature_size = get_clip_image_feature_size(hf_config)
    else:
        image_feature_size = image_feature_size_override

    values = torch.zeros((1, image_feature_size, hf_config.hidden_size),
                         dtype=torch.float16)
    return ImageFeatureData(values)


# Adapted from https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/models/clip/modeling_clip.py#L164 # noqa
class CLIPVisionEmbeddings(nn.Module):

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = get_clip_num_patches(image_size=self.image_size,
                                                patch_size=self.patch_size)
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions,
                                               self.embed_dim)
        self.register_buffer("position_ids",
                             torch.arange(self.num_positions).expand((1, -1)),
                             persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(
            dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class CLIPMLP(nn.Module):

    def __init__(self,
                 config: CLIPVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(config.hidden_size,
                                        config.intermediate_size,
                                        bias=True,
                                        quant_config=quant_config)
        self.fc2 = RowParallelLinear(config.intermediate_size,
                                     config.hidden_size,
                                     bias=True,
                                     quant_config=quant_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class CLIPEncoderLayer(nn.Module):

    def __init__(self,
                 config: CLIPVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()

        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config, quant_config=quant_config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self 
    attention layers. Each layer is a [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self,
                 config: CLIPVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(config=config, quant_config=quant_config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self,
                inputs_embeds: torch.Tensor,
                vision_feature_layer: int = -1):

        # Encoder forward pass only up to the required layer
        num_layer = len(self.layers) + vision_feature_layer + 1
        hidden_states = inputs_embeds
        for encoder_layer in self.layers[:num_layer]:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class CLIPVisionTransformer(nn.Module):

    def __init__(self,
                 config: CLIPVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)

        # NOTE: This typo of "layrnorm" is not fixed on purpose to match
        # the original transformers code and name of the model weights.
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config=config, quant_config=quant_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        vision_feature_layer: int = -1,
    ) -> torch.Tensor:

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        hidden_states = self.encoder(inputs_embeds=hidden_states,
                                     vision_feature_layer=vision_feature_layer)

        return hidden_states


class CLIPVisionModel(nn.Module):

    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self,
                 config: CLIPVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(config=config,
                                                  quant_config=quant_config)

    def forward(self,
                pixel_values: Optional[torch.Tensor] = None,
                vision_feature_layer: int = -1):

        return self.vision_model(pixel_values=pixel_values,
                                 vision_feature_layer=vision_feature_layer)

    @property
    def device(self):
        return next(self.parameters()).device
