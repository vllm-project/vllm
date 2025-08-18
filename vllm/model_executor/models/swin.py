# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Implementation of Swin Transformer intended to be used within a vision
language model.

This code is adapted from the Hugging Face's Swin implementation:
https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/models/swin/modeling_swin.py
"""

import math
from typing import Optional

import torch
from torch import nn
from transformers import SwinConfig
from transformers.pytorch_utils import meshgrid

from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig

from .vision import VisionEncoderInfo


class SwinEncoderInfo(VisionEncoderInfo[SwinConfig]):

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        # For Swin, the number of tokens changes after each stage.
        # This returns the number of tokens after the final stage.
        patch_grid = self._get_patch_grid_size(image_height, image_width)
        num_stages = len(self.vision_config.depths)

        # Each stage with downsampling halves the spatial dimensions
        final_height = patch_grid[0] // (2**(num_stages - 1))
        final_width = patch_grid[1] // (2**(num_stages - 1))

        return final_height * final_width

    def get_image_size(self) -> tuple[int, int]:
        image_size = self.vision_config.image_size
        return (image_size, image_size)

    def get_patch_size(self) -> tuple[int, int]:
        patch_size = self.vision_config.patch_size
        return (patch_size, patch_size)

    def _get_patch_grid_size(self, height: int, width: int) -> tuple[int, int]:
        patch_size_h, patch_size_w = self.get_patch_size()

        # Account for potential padding
        grid_h = math.ceil(height / patch_size_h)
        grid_w = math.ceil(width / patch_size_w)

        return (grid_h, grid_w)


# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature: torch.Tensor,
                     window_size: int) -> torch.Tensor:
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(batch_size, height // window_size,
                                       window_size, width // window_size,
                                       window_size, num_channels)
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, num_channels)
    return windows


# Copied from transformers.models.swin.modeling_swin.window_reverse
def window_reverse(windows: torch.Tensor, window_size: int, height: int,
                   width: int) -> torch.Tensor:
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    windows = windows.view(-1, height // window_size, width // window_size,
                           window_size, window_size, num_channels)
    windows = windows.permute(0, 1, 3, 2, 4,
                              5).contiguous().view(-1, height, width,
                                                   num_channels)
    return windows


class SwinPatchEmbeddings(nn.Module):

    def __init__(self, config: SwinConfig):
        super().__init__()
        self.image_size = (config.image_size, config.image_size)
        self.patch_size = (config.patch_size, config.patch_size)
        self.num_channels = config.num_channels
        self.embed_dim = config.embed_dim

        self.projection = nn.Conv2d(self.num_channels,
                                    self.embed_dim,
                                    kernel_size=self.patch_size,
                                    stride=self.patch_size)

    def maybe_pad(self, pixel_values: torch.Tensor, height: int,
                  width: int) -> torch.Tensor:
        pad_right = (self.patch_size[1] -
                     width % self.patch_size[1]) % self.patch_size[1]
        pad_bottom = (self.patch_size[0] -
                      height % self.patch_size[0]) % self.patch_size[0]
        if pad_right > 0 or pad_bottom > 0:
            pixel_values = nn.functional.pad(pixel_values,
                                             (0, pad_right, 0, pad_bottom))
        return pixel_values

    def forward(
            self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, height, width = pixel_values.shape
        pixel_values = self.maybe_pad(pixel_values, height, width)

        embeddings = self.projection(pixel_values)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        return embeddings, output_dimensions


class SwinEmbeddings(nn.Module):

    def __init__(self, config: SwinConfig):
        super().__init__()
        self.patch_embeddings = SwinPatchEmbeddings(config)
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, output_dimensions


class SwinAttention(nn.Module):

    def __init__(
        self,
        config: SwinConfig,
        dim: int,
        num_heads: int,
        window_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number "
                f"of attention heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.window_size = (window_size, window_size)

        self.qkv_proj = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:,
                                                                      None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index",
                             relative_position_index,
                             persistent=False)

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for attention computation
        # (batch_size * num_windows, num_heads, seq_len, head_dim)
        q = q.view(q.shape[:-1] +
                   (self.num_heads_per_partition, self.head_dim)).permute(
                       0, 2, 1, 3)
        k = k.view(k.shape[:-1] +
                   (self.num_heads_per_partition, self.head_dim)).permute(
                       0, 2, 1, 3)
        v = v.view(v.shape[:-1] +
                   (self.num_heads_per_partition, self.head_dim)).permute(
                       0, 2, 1, 3)

        # Manual attention calculation to support relative position bias
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        rel_pos_bias = self._get_rel_pos_bias()
        # Partition bias tensor for TP
        if self.tp_size > 1:
            head_start = self.num_heads_per_partition \
                * torch.distributed.get_rank()
            head_end = head_start + self.num_heads_per_partition
            rel_pos_bias = rel_pos_bias[:, head_start:head_end, :, :]
        attn_weights = attn_weights + rel_pos_bias

        if attention_mask is not None:
            # The mask is broadcastable to the attention scores
            attn_weights = attn_weights + attention_mask.unsqueeze(1)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(
            v.shape[0], -1, self.num_heads_per_partition * self.head_dim)

        out, _ = self.out_proj(attn_output)
        return out


class SwinMLP(nn.Module):

    def __init__(
        self,
        config: SwinConfig,
        dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        intermediate_size = int(config.mlp_ratio * dim)
        self.fc1 = ColumnParallelLinear(dim,
                                        intermediate_size,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.fc1")
        self.fc2 = RowParallelLinear(intermediate_size,
                                     dim,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.fc2")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinDropPath
class SwinDropPath(nn.Module):

    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class SwinLayer(nn.Module):

    def __init__(
        self,
        config: SwinConfig,
        dim: int,
        num_heads: int,
        input_resolution: tuple[int, int],
        drop_path_rate: float = 0.0,
        shift_size: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution

        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attention = SwinAttention(
            config,
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
        )
        self.drop_path = SwinDropPath(
            drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.mlp = SwinMLP(config,
                           dim,
                           quant_config=quant_config,
                           prefix=f"{prefix}.mlp")

    def get_attn_mask(self, height: int, width: int, dtype: torch.dtype,
                      device: torch.device) -> Optional[torch.Tensor]:
        if self.shift_size > 0:
            img_mask = torch.zeros((1, height, width, 1),
                                   dtype=dtype,
                                   device=device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        return attn_mask

    def forward(self, hidden_states: torch.Tensor,
                input_dimensions: tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.shape
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels)

        # Pad hidden_states to multiples of window size
        pad_r = (self.window_size -
                 width % self.window_size) % self.window_size
        pad_b = (self.window_size -
                 height % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            hidden_states = nn.functional.pad(hidden_states,
                                              (0, 0, 0, pad_r, 0, pad_b))
        _, height_pad, width_pad, _ = hidden_states.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states,
                                               shifts=(-self.shift_size,
                                                       -self.shift_size),
                                               dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # Partition windows
        hidden_states_windows = window_partition(shifted_hidden_states,
                                                 self.window_size)
        hidden_states_windows = hidden_states_windows.view(
            -1, self.window_size * self.window_size, channels)

        attn_mask = self.get_attn_mask(height_pad, width_pad,
                                       hidden_states.dtype,
                                       hidden_states.device)

        attention_output = self.attention(hidden_states_windows, attn_mask)

        # Merge windows
        attention_windows = attention_output.view(-1, self.window_size,
                                                  self.window_size, channels)
        shifted_windows = window_reverse(attention_windows, self.window_size,
                                         height_pad, width_pad)

        # Reverse cyclic shift
        if self.shift_size > 0:
            hidden_states = torch.roll(shifted_windows,
                                       shifts=(self.shift_size,
                                               self.shift_size),
                                       dims=(1, 2))
        else:
            hidden_states = shifted_windows

        if pad_r > 0 or pad_b > 0:
            hidden_states = hidden_states[:, :height, :width, :].contiguous()

        hidden_states = hidden_states.view(batch_size, height * width,
                                           channels)
        hidden_states = shortcut + self.drop_path(hidden_states)

        # FFN
        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.drop_path(hidden_states)

        return hidden_states


class SwinPatchMerging(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # This layer is not a major performance bottleneck and parallelizing it
        # adds complexity. We keep it as a standard nn.Linear.
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, input_feature: torch.Tensor,
                input_dimensions: tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions
        batch_size, _, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width,
                                           num_channels)

        # Padding
        pad_r = (2 - width % 2) % 2
        pad_b = (2 - height % 2) % 2
        if pad_r > 0 or pad_b > 0:
            input_feature = nn.functional.pad(input_feature,
                                              (0, 0, 0, pad_r, 0, pad_b))

        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        input_feature = torch.cat([
            input_feature_0, input_feature_1, input_feature_2, input_feature_3
        ], -1)
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)

        input_feature = self.norm(input_feature)
        input_feature = self.reduction(input_feature)

        return input_feature


class SwinStage(nn.Module):

    def __init__(self,
                 config: SwinConfig,
                 dim: int,
                 depth: int,
                 num_heads: int,
                 input_resolution: tuple[int, int],
                 drop_path_rates: list[float],
                 downsample: bool,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinLayer(
                config=config,
                dim=dim,
                num_heads=num_heads[i],
                input_resolution=input_resolution,
                drop_path_rate=drop_path_rates[i],
                shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{i}",
            ) for i in range(depth)
        ])

        if downsample:
            self.downsample = SwinPatchMerging(dim=dim)
        else:
            self.downsample = None

    def forward(
        self, hidden_states: torch.Tensor,
        input_dimensions: tuple[int,
                                int]) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = input_dimensions
        for block in self.blocks:
            hidden_states = block(hidden_states, input_dimensions)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states, input_dimensions)
            output_dimensions = ((height + 1) // 2, (width + 1) // 2)
        else:
            output_dimensions = (height, width)

        return hidden_states, output_dimensions


class SwinEncoder(nn.Module):

    def __init__(self,
                 config: SwinConfig,
                 grid_size: tuple[int, int],
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.num_layers = len(config.depths)
        dpr = [
            x.item() for x in torch.linspace(0, config.drop_path_rate,
                                             sum(config.depths))
        ]

        self.layers = nn.ModuleList([
            SwinStage(
                config=config,
                dim=int(config.embed_dim * 2**i),
                depth=config.depths[i],
                num_heads=config.num_heads[i],
                input_resolution=(grid_size[0] // (2**i),
                                  grid_size[1] // (2**i)),
                drop_path_rates=dpr[sum(config.depths[:i]
                                        ):sum(config.depths[:i + 1])],
                downsample=(i < self.num_layers - 1),
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}",
            ) for i in range(self.num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor,
                input_dimensions: tuple[int, int]) -> torch.Tensor:
        for layer_module in self.layers:
            hidden_states, input_dimensions = layer_module(
                hidden_states, input_dimensions)
        return hidden_states


class SwinVisionTransformer(nn.Module):

    def __init__(self,
                 config: SwinConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 add_pooling_layer: bool = True):
        super().__init__()
        self.config = config
        self.num_features = int(config.embed_dim * 2**(len(config.depths) - 1))

        self.embeddings = SwinEmbeddings(config)

        patch_grid = (
            config.image_size // config.patch_size,
            config.image_size // config.patch_size,
        )

        self.encoder = SwinEncoder(config,
                                   patch_grid,
                                   quant_config,
                                   prefix=f"{prefix}.encoder")

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states, input_dimensions = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states, input_dimensions)

        return hidden_states
