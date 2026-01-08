# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from
# https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepencoder/sam_vary_sdpa.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from collections.abc import Iterable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionConfig

from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention
from vllm.config import MultiModalConfig
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .clip import CLIPEncoder, CLIPVisionEmbeddings


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """  # noqa: E501
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: nn.Parameter | None = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            Conv2dLayer(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            Conv2dLayer(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        self.net_2 = Conv2dLayer(
            256, 512, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.net_3 = Conv2dLayer(
            512, 1024, kernel_size=3, stride=2, padding=1, bias=False
        )

    def get_abs_pos(self, abs_pos: torch.Tensor, tgt_size: int):
        dtype = abs_pos.dtype

        src_size = abs_pos.size(1)

        if src_size != tgt_size:
            old_pos_embed = abs_pos.permute(0, 3, 1, 2)
            old_pos_embed = old_pos_embed.to(torch.float32)
            new_pos_embed = F.interpolate(
                old_pos_embed,
                size=(tgt_size, tgt_size),
                mode="bicubic",
                antialias=True,
                align_corners=False,
            ).to(dtype)
            new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
            return new_pos_embed
        else:
            return abs_pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.get_abs_pos(self.pos_embed, x.size(1))

        for blk in self.blocks:
            x = blk(x)

        neck_output = self.neck(x.permute(0, 3, 1, 2))
        conv2_output = self.net_2(neck_output)
        conv3_output = self.net_3(conv2_output)

        return conv3_output


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation
    blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """  # noqa: E501
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RelPosAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class RelPosAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """  # noqa: E501
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, (
                "Input size must be provided if using relative positional encoding."
            )
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        rel_h, rel_w = None, None
        if self.use_rel_pos:
            rel_h, rel_w = add_decomposed_rel_pos(
                q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        q = q.view(B, self.num_heads, H * W, -1)
        k = k.view(B, self.num_heads, H * W, -1)
        v = v.view(B, self.num_heads, H * W, -1)

        if self.use_rel_pos:
            rel_h = rel_h.view(
                B, self.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3)
            )
            rel_w = rel_w.view(
                B, self.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3)
            )
            attn_bias = (rel_h + rel_w).view(
                B, self.num_heads, rel_h.size(2), rel_h.size(3) * rel_w.size(4)
            )
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias
            )
        else:
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        x = (
            x.view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )

        x = self.proj(x)

        return x


def window_partition(
    x: torch.Tensor, window_size: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """  # noqa: E501
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: tuple[int, int],
    hw: tuple[int, int],
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """  # noqa: E501
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        dtype = rel_pos.dtype
        rel_pos = rel_pos.to(torch.float32)
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        ).to(dtype)
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(
        k_size / q_size, 1.0
    )
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(
        q_size / k_size, 1.0
    )
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py
    Args:
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """  # noqa: E501
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    rel_h = rel_h.unsqueeze(-1)
    rel_w = rel_w.unsqueeze(-2)
    rel_h = rel_h.reshape(B, q_h * q_w, k_h, 1)
    rel_w = rel_w.reshape(B, q_h * q_w, 1, k_w)

    return rel_h, rel_w


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = Conv2dLayer(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


# TODO(Isotr0py): use vision_config to build sam model
def build_sam_vit_b():
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
    )


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )
    return image_encoder


class DeepCLIPVisionEmbeddings(CLIPVisionEmbeddings):
    def get_abs_pos(self, abs_pos: torch.Tensor, tgt_size: int):
        # abs_pos: L, C
        # tgt_size: M
        # return: M, C
        dim = abs_pos.size(-1)
        abs_pos_new = abs_pos.squeeze(0)
        cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]

        src_size = int(math.sqrt(abs_pos_new.shape[0] - 1))
        tgt_size = int(math.sqrt(tgt_size))
        dtype = abs_pos.dtype

        if src_size != tgt_size:
            old_pos_embed = (
                old_pos_embed.view(1, src_size, src_size, dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            old_pos_embed = old_pos_embed.to(torch.float32)
            new_pos_embed = F.interpolate(
                old_pos_embed,
                size=(tgt_size, tgt_size),
                mode="bicubic",
                antialias=True,
                align_corners=False,
            ).to(dtype)
            new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
            new_pos_embed = new_pos_embed.view(tgt_size * tgt_size, dim)
            vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)
            vision_pos_embed = vision_pos_embed.view(1, tgt_size * tgt_size + 1, dim)
            return vision_pos_embed
        else:
            return abs_pos

    def forward(
        self, pixel_values: torch.Tensor, patch_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        if patch_embeds is not None:
            patch_embeds = patch_embeds
        else:
            patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.get_abs_pos(
            self.position_embedding(self.position_ids), embeddings.size(1)
        )
        return embeddings


class DeepCLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = DeepCLIPVisionEmbeddings(config)

        # NOTE: This typo of "layrnorm" is not fixed on purpose to match
        # the original transformers code and name of the model weights.
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.transformer = CLIPEncoder(
            config=config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.encoder",
            attn_cls=MMEncoderAttention,
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.transformer.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.transformer.layers)} layers."
            )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_embeds: torch.Tensor | None = None,
        *,
        select_layers: list[int] | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values, patch_embeds)
        hidden_states = self.pre_layrnorm(hidden_states)

        # Produces either the last layer output or all of the hidden states,
        # depending on if we have select_layers or not
        encoder_outputs = self.transformer(
            inputs_embeds=hidden_states,
            return_all_hidden_states=select_layers is not None,
        )
        return encoder_outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
