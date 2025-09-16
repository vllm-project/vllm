# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
from collections.abc import Iterable
from itertools import repeat
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PretrainedConfig

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.intern_vit import InternVisionEncoder

input_dim_t = Union[int, tuple[int, int]]
norm_t = Union[tuple[float, float, float], torch.Tensor]


def _ntuple(n):

    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class InputConditioner(nn.Module):

    def __init__(
        self,
        input_scale: float,
        norm_mean: norm_t,
        norm_std: norm_t,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        self.dtype = dtype

        self.register_buffer("norm_mean", _to_tensor(norm_mean) / input_scale)
        self.register_buffer("norm_std", _to_tensor(norm_std) / input_scale)

    def forward(self, x: torch.Tensor):
        y = (x - self.norm_mean) / self.norm_std
        if self.dtype is not None:
            y = y.to(self.dtype)
        return y


def _to_tensor(v: norm_t):
    return torch.as_tensor(v, dtype=torch.float32).view(-1, 1, 1)


class ClsToken(nn.Module):

    def __init__(
        self,
        ndim: int,
        num_tokens: int = 1,
        enabled: bool = True,
        register_multiple: Optional[int] = None,
        num_registers: Optional[int] = None,
    ):
        super().__init__()

        self.ndim = ndim
        self.enabled = enabled
        self.num_registers = 0
        self.num_tokens = num_tokens
        if enabled:
            if num_registers:
                self.num_registers = num_registers
            elif register_multiple:
                self.num_registers = register_multiple - (num_tokens %
                                                          register_multiple)

            scale = ndim**-0.5
            self.token = nn.Parameter(
                torch.randn(num_tokens + self.num_registers, ndim) * scale)

        else:
            self.token = None

        self.num_patches = self.num_tokens + self.num_registers

    def forward(self, x: torch.Tensor):
        if self.token is None:
            return x

        token = self.token.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([
            token,
            x,
        ], dim=1)

        return x


class ViTPatchGenerator(nn.Module):

    def __init__(
        self,
        #  config: PretrainedConfig,
        patch_size: int,
        embed_dim: int,
        input_dims: input_dim_t,
        abs_pos: bool = True,
        normalize_patches: bool = False,
        cls_token: bool = False,
        max_input_dims: Optional[input_dim_t] = None,
        pos_dropout: float = 0.0,
        return_pos_enc: bool = False,
        num_cls_tokens: int = 1,
        register_multiple: Optional[int] = None,
        num_registers: Optional[int] = None,
        patch_bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if isinstance(input_dims, int):
            input_dims = (input_dims, input_dims)

        if max_input_dims is None:
            max_input_dims = input_dims
        if isinstance(max_input_dims, int):
            max_input_dims = (max_input_dims, max_input_dims)

        max_input_dims = tuple(
            int(math.ceil(d / patch_size) * patch_size)
            for d in max_input_dims)

        self.cpe_mode = max_input_dims != input_dims
        self.pos_dropout = pos_dropout
        self.return_pos_enc = return_pos_enc

        factory = dict(device=device, dtype=dtype)

        self.patch_size = patch_size
        self.abs_pos = abs_pos
        self.embed_dim = embed_dim

        self.num_rows = max_input_dims[0] // patch_size
        self.num_cols = max_input_dims[1] // patch_size
        self.input_dims = tuple(d // patch_size for d in input_dims)
        self.num_patches = self.num_rows * self.num_cols
        self.max_input_dims = max_input_dims

        self.im_to_patches = Im2Patches(patch_size)
        self.embedder = ViTPatchLinear(patch_size,
                                       embed_dim,
                                       bias=patch_bias,
                                       **factory)

        if abs_pos:
            scale = embed_dim**-0.5
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.num_patches, embed_dim, **factory) * scale)

        self.cls_token = ClsToken(
            embed_dim,
            num_tokens=num_cls_tokens,
            enabled=cls_token,
            register_multiple=register_multiple,
            num_registers=num_registers,
        )

        self.patch_normalizer = nn.LayerNorm(
            embed_dim) if normalize_patches else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.embed_patches(x)
        patches, pos_enc = self.apply_pos_enc(patches, input_size=x.shape[2:])
        patches = self.cls_token(patches)
        patches = self.patch_normalizer(patches)
        if self.return_pos_enc:
            return patches, pos_enc
        return patches

    @property
    def apply_cls_token(self):
        return self.cls_token.enabled

    @property
    def num_cls_tokens(self):
        return self.cls_token.num_tokens

    @property
    def num_cls_patches(self):
        return self.cls_token.num_patches

    @property
    def num_registers(self):
        return self.cls_token.num_registers

    @property
    def num_skip(self):
        return self.num_cls_tokens + self.num_registers

    def _load_embed(self, src_embed: torch.Tensor, targ_embed: nn.Parameter):
        if src_embed.shape != targ_embed.shape:
            src_size = int(math.sqrt(src_embed.shape[1]))

            assert src_size**2 == src_embed.shape[
                1], 'Unable to interpolate non-square embedding'

            src_embed = rearrange(src_embed,
                                  'b (h w) c -> b c h w',
                                  h=src_size,
                                  w=src_size)
            src_embed = F.interpolate(src_embed,
                                      size=(self.num_rows, self.num_cols),
                                      mode='bicubic',
                                      align_corners=True,
                                      antialias=False)
            src_embed = rearrange(src_embed, 'b c h w -> b (h w) c')
        targ_embed.data.copy_(src_embed)

    def _load_projection(self, src_proj_weight: torch.Tensor,
                         targ_proj_weight: torch.Tensor):
        if src_proj_weight.shape != targ_proj_weight.shape:
            src_patch_size = int(math.sqrt(src_proj_weight.shape[1] // 3))

            assert (src_patch_size**2) * 3 == src_proj_weight.shape[
                1], 'Unable to interpolate non-square patch size'

            src_proj_weight = rearrange(src_proj_weight,
                                        'b (c h w) -> b c h w',
                                        c=3,
                                        h=src_patch_size,
                                        w=src_patch_size)
            src_proj_weight = F.interpolate(src_proj_weight,
                                            size=(self.patch_size,
                                                  self.patch_size),
                                            mode='bicubic',
                                            align_corners=True,
                                            antialias=False)
            src_proj_weight = rearrange(src_proj_weight,
                                        'b c h w -> b (c h w)')
        targ_proj_weight.data.copy_(src_proj_weight)

    def embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.im_to_patches(x)
        patches = self.embedder(patches)
        return patches

    def apply_pos_enc(
        self,
        patches: torch.Tensor,
        patch_idxs: Optional[torch.Tensor] = None,
        input_size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        if not self.abs_pos:
            return patches

        pos_enc = self.get_pos_enc(patches.shape[0], patch_idxs, input_size)

        if self.training and self.pos_dropout > 0:
            keeps = torch.rand(patches.shape[0],
                               1,
                               1,
                               dtype=pos_enc.dtype,
                               device=pos_enc.device) > self.pos_dropout
            pos_enc_drop = torch.where(keeps, pos_enc, 0)
        else:
            pos_enc_drop = pos_enc

        return patches + pos_enc_drop, pos_enc

    def get_pos_enc(
        self,
        batch_size: int,
        patch_idxs: Optional[torch.Tensor] = None,
        input_size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        if input_size is None:
            input_dims = self.input_dims
        else:
            input_dims = tuple(d // self.patch_size for d in input_size)

        pos_embed = self._get_pos_embeddings(batch_size, input_dims)

        if patch_idxs is None:
            return pos_embed

        exp_patch_idxs = patch_idxs.unsqueeze(-1).expand(
            -1, -1, pos_embed.shape[-1])

        pos_embed = torch.gather(pos_embed.expand(patch_idxs.shape[0], -1, -1),
                                 dim=1,
                                 index=exp_patch_idxs)
        return pos_embed

    def _get_pos_embeddings(self, batch_size: int, input_dims: tuple[int,
                                                                     int]):
        if (self.num_rows, self.num_cols) == input_dims:
            return self.pos_embed

        pos_embed = self.pos_embed.reshape(1, self.num_rows, self.num_cols,
                                           -1).permute(0, 3, 1, 2)

        def window_select(pos_embed):
            if input_dims[0] < pos_embed.shape[-2]:
                pos_embed = pos_embed[..., :input_dims[0], :]
            if input_dims[1] < pos_embed.shape[-1]:
                pos_embed = pos_embed[..., :, :input_dims[1]]
            return pos_embed

        if self.cpe_mode:
            if self.training:
                min_scale = math.sqrt(0.1)
                scale = torch.rand(batch_size, 1, 1, device=pos_embed.device
                                   ) * (1 - min_scale) + min_scale
                aspect_min = math.log(3 / 4)
                aspect_max = -aspect_min
                aspect = torch.exp(
                    torch.rand(batch_size, 1, 1, device=pos_embed.device) *
                    (aspect_max - aspect_min) + aspect_min)

                scale_x = scale * aspect
                scale_y = scale * (1 / aspect)
                scale_xy = torch.stack([scale_x, scale_y], dim=-1).clamp_(0, 1)

                pos_xy = torch.rand(
                    batch_size, 1, 1, 2,
                    device=pos_embed.device) * (1 - scale_xy)

                lin_x = torch.linspace(
                    0, 1, steps=input_dims[1],
                    device=pos_embed.device)[None, None].expand(
                        batch_size, input_dims[0], -1)
                lin_y = torch.linspace(
                    0, 1, steps=input_dims[0],
                    device=pos_embed.device)[None, :, None].expand(
                        batch_size, -1, input_dims[1])

                lin_xy = torch.stack([lin_x, lin_y], dim=-1)

                grid_xy = lin_xy * scale_xy + pos_xy

                # Convert to [-1, 1] range
                grid_xy.mul_(2).sub_(1)

                pos_embed = F.grid_sample(
                    pos_embed.float().expand(batch_size, -1, -1, -1),
                    grid=grid_xy,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True,
                ).to(pos_embed.dtype)
            else:
                max_dim = max(input_dims)
                pos_embed = F.interpolate(pos_embed.float(),
                                          size=(max_dim, max_dim),
                                          align_corners=True,
                                          mode='bilinear').to(pos_embed.dtype)

                pos_embed = window_select(pos_embed)
        else:
            pos_embed = window_select(pos_embed)

        if pos_embed.shape[-2:] != input_dims:
            pos_embed = F.interpolate(pos_embed.float(),
                                      size=input_dims,
                                      align_corners=True,
                                      mode='bilinear').to(pos_embed.dtype)

        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)

        return pos_embed


class Im2Patches(nn.Module):

    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size == 1:
            patches = x.flatten(2)
            patches = patches.permute(0, 2, 1)
            return patches

        py = x.shape[-2] // self.patch_size
        px = x.shape[-1] // self.patch_size
        patches = rearrange(
            x,
            'b c (py yy) (px xx) -> b (py px) (c yy xx)',
            py=py,
            yy=self.patch_size,
            px=px,
            xx=self.patch_size,
        )
        return patches


class ViTPatchLinear(nn.Linear):

    def __init__(self,
                 patch_size: int,
                 embed_dim: int,
                 bias: bool = False,
                 **factory):
        super().__init__(3 * (patch_size**2), embed_dim, bias=bias, **factory)
        self.patch_size = patch_size


class RadioInternVisionModel(nn.Module):
    packed_modules_mapping = {
        "qkv": ["qkv"],
    }

    def __init__(
        self,
        config: PretrainedConfig = None,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(
            to_2tuple(config.patch_size), config.image_size)
        max_img_size = int(
            round(config.max_img_size / config.patch_size) * config.patch_size)
        self.patch_generator = ViTPatchGenerator(
            config.patch_size,
            config.hidden_size,
            input_dims=self.img_size,
            max_input_dims=max_img_size,
            cls_token=True,
            register_multiple=config.reg_tokens)

        self.encoder = InternVisionEncoder(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            num_dummy_heads=num_dummy_heads,
            prefix=f"{prefix}.encoder",
        )

    def _init_img_size(self, patch_size, img_size: Union[int, tuple[int,
                                                                    int]]):
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        assert self.patch_generator is not None
        hidden_states = self.patch_generator(x)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        return encoder_outputs


class RadioModel(nn.Module):
    packed_modules_mapping = {
        "qkv": ["qkv"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.input_conditioner = InputConditioner(
            input_scale=1.0,
            norm_mean=config.norm_mean,
            norm_std=config.norm_std,
        )
        self.model = RadioInternVisionModel(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            num_dummy_heads=num_dummy_heads,
            prefix=prefix)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_embeds: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        x = self.input_conditioner(pixel_values)
        y = self.model(x)
        return self._extract_final(y)

    def load_weights(self, weights) -> set[str]:
        loaded_params: set[str] = set()
        params_dict = dict(self.named_parameters())

        if isinstance(weights, dict):
            weights_list = list(weights.items())
        else:
            weights_list = list(weights)

        for name, weight in weights_list:
            if not name.startswith("radio_model."):
                # Skip non-radio weights
                continue

            sub = name[len("radio_model."):]  # drop "radio_model." prefix

            # Skip buffers not used in vLLM
            if sub in {"summary_idxs"}:
                continue

            vllm_key = None
            if sub.startswith("model.patch_generator."):
                vllm_key = f"model.patch_generator.{sub.split('.', 2)[-1]}"
            elif sub.startswith("input_conditioner."):
                vllm_key = f"input_conditioner.{sub.split('.', 1)[-1]}"
            elif sub.startswith("model.blocks."):
                # Encoder blocks: HF 'model.blocks.{i}.' ->
                # vLLM 'model.encoder.layers.{i}.'
                parts = sub.split(".")
                if len(parts) >= 4:
                    layer_idx = parts[2]
                    suffix = ".".join(parts[3:])
                    # Skip layer-scale entries that vLLM doesn't use
                    if suffix in {"ls1", "ls2"} or suffix.startswith(
                        ("ls1.", "ls2.")):
                        continue
                    vllm_key = f"model.encoder.layers.{layer_idx}.{suffix}"

            if vllm_key and vllm_key in params_dict:
                param = params_dict[vllm_key]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, weight)
                loaded_params.add(vllm_key)

        return loaded_params

    def _extract_final(self, y: torch.Tensor):
        # Remove CLS + REGISTERS tokens
        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            all_feat = y[:, patch_gen.num_skip:]

        return all_feat
