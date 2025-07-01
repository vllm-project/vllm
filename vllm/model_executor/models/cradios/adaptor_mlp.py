# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from typing import Dict, Optional

import torch
from einops import rearrange
from torch import nn

from .enable_spectral_reparam import (disable_spectral_reparam,
                                      enable_spectral_reparam)


class MLP(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_inner: int = 0,
                 device: torch.device = None,
                 **kwargs):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, device=device)
        self.norm = nn.LayerNorm(hidden_size, device=device)
        self.relu = nn.ReLU()

        inner = []
        for _ in range(num_inner):
            inner.extend([
                nn.Linear(hidden_size, hidden_size, device=device),
                nn.LayerNorm(hidden_size, device=device),
                nn.ReLU(),
            ])
        if inner:
            self.inner = nn.Sequential(*inner)
        else:
            self.inner = nn.Identity()

        self.fc2 = nn.Linear(hidden_size, output_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.inner(x)
        x = self.fc2(x)
        return x


class MLP2(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_inner: int = 0,
                 pre_norm: bool = False,
                 device: torch.device = None,
                 upsample_factor: int = 1,
                 upsample_rank: int = None,
                 from_config: bool = False,
                 **kwargs):
        super().__init__()

        self.pre_norm = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.GELU(),
        ) if pre_norm else nn.Identity()

        self.upsample_factor = upsample_factor
        sq_ups = upsample_factor**2

        self._real_output_dim = output_size // sq_ups

        # hidden_size *= upsample_factor
        # output_size *= (upsample_factor ** 2)

        self.fc1 = nn.Linear(input_size, hidden_size, device=device)

        blocks = []
        for _ in range(num_inner):
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_size, device=device),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size, device=device),
                ))
        self.blocks = nn.ModuleList(blocks)

        self.final = nn.Sequential(
            nn.LayerNorm(hidden_size, device=device),
            nn.GELU(),
            nn.Linear(hidden_size, output_size, device=device),
        )

    def forward(self,
                x: torch.Tensor,
                images: Optional[torch.Tensor] = None,
                patch_size: Optional[int] = None) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.fc1(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.final(x)

        if self.upsample_factor > 1:
            if images is None:
                raise ValueError(
                    '`images` cannot be `None` when the head\'s `upsample_factor > 1`!'
                )
            if patch_size is None:
                raise ValueError(
                    '`patch_size` cannot be `None` when the head\'s `upsample_factor > 1`!'
                )
            h, w = tuple(d // patch_size for d in images.shape[-2:])
            x = rearrange(x,
                          'b (h w) (u1 u2 c) -> b (h u1 w u2) c',
                          h=h,
                          w=w,
                          u1=self.upsample_factor,
                          u2=self.upsample_factor,
                          c=self._real_output_dim)

        return x


MLP_FACTORY = {
    'v1': MLP,
    'v2': MLP2,
}


def strip_prefix(state: Dict[str, torch.Tensor], prefix: str):
    state = {
        k[len(prefix):]: v
        for k, v in state.items() if k.startswith(prefix)
    }
    return state


def get_mlp_info_from_state(version: str,
                            state: Dict[str, torch.Tensor],
                            prefix: str = '',
                            spectral_weights: bool = False):
    state = strip_prefix(state, prefix)

    weight_suffix = 'weight' if not spectral_weights else 'parametrizations.weight.original'

    if version == 'v1':
        hidden_dim, input_dim = state[f'fc1.{weight_suffix}'].shape
        output_dim = state[f'fc2.{weight_suffix}'].shape[0]

        for num_inner in range(1000):
            k = f'inner.{num_inner}.0.weight'
            if k not in state:
                break
    elif version == 'v2':
        hidden_dim, input_dim = state[f'fc1.{weight_suffix}'].shape
        output_dim = state[f'final.2.{weight_suffix}'].shape[0]

        for num_inner in range(1000):
            k = f'blocks.{num_inner}.0.weight'
            if k not in state:
                break
    else:
        raise ValueError(f'Unsupported MLP version: {version}')

    return input_dim, hidden_dim, output_dim, num_inner


def create_mlp_from_config(version: str, input_dim: int, hidden_dim: int,
                           output_dim: int, num_inner: int, **kwargs):
    ret: nn.Module = MLP_FACTORY[version](input_dim,
                                          hidden_dim,
                                          output_dim,
                                          num_inner,
                                          from_config=True,
                                          **kwargs)

    return ret


def create_mlp_from_state(version: str,
                          state: Dict[str, torch.Tensor],
                          prefix: str = '',
                          spectral_weights: bool = False,
                          **kwargs):
    state = strip_prefix(state, prefix)

    input_dim, hidden_dim, output_dim, num_inner = get_mlp_info_from_state(
        version, state, spectral_weights=spectral_weights)

    ret: nn.Module = create_mlp_from_config(version, input_dim, hidden_dim,
                                            output_dim, num_inner, **kwargs)

    if spectral_weights:
        enable_spectral_reparam(ret,
                                init_norm_to_current=False,
                                state_dict_guidance=state)

    ret.load_state_dict(state)

    if spectral_weights:
        disable_spectral_reparam(ret)

    return ret
