# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from vllm.forward_context import get_forward_context, is_forward_context_available

from .base import RotaryEmbedding


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        max_trained_positions: int,
        base: float,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.max_trained_positions = max_trained_positions
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        # NOTE(woosuk): self.max_position_embeddings is the original
        # maximum length before applying the rope scaling.
        # Thus, the maximum length after applying the rope scaling is
        # self.max_position_embeddings * self.scaling_factor.
        base = self.base * (
            (
                self.scaling_factor
                * self.max_position_embeddings
                / self.max_trained_positions
            )
            - (self.scaling_factor - 1)
        ) ** (self.rotary_dim / (self.rotary_dim - 2))
        inv_freq = self._compute_inv_freq(base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache


class DynamicNTKScalingRotaryEmbeddingForEncoder(RotaryEmbedding):
    """Dynamic NTK scaling for packed, complete encoder sequences.

    Each sequence starts at position zero. Unlike autoregressive decoding,
    the final length is known, and no rotated keys are cached across forwards.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        max_trained_positions: int,
        base: float,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
        )
        self.max_trained_positions = max_trained_positions
        # Cache only inverse frequencies by length, not a quadratic table of
        # (length, position) pairs. Short sequences retain the original base.
        lengths = torch.arange(max_position_embeddings + 1, dtype=torch.float64)
        lengths = lengths.clamp_min(max_trained_positions)
        bases = base * (
            scaling_factor * lengths / max_trained_positions - (scaling_factor - 1)
        ) ** (rotary_dim / (rotary_dim - 2))
        powers = torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
        inv_freq = 1.0 / (bases.float()[:, None] ** powers)
        self.register_buffer("inv_freq_by_length", inv_freq, persistent=False)

    def _cos_sin_for_positions(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.flatten()
        if is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                positions = positions.masked_fill(is_padding, 0)
        indices = torch.arange(positions.numel(), device=positions.device)
        # A zero next position marks the end of a document. Padding positions
        # are also zero, so they cannot extend the final real document.
        next_positions = torch.cat((positions, positions.new_zeros(1)))[1:]
        ends = torch.where(next_positions == 0, indices + 1, positions.numel())
        ends = ends.flip(0).cummin(0).values.flip(0)
        lengths = ends - indices + positions
        inv_freq = self.inv_freq_by_length[lengths]
        freqs = positions.float()[:, None] * inv_freq
        cos_sin = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)
        # Keep short-input rounding identical to the original RoPE cache,
        # including when this forward is compiled with low-precision Q/K.
        cos_sin = torch.where(
            (lengths <= self.max_trained_positions)[:, None],
            self.cos_sin_cache[positions].to(dtype),
            cos_sin,
        )
        return indices, cos_sin

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        indices, cos_sin = self._cos_sin_for_positions(positions, query.dtype)
        return self.forward_static(
            indices,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            cos_sin,
            self.is_neox_style,
        )

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from vllm import _custom_ops as ops

        indices, cos_sin = self._cos_sin_for_positions(positions, query.dtype)
        ops.rotary_embedding(
            indices, query, key, self.head_size, cos_sin, self.is_neox_style
        )
        return query, key

    forward_cpu = forward_cuda
    forward_hip = forward_cuda

    def forward_xpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if key is None:
            return self.forward_native(positions, query, key)
        return self.forward_cuda(positions, query, key)
