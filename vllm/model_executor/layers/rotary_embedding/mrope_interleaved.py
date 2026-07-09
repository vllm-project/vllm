# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from vllm/model_executor/layers/rotary_embedding/__init__.py
# Copyright 2023 The vLLM team.
#
# This file is a part of the vllm-ascend project.
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

from .mrope import MRotaryEmbedding


# MRotaryEmbedding with interleaved
class MRotaryEmbeddingInterleaved(MRotaryEmbedding):
    """Rotary Embedding with Multimodal Sections and Interleaved Support."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: list[int],
        mrope_interleaved: bool = True,
    ) -> None:
        # Enlarge max_position_embeddings for video inputs
        self.cache_max_position_num = max_position_embeddings
        super().__init__(
            head_size,
            rotary_dim,
            self.cache_max_position_num,
            base,
            is_neox_style,
            dtype,
        )

        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved

        if self.mrope_section is None:
            raise ValueError("mrope_section cannot be None.")
        if sum(self.mrope_section) != rotary_dim // 2:
            raise ValueError("Sum of mrope_section must equal rotary_dim // 2.")
        if not self.mrope_interleaved:
            raise ValueError(
                "mrope_interleaved must be True when mrope_section is provided."
            )

        # Generate interleaved indices
        if len(mrope_section) == 2:
            h_num, w_num = mrope_section[0], mrope_section[1]
            mrope_dim = self.get_mrope_interleaved_id_list(h_num, w_num, 0)
        elif len(mrope_section) == 3:
            t_num, h_num, w_num = mrope_section[0], mrope_section[1], mrope_section[2]
            mrope_dim = self.get_mrope_interleaved_id_list(
                t_num, h_num, w_num, force_last=True
            )
        else:
            raise AssertionError(
                "Cannot support the length of mrope section is not 2 or 3."
            )

        mrope_dim = mrope_dim * 2
        self.mrope_dim = mrope_dim

        self.layer_cache = None

    def _rebuild_pos_emb(
        self,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interleave the rotary embedding"""
        cos_sin = self.cos_sin_cache[positions]
        mrope_section_3d = [1] * len(self.mrope_dim)
        mrope_dim = self.mrope_dim
        cos_sin = torch.cat(
            [
                m[mrope_dim[i]]
                for i, m in enumerate(cos_sin.split(mrope_section_3d, dim=-1))
            ],
            dim=-1,
        )
        return cos_sin, torch.arange(cos_sin.shape[0], device=positions.device)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with interleaved rotary embedding."""
        cos_sin, positions = self._rebuild_pos_emb(positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = self.apply_rotary_emb.forward_native(
            query_rot,
            cos,
            sin,
        )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        # key may be None in some cases, e.g. cross-layer KV sharing
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., : self.rotary_dim]
            key_pass = key[..., self.rotary_dim :]
            key_rot = self.apply_rotary_emb.forward_native(
                key_rot,
                cos,
                sin,
            )
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    @staticmethod
    def get_mrope_interleaved_id_list(
        a: int, b: int, c: int, force_last: bool = False
    ) -> list[int]:
        """
        Generate an interleaved list of indices for multi-modal rotary embedding.

        Args:
            a: Number of indices for first modality
            b: Number of indices for second modality
            c: Number of indices for third modality
            force_last: Whether to force the last element to be from the first modality

        Returns:
            List of interleaved indices
        """
        if force_last:
            a -= 1

        counts = {0: a, 1: b, 2: c}
        placed = {k: 0 for k in counts}
        rem = counts.copy()
        seq: list[int] = []
        last = None

        total = a + b + c
        for _ in range(total):
            # Candidates: remaining > 0 and ≠ last
            cands = [k for k in rem if rem[k] > 0 and k != last]
            if not cands:
                # If only last remains, relax the condition
                cands = [k for k in rem if rem[k] > 0]

            # Select the rarest candidate
            try:
                best = min(cands, key=lambda k: (placed[k] / counts[k], k))
            except KeyError:
                best = 0

            seq.append(best)
            placed[best] += 1
            rem[best] -= 1
            last = best

        if force_last:
            seq.append(0)

        return seq
