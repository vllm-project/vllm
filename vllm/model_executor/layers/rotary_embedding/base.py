# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rotary Positional Embeddings Base Class."""
from typing import Optional

import torch

from vllm.model_executor.custom_op import CustomOp

from .common import apply_rotary_emb_dispatch, apply_rotary_emb_torch
from .rocm_aiter_rope_ops import is_rocm_rotary_embedding_enabled


@CustomOp.register("rotary_embedding")
class RotaryEmbedding(CustomOp):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        self.is_rocm_aiter_enabled = is_rocm_rotary_embedding_enabled()

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = apply_rotary_emb_torch(query_rot, cos, sin,
                                           self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        # key may be None in some cases, e.g. cross-layer KV sharing
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., :self.rotary_dim]
            key_pass = key[..., self.rotary_dim:]
            key_rot = apply_rotary_emb_torch(key_rot, cos, sin,
                                             self.is_neox_style)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        from vllm import _custom_ops as ops

        # __setattr__ in nn.Module (called by `self.cos_sin_cache = ...`)
        # is expensive, so avoid calling it if possible
        if self.cos_sin_cache.device != query.device or \
            self.cos_sin_cache.dtype != query.dtype:
            self.cos_sin_cache = self.cos_sin_cache.to(query.device,
                                                       dtype=query.dtype)

        # ops.rotary_embedding()/batched_rotary_embedding()
        # are in-place operations that update the query and key tensors.
        if offsets is not None:
            ops.batched_rotary_embedding(positions, query, key, self.head_size,
                                         self.cos_sin_cache,
                                         self.is_neox_style, self.rotary_dim,
                                         offsets)
        else:
            ops.rotary_embedding(positions, query, key, self.head_size,
                                 self.cos_sin_cache, self.is_neox_style)
        return query, key

    def forward_hip(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        is_nope_first=False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # currently only rotary embedding ops from AITER package are
        # supported for HiP forward.
        if self.is_rocm_aiter_enabled:
            return self.forward_hip_rocm_aiter(positions, query, key, offsets,
                                               is_nope_first)
        return self.forward_native(positions, query, key, offsets)

    def forward_hip_rocm_aiter(
        self,
        positions: torch.Tensor,
        # if     is_nope_first
        # [[batch_size, seq_len, num_heads, nope_size+rope_size]
        # if NOT is_nope_first
        # [[batch_size, seq_len, num_heads, rope_size+nope_size],
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        is_nope_first: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.cos_sin_cache.device != query.device or \
            self.cos_sin_cache.dtype != query.dtype:
            self.cos_sin_cache = self.cos_sin_cache.to(query.device,
                                                       dtype=query.dtype)
        cos, sin = self.cos_sin_cache.chunk(2, dim=-1)

        cos = cos.unsqueeze(-2).unsqueeze(-2)
        sin = sin.unsqueeze(-2).unsqueeze(-2)

        rotate_style = 0 if self.is_neox_style else 1

        num_tokens = positions.numel()

        query_shape = query.shape
        query = query.view(1, num_tokens, -1, self.head_size)
        if key is not None:
            key_shape = key.shape
            key = key.view(1, num_tokens, -1, self.head_size)

        positions = positions.view(*query.shape[:2])
        if offsets is not None:
            offsets = offsets.view(*query.shape[:2])

        if not is_nope_first:
            query_ = query[..., :self.rotary_dim]
            key_ = key[..., :self.rotary_dim] if key is not None else None
        else:
            query_ = query[..., -self.rotary_dim:]
            key_ = key[..., -self.rotary_dim:] if key is not None else None

        if key_ is None:
            torch.ops.vllm.rocm_aiter_rotary_emb_without_key_forward_hip(
                positions, sin, cos, query_, offsets, rotate_style,
                is_nope_first)
            return query.view(query_shape), None

        torch.ops.vllm.rocm_aiter_rotary_emb_with_key_forward_hip(
            positions, sin, cos, query_, key_, offsets, rotate_style,
            is_nope_first)

        return query.view(query_shape), key.view(key_shape)

    def forward_xpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        from vllm._ipex_ops import ipex_ops as ops

        self.cos_sin_cache = self.cos_sin_cache.to(positions.device,
                                                   dtype=query.dtype)
        # ops.rotary_embedding()/batched_rotary_embedding()
        # are in-place operations that update the query and key tensors.
        if key is None:
            # XPU kernel doesn't support key=None so fall back to native impl
            # TODO(sarckk): add support for optional key in
            # ipex.llm.functional.rotary_embedding_batched
            return self.forward_native(positions, query, key, offsets)
        else:
            if offsets is not None:
                ops.batched_rotary_embedding(positions, query, key,
                                             self.head_size,
                                             self.cos_sin_cache,
                                             self.is_neox_style,
                                             self.rotary_dim, offsets)
            else:
                ops.rotary_embedding(positions, query, key, self.head_size,
                                     self.cos_sin_cache, self.is_neox_style)
        return query, key

    def forward_neuron(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        def _apply_rotary_emb_neuron(
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            is_neox_style: bool,
        ) -> torch.Tensor:
            cos = cos.unsqueeze(-2).to(x.dtype)
            sin = sin.unsqueeze(-2).to(x.dtype)
            if is_neox_style:
                x1, x2 = torch.chunk(x, 2, dim=-1)
            else:
                # x1 = x[..., ::2]

                # x2 = x[..., 1::2]
                d = x.shape[-1] // 2
                x_reshaped = x.view(-1, x.shape[-1])
                x1 = x_reshaped[:, ::2].view(*x.shape[:-1], d)
                x2 = x_reshaped[:, 1::2].view(*x.shape[:-1], d)
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            if is_neox_style:
                return torch.cat((o1, o2), dim=-1)
            else:
                return torch.stack((o1, o2), dim=-1).flatten(-2)

        if offsets is not None:
            positions = positions + offsets

        self.cos_sin_cache = self.cos_sin_cache.to(query.device,
                                                   dtype=query.dtype)

        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)

        if self.rotary_dim == self.head_size:
            query = apply_rotary_emb_dispatch(query, cos, sin,
                                              self.is_neox_style)
            query = query.reshape(query_shape)
            if key is not None:
                key = apply_rotary_emb_dispatch(key, cos, sin,
                                                self.is_neox_style)
                key = key.reshape(key_shape)
        else:
            head_size = query.shape[-1]
            query_reshaped = query.view(-1, head_size)
            query_pass = query_reshaped[:, self.rotary_dim:].view(
                *query.shape[:-1], head_size - self.rotary_dim)
            query_rot = query_reshaped[:, :self.rotary_dim].view(
                *query.shape[:-1], self.rotary_dim)
            query_rot = _apply_rotary_emb_neuron(query_rot, cos, sin,
                                                 self.is_neox_style)
            query = torch.cat((query_rot, query_pass),
                              dim=-1).reshape(query_shape)

            if key is not None:
                key_reshaped = key.view(-1, head_size)
                key_pass = key_reshaped[:, self.rotary_dim:].view(
                    *key.shape[:-1], head_size - self.rotary_dim)
                key_rot = key_reshaped[:, :self.rotary_dim].view(
                    *key.shape[:-1], self.rotary_dim)
                key_rot = _apply_rotary_emb_neuron(key_rot, cos, sin,
                                                   self.is_neox_style)
                key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s
