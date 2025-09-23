# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rotary Positional Embeddings Base Class."""
from typing import Optional

import torch

from vllm.model_executor.custom_op import CustomOp

from .common import apply_rotary_emb_torch
from .rocm_aiter_rope_ops import (is_rocm_rotary_embedding_enabled,
                                  is_rocm_triton_rotary_embedding_enabled,
                                  rocm_aiter_rotary_emb)


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
        # TODO(mgoin): disabled for now due to failures
        # Flashinfer only supports head_size=64, 128, 256, 512.
        # https://github.com/flashinfer-ai/flashinfer/blob/ebfd655efe830048dba5d582aaa61d61d1cf9a87/include/flashinfer/utils.cuh#L174-L202
        # self.use_flashinfer = (self.enabled()
        #                        and dtype in (torch.float16, torch.bfloat16)
        #                        and current_platform.is_cuda()
        #                        and has_flashinfer()
        #                        and self.head_size in [64, 128, 256, 512])
        self.use_flashinfer = False

        cache = self._compute_cos_sin_cache()
        if not self.use_flashinfer:
            cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        self.is_rocm_aiter_enabled = \
            is_rocm_rotary_embedding_enabled()
        self.is_rocm_aiter_triton_enabled = \
            is_rocm_triton_rotary_embedding_enabled(
        )

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

    def _match_cos_sin_cache_dtype(self, query: torch.Tensor) -> None:
        # __setattr__ in nn.Module (called by `self.cos_sin_cache = ...`)
        # is expensive, so avoid calling it if possible
        if self.cos_sin_cache.device != query.device or \
            self.cos_sin_cache.dtype != query.dtype:
            self.cos_sin_cache = self.cos_sin_cache.to(query.device,
                                                       dtype=query.dtype)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """A PyTorch-native implementation of forward()."""
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
        if self.use_flashinfer:
            torch.ops.vllm.flashinfer_rotary_embedding(positions, query, key,
                                                       self.head_size,
                                                       self.cos_sin_cache,
                                                       self.is_neox_style)
            return query, key

        from vllm import _custom_ops as ops

        self._match_cos_sin_cache_dtype(query)

        if is_rocm_triton_rotary_embedding_enabled() and \
            positions.numel() <= 128:
            rocm_aiter_rotary_emb(positions, query, key, self.cos_sin_cache,
                                  self.head_size, self.rotary_dim, offsets)
        else:
            # ops.rotary_embedding() is an in-place operation
            # that updates the query and key tensors.
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
        if self.is_rocm_aiter_triton_enabled:
            return self.forward_cuda(positions, query, key, offsets)
        elif self.is_rocm_aiter_enabled:
            return self.forward_hip_rocm_aiter(positions, query, key, offsets,
                                               is_nope_first)
        return self.forward_native(positions, query, key)

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
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        from vllm._ipex_ops import ipex_ops as ops

        self._match_cos_sin_cache_dtype(query)
        # ops.rotary_embedding() is an in-place operation
        # that updates the query and key tensors.
        if key is None:
            # XPU kernel doesn't support key=None so fall back to native impl
            # TODO(sarckk): add support for optional key in
            # ipex.llm.functional.rotary_embedding_batched
            return self.forward_native(positions, query, key)
        else:
            ops.rotary_embedding(positions, query, key, self.head_size,
                                 self.cos_sin_cache, self.is_neox_style)
        return query, key

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s
