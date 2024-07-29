###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.utils import is_hpu

logger = init_logger(__name__)

if is_hpu():
    try:
        from habana_frameworks.torch.hpex.kernels import (
            RotaryPosEmbeddingHelperV1 as FusedRoPE)
    except ImportError:
        logger.warning("Could not import HPU FusedRoPE kernel. "
                       "vLLM will use forward_native implementation of RoPE.")
    FusedRoPE = None
else:
    FusedRoPE = None


class HpuRotaryEmbedding(nn.Module):

    def __init__(self,
                 head_size,
                 rotary_dim,
                 max_position_embeddings=2048,
                 base=10000,
                 is_neox_style=None,
                 device='hpu',
                 RoPEFallback=None):
        super().__init__()

        self.head_size = head_size
        self.dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base**(
            torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings,
                                device=self.inv_freq.device,
                                dtype=torch.get_default_dtype())
        if FusedRoPE is None:
            assert RoPEFallback is not None, (
                "HPU FusedRoPE kernel could not be imported, and "
                "fallback RoPE implementation was not provided!")
            self.fallback_impl = RoPEFallback(head_size,
                                              rotary_dim,
                                              max_position_embeddings,
                                              base,
                                              is_neox_style,
                                              dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order
        # to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached",
                             emb.cos().to(dtype),
                             persistent=False)
        self.register_buffer("sin_cached",
                             emb.sin().to(dtype),
                             persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor,
                key: torch.Tensor):
        if FusedRoPE is None:
            return self.fallback_impl(positions, query, key)
        if query.dim() == 2:
            query = query.unsqueeze(0)
        if key.dim() == 2:
            key = key.unsqueeze(0)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        seq_len = key.shape[-2]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len,
                                    device=query.device,
                                    dtype=query.dtype)

        cos, sin = self.cos_cached[:seq_len].to(
            dtype=query.dtype), self.sin_cached[:seq_len].to(dtype=query.dtype)
        query = query.reshape(
            (query.shape[0], query.shape[1], query.shape[2] // self.head_size,
             self.head_size))
        key = key.reshape((key.shape[0], key.shape[1],
                           key.shape[2] // self.head_size, self.head_size))

        if len(positions[0]) == 1:
            cos = self.cos_cached[positions].unsqueeze(2).to(dtype=query.dtype)
            sin = self.sin_cached[positions].unsqueeze(2).to(dtype=query.dtype)
        else:
            cos = cos[positions].unsqueeze(2)
            sin = sin[positions].unsqueeze(2)
        query, key = FusedRoPE.apply(query, cos, sin,
                                     0), FusedRoPE.apply(key, cos, sin, 0)
        return query.reshape(
            (query.shape[0], query.shape[1],
             query.shape[2] * query.shape[3])), key.reshape(
                 (key.shape[0], key.shape[1], key.shape[2] * key.shape[3]))
