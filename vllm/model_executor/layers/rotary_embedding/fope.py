# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from .base import RotaryEmbedding
from .common import rotate_neox


class FourierRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        init_cache: bool,
        # extra parameters for FoPE
        num_key_value_heads: int,
        num_inv_freq: int,
        fope_sep_head: bool,
        fope_init_factor: float,
    ):
        # fope related parameters
        self.num_key_value_heads = num_key_value_heads
        self.num_inv_freq = num_inv_freq
        self.fope_sep_head = fope_sep_head
        self.fope_init_factor = fope_init_factor

        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
            init_cache=init_cache,
        )

        # setup buffers and parameters
        self.inv_freq: torch.Tensor
        self.register_buffer(
            "inv_freq", self._compute_inv_freq(self.base), persistent=False
        )

        self.input_dim = self.inv_freq.shape[-1]
        self.output_dim = self.inv_freq.shape[-1]
        self.cos_coef = nn.Parameter(
            torch.empty(num_key_value_heads, self.input_dim, self.output_dim),
            requires_grad=False,
        )
        self.sin_coef = nn.Parameter(
            torch.empty(num_key_value_heads, self.input_dim, self.output_dim),
            requires_grad=False,
        )
        self.sin_coef.weight_loader = self.weight_loader
        self.cos_coef.weight_loader = self.weight_loader

        self.cos_sin_cache: torch.Tensor
        cache = self._compute_cos_sin_cache().to(dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

        # update cache in the first forward, where sin/cos_coef weights are ready
        self.update_cache = True

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )

        inv_freq_idx_selected = torch.ones_like(inv_freq, dtype=torch.bool)
        if self.num_inv_freq is not None:
            inv_freq_idx_selected[self.num_inv_freq :] = False
        else:
            inv_freq_idx_selected = inv_freq > (
                2.0 * torch.pi / self.max_position_embeddings
            )

        inv_freq = inv_freq[inv_freq_idx_selected]
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        device = self.inv_freq.device
        t = torch.arange(self.max_position_embeddings, dtype=torch.float, device=device)

        freqs = torch.einsum("j,i -> ji", t, self.inv_freq)
        if self.fope_sep_head:
            pos_cos = freqs.cos().unsqueeze(0).expand(self.num_key_value_heads, -1, -1)
            pos_sin = freqs.sin().unsqueeze(0).expand(self.num_key_value_heads, -1, -1)
        else:
            pos_cos = freqs.cos()
            pos_sin = freqs.sin()

        if self.fope_sep_head:
            sin = torch.einsum("htD, hDd -> thd", pos_sin, self.sin_coef.float())
            cos = torch.einsum("htD, hDd -> thd", pos_cos, self.cos_coef.float())
        else:
            sin = torch.einsum("tD, Dd -> td", pos_sin, self.sin_coef.float())
            cos = torch.einsum("tD, Dd -> td", pos_cos, self.cos_coef.float())

        sin = F.pad(
            input=sin,
            pad=(0, self.head_size // 2 - sin.size(-1)),
            mode="constant",
            value=1,
        )
        cos = F.pad(
            input=cos,
            pad=(0, self.head_size // 2 - cos.size(-1)),
            mode="constant",
            value=1,
        )

        sin = torch.cat((sin, sin), dim=-1)
        cos = torch.cat((cos, cos), dim=-1)

        # cache: (max_position_embeddings, num_kv_heads, kv_size * 2)
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # update cos/sin cache in the first forward
        if self.update_cache:
            cache = self._compute_cos_sin_cache().to(self.dtype)
            self.cos_sin_cache.copy_(cache)
            self.update_cache = False

        positions = positions.flatten()
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        # apply rotary embedding
        # query: (seq_len, num_heads, head_size)
        # key: (seq_len, num_kv_heads, head_size)
        query = query.unflatten(-1, (-1, self.head_size))
        assert key is not None, "Key tensor is required for FoPE."
        key = key.unflatten(-1, (-1, self.head_size))

        assert query.dim() == key.dim() == 3, (
            "Expected query key (seq_len, heads, head_dim)"
        )
        assert cos.dim() <= 3 and sin.dim() <= 3

        need_reshape = False
        if cos.dim() == 3:
            # for fope
            need_reshape = True
            query_shape = query.shape
            key_shape = key.shape
            cos = cos.flatten(0, 1)
            sin = sin.flatten(0, 1)
            seq_len = cos.size(0)
            query = query.view(seq_len, -1, query.size(-1))
            key = key.view(seq_len, -1, key.size(-1))

        # native implementation of apply rope for neox style
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query = (query * cos) + (rotate_neox(query) * sin)
        key = (key * cos) + (rotate_neox(key) * sin)

        if need_reshape:
            query = query.view(query_shape)
            key = key.view(key_shape)

        return query, key

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """load fope weights"""
        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        num_key_value_heads = loaded_weight.size(0)

        if num_key_value_heads < world_size:
            n_replicate = world_size // num_key_value_heads
            world_size = num_key_value_heads
            rank = rank // n_replicate

        loaded_weight = loaded_weight.chunk(world_size, dim=0)[rank]
        param.data.copy_(loaded_weight)
