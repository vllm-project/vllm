###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import torch
import torch.nn as nn
import habana_frameworks.torch.utils.experimental as htexp


def get_device_type():
    return htexp._get_device_type()


def is_gaudi1():
    return get_device_type() == htexp.synDeviceType.synDeviceGaudi


def is_gaudi2():
    return get_device_type() == htexp.synDeviceType.synDeviceGaudi2


def is_gaudi3():
    return get_device_type() == htexp.synDeviceType.synDeviceGaudi3


# TODO: remove this workaround when FusedRoPE properly works on Gaudi
if not is_gaudi1() and (is_gaudi2() or is_gaudi3()):
    try:
        from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV1 as FusedRoPE
    except ImportError:
        print("Not using HPU fused kernel for apply_rotary_pos_emb")
        FusedRoPE = None
else:
    FusedRoPE = None


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids]  #.unsqueeze(unsqueeze_dim)
    sin = sin[position_ids]  #.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class HpuRotaryEmbedding(nn.Module):

    def __init__(self,
                 head_size,
                 rotary_dim,
                 max_position_embeddings=2048,
                 base=10000,
                 is_neox_style=None,
                 device='hpu'):
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

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached",
                             emb.cos().to(dtype),
                             persistent=False)
        self.register_buffer("sin_cached",
                             emb.sin().to(dtype),
                             persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor,
                key: torch.Tensor):
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
        if query.device.type == "hpu" and FusedRoPE:
            if len(positions[0]) == 1:
                cos = self.cos_cached[positions].unsqueeze(2).to(
                    dtype=query.dtype)
                sin = self.sin_cached[positions].unsqueeze(2).to(
                    dtype=query.dtype)
            else:
                cos = cos[positions].unsqueeze(2)
                sin = sin[positions].unsqueeze(2)
            query, key = FusedRoPE.apply(query, cos, sin,
                                         0), FusedRoPE.apply(key, cos, sin, 0)
        else:
            query, key = apply_rotary_pos_emb(query, key, cos, sin, positions)
        return query.reshape(
            (query.shape[0], query.shape[1],
             query.shape[2] * query.shape[3])), key.reshape(
                 (key.shape[0], key.shape[1], key.shape[2] * key.shape[3]))
