from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RefRotaryEmbedding(nn.Module):
    """Reference implementation of the GPT-NeoX style rotary embedding."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
    ) -> None:
        super().__init__()
        self.max_position_embeddings = max_position_embeddings

        # Create cos and sin embeddings.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
        t = torch.arange(max_position_embeddings).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=inv_freq.dtype)
        sin = emb.sin().to(dtype=inv_freq.dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        positions: torch.LongTensor,
        query: torch.Tensor,            # [num_tokens, num_heads, head_size]
        key: torch.Tensor,              # [num_tokens, num_heads, head_size]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = F.embedding(positions, self.cos_cached)
        sin = F.embedding(positions, self.sin_cached)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        query = query.transpose(0, 1).contiguous()
        key = key.transpose(0, 1).contiguous()
        # Output query/key shape: [num_tokens, num_tokens, head_size]
        return query, key
