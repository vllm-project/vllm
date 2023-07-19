from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import pos_encoding_ops


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
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


class RefRotaryEmbeddingNeox(nn.Module):
    """Reference implementation of the GPT-NeoX style rotary embedding."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
    ) -> None:
        super().__init__()
        self.rotary_dim = dim
        self.max_position_embeddings = max_position_embeddings

        # Create cos and sin embeddings.
        inv_freq = 1.0 / (base**(torch.arange(0, dim, 2) / dim))
        t = torch.arange(max_position_embeddings).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=inv_freq.dtype)
        sin = emb.sin().to(dtype=inv_freq.dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,  # [num_tokens]
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]

        query_rot = query_rot.transpose(0, 1)
        key_rot = key_rot.transpose(0, 1)
        cos = F.embedding(positions, self.cos_cached)
        sin = F.embedding(positions, self.sin_cached)
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)
        query_rot = query_rot.transpose(0, 1).contiguous()
        key_rot = key_rot.transpose(0, 1).contiguous()

        query = torch.cat((query_rot, query_pass), dim=-1)
        key = torch.cat((key_rot, key_pass), dim=-1)

        # Output query/key shape: [num_tokens, num_tokens, head_size]
        return query, key


@torch.inference_mode()
def run_rotary_embedding_neox(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    max_position: int,
    rotary_dim: int,
    dtype: torch.dtype,
    base: int = 10000,
) -> None:
    positions = torch.randint(0, max_position, (num_tokens, ), device='cuda')
    query = torch.randn(num_tokens,
                        num_heads * head_size,
                        dtype=dtype,
                        device='cuda')
    key = torch.randn(num_tokens,
                      num_heads * head_size,
                      dtype=dtype,
                      device='cuda')

    # Create the rotary embedding.
    inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2) / rotary_dim))
    t = torch.arange(max_position).float()
    freqs = torch.einsum('i,j -> ij', t, inv_freq.float())
    cos = freqs.cos()
    sin = freqs.sin()
    cos_sin_cache = torch.cat((cos, sin), dim=-1)
    cos_sin_cache = cos_sin_cache.to(dtype=dtype, device='cuda')

    # Run the kernel. The kernel is in-place, so we need to clone the inputs.
    out_query = query.clone()
    out_key = key.clone()
    pos_encoding_ops.rotary_embedding_neox(
        positions,
        out_query,
        out_key,
        head_size,
        cos_sin_cache,
    )

    # Run the reference implementation.
    ref_rotary_embedding = RefRotaryEmbeddingNeox(
        dim=rotary_dim,
        max_position_embeddings=max_position,
        base=base,
    ).to(dtype=dtype, device='cuda')
    ref_query, ref_key = ref_rotary_embedding(
        positions,
        query.view(num_tokens, num_heads, head_size),
        key.view(num_tokens, num_heads, head_size),
    )
    ref_query = ref_query.view(num_tokens, num_heads * head_size)
    ref_key = ref_key.view(num_tokens, num_heads * head_size)

    # Compare the results.
    assert torch.allclose(out_query, ref_query, atol=1e-3, rtol=1e-5)
    assert torch.allclose(out_key, ref_key, atol=1e-3, rtol=1e-5)


def test_rotary_embedding_neox() -> None:
    for dtype in [torch.half, torch.bfloat16, torch.float]:
        for head_size in [32, 64, 80, 96, 128, 160, 192, 256]:
            print(f'Running tests for head_size={head_size} and dtype={dtype}')
            run_rotary_embedding_neox(
                num_tokens=2145,
                num_heads=5,
                head_size=head_size,
                max_position=8192,
                rotary_dim=head_size,
                dtype=dtype,
            )
