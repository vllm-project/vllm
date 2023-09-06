from typing import Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import pos_encoding_ops

IS_NEOX_STYLE = [True, False]
DTYPES = [torch.half, torch.bfloat16, torch.float]
HEAD_SIZES = [64, 80, 96, 112, 128, 256]
ROTARY_DIMS = [None, 32]  # None means rotary dim == head size
NUM_HEADS = [7, 12, 40, 52]  # Arbitrary values for testing
NUM_TOKENS = [11, 83, 2048]  # Arbitrary values for testing
SEEDS = [0]


def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotate_fn = rotate_neox if is_neox_style else rotate_gptj
    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)
    return q_embed, k_embed


class RefRotaryEmbedding(nn.Module):
    """Reference implementation of rotary embedding."""

    def __init__(
        self,
        dim: int,
        is_neox_style: bool,
        max_position_embeddings: int = 8192,
        base: int = 10000,
    ) -> None:
        super().__init__()
        self.rotary_dim = dim
        self.is_neox_style = is_neox_style
        self.max_position_embeddings = max_position_embeddings

        # Create cos and sin embeddings.
        inv_freq = 1.0 / (base**(torch.arange(0, dim, 2) / dim))
        t = torch.arange(max_position_embeddings).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq.float())
        if is_neox_style:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.repeat_interleave(freqs, 2, -1)
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

        query_rot, key_rot = apply_rope(query_rot, key_rot, cos, sin,
                                        self.is_neox_style)
        query_rot = query_rot.transpose(0, 1).contiguous()
        key_rot = key_rot.transpose(0, 1).contiguous()

        query = torch.cat((query_rot, query_pass), dim=-1)
        key = torch.cat((key_rot, key_pass), dim=-1)

        # Output query/key shape: [num_tokens, num_tokens, head_size]
        return query, key


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_rotary_embedding(
    is_neox_style: bool,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    if rotary_dim is None:
        rotary_dim = head_size
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    positions = torch.randint(0, max_position, (num_tokens, ), device="cuda")
    query = torch.randn(num_tokens,
                        num_heads * head_size,
                        dtype=dtype,
                        device="cuda")
    key = torch.randn(num_tokens,
                      num_heads * head_size,
                      dtype=dtype,
                      device="cuda")

    # Create the rotary embedding.
    inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2) / rotary_dim))
    t = torch.arange(max_position).float()
    freqs = torch.einsum("i,j -> ij", t, inv_freq.float())
    cos = freqs.cos()
    sin = freqs.sin()
    cos_sin_cache = torch.cat((cos, sin), dim=-1)
    cos_sin_cache = cos_sin_cache.to(dtype=dtype, device='cuda')

    # Run the kernel. The kernel is in-place, so we need to clone the inputs.
    out_query = query.clone()
    out_key = key.clone()
    pos_encoding_ops.rotary_embedding(
        positions,
        out_query,
        out_key,
        head_size,
        cos_sin_cache,
        is_neox_style,
    )

    # Run the reference implementation.
    ref_rotary_embedding = RefRotaryEmbedding(
        dim=rotary_dim,
        is_neox_style=is_neox_style,
        max_position_embeddings=max_position,
        base=base,
    ).to(dtype=dtype, device="cuda")
    ref_query, ref_key = ref_rotary_embedding(
        positions,
        query.view(num_tokens, num_heads, head_size),
        key.view(num_tokens, num_heads, head_size),
    )
    ref_query = ref_query.view(num_tokens, num_heads * head_size)
    ref_key = ref_key.view(num_tokens, num_heads * head_size)

    # Compare the results.
    assert torch.allclose(out_query, ref_query, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_key, ref_key, atol=1e-5, rtol=1e-5)
