# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4's shared M-RoPE helpers must rotate exactly like vLLM's native M-RoPE.

`set_token_rope` folds the T/H/W cos/sin channels into one row per token so the
1-D RoPE kernels (`rope_into`, `apply_mrope`, `fused_mrope_norm`) can serve
[3, T] positions; any channel mixup silently corrupts vision-language tokens.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_cuda_alike(), reason="HY4 M-RoPE kernels need a GPU"
    ),
    pytest.mark.usefixtures("default_vllm_config"),
]

DEVICE = "cuda"
ROPE_DIM = 64
MAX_POSITION = 4096
# The kernels rotate in fp32 from a bf16 cos/sin cache; one bf16 ulp at |x| ~ 4.
BF16_TOL = dict(atol=2e-2, rtol=2e-2)


def _rotary_emb(interleaved: bool):
    from vllm.model_executor.layers.rotary_embedding import get_rope

    return get_rope(
        ROPE_DIM,
        max_position=MAX_POSITION,
        rope_parameters={
            "mrope_interleaved": interleaved,
            "mrope_section": [12, 10, 10],
            "rope_theta": 1e7,
            "rope_type": "default",
        },
        is_neox_style=False,
    ).to(DEVICE)


def _positions(num_tokens: int, gen: torch.Generator) -> torch.Tensor:
    """Distinct T/H/W rows, so a swapped channel changes the rotation."""
    return torch.randint(0, MAX_POSITION, (3, num_tokens), generator=gen).to(DEVICE)


def _native_mrope(rotary_emb, positions, x):
    """VLLM's native M-RoPE of `x` (tokens, heads, rot_dim), in fp32."""
    x = x.float()
    out, _ = rotary_emb.forward_native(positions, x.clone(), x[:, :1].clone())
    return out.view_as(x)


def _rope_reference(x, positions, cos_sin_cache):
    """Interleaved (GPT-J) RoPE in fp32: pairs (2i, 2i+1) rotate by row i."""
    cos, sin = cos_sin_cache[positions].float().chunk(2, dim=-1)
    cos, sin = cos[:, None], sin[:, None]
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    out = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return out.flatten(-2).to(x.dtype)


@pytest.mark.parametrize("interleaved", [True, False])
@pytest.mark.parametrize("num_tokens", [1, 37, 300])
@torch.inference_mode()
def test_apply_mrope_matches_native_mrope(interleaved, num_tokens):
    """Strided q_pe/k_pe rotated through the token rows equal forward_native."""
    from vllm.models.hy_v4.common.mrope import apply_mrope, set_token_rope

    gen = torch.Generator().manual_seed(num_tokens)
    rotary_emb = _rotary_emb(interleaved)
    positions = _positions(num_tokens, gen)
    q = torch.randn(num_tokens, 8, 256, generator=gen).to(DEVICE, torch.bfloat16)
    k = torch.randn(num_tokens, 1, ROPE_DIM, generator=gen).to(DEVICE, torch.bfloat16)
    q_pe = q[..., 192:]
    expected_q = _native_mrope(rotary_emb, positions, q_pe)
    expected_k = _native_mrope(rotary_emb, positions, k)
    q_nope = q[..., :192].clone()

    set_token_rope(rotary_emb, positions, q)
    apply_mrope(rotary_emb, positions, q_pe, k)

    torch.testing.assert_close(q_pe.float(), expected_q, **BF16_TOL)
    torch.testing.assert_close(k.float(), expected_k, **BF16_TOL)
    torch.testing.assert_close(q[..., :192], q_nope, rtol=0, atol=0)


@torch.inference_mode()
def test_token_rope_passes_text_positions_through():
    """1-D text positions keep the shared cos/sin cache; no rows are built."""
    from vllm.models.hy_v4.common.mrope import set_token_rope, token_rope

    rotary_emb = _rotary_emb(interleaved=True)
    positions = torch.arange(5, device=DEVICE)
    x = torch.empty(5, ROPE_DIM, device=DEVICE, dtype=torch.bfloat16)
    cache = rotary_emb._match_cos_sin_cache_dtype(x)

    set_token_rope(rotary_emb, positions, x)
    rope_pos, rows = token_rope(rotary_emb, positions, cache)

    assert rotary_emb._hy4_token_rope is None
    assert rope_pos is positions and rows is cache


@torch.inference_mode()
def test_token_rope_rejects_stale_rows():
    """Rows built for another forward's token count must not be reused."""
    from vllm.models.hy_v4.common.mrope import set_token_rope, token_rope

    gen = torch.Generator().manual_seed(0)
    rotary_emb = _rotary_emb(interleaved=True)
    x = torch.empty(4, ROPE_DIM, device=DEVICE, dtype=torch.bfloat16)
    set_token_rope(rotary_emb, _positions(4, gen), x)

    with pytest.raises(AssertionError, match="set_token_rope must run"):
        token_rope(rotary_emb, _positions(5, gen), rotary_emb.cos_sin_cache)


@pytest.mark.parametrize("num_heads", [1, 5, 16, 24, 64])
@pytest.mark.parametrize("num_tokens", [0, 1, 129])
@torch.inference_mode()
def test_rope_into_matches_reference(num_heads, num_tokens):
    """In-place on a strided q_pe slice, as HYV4 attention calls it; the head
    count picks the per-program head block, so cover non-multiples of 16."""
    from vllm.models.hy_v4.common.mrope import rope_into

    gen = torch.Generator().manual_seed(num_heads * 1000 + num_tokens)
    cache = _rotary_emb(interleaved=True).cos_sin_cache.to(torch.bfloat16)
    positions = torch.randint(0, MAX_POSITION, (num_tokens,), generator=gen)
    positions = positions.to(DEVICE)
    q = torch.randn(num_tokens, num_heads, 256, generator=gen)
    q = q.to(DEVICE, torch.bfloat16)
    q_pe = q[..., 192:]
    expected = _rope_reference(q_pe, positions, cache)
    q_nope = q[..., :192].clone()

    rope_into(positions, q_pe, cache, q_pe)

    torch.testing.assert_close(q_pe, expected)
    torch.testing.assert_close(q[..., :192], q_nope, rtol=0, atol=0)


@torch.inference_mode()
def test_rope_into_graph_replay_reads_new_inputs():
    """A captured `rope_into` re-reads positions and src on every replay."""
    from vllm.models.hy_v4.common.mrope import rope_into

    gen = torch.Generator().manual_seed(0)
    cache = _rotary_emb(interleaved=True).cos_sin_cache.to(torch.bfloat16)
    positions = torch.zeros(16, dtype=torch.int64, device=DEVICE)
    src = torch.zeros(16, 16, ROPE_DIM, dtype=torch.bfloat16, device=DEVICE)
    dst = torch.empty_like(src)
    rope_into(positions, src, cache, dst)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rope_into(positions, src, cache, dst)

    for _ in range(2):
        positions.copy_(torch.randint(0, MAX_POSITION, (16,), generator=gen))
        src.copy_(torch.randn(16, 16, ROPE_DIM, generator=gen))
        graph.replay()
        torch.testing.assert_close(dst, _rope_reference(src, positions, cache))


@pytest.mark.parametrize("num_tokens", [1, 70])
@torch.inference_mode()
def test_fused_mrope_norm_matches_reference(num_tokens):
    """RMS-normed q_c/kv_c and an M-RoPE'd k_pe from one fused qkv_a output,
    with the token rows `rope_into` then reuses for q_pe."""
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.hy_v4.common.mrope import fused_mrope_norm, set_token_rope

    gen = torch.Generator().manual_seed(num_tokens)
    q_lora_rank = 128
    rotary_emb = _rotary_emb(interleaved=True)
    positions = _positions(num_tokens, gen)
    qkv = torch.randn(num_tokens, q_lora_rank + 512 + ROPE_DIM, generator=gen)
    qkv = qkv.to(DEVICE, torch.bfloat16)
    q_c, kv_lora = qkv.split([q_lora_rank, 512 + ROPE_DIM], dim=-1)
    norms = []
    for width in (q_lora_rank, 512):
        norm = RMSNorm(width, eps=1e-5).to(DEVICE, torch.bfloat16)
        norm.weight.data.copy_(torch.rand(width, generator=gen) + 0.5)
        norms.append(norm)
    q_norm, kv_norm = norms
    k_pe = kv_lora[:, 512:]
    expected_k = _native_mrope(rotary_emb, positions, k_pe.unsqueeze(1))

    set_token_rope(rotary_emb, positions, qkv)
    q_out, kv_out, k_out, rope_pos, cache = fused_mrope_norm(
        rotary_emb, positions, q_c, kv_lora, q_norm, kv_norm
    )

    for out, x, norm in ((q_out, q_c, q_norm), (kv_out, kv_lora[:, :512], kv_norm)):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        expected = x.float() * torch.rsqrt(variance + 1e-5)
        expected = expected * norm.weight.float()
        torch.testing.assert_close(out.float(), expected, **BF16_TOL)
    torch.testing.assert_close(k_out.float(), expected_k, **BF16_TOL)
    torch.testing.assert_close(
        _rope_reference(k_pe.unsqueeze(1), rope_pos, cache), k_out
    )
