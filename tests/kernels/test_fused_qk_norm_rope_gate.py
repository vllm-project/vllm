# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# Qwen/Qwen3.6-27B config (huggingface.co/Qwen/Qwen3.6-27B), TP=1 shapes.
NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
PARTIAL_ROTARY_FACTOR = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)  # 64
RMS_NORM_EPS = 1e-6
MAX_POSITION_EMBEDDINGS = 262144
ROPE_THETA = 10000000.0

DTYPES = [torch.bfloat16]
SEEDS = [13]
NUM_TOKENS = [1, 4, 37]


def _ref_qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference: split + RMSNorm + partial NeoX RoPE + gate extraction.

    Matches ``fused_qk_rmsnorm_rope_gate``'s contract: ``q_gamma`` / ``k_gamma``
    are the already-adjusted effective gammas (for GemmaRMSNorm the caller
    has done ``weight + 1`` before passing them in).
    """
    n_tokens = q_gate.shape[0]
    half = rotary_dim // 2

    # Per head the q projection is laid out as [q | gate].
    q_gate = q_gate.view(n_tokens, num_q_heads, 2 * head_dim)
    q = q_gate[..., :head_dim]
    gate = q_gate[..., head_dim:].reshape(n_tokens, num_q_heads * head_dim)
    k = k.view(n_tokens, num_kv_heads, head_dim)

    def rms_norm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + eps)
        return (x * gamma.float()).to(orig_dtype)

    q = rms_norm(q, q_gamma)
    k = rms_norm(k, k_gamma)

    # Partial NeoX RoPE on the first ``rotary_dim`` elements of each head;
    # cos_sin_cache row is packed as [cos(half) | sin(half)].
    pos = positions.view(-1)
    cos = cos_sin_cache[pos, :half].float()[:, None, :]
    sin = cos_sin_cache[pos, half:rotary_dim].float()[:, None, :]

    def rope(x: torch.Tensor) -> torch.Tensor:
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        x1 = x_rot[..., :half].float()
        x2 = x_rot[..., half:].float()
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        rotated = torch.cat([o1, o2], dim=-1).to(x.dtype)
        return torch.cat([rotated, x_pass], dim=-1)

    q = rope(q).reshape(n_tokens, num_q_heads * head_dim)
    k = rope(k).reshape(n_tokens, num_kv_heads * head_dim)
    return q, k, gate


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_qk_rmsnorm_rope_gate Triton kernel requires CUDA/ROCm",
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@torch.inference_mode()
def test_fused_qk_norm_rope_gate_matches_reference(
    default_vllm_config,
    dtype: torch.dtype,
    seed: int,
    num_tokens: int,
):
    device = torch.device("cuda", torch.accelerator.current_device_index())
    torch.set_default_device(device)
    set_random_seed(seed)

    q_gate = torch.randn(
        num_tokens, NUM_Q_HEADS * 2 * HEAD_DIM, dtype=dtype, device=device
    )
    k = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=dtype, device=device)
    # GemmaRMSNorm-style: the kernel takes the effective gamma (weight + 1).
    q_gamma = (
        torch.empty(HEAD_DIM, dtype=dtype, device=device).normal_(mean=0.0, std=0.1)
        + 1.0
    )
    k_gamma = (
        torch.empty(HEAD_DIM, dtype=dtype, device=device).normal_(mean=0.0, std=0.1)
        + 1.0
    )

    # fused_qk_rmsnorm_rope_gate only handles NeoX-style RoPE.
    rope = RotaryEmbedding(
        head_size=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        base=ROPE_THETA,
        is_neox_style=True,
        dtype=dtype,
    ).to(device)
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)

    q_ref, k_ref, gate_ref = _ref_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        q_gamma,
        k_gamma,
        rope.cos_sin_cache,
        positions,
        RMS_NORM_EPS,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        ROTARY_DIM,
    )
    q_out, k_out, gate_out = fused_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        q_gamma,
        k_gamma,
        rope.cos_sin_cache,
        positions,
        RMS_NORM_EPS,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        ROTARY_DIM,
    )

    atol, rtol = 2e-3, 5e-3
    torch.testing.assert_close(q_out, q_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_out, k_ref, atol=atol, rtol=rtol)
    # gate is a verbatim copy of the source slice — must match bit-exactly.
    torch.testing.assert_close(gate_out, gate_ref, atol=0, rtol=0)
