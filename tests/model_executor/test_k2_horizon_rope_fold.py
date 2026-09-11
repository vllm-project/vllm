# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.k2_horizon import (
    K2HorizonRMSNorm,
    _rope_weight_perm,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type


def split_to_interleaved(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(*x.shape[:-1], 2, -1).transpose(-1, -2).reshape(*x.shape[:-1], -1)


def interleaved_to_split(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(*x.shape[:-1], -1, 2).transpose(-1, -2).reshape(*x.shape[:-1], -1)


def get_gptj_rope(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_head_dim: int,
    max_position: int,
    base: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_emb = get_rope(
        rope_head_dim,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters={"rope_theta": base},
        dtype=dtype,
    )

    if rope_head_dim == head_dim:
        return rotary_emb(positions, q, k)

    q = q.reshape(*q.shape[:-1], num_heads, head_dim)
    k = k.reshape(*k.shape[:-1], num_kv_heads, head_dim)

    q_rope, q_nope = torch.split(
        split_to_interleaved(q),
        split_size_or_sections=[rope_head_dim, head_dim - rope_head_dim],
        dim=-1,
    )
    k_rope, k_nope = torch.split(
        split_to_interleaved(k),
        split_size_or_sections=[rope_head_dim, head_dim - rope_head_dim],
        dim=-1,
    )

    q_rope, k_rope = rotary_emb(
        positions,
        interleaved_to_split(q_rope).contiguous(),
        interleaved_to_split(k_rope).contiguous(),
    )

    q = interleaved_to_split(
        torch.cat([split_to_interleaved(q_rope), q_nope], dim=-1)
    ).reshape(*q.shape[:-2], -1)
    k = interleaved_to_split(
        torch.cat([split_to_interleaved(k_rope), k_nope], dim=-1)
    ).reshape(*k.shape[:-2], -1)
    return q, k


def _apply_head_perm(
    x: torch.Tensor, num: int, head_dim: int, idx: torch.Tensor
) -> torch.Tensor:
    return (
        x.reshape(*x.shape[:-1], num, head_dim)[..., idx]
        .reshape(*x.shape)
        .contiguous()
    )


def get_neox_rope(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    rope_head_dim: int,
    head_dim: int,
    max_position: int,
    base: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_emb = get_rope(
        head_dim,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters={"rope_theta": base, "rope_dim": rope_head_dim},
        dtype=dtype,
    )
    return rotary_emb(positions, q, k)


def fold_qk_proj_weight(
    weight: torch.Tensor, head_dim: int, idx: torch.Tensor
) -> torch.Tensor:
    hidden = weight.shape[-1]
    return (
        weight.view(-1, head_dim, hidden)[:, idx, :].reshape(-1, hidden).contiguous()
    )


def fold_qk_norm_weight(
    weight: torch.Tensor, head_dim: int, idx: torch.Tensor
) -> torch.Tensor:
    return weight.view(-1, head_dim)[:, idx].reshape(-1).contiguous()


SHAPES = [
    (128, 64),
    (128, 128),
    (192, 128),
    (96, 32),
]
DTYPES = [torch.float32, torch.bfloat16]


@pytest.mark.parametrize("head_dim,rope_head_dim", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rope_fold_matches_permute_then_rope(
    head_dim, rope_head_dim, dtype, default_vllm_config
):
    set_random_seed(0)
    num_tokens = 17
    num_heads = 8
    num_kv_heads = 2
    hidden_size = 512
    max_position = 4096
    base = 10000.0

    torch.set_default_dtype(dtype)

    positions = torch.randint(0, max_position, (num_tokens,), device=DEVICE)

    scale = hidden_size**-0.5
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=DEVICE)
    w_q = (
        torch.randn(num_heads * head_dim, hidden_size, dtype=dtype, device=DEVICE)
        * scale
    )
    w_k = (
        torch.randn(num_kv_heads * head_dim, hidden_size, dtype=dtype, device=DEVICE)
        * scale
    )

    idx = _rope_weight_perm(head_dim, rope_head_dim).to(DEVICE)

    q = torch.nn.functional.linear(x, w_q)
    k = torch.nn.functional.linear(x, w_k)
    q_gptj, k_gptj = get_gptj_rope(
        positions,
        q,
        k,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_head_dim,
        max_position,
        base,
        dtype,
    )

    q_folded = torch.nn.functional.linear(x, fold_qk_proj_weight(w_q, head_dim, idx))
    k_folded = torch.nn.functional.linear(x, fold_qk_proj_weight(w_k, head_dim, idx))
    q_neox, k_neox = get_neox_rope(
        positions,
        q_folded,
        k_folded,
        rope_head_dim,
        head_dim,
        max_position,
        base,
        dtype,
    )

    q_gptj_perm = _apply_head_perm(q_gptj, num_heads, head_dim, idx)
    k_gptj_perm = _apply_head_perm(k_gptj, num_kv_heads, head_dim, idx)

    if dtype == torch.float32:
        atol, rtol = 1e-3, 1e-3
    else:
        atol, rtol = 1e-2, 1e-2

    # Two complementary checks. First, an element-wise check that the folded
    # NeoX path equals the GPT-J path up to the channel permutation P
    # (neox_rope(Pq, Pk) == P @ gptj_rope(q, k)), hence permuting q_gptj/k_gptj.
    # Second, the scores check below: P cancels in q @ k.T.
    torch.testing.assert_close(q_neox, q_gptj_perm, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_neox, k_gptj_perm, atol=atol, rtol=rtol)

    def scores(qt, kt):
        qh = qt.reshape(num_tokens, num_heads, head_dim)
        kh = kt.reshape(num_tokens, num_kv_heads, head_dim)
        g = num_heads // num_kv_heads
        qh = qh.reshape(num_tokens, num_kv_heads, g, head_dim).float()
        kh = kh.float()
        return torch.einsum("tkgd,skd->tksg", qh, kh)

    torch.testing.assert_close(
        scores(q_neox, k_neox), scores(q_gptj, k_gptj), atol=atol, rtol=rtol
    )
