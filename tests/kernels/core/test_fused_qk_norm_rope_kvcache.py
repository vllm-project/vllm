# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fused_qk_norm_rope_kvcache must reproduce fused_qk_norm_rope followed by
reshape_and_cache_flash bit for bit: same Q/K values (now in q_out/k_out
instead of qkv), same K/V rows in the paged cache, qkv left untouched."""

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

BLOCK_SIZE = 16
MAX_POSITION = 4096


def _make_inputs(
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    is_neox: bool,
    dtype: torch.dtype,
    device: str,
):
    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    positions = torch.randint(0, MAX_POSITION, (num_tokens,), device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    rope = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=MAX_POSITION,
        base=10000.0,
        is_neox_style=is_neox,
        dtype=dtype,
    ).to(device)
    return qkv, positions, q_weight, k_weight, rope.cos_sin_cache


def _make_flash_kv_cache(
    num_tokens: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype, device
):
    """One (num_blocks, H, N, 2*D) buffer split into strided key/value views,
    exactly as the FlashAttention backend hands them to the cache-write op."""
    num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 2
    kv = torch.randn(
        num_blocks, num_kv_heads, BLOCK_SIZE, 2 * head_dim, dtype=dtype, device=device
    )
    return kv


def _split_kv(kv: torch.Tensor, head_dim: int):
    return kv.transpose(1, 2).split(head_dim, dim=-1)


def _make_slot_mapping(num_tokens: int, num_slots: int, num_padded: int, device):
    slots = torch.randperm(num_slots, device=device)[:num_tokens].to(torch.int64)
    if num_padded:
        slots[num_tokens - num_padded :] = -1
    return slots


def _reference(
    qkv, positions, q_weight, k_weight, cos_sin_cache, is_neox, kv, geometry, slots
):
    num_heads, num_kv_heads, head_dim, eps = geometry
    qkv_ref = qkv.clone()
    ops.fused_qk_norm_rope(
        qkv_ref,
        num_heads,
        num_kv_heads,
        num_kv_heads,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        positions,
    )
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q_ref = qkv_ref[:, :q_size].view(-1, num_heads, head_dim)
    k_ref = qkv_ref[:, q_size : q_size + kv_size].view(-1, num_kv_heads, head_dim)
    v_ref = qkv_ref[:, q_size + kv_size :].view(-1, num_kv_heads, head_dim)
    kv_ref = kv.clone()
    key_cache, value_cache = _split_kv(kv_ref, head_dim)
    one = torch.tensor(1.0, dtype=torch.float32, device=qkv.device)
    ops.reshape_and_cache_flash(
        k_ref[: slots.shape[0]],
        v_ref[: slots.shape[0]],
        key_cache,
        value_cache,
        slots,
        "auto",
        one,
        one,
    )
    return q_ref, k_ref, kv_ref


def _run_and_check(
    qkv,
    positions,
    q_weight,
    k_weight,
    cos_sin_cache,
    is_neox,
    kv,
    geometry,
    slots,
    token_heads_per_warp=-1,
    check_op=False,
):
    num_heads, num_kv_heads, head_dim, eps = geometry
    q_ref, k_ref, kv_ref = _reference(
        qkv, positions, q_weight, k_weight, cos_sin_cache, is_neox, kv, geometry, slots
    )
    qkv_in = qkv.clone()
    q_out = torch.empty_like(q_ref)
    k_out = torch.empty_like(k_ref)
    kv_out = kv.clone()
    key_cache, value_cache = _split_kv(kv_out, head_dim)
    args = (
        qkv_in,
        q_out,
        k_out,
        num_heads,
        num_kv_heads,
        num_kv_heads,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        positions,
        key_cache,
        value_cache,
        slots,
        token_heads_per_warp,
    )
    if check_op:
        opcheck(torch.ops._C.fused_qk_norm_rope_kvcache, args)
    ops.fused_qk_norm_rope_kvcache(*args)
    assert torch.equal(qkv_in, qkv), "qkv must be read-only"
    assert torch.equal(q_out, q_ref)
    assert torch.equal(k_out, k_ref)
    assert torch.equal(kv_out, kv_ref)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="fused_qk_norm_rope_kvcache is CUDA-only"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_dim, rotary_dim",
    [
        (32, 8, 128, 128),
        (16, 4, 64, 64),
        (8, 2, 256, 256),
        (8, 8, 128, 128),
        (32, 8, 128, 64),
    ],
)
@pytest.mark.parametrize("num_tokens", [1, 5, 300, 2048])
@pytest.mark.parametrize("eps", [1e-6])
@torch.inference_mode()
def test_matches_unfused_pair(
    default_vllm_config,
    dtype,
    is_neox,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    num_tokens,
    eps,
):
    device = "cuda:0"
    set_random_seed(13)
    qkv, positions, q_weight, k_weight, cos_sin_cache = _make_inputs(
        num_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        is_neox,
        dtype,
        device,
    )
    kv = _make_flash_kv_cache(num_tokens, num_kv_heads, head_dim, dtype, device)
    slots = _make_slot_mapping(num_tokens, kv.shape[0] * BLOCK_SIZE, 0, device)
    _run_and_check(
        qkv,
        positions,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        kv,
        (num_heads, num_kv_heads, head_dim, eps),
        slots,
        check_op=num_tokens == 5,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="fused_qk_norm_rope_kvcache is CUDA-only"
)
@pytest.mark.parametrize("token_heads_per_warp", [1, 2, 4, 8])
@pytest.mark.parametrize("num_tokens", [5, 2048])
@pytest.mark.parametrize("is_neox", [True, False])
@torch.inference_mode()
def test_all_token_heads_per_warp(
    default_vllm_config, token_heads_per_warp, num_tokens, is_neox
):
    device, dtype = "cuda:0", torch.bfloat16
    set_random_seed(13)
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    qkv, positions, q_weight, k_weight, cos_sin_cache = _make_inputs(
        num_tokens, num_heads, num_kv_heads, head_dim, head_dim, is_neox, dtype, device
    )
    kv = _make_flash_kv_cache(num_tokens, num_kv_heads, head_dim, dtype, device)
    slots = _make_slot_mapping(num_tokens, kv.shape[0] * BLOCK_SIZE, 0, device)
    _run_and_check(
        qkv,
        positions,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        kv,
        (num_heads, num_kv_heads, head_dim, 1e-6),
        slots,
        token_heads_per_warp=token_heads_per_warp,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="fused_qk_norm_rope_kvcache is CUDA-only"
)
@pytest.mark.parametrize(
    "num_tokens, num_slot_tokens, num_padded",
    [
        (64, 64, 7),  # CUDA-graph padding: trailing slots are -1
        (64, 40, 0),  # slot_mapping shorter than qkv (num_actual_tokens < padded)
        (300, 290, 5),
    ],
)
@torch.inference_mode()
def test_padded_and_short_slot_mapping(
    default_vllm_config, num_tokens, num_slot_tokens, num_padded
):
    """Rows without a slot are still normalised/rotated but never cached."""
    device, dtype = "cuda:0", torch.bfloat16
    set_random_seed(13)
    num_heads, num_kv_heads, head_dim = 16, 4, 128
    qkv, positions, q_weight, k_weight, cos_sin_cache = _make_inputs(
        num_tokens, num_heads, num_kv_heads, head_dim, head_dim, True, dtype, device
    )
    kv = _make_flash_kv_cache(num_tokens, num_kv_heads, head_dim, dtype, device)
    slots = _make_slot_mapping(
        num_slot_tokens, kv.shape[0] * BLOCK_SIZE, num_padded, device
    )
    _run_and_check(
        qkv,
        positions,
        q_weight,
        k_weight,
        cos_sin_cache,
        True,
        kv,
        (num_heads, num_kv_heads, head_dim, 1e-6),
        slots,
    )
