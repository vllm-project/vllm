# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-identical correctness test for the fused RoPE + reshape_and_cache_flash
CUDA kernel.

The fused op is compared against the unfused reference pipeline
``rotary_embedding`` + ``reshape_and_cache_flash``. Both paths share the same
fp32-intermediate RoPE math and the same ``fp8::scaled_convert`` template, so
outputs should match bit-exactly. Tests assert ``rtol=0, atol=0``; any
non-zero diff indicates a real divergence rather than a tolerance issue.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# (qkv dtype, kv_cache_dtype string)
DTYPE_CACHE = [
    (torch.bfloat16, "auto"),
    (torch.float16, "auto"),
    (torch.bfloat16, "fp8_e4m3"),
    (torch.float16, "fp8_e4m3"),
]

# (num_q_heads, num_kv_heads, head_size): one MHA, one GQA.
HEAD_CONFIGS = [
    (32, 32, 128),  # MHA — Llama-3 8B
    (32, 8, 128),  # GQA — Llama-3 70B
]

IS_NEOX = [True, False]
NUM_TOKENS = [1, 8, 128, 2048]

BLOCK_SIZE = 16
NUM_BLOCKS = 256
MAX_POS = 4096
SEED = 0


def _maxdiff_uint8(a: torch.Tensor, b: torch.Tensor) -> int:
    """Bytewise max-abs diff, for fp8 tensors that don't support arithmetic."""
    return (a.view(torch.uint8).int() - b.view(torch.uint8).int()).abs().max().item()


@pytest.mark.parametrize("dtype_cache", DTYPE_CACHE)
@pytest.mark.parametrize("head_config", HEAD_CONFIGS)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@torch.inference_mode()
def test_fused_rope_and_reshape_cache_flash(
    dtype_cache: tuple[torch.dtype, str],
    head_config: tuple[int, int, int],
    is_neox: bool,
    num_tokens: int,
) -> None:
    dtype, kv_cache_dtype = dtype_cache
    num_q_heads, num_kv_heads, head_size = head_config
    rot_dim = head_size
    device = torch.device("cuda")

    set_random_seed(SEED)

    query = torch.randn(num_tokens, num_q_heads, head_size, dtype=dtype, device=device)
    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device=device)
    value = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device=device)
    cos_sin_cache = torch.randn(MAX_POS, rot_dim, dtype=dtype, device=device)
    positions = torch.randint(
        0, MAX_POS, (num_tokens,), dtype=torch.long, device=device
    )

    # Unique slot ids so that "last write wins" cannot make fused vs unfused
    # diverge under non-deterministic CUDA block scheduling.
    total_slots = NUM_BLOCKS * BLOCK_SIZE
    assert num_tokens <= total_slots
    slot_mapping = torch.randperm(total_slots, device=device)[:num_tokens].to(
        torch.long
    )
    if num_tokens >= 4:
        # Exercise the padded-slot branch.
        slot_mapping[0] = -1

    cache_dtype = current_platform.fp8_dtype() if kv_cache_dtype != "auto" else dtype
    key_cache = torch.zeros(
        NUM_BLOCKS,
        BLOCK_SIZE,
        num_kv_heads,
        head_size,
        dtype=cache_dtype,
        device=device,
    )
    value_cache = torch.zeros_like(key_cache)

    # Non-trivial scales so the FP8 path actually exercises scaled_convert.
    k_scale = torch.tensor([0.7], dtype=torch.float32, device=device)
    v_scale = torch.tensor([1.3], dtype=torch.float32, device=device)

    # Reference path uses clones (rotary_embedding mutates query/key in place).
    q_ref = query.clone()
    k_ref = key.clone()
    kc_ref = key_cache.clone()
    vc_ref = value_cache.clone()

    # Fused path.
    ops.fused_rope_and_reshape_cache_flash(
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        is_neox,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale,
        v_scale,
        kv_cache_dtype,
    )

    # Reference: separate rotary_embedding + reshape_and_cache_flash.
    torch.ops._C.rotary_embedding(
        positions,
        q_ref,
        k_ref,
        head_size,
        cos_sin_cache,
        is_neox,
        0,
        False,
    )
    ops.reshape_and_cache_flash(
        k_ref,
        value,
        kc_ref,
        vc_ref,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )

    # Bit-identical assertions. For fp8 cache, compare as uint8 bytes since
    # torch.float8_e4m3fn does not support arithmetic.
    torch.testing.assert_close(query, q_ref, rtol=0, atol=0)
    torch.testing.assert_close(key, k_ref, rtol=0, atol=0)

    if cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        assert _maxdiff_uint8(key_cache, kc_ref) == 0, "key_cache bytes differ"
        assert _maxdiff_uint8(value_cache, vc_ref) == 0, "value_cache bytes differ"
    else:
        torch.testing.assert_close(key_cache, kc_ref, rtol=0, atol=0)
        torch.testing.assert_close(value_cache, vc_ref, rtol=0, atol=0)
