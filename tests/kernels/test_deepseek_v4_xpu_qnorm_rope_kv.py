# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for the DeepSeek V4 XPU qnorm/RoPE KV kernel."""

import pytest
import torch

from tests.kernels.test_fused_deepseek_v4_qnorm_rope_kv_insert import (
    HEAD_BYTES,
    HEAD_DIM,
    NOPE_DIM,
    ROPE_DIM,
    apply_rope_gptj_last_k,
    make_cos_sin_cache,
)
from vllm.models.deepseek_v4.common.ops import (
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (
    xpu_qnorm_rope_kv_fp8_insert,
)
from vllm.platforms import current_platform


pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="skip for non-XPU platform",
)


def test_xpu_kv_rope_region_matches_reference():
    """The KV bulk copy must not overwrite the separately rotated RoPE region."""
    torch.manual_seed(1)

    device = "xpu"
    dtype = torch.bfloat16
    num_tokens = 16
    num_heads = 1
    block_size = 16
    eps = 1e-6
    max_pos = 16

    kv = torch.randn(
        num_tokens,
        HEAD_DIM,
        dtype=dtype,
        device=device,
    )

    q = torch.zeros(
        num_tokens,
        num_heads,
        HEAD_DIM,
        dtype=dtype,
        device=device,
    )

    positions = torch.ones(
        num_tokens,
        dtype=torch.int64,
        device=device,
    )

    cos_sin_cache = make_cos_sin_cache(
        max_pos,
        ROPE_DIM,
        torch.float32,
        device,
    )

    slot_mapping = torch.arange(
        num_tokens,
        dtype=torch.int64,
        device=device,
    )

    num_blocks = (num_tokens + block_size - 1) // block_size + 1

    k_cache = torch.zeros(
        num_blocks,
        block_size * HEAD_BYTES,
        dtype=torch.uint8,
        device=device,
    )

    # Reference: apply GPT-J RoPE first, then use the same cache
    # quantization/insertion path as the XPU implementation.
    kv_ref = apply_rope_gptj_last_k(
        kv,
        positions,
        cos_sin_cache,
    )

    k_cache_ref = torch.zeros_like(k_cache)

    quantize_and_insert_k_cache(
        kv_ref,
        k_cache_ref,
        slot_mapping,
        block_size=block_size,
    )

    # XPU implementation under test.
    xpu_qnorm_rope_kv_fp8_insert(
        q,
        kv,
        k_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps,
        block_size,
    )

    def dequantize(k_cache_tensor):
        out = torch.zeros(
            1,
            num_tokens,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )

        seq_lens = torch.tensor(
            [num_tokens],
            dtype=torch.int32,
            device=device,
        )

        block_table = torch.arange(
            num_blocks,
            dtype=torch.int32,
            device=device,
        ).unsqueeze(0)

        k_cache_3d = k_cache_tensor.view(
            num_blocks,
            block_size,
            HEAD_BYTES,
        )

        dequantize_and_gather_k_cache(
            out,
            k_cache_3d,
            seq_lens,
            None,
            block_table,
            block_size,
            offset=0,
        )

        return out[0, :num_tokens]

    actual = dequantize(k_cache)
    expected = dequantize(k_cache_ref)

    # The regression is specifically about the RoPE region. The fix prevents
    # the unrotated bulk store from racing with these addresses.
    torch.testing.assert_close(
        actual[:, NOPE_DIM:],
        expected[:, NOPE_DIM:],
        rtol=1e-2,
        atol=1e-2,
    )
