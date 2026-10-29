# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the DeepSeek-V4 XPU qnorm/RoPE kernel on large batches.

``token_idx * num_heads * HEAD_DIM`` used to be computed in int32, so the Q
address wrapped once a step held more than 2^31 Q elements (65536 tokens with
64 heads) and the device faulted (UR_RESULT_ERROR_DEVICE_LOST).
"""

import pytest
import torch

from tests.kernels.test_fused_deepseek_v4_qnorm_rope_kv_insert import (
    HEAD_DIM,
    ROPE_DIM,
    apply_rope_gptj_last_k,
    make_cos_sin_cache,
    rmsnorm_no_weight,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="skip for non-XPU platform",
)


def test_xpu_qnorm_rope_past_int32_q_elements():
    from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (
        xpu_qnorm_rope_kv_fp8_insert,
    )

    num_heads = 64
    # First token whose Q offset no longer fits in int32, plus a margin.
    num_tokens = 2**31 // (num_heads * HEAD_DIM) + 64
    q_bytes = num_tokens * num_heads * HEAD_DIM * 2
    free, _ = torch.xpu.mem_get_info()
    if free < 2 * q_bytes:
        pytest.skip(f"needs {2 * q_bytes / 2**30:.1f} GiB of free device memory")

    torch.manual_seed(0)
    device = "xpu"
    dtype = torch.bfloat16
    eps = 1e-6
    max_pos = 4096

    q = torch.randn(num_tokens, num_heads, HEAD_DIM, dtype=dtype, device=device)
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=device
    )
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, device)

    # The kernel updates q in place: build the reference for a few rows on both
    # sides of the int32 boundary first.
    boundary = 2**31 // (num_heads * HEAD_DIM)
    rows = torch.cat(
        [
            torch.arange(0, 8, device=device),
            torch.arange(boundary - 8, num_tokens, device=device),
        ]
    )
    q_ref = rmsnorm_no_weight(q[rows], eps)
    q_ref = apply_rope_gptj_last_k(q_ref, positions[rows], cos_sin_cache).to(dtype)

    block_size = 64
    num_blocks = triton.cdiv(num_tokens, block_size)
    swa_kv_cache = torch.zeros(
        num_blocks, block_size, 584, dtype=torch.uint8, device=device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    xpu_qnorm_rope_kv_fp8_insert(
        q,
        kv,
        swa_kv_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps,
        block_size,
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(q[rows], q_ref, rtol=1e-2, atol=1e-2)
