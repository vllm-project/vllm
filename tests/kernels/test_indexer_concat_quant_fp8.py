# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    indexer_concat_quant_fp8,
    per_token_group_quant_fp8,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("num_tokens", [1, 4, 16, 64, 128])
@pytest.mark.parametrize("num_heads", [64])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("rope_dim", [64])
@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize("use_ue8m0", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_indexer_concat_quant_fp8(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    group_size: int,
    use_ue8m0: bool,
    dtype: torch.dtype,
):
    set_random_seed(0)
    nope_dim = head_dim - rope_dim
    q_pe = torch.randn(num_tokens, num_heads, rope_dim, dtype=dtype, device="cuda")
    q_nope = torch.randn(num_tokens, num_heads, nope_dim, dtype=dtype, device="cuda")

    q_ref = torch.cat([q_pe, q_nope], dim=-1)
    q_ref = q_ref.view(-1, head_dim)
    q_fp8_ref, q_scale_ref = per_token_group_quant_fp8(
        q_ref, group_size, column_major_scales=False, use_ue8m0=use_ue8m0
    )
    q_fp8_ref = q_fp8_ref.view(-1, num_heads, head_dim)
    q_scale_ref = q_scale_ref.view(-1, num_heads, 1)

    q_fp8, q_scale = indexer_concat_quant_fp8(
        q_pe, q_nope, group_size, column_major_scales=False, use_ue8m0=use_ue8m0
    )

    torch.testing.assert_close(q_fp8.float(), q_fp8_ref.float(), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(q_scale, q_scale_ref, atol=2e-3, rtol=2e-3)
