# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("use_ue8m0", [False, True])
@pytest.mark.parametrize("clamp_limit", [None, 7.0])
def test_fused_quant_is_independent_of_cobatched_tokens(use_ue8m0, clamp_limit):
    assert fp8_utils.is_batch_invariant_quant_kernel_enabled()
    torch.manual_seed(42)
    x = torch.randn(5, 1024, device="cuda", dtype=torch.bfloat16) * 3
    inputs = (x[:2].contiguous(), x.contiguous())
    results = [
        fp8_utils.fused_silu_mul_per_token_group_quant_fp8(
            inp,
            use_ue8m0=use_ue8m0,
            clamp_limit=clamp_limit,
            masked_m=None,
        )
        for inp in inputs
    ]
    (small_q, small_s), (large_q, large_s) = results
    assert torch.equal(small_q.view(torch.uint8), large_q[:2].view(torch.uint8))
    assert torch.equal(small_s, large_s[:2])


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("groups", [16, 32])
@pytest.mark.parametrize("use_ue8m0", [False, True])
def test_masked_silu_quant_matches_contiguous(groups: int, use_ue8m0: bool):
    assert fp8_utils.is_batch_invariant_quant_kernel_enabled()
    torch.manual_seed(42)
    x = torch.randn(3, 5, 2 * groups * 128, device="cuda", dtype=torch.bfloat16)
    masked_m = torch.tensor([5, 2, 0], device="cuda", dtype=torch.int32)

    q, s = fp8_utils.fused_silu_mul_per_token_group_quant_fp8(
        x, use_ue8m0=use_ue8m0, masked_m=masked_m
    )
    for expert, count in enumerate(masked_m.tolist()):
        if count == 0:
            continue
        ref_q, ref_s = fp8_utils.fused_silu_mul_per_token_group_quant_fp8(
            x[expert, :count].contiguous(), use_ue8m0=use_ue8m0, masked_m=None
        )
        assert torch.equal(q[expert, :count].view(torch.uint8), ref_q.view(torch.uint8))
        assert torch.equal(s[expert, :count], ref_s)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("groups", [1, 2, 3, 8, 17])
def test_masked_silu_quant_rejects_nonmultiple_of_16_groups(groups: int):
    assert fp8_utils.is_batch_invariant_quant_kernel_enabled()
    x = torch.zeros(1, 1, 2 * groups * 128, device="cuda", dtype=torch.bfloat16)
    masked_m = torch.ones(1, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        fp8_utils.fused_silu_mul_per_token_group_quant_fp8(
            x, use_ue8m0=False, masked_m=masked_m
        )
