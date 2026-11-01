# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped W8A16 skinny GEMM for 128x128 block-scaled FP8 experts."""

import pytest
import torch

from vllm.model_executor.kernels.linear.cute_dsl.skinny_fp8_moe_gemm import (
    QUANT_BLOCK,
    can_use_skinny_fp8_moe_gemm,
    shape_dynamic_skinny_fp8_moe_gemm,
)
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import SkinnyGemmConfig
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm_plans import (
    SKINNY_FP8_MOE_GEMM_PLANS_BY_CAPABILITY,
)
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA only", allow_module_level=True)
if not shape_dynamic_skinny_fp8_moe_gemm.is_available():
    pytest.skip("cuteDSL is not available", allow_module_level=True)

_CAPABILITY = current_platform.get_device_capability()
if _CAPABILITY not in SKINNY_FP8_MOE_GEMM_PLANS_BY_CAPABILITY:
    pytest.skip(
        f"no skinny FP8 MoE GEMM plans for {_CAPABILITY}", allow_module_level=True
    )

_PLANS = SKINNY_FP8_MOE_GEMM_PLANS_BY_CAPABILITY[_CAPABILITY]
_CASES = [
    (n, k, num_tokens, config)
    for (n, k), plan in sorted(_PLANS.items())
    for num_tokens, config in sorted(plan.items())
]


def _block_fp8(n: int, k: int, num_experts: int):
    weight = torch.randn(num_experts, n, k, device="cuda") * 0.02
    scale = (
        torch.rand(num_experts, n // QUANT_BLOCK, k // QUANT_BLOCK, device="cuda")
        * 0.01
        + 0.005
    )
    scale = torch.exp2(torch.round(torch.log2(scale)))  # exact in FP8 * FP32
    weight_q = (
        weight
        / scale.repeat_interleave(QUANT_BLOCK, 1).repeat_interleave(QUANT_BLOCK, 2)
    ).to(torch.float8_e4m3fn)
    dequant = weight_q.float() * scale.repeat_interleave(
        QUANT_BLOCK, 1
    ).repeat_interleave(QUANT_BLOCK, 2)
    return weight_q.contiguous(), scale.contiguous(), dequant


@pytest.mark.parametrize("n,k,num_tokens,config", _CASES)
@pytest.mark.parametrize("a_per_token", [True, False])
@pytest.mark.parametrize("mul_routed_weight", [False, True])
@torch.inference_mode()
def test_skinny_fp8_moe_gemm_matches_reference(
    n: int,
    k: int,
    num_tokens: int,
    config: SkinnyGemmConfig,
    a_per_token: bool,
    mul_routed_weight: bool,
) -> None:
    torch.manual_seed(0)
    num_experts, top_k = 16, 4
    weight, scale, dequant = _block_fp8(n, k, num_experts)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device="cuda")[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.rand(num_tokens, top_k, device="cuda"), -1)
    a_rows = num_tokens if a_per_token else num_tokens * top_k
    a = torch.randn(a_rows, k, device="cuda", dtype=torch.bfloat16)

    assert can_use_skinny_fp8_moe_gemm(
        a, weight, scale, topk_ids, config, a_per_token=a_per_token
    )
    out = shape_dynamic_skinny_fp8_moe_gemm(
        a,
        weight,
        scale,
        topk_ids,
        topk_weights if mul_routed_weight else None,
        config,
        a_per_token=a_per_token,
    )

    ids = topk_ids.view(-1).long()
    rows = a[torch.arange(ids.numel(), device="cuda") // top_k] if a_per_token else a
    ref = torch.einsum("pk,pnk->pn", rows.float(), dequant[ids])
    if mul_routed_weight:
        ref = ref * topk_weights.view(-1, 1)
    assert out.shape == (ids.numel(), n) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)

    # Padded pairs (expert id -1, as in warmup and profiling runs) and ids
    # outside the stack give zeros.
    padded_ids = topk_ids.clone()
    padded_ids[0, 0] = -1
    if top_k > 1:
        padded_ids[0, 1] = num_experts
    out = shape_dynamic_skinny_fp8_moe_gemm(
        a,
        weight,
        scale,
        padded_ids,
        topk_weights if mul_routed_weight else None,
        config,
        a_per_token=a_per_token,
    )
    assert torch.all(out[:2] == 0)
    torch.testing.assert_close(out[2:].float(), ref[2:], atol=1e-2, rtol=1e-2)


@torch.inference_mode()
def test_skinny_fp8_moe_gemm_rejects_bad_inputs() -> None:
    (n, k), plan = next(iter(sorted(_PLANS.items())))
    num_tokens, config = next(iter(sorted(plan.items())))
    weight, scale, _ = _block_fp8(n, k, 8)
    topk_ids = torch.zeros(num_tokens, 2, dtype=torch.int32, device="cuda")
    a = torch.randn(num_tokens, k, device="cuda", dtype=torch.bfloat16)
    ok = dict(a_per_token=True)
    assert can_use_skinny_fp8_moe_gemm(a, weight, scale, topk_ids, config, **ok)
    # int64 ids, fp16 activations, a pair-major activation of the wrong row
    # count, and a plan for several activation rows are all rejected.
    assert not can_use_skinny_fp8_moe_gemm(
        a, weight, scale, topk_ids.long(), config, **ok
    )
    assert not can_use_skinny_fp8_moe_gemm(
        a.half(), weight, scale, topk_ids, config, **ok
    )
    assert not can_use_skinny_fp8_moe_gemm(
        a, weight, scale, topk_ids, config, a_per_token=False
    )
    assert not can_use_skinny_fp8_moe_gemm(
        a, weight, scale, topk_ids, SkinnyGemmConfig(2, 64, 8), **ok
    )
    # Routing weights must be FP32 with the ids' shape.
    assert not can_use_skinny_fp8_moe_gemm(
        a, weight, scale, topk_ids, config, topk_weights=torch.ones_like(a[:, :2]), **ok
    )
    with pytest.raises(ValueError):
        shape_dynamic_skinny_fp8_moe_gemm(
            a, weight, scale, topk_ids.long(), None, config, a_per_token=True
        )


@torch.inference_mode()
def test_skinny_fp8_moe_gemm_equals_dense_skinny_gemm() -> None:
    """One expert per pair reproduces the dense W8A16 skinny GEMM's rounding."""
    from vllm.model_executor.kernels.linear.cute_dsl.skinny_fp8_gemm import (
        shape_dynamic_skinny_fp8_gemm,
    )

    (n, k), plan = next(iter(sorted(_PLANS.items())))
    config = plan[1]
    weight, scale, _ = _block_fp8(n, k, 4)
    a = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.tensor([[2]], dtype=torch.int32, device="cuda")
    grouped = shape_dynamic_skinny_fp8_moe_gemm(
        a, weight, scale, topk_ids, None, config, a_per_token=True
    )
    dense = shape_dynamic_skinny_fp8_gemm(a, weight[2], scale[2], config)
    torch.testing.assert_close(grouped.float(), dense.float(), atol=2e-2, rtol=2e-2)
