# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT4 kernels of the SDMA all-reduce against a torch model (one ROCm GPU)."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="SDMA all-reduce is ROCm only"
)


def _reference_int4_one_shot(xs: list[torch.Tensor], group: int = 32) -> torch.Tensor:
    """What DmaAllReduce computes: the sum over ranks of dq(q(x_r)) in fp32,
    then x's dtype."""
    acc = None
    for x in xs:
        g = x.float().reshape(-1, group)
        s = (
            torch.clamp(g.abs().amax(dim=1) / 7.0, max=65504.0)
            .to(torch.float16)
            .float()
        )
        inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
        q = torch.clamp(torch.floor(g * inv[:, None] + 0.5), -7, 7)
        d = q * s[:, None]
        acc = d if acc is None else acc + d
    return acc.reshape(xs[0].shape).to(xs[0].dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [1, 33, 2048])
def test_quantize_and_sum_matches_reference(dtype, tokens):
    from vllm.distributed.device_communicators.dma_all_reduce import (
        GROUP,
        _dq_sum_kernel,
        _quant_int4_kernel,
    )
    from vllm.triton_utils import triton

    hidden = 256
    torch.manual_seed(0)
    xs = [torch.randn(tokens, hidden, dtype=dtype, device="cuda") for _ in range(2)]
    ng = xs[0].numel() // GROUP
    q = [torch.empty(ng * 16, dtype=torch.uint8, device="cuda") for _ in xs]
    s = [torch.empty(ng, dtype=torch.float16, device="cuda") for _ in xs]
    for x, qi, si in zip(xs, q, s):
        _quant_int4_kernel[(triton.cdiv(ng, 128),)](
            x, qi, si, ng, BLOCK_G=128, enable_fp_fusion=False
        )
    out = torch.empty_like(xs[0])
    _dq_sum_kernel[(triton.cdiv(ng, 128),)](
        q[0], s[0], q[1], s[1], out, ng, BLOCK_G=128, num_warps=2
    )
    torch.testing.assert_close(out, _reference_int4_one_shot(xs), atol=0, rtol=0)


def _int4_messages(
    xs: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Each rank's message as DmaAllReduce sends it: INT4 values, fp16 scales."""
    from vllm.distributed.device_communicators.dma_all_reduce import (
        GROUP,
        _quant_int4_kernel,
    )
    from vllm.triton_utils import triton

    ng = xs[0].numel() // GROUP
    q = [torch.empty(ng * 16, dtype=torch.uint8, device="cuda") for _ in xs]
    s = [torch.empty(ng, dtype=torch.float16, device="cuda") for _ in xs]
    for x, qi, si in zip(xs, q, s):
        _quant_int4_kernel[(triton.cdiv(ng, 128),)](
            x, qi, si, ng, BLOCK_G=128, enable_fp_fusion=False
        )
    return q, s


@pytest.mark.parametrize("also_bf16", [False, True])
@pytest.mark.parametrize("tokens", [257, 2048])
def test_fused_norm_matches_sum_then_norm(tokens, also_bf16):
    """The fused consumer against the two kernels it replaces (dequantize-add,
    then the MiniMax-M3 Triton norm): the new residual and the MXFP8 scales are
    identical; the variance's summation order moves a few FP8 values and normed
    values by one step."""
    from vllm.distributed.device_communicators.dma_all_reduce import _dq_sum_kernel
    from vllm.distributed.device_communicators.dma_all_reduce_norm import (
        dq_sum_add_gemma_rmsnorm_mxfp8,
    )
    from vllm.models.minimax_m3.amd.ops import (
        gemma_fused_add_rmsnorm_mxfp8,
        gemma_fused_add_rmsnorm_mxfp8_and_bf16,
    )
    from vllm.triton_utils import triton

    hidden, eps = 6144, 1e-6
    torch.manual_seed(0)
    xs = []
    for _ in range(2):
        x = torch.randn(tokens, hidden, device="cuda") * 0.5
        x[:, :8] *= 20
        xs.append(x.bfloat16())
    residual = torch.randn(tokens, hidden, device="cuda") * 2.0
    residual[:, 8:12] *= 200
    residual = residual.bfloat16()
    weight = (torch.randn(hidden, device="cuda") * 0.3).bfloat16()
    q, s = _int4_messages(xs)

    ng = xs[0].numel() // 32
    summed = torch.empty_like(xs[0])
    _dq_sum_kernel[(triton.cdiv(ng, 128),)](
        q[0], s[0], q[1], s[1], summed, ng, BLOCK_G=128, num_warps=2
    )
    ref_normed: torch.Tensor | None = None
    if also_bf16:
        ref_q, ref_scale, ref_normed, ref_res = gemma_fused_add_rmsnorm_mxfp8_and_bf16(
            summed, residual, weight, eps
        )
    else:
        ref_q, ref_scale, ref_res = gemma_fused_add_rmsnorm_mxfp8(
            summed, residual, weight, eps
        )
    x_q, x_scale, normed, res_out = dq_sum_add_gemma_rmsnorm_mxfp8(
        q[0], s[0], q[1], s[1], residual, weight, eps, also_bf16
    )

    assert torch.equal(res_out, ref_res)
    assert torch.equal(x_scale, ref_scale)
    assert (normed is None) == (ref_normed is None)
    pairs = [(x_q, ref_q)]
    if normed is not None and ref_normed is not None:
        pairs.append((normed, ref_normed))
    for got, want in pairs:
        bits = torch.int8 if got.element_size() == 1 else torch.int16
        steps = (got.view(bits).int() - want.view(bits).int()).abs()
        assert steps.max() <= 1
        assert (steps > 0).sum() <= 1e-4 * steps.numel()
