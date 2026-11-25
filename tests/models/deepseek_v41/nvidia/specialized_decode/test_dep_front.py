# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-dispatch routing/quantization must not build a TP-local routing record."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.models.deepseek_v41.nvidia.specialized_decode.dep.front import (
    DEPFront,
    FrontOutput,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.dep.workspace import (
    ExpertWorkspace,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_gemm import (
    FrontConfig,
)

from .test_kernels import SENTINEL, _front_inputs


@pytest.mark.parametrize("rows", [1, 6, 9, 32, 48, 96, 128, 129, 192, 193, 288, 384])
@pytest.mark.parametrize("image", [False, True])
def test_front_before_ep(rows, image):
    generator = torch.Generator().manual_seed(17 + rows)
    bias = 0.05 * torch.randn(384, generator=generator, dtype=torch.float32)
    bias[10:20] = bias[10]
    config = FrontConfig(
        bias=bias.cuda(),
        bias_vl=(bias * 2).cuda() if image else None,
        image_sentinel_lo=SENTINEL if image else 0,
        scale=1.5,
    )
    front = DEPFront(config, ExpertWorkspace("cuda", max_rows=384))
    front.warmup(rows)
    logits, x, ids = _front_inputs(rows, generator, image_rows=rows // 2)
    mask = torch.zeros(rows, dtype=torch.bool, device="cuda")

    def check(output):
        weights, indices = dsv4_topk(
            logits,
            config.bias,
            torch.int32,
            config.scale,
            is_padding=mask,
            input_ids=ids,
            bias_vl=config.bias_vl,
            image_sentinel_lo=config.image_sentinel_lo,
        )
        quant, scales = moe_kernel_quantize_input(
            x,
            None,
            quant_dtype="mxfp8",
            per_act_token_quant=False,
            is_scale_swizzled=False,
        )
        for got, expected in zip(output, (quant, scales, indices, weights)):
            assert got.shape == expected.shape
            assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))

    check(front(x, logits, ids, mask))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = front(x, logits, ids, mask)
    # Reuse the same graph/workspace with changed data and all-padding rows.
    for padding in (rows // 2, rows, 0):
        fresh_logits, fresh_x, fresh_ids = _front_inputs(
            rows,
            generator,
            image_rows=rows // 3,
        )
        logits.copy_(fresh_logits)
        x.copy_(fresh_x)
        ids.copy_(fresh_ids)
        mask.copy_(torch.arange(rows, device="cuda") >= rows - padding)
        graph.replay()
        check(output)


@pytest.mark.parametrize("rows", [6, 12, 48, 126, 192, 288, 384])
def test_front_into_dispatch_slice(rows):
    gen = torch.Generator().manual_seed(351)
    logits, x, ids = _front_inputs(rows, gen, image_rows=0)
    front = DEPFront(FrontConfig(torch.zeros(384, device="cuda"), None, 0, 1.5))
    front.warmup(rows)
    original = front(x, logits, ids)
    global_output = FrontOutput(
        *(
            torch.full((4 * rows, *t.shape[1:]), 0, device=t.device, dtype=t.dtype)
            for t in original
        )
    )
    local_output = FrontOutput(*(t.narrow(0, 2 * rows, rows) for t in global_output))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = front(x, logits, ids, output=local_output)
    for _ in range(3):
        x.mul_(0.5)
        logits.neg_()
        reference = front(x, logits, ids)
        graph.replay()
        for actual, expected, whole in zip(result, reference, global_output):
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            assert not torch.count_nonzero(whole[: 2 * rows].view(torch.uint8))
            assert not torch.count_nonzero(whole[3 * rows :].view(torch.uint8))
    with pytest.raises(ValueError, match="front output"):
        front(
            x, logits, ids, output=FrontOutput(*(t[: rows - 1] for t in local_output))
        )


@pytest.mark.parametrize("rows", [1, 6, 48, 128, 129, 192, 193, 288, 384])
@pytest.mark.parametrize("image", [False, True])
def test_ffn_boundary_quantization_preserves_dispatch(rows, image):
    """Quantize the BF16 norm result in registers, without changing routing."""
    from flashinfer import mxfp8_quantize

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.deepseek_v41.nvidia.specialized_decode.dep.local_boundary import (
        LocalPostPre,
    )

    gen = torch.Generator().manual_seed(351 + rows)
    logits, x, ids = _front_inputs(rows, gen, image_rows=rows // 2)
    residual = torch.randn(rows, 4, 5120, generator=gen).bfloat16().cuda()
    post = torch.sigmoid(torch.randn(rows, 4, 1, generator=gen)).cuda()
    comb = torch.softmax(torch.randn(rows, 4, 4, generator=gen), dim=-1).cuda()
    pre = torch.softmax(torch.randn(rows, 4, generator=gen), dim=-1).cuda()
    bias = torch.randn(384, generator=gen).cuda() * 0.05
    config = FrontConfig(bias, bias * 2 if image else None, SENTINEL, 1.5)
    full, route = DEPFront(config), DEPFront(config, quantize=False)
    full.warmup(rows)
    route.warmup(rows)
    padding = torch.zeros(rows, device="cuda", dtype=torch.bool)
    q_storage = torch.full((rows + 2, 5120), 90, device="cuda", dtype=torch.uint8)
    s_storage = torch.full((rows + 2, 160), 90, device="cuda", dtype=torch.uint8)
    output = FrontOutput(
        q_storage[1:-1].view(torch.float8_e4m3fn),
        s_storage[1:-1],
        torch.empty(rows, 6, device="cuda", dtype=torch.int32),
        torch.empty(rows, 6, device="cuda", dtype=torch.float32),
    )
    with torch.inference_mode(), set_current_vllm_config(VllmConfig()):
        norm = RMSNorm(5120, eps=1e-6).to("cuda", torch.bfloat16)
        norm.weight.copy_((1 + torch.randn(5120, generator=gen) * 0.05).bfloat16())
        boundary = LocalPostPre()
        args = (x, residual, post, comb, pre, norm)

        def candidate():
            result = boundary(
                *args,
                (output.x_quant.view(torch.uint8), output.x_scale),
                linear_scales=True,
            )
            assert result is not None
            return result, route(result[1], logits, ids, padding, output=output)

        candidate()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got_boundary, got_front = candidate()
        for valid in (rows, rows // 2, 0, rows):
            if valid == 0:
                x.zero_()
                residual.zero_()
            else:
                x.copy_(torch.randn(x.shape, generator=gen).bfloat16())
                residual.copy_(torch.randn(residual.shape, generator=gen).bfloat16())
            logits.neg_()
            padding.copy_(torch.arange(rows, device="cuda") >= valid)
            expected_boundary = boundary(*args, None)
            expected_front = full(expected_boundary[1], logits, ids, padding)
            quant, scales = moe_kernel_quantize_input(
                expected_boundary[1],
                None,
                quant_dtype="mxfp8",
                per_act_token_quant=False,
                is_scale_swizzled=False,
            )
            graph.replay()
            for actual, expected in zip(got_boundary[:2], expected_boundary[:2]):
                assert torch.equal(actual, expected)
            for actual, expected in zip(got_front, expected_front):
                assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            assert torch.equal(
                got_front.x_quant.view(torch.uint8), quant.view(torch.uint8)
            )
            assert torch.equal(got_front.x_scale, scales)
            for storage in (q_storage, s_storage):
                assert (storage[0] == 90).all() and (storage[-1] == 90).all()

        # The default attention-scale layout must remain independent of the
        # linear-scale compiled plan for the same row count.
        want_q, want_sf = mxfp8_quantize(
            expected_boundary[1], is_sf_swizzled_layout=True
        )
        swizzled = (torch.empty_like(want_q), torch.empty_like(want_sf))
        boundary(*args, swizzled)
        for actual, expected in zip(swizzled, (want_q, want_sf)):
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))

        with pytest.raises(ValueError, match="prequantized output"):
            route(x, logits, ids, padding)
        with pytest.raises(ValueError, match="linear MXFP8"):
            boundary(*args, None, linear_scales=True)
