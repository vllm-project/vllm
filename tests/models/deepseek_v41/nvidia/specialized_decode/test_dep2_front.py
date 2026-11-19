# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-dispatch routing/quantization must not build a TP-local routing record."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.models.deepseek_v41.nvidia.specialized_decode.dep2.front import (
    DEP2Front,
    FrontOutput,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.dep2.workspace import (
    ExpertWorkspace,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_gemm import (
    FrontConfig,
)

from .test_kernels import SENTINEL, _front_inputs


@pytest.mark.parametrize("rows", [1, 6, 9, 32, 48, 96, 128])
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
    front = DEP2Front(config, ExpertWorkspace("cuda"))
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


@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_front_preserves_global_expert_ids(ep_size):
    # Force every linear shard boundary, including the wrap from expert 383 to 0.
    gen = torch.Generator().manual_seed(31)
    logits, x, ids = _front_inputs(ep_size, gen, image_rows=0)
    rows, experts = logits.shape
    boundary = torch.arange(rows, device="cuda")[:, None] * (experts // ep_size)
    expected = (boundary + torch.arange(-3, 3, device="cuda")) % experts
    logits.fill_(-8).scatter_(1, expected, 8.0)
    config = FrontConfig(
        bias=torch.zeros(experts, device="cuda"),
        bias_vl=None,
        image_sentinel_lo=0,
        scale=1.5,
    )
    front = DEP2Front(config, ExpertWorkspace("cuda"))
    front.warmup(rows)
    mask = torch.zeros(rows, device="cuda", dtype=torch.bool)
    out = front(x, logits, ids, mask)
    assert torch.equal(out.topk_ids.sort(dim=1).values, expected.sort(dim=1).values)


@pytest.mark.parametrize("rows", [6, 12, 48, 126])
def test_front_into_dispatch_slice(rows):
    gen = torch.Generator().manual_seed(351)
    logits, x, ids = _front_inputs(rows, gen, image_rows=0)
    front = DEP2Front(FrontConfig(torch.zeros(384, device="cuda"), None, 0, 1.5))
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
