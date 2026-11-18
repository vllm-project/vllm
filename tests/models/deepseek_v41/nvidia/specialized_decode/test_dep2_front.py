# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-dispatch routing/quantization must not build a TP-local routing record."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.models.deepseek_v41.nvidia.specialized_decode.dep2.front import DEP2Front
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
