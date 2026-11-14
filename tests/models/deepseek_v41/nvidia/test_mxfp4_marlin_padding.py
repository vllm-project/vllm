# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's MXFP4 routed experts on Marlin with the SM80 padding: the
288 intermediate columns per rank at TP8, zero-padded to 320, give the same
routed outputs as Marlin's stock padding to 384 up to GEMM2's FP32
summation order."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(80),
    reason="needs an SM80-or-newer CUDA GPU",
)


def test_marlin_mxfp4_moe_padding_320_matches_384():
    from vllm.model_executor.layers.fused_moe.activation import (
        ApplyMoEActivationConfig,
        MoEActivation,
    )
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        _fused_marlin_moe,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_moe_mxfp4_layer_for_marlin,
    )
    from vllm.scalar_type import scalar_types

    device = "cuda"
    torch.manual_seed(0)
    experts, hidden, intermediate, rows, top_k = 32, 1024, 288, 8, 6
    gate_up = torch.randint(0, 256, (experts, 2, intermediate, hidden // 2)).byte()
    gate_up_scale = torch.randint(118, 124, (experts, 2, intermediate, hidden // 32))
    down = torch.randint(0, 256, (experts, hidden, intermediate // 2)).byte()
    down_scale = torch.randint(118, 124, (experts, hidden, intermediate // 32))

    def padded(n: int):
        """Marlin's weights with the gate, up and down zero-padded to n."""
        w13 = torch.zeros(experts, 2, n, hidden // 2, dtype=torch.uint8)
        w13[:, :, :intermediate] = gate_up
        s13 = torch.zeros(experts, 2, n, hidden // 32, dtype=torch.uint8)
        s13[:, :, :intermediate] = gate_up_scale
        w2 = torch.zeros(experts, hidden, n // 2, dtype=torch.uint8)
        w2[..., : intermediate // 2] = down
        s2 = torch.zeros(experts, hidden, n // 32, dtype=torch.uint8)
        s2[..., : intermediate // 32] = down_scale
        return prepare_moe_mxfp4_layer_for_marlin(
            SimpleNamespace(params_dtype=torch.bfloat16),
            w13.view(experts, 2 * n, hidden // 2).to(device),
            w2.to(device),
            s13.view(experts, 2 * n, hidden // 32).to(device),
            s2.to(device),
            None,
            None,
        )[:4]

    x = torch.randn(rows, hidden, device=device).bfloat16()
    scores = torch.randn(rows, experts, device=device).softmax(dim=-1)
    weights, ids = scores.topk(top_k, dim=-1)
    ids = ids.int()
    sorted_ids, expert_ids, num_padded = moe_align_block_size(ids, 8, experts, None)

    def routed(n: int) -> torch.Tensor:
        """The weighted expert outputs `[rows * 6, hidden]` at padding n."""
        w13, w2, s13, s2 = padded(n)
        return _fused_marlin_moe(
            x, w13, w2, None, None, s13, s2, weights, top_k,
            scalar_types.float4_e2m1f, False, None, 8,
            sorted_ids, expert_ids, num_padded,
            activation=MoEActivation.SILU,
            activation_config=ApplyMoEActivationConfig(clamp_limit=10.0),
            topk_ids=ids,
        ).float()  # fmt: skip

    narrow, stock = routed(320), routed(384)
    assert (narrow - stock).norm() <= 1e-2 * stock.norm()
