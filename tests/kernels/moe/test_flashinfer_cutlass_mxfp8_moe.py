# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 W8A8 MoE on FlashInfer CUTLASS."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    make_fp8_moe_kernel,
    make_fp8_moe_quant_config,
)
from vllm.model_executor.layers.quantization.online.mxfp8 import Mxfp8OnlineMoEMethod
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    prepare_fp8_moe_layer_for_fi,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe

if not (
    has_flashinfer_cutlass_fused_moe()
    and current_platform.is_device_capability_family(100)
):
    pytest.skip("FlashInfer CUTLASS MXFP8 MoE requires SM10x", allow_module_level=True)

NUM_EXPERTS = 4
TOP_K = 2
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 256
NUM_TOKENS = 40


class _ExpertsLayer(torch.nn.Module):
    def __init__(self, moe_config: FusedMoEConfig) -> None:
        super().__init__()
        self.moe_config = moe_config
        self.activation = moe_config.activation
        self.weight_block_size = [1, MXFP8_BLOCK_SIZE]


def _mxfp8_dequantize(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    block_scales = torch.exp2(scales.view(torch.uint8).float() - 127)
    return values.float() * block_scales.repeat_interleave(MXFP8_BLOCK_SIZE, dim=-1)


def _mxfp8_quantize_dequantize(x: torch.Tensor) -> torch.Tensor:
    blocks = x.view(*x.shape[:-1], -1, MXFP8_BLOCK_SIZE)
    amax = blocks.abs().amax(-1, keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0))).clamp(min=2.0**-127)
    return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).view_as(x)


def _reference(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
) -> torch.Tensor:
    output = torch.zeros(NUM_TOKENS, HIDDEN_SIZE, device=x.device)
    for k in range(TOP_K):
        experts = topk_ids[:, k].long()
        gemm1 = torch.einsum("td,tid->ti", x, w13[experts]).bfloat16().float()
        if activation == MoEActivation.SILU:
            gate, up = gemm1.chunk(2, dim=-1)
            intermediate = F.silu(gate) * up
        else:
            intermediate = F.relu(gemm1).square()
        intermediate = _mxfp8_quantize_dequantize(intermediate.bfloat16().float())
        expert_output = torch.einsum("ti,tdi->td", intermediate, w2[experts])
        output += topk_weights[:, k, None] * expert_output
    return output


@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
def test_flashinfer_cutlass_mxfp8_moe_matches_reference(
    workspace_init: None, activation: MoEActivation
) -> None:
    torch.manual_seed(0)
    moe_config = FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=TOP_K,
        hidden_dim=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_local_experts=NUM_EXPERTS,
        num_logical_experts=NUM_EXPERTS,
        activation=activation,
        device="cuda",
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.TopK,
        moe_backend="flashinfer_cutlass",
    )
    w13_rows = 2 * INTERMEDIATE_SIZE if activation.is_gated else INTERMEDIATE_SIZE
    x = torch.randn(NUM_TOKENS, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    w13 = (
        4
        * HIDDEN_SIZE**-0.5
        * torch.randn(NUM_EXPERTS, w13_rows, HIDDEN_SIZE, device="cuda")
    ).to(torch.bfloat16)
    w2 = (
        INTERMEDIATE_SIZE**-0.5
        * torch.randn(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, device="cuda")
    ).to(torch.bfloat16)
    topk_ids = (
        torch.rand(NUM_TOKENS, NUM_EXPERTS, device="cuda").topk(TOP_K).indices.int()
    )
    topk_weights = torch.rand(NUM_TOKENS, TOP_K, device="cuda")

    method = Mxfp8OnlineMoEMethod(moe=moe_config)
    assert method.fp8_backend == Fp8MoeBackend.FLASHINFER_CUTLASS
    assert method.experts_cls is FlashInferExperts
    w13_q, w13_scale = method._quantize_mxfp8_moe_weight(w13)
    w2_q, w2_scale = method._quantize_mxfp8_moe_weight(w2)
    x_q, x_scale = mxfp8_e4m3_quantize(x)
    expected = _reference(
        _mxfp8_dequantize(x_q, x_scale),
        _mxfp8_dequantize(w13_q, w13_scale),
        _mxfp8_dequantize(w2_q, w2_scale),
        topk_weights,
        topk_ids,
        activation,
    )

    w13_q, w2_q, w13_scale, w2_scale = prepare_fp8_moe_layer_for_fi(
        layer=_ExpertsLayer(moe_config),
        w13=w13_q,
        w2=w2_q,
        w13_scale=w13_scale,
        w13_input_scale=None,
        w2_scale=w2_scale,
        w2_input_scale=None,
    )
    quant_config = make_fp8_moe_quant_config(
        fp8_backend=method.fp8_backend,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        a1_scale=None,
        a2_scale=None,
        block_shape=[1, MXFP8_BLOCK_SIZE],
    )
    assert quant_config.is_mx_quantized
    kernel = make_fp8_moe_kernel(
        quant_config, moe_config, method.experts_cls, method.fp8_backend
    )
    output = kernel.apply(
        x,
        w13_q,
        w2_q,
        topk_weights,
        topk_ids,
        activation=activation,
        global_num_experts=NUM_EXPERTS,
        expert_map=None,
        apply_router_weight_on_input=False,
    )

    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(
        output.float(),
        expected,
        rtol=2e-2,
        atol=2e-2 * expected.abs().max().item(),
    )
