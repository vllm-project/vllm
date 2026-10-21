# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The OCP MX emulation MoE dequantizes only the experts a batch routes to."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts import (
    ocp_mx_emulation_moe as emulation,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="emulation backend targets ROCm"
)
@pytest.mark.parametrize(
    "topk_ids, num_routed",
    [
        # 5 of 32 experts: only those are dequantized.
        ([[30, 2], [2, 17], [9, 30], [17, 4]], 5),
        # 9 of 32, past a quarter: every expert is dequantized.
        ([[0, 1], [2, 3], [4, 5], [6, 7], [8, 0]], 32),
    ],
)
@torch.inference_mode()
def test_routed_dequant_moe_forward_is_bit_identical(monkeypatch, topk_ids, num_routed):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEParallelConfig,
        RoutingMethodType,
        mxfp4_w4a16_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        make_mxfp4_moe_kernel,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        mxfp4_quantize,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.accelerator.current_device_index())

    num_experts, topk = 32, 2
    hidden_size, intermediate_size = 256, 256
    dtype, device = torch.bfloat16, "cuda:0"

    torch.manual_seed(0)
    w13, w13_scale = mxfp4_quantize(
        torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, dtype=dtype, device=device
        )
        / 8
    )
    w2, w2_scale = mxfp4_quantize(
        torch.randn(
            num_experts, hidden_size, intermediate_size, dtype=dtype, device=device
        )
        / 8
    )
    w13, w2 = w13.contiguous(), w2.contiguous()
    w13_scale, w2_scale = w13_scale.contiguous(), w2_scale.contiguous()

    topk_ids = torch.tensor(topk_ids, dtype=torch.int32, device=device)
    num_tokens = topk_ids.size(0)
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, topk, device=device), dim=-1
    ).to(dtype)

    dequantized: list[int] = []
    dequantize = emulation.OCP_MXQuantizationEmulationTritonExperts._dequantize_weights

    def recording_dequantize(self, w, w_scale, dtype):
        dequantized.append(w.size(0))
        return dequantize(self, w, w_scale, dtype)

    monkeypatch.setattr(
        emulation.OCP_MXQuantizationEmulationTritonExperts,
        "_dequantize_weights",
        recording_dequantize,
    )

    def run():
        moe_config = FusedMoEConfig(
            num_experts=num_experts,
            experts_per_token=topk,
            hidden_dim=hidden_size,
            intermediate_size=intermediate_size,
            num_local_experts=num_experts,
            num_logical_experts=num_experts,
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            activation=MoEActivation.SILU,
            in_dtype=dtype,
            device=device,
            routing_method=RoutingMethodType.Renormalize,
        )
        # A fresh quant config per run: the experts null the weight scales on it.
        quant_config = mxfp4_w4a16_moe_quant_config(
            w1_scale=w13_scale, w2_scale=w2_scale
        )
        with set_current_vllm_config(VllmConfig()):
            kernel = make_mxfp4_moe_kernel(
                moe_quant_config=quant_config,
                moe_config=moe_config,
                mxfp4_backend=Mxfp4MoeBackend.EMULATION,
                experts_cls=emulation.OCP_MXQuantizationEmulationTritonExperts,
                routing_tables=None,
            )
            return kernel.apply(
                hidden_states=hidden_states,
                w1=w13,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=MoEActivation.SILU,
                global_num_experts=num_experts,
                expert_map=None,
                apply_router_weight_on_input=False,
            )

    routed = run()
    assert dequantized == [num_routed, num_routed]
    # The routed path is skipped under graph capture; use that to get the
    # full-dequant reference.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    dequantized.clear()
    full = run()
    assert dequantized == [num_experts, num_experts]
    assert torch.equal(routed, full)
