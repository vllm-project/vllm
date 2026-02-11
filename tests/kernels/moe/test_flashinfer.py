# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    apply_fi_trtllm_fp8_per_tensor_moe,
    register_scales_for_trtllm_fp8_per_tensor_moe,
    rotate_weights_for_fi_trtllm_fp8_per_tensor_moe,
    swap_w13_to_w31,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import input_to_float8
from vllm.model_executor.models.llama4 import Llama4MoE
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

try:
    from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe
except ImportError:
    if current_platform.is_rocm():
        pytest.skip(
            "flashinfer not supported for vLLM on ROCm", allow_module_level=True
        )

if not has_flashinfer_cutlass_fused_moe() or not current_platform.has_device_capability(
    90
):
    pytest.skip(
        "Supported for sm >= 90",
        allow_module_level=True,
    )

NUM_EXPERTS = [16]
TOP_KS = [1]

MNK_FACTORS = [
    (256, 8192, 5120),
    (127, 4096, 5120),
    (10, 8192, 5120),
    (10, 4096, 5120),
    (1, 8192, 5120),
    (1, 4096, 5120),
]

vllm_config = VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))


def quant_fp8_per_tensor_batches(a):
    num_batches = a.size(0)
    a_quant = []
    a_scales = []

    for i in range(num_batches):
        a_fp8, a_global_sf = input_to_float8(a[i])
        a_global_sf = 1.0 / a_global_sf
        a_quant.append(a_fp8)
        a_scales.append(a_global_sf)

    result_a_quant = torch.stack(a_quant)
    result_a_scales = torch.stack(a_scales)

    return result_a_quant, result_a_scales


@dataclass
class TestData:
    hidden_states: torch.Tensor
    w13_quantized: torch.Tensor
    w2_quantized: torch.Tensor
    a1_scale: torch.Tensor
    a2_scale: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight_scale: torch.Tensor
    layer: torch.nn.Module

    @staticmethod
    def make_moe_tensors_8bit(
        m: int, k: int, n: int, e: int, is_trtllm: bool, activation: str = "silu"
    ) -> "TestData":
        is_gated = activation != "relu2_no_mul"

        hidden_states = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
        w13 = torch.randn(
            (e, (2 * n) if is_gated else n, k), device="cuda", dtype=torch.bfloat16
        )
        w2 = torch.randn((e, k, n), device="cuda", dtype=torch.bfloat16)

        # Scale to fp8
        _, a1_scale = input_to_float8(hidden_states)
        a1_scale = 1.0 / a1_scale
        a2_scale = torch.scalar_tensor(1.0).to(device="cuda").to(dtype=torch.float32)
        w13_quantized, w13_weight_scale = quant_fp8_per_tensor_batches(w13)
        w2_quantized, w2_weight_scale = quant_fp8_per_tensor_batches(w2)

        layer = torch.nn.Module()
        layer.orig_dtype = torch.bfloat16
        layer.w13_weight = w13_quantized.clone()
        layer.w2_weight = w2_quantized.clone()
        layer.w13_input_scale = a1_scale
        layer.w2_input_scale = a2_scale
        layer.w13_weight_scale = w13_weight_scale
        layer.w2_weight_scale = w2_weight_scale
        # Setup dummy config.
        layer.moe_parallel_config = mk.FusedMoEParallelConfig.make_no_parallel()

        # flashinfer expects swapped rows for w13
        layer.w13_weight.data = swap_w13_to_w31(layer.w13_weight.data)
        if is_trtllm:
            rotate_weights_for_fi_trtllm_fp8_per_tensor_moe(
                layer.w13_weight, layer.w2_weight
            )
            register_scales_for_trtllm_fp8_per_tensor_moe(
                layer,
                layer.w13_weight_scale,
                layer.w13_input_scale,
                layer.w2_weight_scale,
                layer.w2_input_scale,
            )
        layer.custom_routing_function = Llama4MoE.custom_routing_function
        layer.routing_method_type = RoutingMethodType.Llama4
        layer.renormalize = False
        layer.intermediate_size_per_partition = n
        layer.ep_rank = 0
        layer.local_num_experts = e

        return TestData(
            hidden_states=hidden_states,
            w13_quantized=w13_quantized,
            w2_quantized=w2_quantized,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            w13_weight_scale=w13_weight_scale,
            w2_weight_scale=w2_weight_scale,
            layer=layer,
        )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
def test_flashinfer_per_tensor_moe_fp8_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    monkeypatch,
):
    if not current_platform.has_device_capability(100):
        pytest.skip("Test is only supported for sm >= 100")
    set_random_seed(7)
    monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", "8192")
    with set_current_vllm_config(vllm_config):
        td = TestData.make_moe_tensors_8bit(m, k, n, e, is_trtllm=True)

        score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
        topk_weights, topk_ids = Llama4MoE.custom_routing_function(
            hidden_states=td.hidden_states,
            gating_output=score,
            topk=topk,
            renormalize=False,
        )

        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=td.w13_weight_scale,
            w2_scale=td.w2_weight_scale,
            a1_scale=td.a1_scale,
            a2_scale=td.a2_scale,
            per_act_token_quant=False,
        )

        output = fused_experts(
            td.hidden_states,
            td.w13_quantized,
            td.w2_quantized,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation="silu",
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=True,
            quant_config=quant_config,
        )

        flashinfer_output = apply_fi_trtllm_fp8_per_tensor_moe(
            layer=td.layer,
            hidden_states=td.hidden_states,
            router_logits=score,
            routing_bias=None,
            global_num_experts=e,
            top_k=topk,
            num_expert_group=None,
            topk_group=None,
            apply_router_weight_on_input=True,
        )

        torch.testing.assert_close(output, flashinfer_output, atol=5.5e-2, rtol=1e-2)


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("activation", ["silu", "relu2_no_mul"])
def test_flashinfer_cutlass_moe_fp8_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    activation: str,
    monkeypatch,
    workspace_init,
):
    set_random_seed(7)
    monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", "8192")
    assert activation in ["silu", "relu2_no_mul"]
    is_act_and_mul = activation == "silu_and_mul"
    with set_current_vllm_config(vllm_config):
        td = TestData.make_moe_tensors_8bit(
            m, k, n, e, is_trtllm=False, activation=activation
        )

        score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
        topk_weights, topk_ids = Llama4MoE.custom_routing_function(
            hidden_states=td.hidden_states,
            gating_output=score,
            topk=topk,
            renormalize=False,
        )

        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=td.w13_weight_scale,
            g1_alphas=(td.w13_weight_scale * td.a1_scale).squeeze(),
            w2_scale=td.w2_weight_scale,
            g2_alphas=(td.w2_weight_scale * td.a2_scale).squeeze(),
            a1_scale=td.a1_scale,
            a1_gscale=td.a1_scale,
            a2_scale=td.a2_scale,
            a2_gscale=1.0 / td.a2_scale,
            per_act_token_quant=False,
        )

        output = fused_experts(
            td.hidden_states,
            td.w13_quantized,
            td.w2_quantized,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=True,
            quant_config=quant_config,
        )

        td.layer.dp_size = 1

        def get_fused_moe_quant_config(n: torch.nn.Module) -> FusedMoEQuantConfig:
            return quant_config

        td.layer.get_fused_moe_quant_config = get_fused_moe_quant_config
        td.layer.quant_method = td.layer

        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size_per_partition=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=torch.bfloat16,
            is_act_and_mul=is_act_and_mul,
            routing_method=RoutingMethodType.TopK,
        )

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            FlashInferExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
            inplace=False,
        )

        flashinfer_cutlass_output = kernel(
            td.hidden_states,
            td.layer.w13_weight,
            td.layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=True,
        )
        torch.testing.assert_close(
            output, flashinfer_cutlass_output, atol=5.5e-2, rtol=1e-2
        )
