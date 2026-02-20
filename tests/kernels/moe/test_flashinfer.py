# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_test_quant_config
from tests.kernels.quant_utils import (
    native_per_token_group_quant_fp8,
    native_w8a8_block_matmul,
)
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
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

BLOCK_SHAPE = [128, 128]
BLOCK_MNK_FACTORS = [
    (64, 1024, 1024),
    (127, 1024, 1024),
]
BF16_MNK_FACTORS = [
    (64, 1024, 1024),
    (127, 1024, 1024),
]

vllm_config = VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))


def quant_fp8_per_tensor_batches(a):
    num_batches = a.size(0)
    a_quant = []
    a_scales = []

    for i in range(num_batches):
        a_fp8, a_global_sf = input_to_float8(a[i])
        if a_global_sf.numel() == 1:
            a_global_sf = a_global_sf.view(1, 1)
        a_quant.append(a_fp8)
        a_scales.append(a_global_sf)

    result_a_quant = torch.stack(a_quant)
    result_a_scales = torch.stack(a_scales)

    return result_a_quant, result_a_scales


def check_accuracy(ref_output, actual_output, atol=0.1, rtol=0.85, percent=0.925):
    close = torch.isclose(ref_output, actual_output, atol=atol, rtol=rtol)
    match_ratio = close.float().mean()
    assert match_ratio >= percent, (
        f"Match ratio {match_ratio:.4f} is below the threshold {percent:.4f}"
    )

    mismatch_percent = 1.0 - match_ratio.item()
    assert mismatch_percent <= 1 - percent, (
        f"Mismatch percentage {mismatch_percent:.4f} is above the threshold "
        f"{1 - percent:.4f}"
    )


def torch_bf16_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    apply_router_weight_on_input: bool = True,
) -> torch.Tensor:
    """Reference MoE implementation using native torch matmul in BF16."""
    assert activation in [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL]

    batch_size, hidden_dim = hidden_states.shape
    topk = topk_ids.size(1)
    expanded_hidden_states = hidden_states.view(batch_size, 1, hidden_dim).repeat(
        1, topk, 1
    )
    expanded_hidden_states = expanded_hidden_states.reshape(-1, hidden_dim)

    output = torch.zeros(
        batch_size * topk,
        w2.shape[1],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    flat_topk_weights = topk_weights.view(-1)
    flat_topk_ids = topk_ids.view(-1)
    silu_and_mul = SiluAndMul()

    if apply_router_weight_on_input:
        expanded_hidden_states = expanded_hidden_states * flat_topk_weights.unsqueeze(
            1
        ).to(expanded_hidden_states.dtype)

    for expert_idx in range(w1.shape[0]):
        mask = flat_topk_ids == expert_idx
        if not mask.any():
            continue

        inter_out = expanded_hidden_states[mask] @ w1[expert_idx].t()
        if activation == MoEActivation.SILU:
            act_out = silu_and_mul.forward_native(inter_out)
        else:
            act_out = torch.square(torch.relu(inter_out))

        output[mask] = act_out @ w2[expert_idx].t()

    if apply_router_weight_on_input:
        return output.view(batch_size, -1, w2.shape[1]).sum(dim=1)
    return (
        output.view(batch_size, -1, w2.shape[1])
        * flat_topk_weights.view(batch_size, -1, 1).to(output.dtype)
    ).sum(dim=1)


def torch_w8a8_block_fp8_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    block_shape: list[int],
    activation: MoEActivation,
) -> torch.Tensor:
    """Fused MoE with block-wise FP8 quantization using native torch."""
    assert activation in [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL]

    batch_size, hidden_dim = hidden_states.shape
    topk = topk_ids.size(1)
    expanded_hidden_states = hidden_states.view(batch_size, 1, hidden_dim).repeat(
        1, topk, 1
    )
    expanded_hidden_states = expanded_hidden_states.reshape(-1, hidden_dim)

    output = torch.zeros(
        batch_size * topk,
        w2.shape[1],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    flat_topk_weights = topk_weights.view(-1)
    flat_topk_ids = topk_ids.view(-1)
    _, block_k = block_shape
    hidden_states_q, hidden_states_s = native_per_token_group_quant_fp8(
        expanded_hidden_states.contiguous(), block_k
    )
    hidden_states_q = hidden_states_q.to(torch.float32)
    silu_and_mul = SiluAndMul()

    for expert_idx in range(w1.shape[0]):
        mask = flat_topk_ids == expert_idx
        if not mask.any():
            continue

        inter_out = native_w8a8_block_matmul(
            hidden_states_q[mask],
            w1[expert_idx],
            hidden_states_s[mask],
            w1_scale[expert_idx],
            block_shape,
            output_dtype=hidden_states.dtype,
        )
        if activation == MoEActivation.SILU:
            act_out = silu_and_mul.forward_native(inter_out)
        else:
            act_out = torch.square(torch.relu(inter_out))

        act_out_q, act_out_s = native_per_token_group_quant_fp8(
            act_out.contiguous(), block_k
        )
        output[mask] = native_w8a8_block_matmul(
            act_out_q,
            w2[expert_idx],
            act_out_s,
            w2_scale[expert_idx],
            block_shape,
            output_dtype=hidden_states.dtype,
        )

    return (
        output.view(batch_size, -1, w2.shape[1])
        * flat_topk_weights.view(batch_size, -1, 1).to(output.dtype)
    ).sum(dim=1)


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
        m: int,
        k: int,
        n: int,
        e: int,
        is_trtllm: bool,
        activation: MoEActivation = MoEActivation.SILU,
    ) -> "TestData":
        is_gated = activation.is_gated

        hidden_states = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
        w13 = (
            torch.randn(
                (e, (2 * n) if is_gated else n, k), device="cuda", dtype=torch.bfloat16
            )
            / 10
        )
        w2 = torch.randn((e, k, n), device="cuda", dtype=torch.bfloat16) / 10

        # Scale to fp8
        _, a1_scale = input_to_float8(hidden_states)
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
        layer.activation = activation
        # Setup dummy config.
        layer.moe_parallel_config = mk.FusedMoEParallelConfig.make_no_parallel()

        # flashinfer expects swapped rows for w13
        if is_gated:
            layer.w13_weight.data = swap_w13_to_w31(layer.w13_weight.data)
        if is_trtllm:
            rotate_weights_for_fi_trtllm_fp8_per_tensor_moe(
                layer.w13_weight, layer.w2_weight, is_gated
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
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
def test_flashinfer_per_tensor_moe_fp8_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    activation: MoEActivation,
    monkeypatch,
):
    if not current_platform.has_device_capability(100):
        pytest.skip("Test is only supported for sm >= 100")
    set_random_seed(7)
    monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", "8192")
    with set_current_vllm_config(vllm_config):
        td = TestData.make_moe_tensors_8bit(
            m, k, n, e, is_trtllm=True, activation=activation
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
            activation=activation,
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

        check_accuracy(
            ref_output=output,
            actual_output=flashinfer_output,
            atol=0.1,
            rtol=0.85,
            percent=0.925,
        )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
def test_flashinfer_cutlass_moe_fp8_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    activation: MoEActivation,
    monkeypatch,
    workspace_init,
):
    set_random_seed(7)
    monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", "8192")
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
            is_act_and_mul=activation.is_gated,
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

        check_accuracy(
            ref_output=output,
            actual_output=flashinfer_cutlass_output,
            atol=0.1,
            rtol=0.85,
            percent=0.925,
        )


@pytest.mark.parametrize("m,n,k", BF16_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
def test_flashinfer_cutlass_moe_bf16_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    activation: MoEActivation,
    monkeypatch,
    workspace_init,
):
    set_random_seed(7)
    monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", "2048")

    with set_current_vllm_config(vllm_config):
        hidden_states = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
        w1 = (
            torch.randn(
                (e, (2 * n) if activation.is_gated else n, k),
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 10
        )
        w2 = torch.randn((e, k, n), device="cuda", dtype=torch.bfloat16) / 10
        score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
        topk_weights, topk_ids = Llama4MoE.custom_routing_function(
            hidden_states=hidden_states,
            gating_output=score,
            topk=topk,
            renormalize=False,
        )

        ref_output = torch_bf16_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation=activation,
            apply_router_weight_on_input=True,
        )

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
            is_act_and_mul=activation.is_gated,
            routing_method=RoutingMethodType.TopK,
        )

        flashinfer_experts = FlashInferExperts(
            moe_config=moe_config,
            quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
        )
        assert not flashinfer_experts.use_deepseek_fp8_block_scale

        w1_fi = swap_w13_to_w31(w1) if activation.is_gated else w1
        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            flashinfer_experts,
            inplace=False,
        )
        flashinfer_cutlass_output = kernel(
            hidden_states,
            w1_fi,
            w2,
            topk_weights,
            topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=True,
        )

        check_accuracy(
            ref_output=ref_output,
            actual_output=flashinfer_cutlass_output,
            atol=0.05,
            rtol=0.05,
            percent=0.85,
        )


@pytest.mark.parametrize("m,n,k", BLOCK_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
def test_flashinfer_cutlass_moe_fp8_block_scale_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    activation: MoEActivation,
    monkeypatch,
    workspace_init,
):
    if not current_platform.is_device_capability(90):
        pytest.skip("FP8 block-scale MoE requires exact SM 9.0 (H100)")

    set_random_seed(7)
    monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", "2048")

    with set_current_vllm_config(vllm_config):
        hidden_states = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
        score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, score.float(), topk, renormalize=False
        )

        w1, w2, quant_config_ref = make_test_quant_config(
            e=e,
            n=n,
            k=k,
            in_dtype=torch.bfloat16,
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=False,
            block_shape=BLOCK_SHAPE,
            make_gate=activation.is_gated,
        )

        ref_out = torch_w8a8_block_fp8_moe(
            hidden_states,
            w1,
            w2,
            quant_config_ref.w1_scale,
            quant_config_ref.w2_scale,
            topk_weights,
            topk_ids,
            block_shape=BLOCK_SHAPE,
            activation=activation,
        )

        w1_fi = w1
        w1_scale_fi = quant_config_ref.w1_scale
        if activation.is_gated:
            w1_fi = swap_w13_to_w31(w1)
            w1_scale_fi = swap_w13_to_w31(w1_scale_fi)

        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale_fi,
            w2_scale=quant_config_ref.w2_scale,
            per_act_token_quant=False,
            block_shape=BLOCK_SHAPE,
        )

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
            is_act_and_mul=activation.is_gated,
            routing_method=RoutingMethodType.TopK,
        )

        flashinfer_experts = FlashInferExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        assert flashinfer_experts.use_deepseek_fp8_block_scale

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            flashinfer_experts,
            inplace=False,
        )
        flashinfer_cutlass_output = kernel(
            hidden_states,
            w1_fi,
            w2,
            topk_weights,
            topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

        check_accuracy(
            ref_output=ref_out,
            actual_output=flashinfer_cutlass_output,
            atol=0.1,
            rtol=0.85,
            percent=0.925,
        )


@pytest.mark.parametrize(
    "num_experts,intermediate,hidden",
    [
        (8, 2048, 1536),
        (64, 4096, 4096),
    ],
)
def test_convert_moe_weights_to_flashinfer_trtllm_block_layout(
    num_experts, intermediate, hidden
):
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        convert_moe_weights_to_flashinfer_trtllm_block_layout,
    )

    w13 = torch.randn(
        (num_experts, 2 * intermediate, hidden), dtype=torch.bfloat16, device="cuda"
    )
    w2 = torch.randn(
        (num_experts, hidden, intermediate), dtype=torch.bfloat16, device="cuda"
    )

    cache: dict[torch.Size, torch.Tensor] = {}
    w13_converted, w2_converted = convert_moe_weights_to_flashinfer_trtllm_block_layout(
        cache, w13, w2
    )

    assert w13_converted.ndim == 4, (
        f"Expected 4D tensor, got shape {w13_converted.shape}"
    )
    assert w2_converted.ndim == 4, f"Expected 4D tensor, got shape {w2_converted.shape}"

    assert w13_converted.numel() == w13.numel(), "W13 element count should be preserved"
    assert w2_converted.numel() == w2.numel(), "W2 element count should be preserved"

    assert w13_converted.dtype == torch.bfloat16
    assert w2_converted.dtype == torch.bfloat16

    assert w13_converted.shape[0] == num_experts
    assert w2_converted.shape[0] == num_experts
