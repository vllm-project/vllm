# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
    TrtLlmFp8ExpertsModular,
    TrtLlmFp8ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    rotate_weights_for_fi_trtllm_fp8_per_tensor_moe,
    swap_w13_to_w31,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import input_to_float8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.model_executor.models.llama4 import Llama4MoE
from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
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
        topk: int = 1,
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
        layer.custom_routing_function = Llama4MoE.custom_routing_function
        layer.routing_method_type = RoutingMethodType.Llama4
        layer.renormalize = False
        layer.intermediate_size_per_partition = n
        layer.ep_rank = 0
        layer.local_num_experts = e

        layer.moe = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            num_local_experts=e,
            num_logical_experts=e,
            moe_parallel_config=layer.moe_parallel_config,
            in_dtype=hidden_states.dtype,
            routing_method=layer.routing_method_type,
            activation=activation,
            device=w13_quantized.device,
            max_num_tokens=next_power_of_2(m),
        )

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
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=True,
            quant_config=quant_config,
        )

        kernel = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=td.layer.moe,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=True,
            ),
            TrtLlmFp8ExpertsMonolithic(
                moe_config=td.layer.moe,
                quant_config=quant_config,
            ),
        )

        flashinfer_output = kernel.apply_monolithic(
            hidden_states=td.hidden_states,
            w1=td.layer.w13_weight,
            w2=td.layer.w2_weight,
            router_logits=score,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=True,
            routed_scaling_factor=1.0,
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
            intermediate_size=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=torch.bfloat16,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=next_power_of_2(m),
        )

        kernel = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            FlashInferExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
        )

        flashinfer_cutlass_output = kernel.apply(
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
    assert w13_converted.data_ptr() == w13.data_ptr()
    assert w2_converted.data_ptr() == w2.data_ptr()


@pytest.mark.parametrize("is_gated_act_gemm", [True, False])
def test_convert_moe_weights_to_flashinfer_trtllm_block_layout_values(
    is_gated_act_gemm,
):
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        convert_moe_weights_to_flashinfer_trtllm_block_layout,
    )

    num_experts, intermediate, hidden = 2, 256, 256
    w13_multiplier = 2 if is_gated_act_gemm else 1
    w13 = torch.randn(
        (num_experts, w13_multiplier * intermediate, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    w2 = torch.randn(
        (num_experts, hidden, intermediate),
        dtype=torch.bfloat16,
        device="cuda",
    )

    def _reference_block_layout(
        weight: torch.Tensor,
        is_w13: bool,
        cache: dict[torch.Size, torch.Tensor],
    ) -> torch.Tensor:
        outputs = []
        for expert in weight:
            expert_uint8 = expert.view(torch.uint8)
            if is_w13:
                indices = _maybe_get_cached_w3_w1_permute_indices(
                    cache,
                    expert_uint8,
                    128,
                    is_gated_act_gemm=is_gated_act_gemm,
                )
                if is_gated_act_gemm:
                    indices = (indices + expert_uint8.shape[0] // 2) % (
                        expert_uint8.shape[0]
                    )
            else:
                indices = get_w2_permute_indices_with_cache(
                    cache,
                    expert_uint8,
                    128,
                )
            rows, cols = expert_uint8.shape
            blocks = expert_uint8.view(rows, cols // 128, 128).permute(1, 0, 2)
            outputs.append(torch.index_select(blocks, 1, indices.to(weight.device)))
        return torch.stack(outputs).view(torch.bfloat16)

    reference_cache: dict[torch.Size, torch.Tensor] = {}
    expected_w13 = _reference_block_layout(w13, is_w13=True, cache=reference_cache)
    expected_w2 = _reference_block_layout(w2, is_w13=False, cache=reference_cache)
    w13_ptr = w13.data_ptr()
    w2_ptr = w2.data_ptr()

    actual_w13, actual_w2 = convert_moe_weights_to_flashinfer_trtllm_block_layout(
        {},
        w13,
        w2,
        is_gated_act_gemm=is_gated_act_gemm,
    )

    assert actual_w13.data_ptr() == w13_ptr
    assert actual_w2.data_ptr() == w2_ptr
    assert torch.equal(actual_w13, expected_w13)
    assert torch.equal(actual_w2, expected_w2)


def test_unquantized_flashinfer_trtllm_weights_can_be_reprocessed(monkeypatch):
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoeBackend,
    )
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        convert_moe_weights_to_flashinfer_trtllm_block_layout,
    )

    moe_config = make_dummy_moe_config(
        num_experts=2,
        hidden_dim=256,
        intermediate_size=256,
    )
    method = object.__new__(UnquantizedFusedMoEMethod)
    method.moe = moe_config
    method.unquantized_backend = UnquantizedMoeBackend.FLASHINFER_TRTLLM
    method.moe_kernel = None
    monkeypatch.setattr(
        method,
        "_init_moe_kernel",
        lambda _: setattr(method, "moe_kernel", object()),
    )

    layer = torch.nn.Module()
    layer.moe_config = moe_config
    w13_shape = (2, 512, 256)
    w2_shape = (2, 256, 256)
    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(
            torch.randn(w13_shape, dtype=torch.bfloat16, device="cuda"),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight",
        torch.nn.Parameter(
            torch.randn(w2_shape, dtype=torch.bfloat16, device="cuda"),
            requires_grad=False,
        ),
    )
    w13_ptr = layer.w13_weight.data_ptr()
    w2_ptr = layer.w2_weight.data_ptr()

    for _ in range(2):
        reloaded_w13 = torch.randn_like(layer.w13_weight)
        reloaded_w2 = torch.randn_like(layer.w2_weight)
        expected_w13, expected_w2 = (
            convert_moe_weights_to_flashinfer_trtllm_block_layout(
                {},
                reloaded_w13.clone(),
                reloaded_w2.clone(),
            )
        )

        layer.w13_weight.copy_(reloaded_w13)
        layer.w2_weight.copy_(reloaded_w2)
        method._setup_kernel(layer, layer.w13_weight, layer.w2_weight)

        assert layer.w13_weight.shape == w13_shape
        assert layer.w2_weight.shape == w2_shape
        assert layer.w13_weight.data_ptr() == w13_ptr
        assert layer.w2_weight.data_ptr() == w2_ptr
        kernel_w13, kernel_w2 = method._kernel_weights(layer)
        torch.testing.assert_close(kernel_w13, expected_w13)
        torch.testing.assert_close(kernel_w2, expected_w2)


@pytest.mark.parametrize(
    ("weight_key", "activation_key", "activation", "expected"),
    [
        (kMxfp8Static, kMxfp8Dynamic, MoEActivation.SILU, True),
        (kFp8Static128BlockSym, kFp8Dynamic128Sym, MoEActivation.SILU, True),
        (
            kFp8Static128BlockSym,
            kFp8Dynamic128Sym,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            True,
        ),
        (kFp8StaticTensorSym, kFp8DynamicTensorSym, MoEActivation.SILU, False),
        # FlashInfer takes the clamp only with a SwiGLU activation.
        (
            kFp8Static128BlockSym,
            kFp8Dynamic128Sym,
            MoEActivation.RELU2_NO_MUL,
            False,
        ),
    ],
)
def test_trtllm_fp8_swiglu_clamp_support(
    weight_key: QuantKey,
    activation_key: QuantKey,
    activation: MoEActivation,
    expected: bool,
):
    """FlashInfer >= 0.6.18 applies the SwiGLU clamp for both block-scaled
    kernels with a SwiGLU activation (DeepSeek-V4 sets swiglu_limit); the
    per-tensor kernel has no clamp, and Relu2 rejects the parameters."""

    class _Experts(TrtLlmFp8ExpertsModular):
        @staticmethod
        def _supports_current_device() -> bool:
            return True

        @staticmethod
        def _supports_quant_scheme(
            weight_key: QuantKey | None, activation_key: QuantKey | None
        ) -> bool:
            return True

    moe_config = make_dummy_moe_config()
    moe_config.swiglu_limit = 7.0
    moe_config.activation = activation

    supported, reason = _Experts.is_supported_config(
        _Experts,
        moe_config,
        weight_key,
        activation_key,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert supported == expected, reason
    if not expected:
        assert "SwiGLU" in reason
