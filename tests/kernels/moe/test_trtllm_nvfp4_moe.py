# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the FlashInfer TRTLLM NvFP4 MoE backend
(`TrtLlmNvFp4ExpertsModular`).

Covers the activations the wrapper claims to support — SiLU, RELU^2 (non-gated),
and GELU — including a Gemma4-shaped case (128 experts, top-k 8,
intermediate_size 704) that exercises the non-256-aligned padding path.
"""

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_test_quant_config
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    break_fp4_bytes,
    convert_swizzled_to_linear,
    dequantize_nvfp4_to_dtype,
)
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.custom_op import CustomOp, op_registry
from vllm.model_executor.layers.activation import (
    SiluAndMul,
    SiluAndMulWithClamp,
    SituAndMul,
)
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsModular,
    TrtLlmNvFp4ExpertsMonolithic,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    prepare_static_weights_for_trtllm_fp4_moe,
    reorder_w1w3_to_w3w1,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed

if pytest and (
    not has_flashinfer_trtllm_fused_moe()
    or not current_platform.has_device_capability(100)
):
    pytest.skip(
        "Requires flashinfer TRTLLM fused MoE and NvFP4 (SM100)",
        allow_module_level=True,
    )

# (m, n, k) = (tokens, intermediate_size_per_partition, hidden_dim).
# The (64, 704, 4096) row matches Gemma4's MoE shape and exercises the
# non-256-aligned intermediate (padded inside the wrapper).
MNK_FACTORS = [
    (2, 1024, 1024),
    (64, 2048, 1536),
    (64, 704, 4096),
]

_SWIGLU_LIMIT = 0.1
_LARGE_OUTPUT1_SCALE = 32768.0
_CLAMP_OP_NAME = "test_silu_and_mul_with_clamp"
_SITU_OP_NAME = "test_situ_and_mul"

# Test-only fixed-limit clamp. ``custom_op_name`` makes the class itself
# valid as an ``activation=`` argument to ``torch_moe`` (which only looks
# up ``activation.custom_op_name`` in ``op_registry``), so no
# ``MoEActivation`` enum extension is needed.
if _CLAMP_OP_NAME not in op_registry:

    @CustomOp.register(_CLAMP_OP_NAME)
    class _SiluAndMulWithClampTest(SiluAndMulWithClamp):
        custom_op_name = _CLAMP_OP_NAME

        def __init__(self, *, compile_native: bool = True) -> None:
            super().__init__(_SWIGLU_LIMIT, compile_native=compile_native)


if _SITU_OP_NAME not in op_registry:

    @CustomOp.register(_SITU_OP_NAME)
    class _SituAndMulTest(SituAndMul):
        custom_op_name = _SITU_OP_NAME

        def __init__(self, *, compile_native: bool = True) -> None:
            super().__init__(4.0, 25.0, compile_native=compile_native)


SILU_WITH_CLAMP = op_registry[_CLAMP_OP_NAME]
SITU = op_registry[_SITU_OP_NAME]


ACTIVATION_CASES = [
    pytest.param(MoEActivation.SILU, MoEActivation.SILU, None, id="silu"),
    pytest.param(MoEActivation.SILU, SILU_WITH_CLAMP, _SWIGLU_LIMIT, id="silu_clamp"),
    pytest.param(MoEActivation.SITU, SITU, None, id="situ"),
    pytest.param(
        MoEActivation.RELU2_NO_MUL,
        MoEActivation.RELU2_NO_MUL,
        None,
        id="relu2_no_mul",
    ),
    pytest.param(MoEActivation.GELU, MoEActivation.GELU, None, id="gelu"),
]


def _dequantize_linear_nvfp4(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_decode_scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize row-major K16 NVFP4 scales without layout conversion."""
    m, packed_k = tensor_fp4.shape
    values = break_fp4_bytes(tensor_fp4, torch.float32).reshape(m, -1, 16)
    block_scales = tensor_sf.view(torch.float8_e4m3fn).float().reshape(m, -1)
    return (
        values * block_scales.unsqueeze(-1) * global_decode_scale.reshape(m, 1, 1)
    ).reshape(m, packed_k * 2)


def _dynamic_gemm2_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gemm1_encode_scale: torch.Tensor,
) -> torch.Tensor:
    """Reference static GEMM1 input and per-routed-row dynamic GEMM2 input."""
    from flashinfer import SfLayout, nvfp4_quantize

    hidden_fp4, hidden_sf = ops.scaled_fp4_quant(
        hidden_states,
        gemm1_encode_scale,
        is_sf_swizzled_layout=False,
    )
    hidden_dequant = _dequantize_linear_nvfp4(
        hidden_fp4,
        hidden_sf,
        (1.0 / gemm1_encode_scale).expand(hidden_states.shape[0]),
    ).to(hidden_states.dtype)

    m, topk = topk_ids.shape
    expanded_hidden = (
        hidden_dequant[:, None, :].expand(m, topk, -1).reshape(m * topk, -1)
    )
    flat_topk_ids = topk_ids.reshape(-1)
    route_outputs = torch.zeros(
        (m * topk, hidden_states.shape[1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    for expert_idx in range(w1.shape[0]):
        expert_mask = flat_topk_ids == expert_idx
        if not expert_mask.any():
            continue
        gemm1_output = expanded_hidden[expert_mask] @ w1[expert_idx].T
        gemm2_input = SiluAndMul()(gemm1_output)
        gemm2_fp4, gemm2_sf, gemm2_decode_scale = nvfp4_quantize(
            gemm2_input,
            1.0 / (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX),
            sfLayout=SfLayout.layout_linear,
            per_token_activation=True,
        )
        gemm2_input_dequant = _dequantize_linear_nvfp4(
            gemm2_fp4, gemm2_sf, gemm2_decode_scale
        ).to(hidden_states.dtype)
        route_outputs[expert_mask] = gemm2_input_dequant @ w2[expert_idx].T

    return (
        (route_outputs.view(m, topk, -1).float() * topk_weights.unsqueeze(-1).float())
        .sum(dim=1)
        .to(hidden_states.dtype)
    )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [128])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("activation,torch_activation,swiglu_limit", ACTIVATION_CASES)
@torch.inference_mode()
def test_trtllm_fp4_moe_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    activation: MoEActivation,
    torch_activation: MoEActivation | type[SiluAndMulWithClamp] | type[SituAndMul],
    swiglu_limit: float | None,
    workspace_init,
):
    # FlashInfer's trtllm_batched_gemm_runner has no precompiled tile
    # config for non-gated RELU^2 at non-256-aligned intermediate_size
    # (e.g. Gemma4's 704). Other activations (SiLU/GELU) work at the
    # same shape. Tracked upstream in FlashInfer; unrelated to this
    # PR's GELU enablement (Gemma4 uses GeGLU, not non-gated RELU^2).
    if activation == MoEActivation.RELU2_NO_MUL and (m, n, k) == (64, 704, 4096):
        pytest.skip(
            "FlashInfer trtllm_batched_gemm_runner: no valid tile config "
            "for non-gated RELU^2 at intermediate_size=704 "
            "(getValidConfigIndices throws). Tracked upstream."
        )

    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        quant_blocksize = 16
        is_gated_act = activation.is_gated

        w1_q, w2_q, quant_config = make_test_quant_config(
            e,
            n,
            k,
            in_dtype=dtype,
            quant_dtype="nvfp4",
            block_shape=None,
            per_act_token_quant=False,
            make_gate=is_gated_act,
            # The TRT-LLM FP4 MoE kernel rejects swizzled (padded) activation
            # scales — its numel-based vec_size check requires numel == M*K/16.
            # Match what oracle/nvfp4.py does for this backend.
            is_scale_swizzled=False,
        )
        quant_config.gemm1_clamp_limit = swiglu_limit
        if swiglu_limit is not None:
            assert quant_config.g1_alphas is not None
            assert quant_config.a2_gscale is not None
            assert torch.all(quant_config.a2_gscale == 1)
            # With a2_gscale == 1, g1_alphas is the TRTLLM
            # output1_scale_gate_scalar. Make it large enough to catch
            # clamp/output-scale coupling in the FlashInfer kernel wrapper.
            quant_config.g1_alphas.fill_(_LARGE_OUTPUT1_SCALE)

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

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
            in_dtype=dtype,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=next_power_of_2(m),
            activation_situ_beta=4.0 if activation == MoEActivation.SITU else None,
            activation_situ_linear_beta=(
                25.0 if activation == MoEActivation.SITU else None
            ),
        )

        trtllm_inner = TrtLlmNvFp4ExpertsModular(
            moe_config=moe_config, quant_config=quant_config
        )
        # Mimic the production weight-loader path so per-expert tensors that
        # are normally precomputed in process_weights_after_loading (g1_scale_c
        # and the rescaled gemm1_clamp_limit) get materialized. The test's
        # synthetic quant_config has g1_alphas/g2_alphas already at their
        # post-fusion values, so we set w13_weight_scale_2 to alias g1_alphas
        # (same tensor) and use input_scale=1 to make the in-place
        # weight_scale_2 *= input_scale step a no-op.
        fake_layer = torch.nn.Module()
        fake_layer.w13_weight_scale_2 = quant_config.g1_alphas
        fake_layer.w2_weight_scale_2 = quant_config.g2_alphas
        fake_layer.w13_input_scale = torch.ones_like(quant_config.g1_alphas)
        fake_layer.w2_input_scale = torch.ones_like(quant_config.g2_alphas)
        trtllm_inner.process_weights_after_loading(fake_layer)
        if activation == MoEActivation.SITU:
            torch.testing.assert_close(trtllm_inner.g1_scale_c, quant_config.a2_gscale)

        trtllm_experts = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            trtllm_inner,
        )

        trtllm_output = trtllm_experts.apply(
            hidden_states=a,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

        # Reference: round-trip activations and weights through FP4
        # quant/dequant so the comparison isolates kernel/activation behavior
        # from quantization error.
        a_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / a.abs().max()).to(
            torch.float32
        )
        a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)
        a_in_dtype = dequantize_nvfp4_to_dtype(
            a_fp4,
            a_scale_interleaved,
            a_global_scale,
            dtype=a.dtype,
            device=a.device,
            block_size=quant_blocksize,
        )

        w1_d = torch.empty(
            (e, (2 if is_gated_act else 1) * n, k), device="cuda", dtype=dtype
        )
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)
        for idx in range(e):
            w1_d[idx] = dequantize_nvfp4_to_dtype(
                w1_q[idx],
                quant_config.w1_scale[idx],
                (1 / quant_config.g1_alphas[idx]),
                dtype=dtype,
                device=w1_q.device,
                block_size=quant_blocksize,
            )
            w2_d[idx] = dequantize_nvfp4_to_dtype(
                w2_q[idx],
                quant_config.w2_scale[idx],
                (1 / quant_config.g2_alphas[idx]),
                dtype=dtype,
                device=w2_q.device,
                block_size=quant_blocksize,
            )

        torch_output = torch_moe(
            a_in_dtype, w1_d, w2_d, score, topk, activation=torch_activation
        )

        torch.testing.assert_close(torch_output, trtllm_output, atol=2e-1, rtol=2e-1)


@pytest.mark.parametrize("use_monolithic", [False, True], ids=["modular", "monolithic"])
@torch.inference_mode()
def test_trtllm_fp4_moe_dynamic_gemm2_matches_rowwise_reference(
    use_monolithic: bool, workspace_init
):
    """Dynamic GEMM2 ignores its static scale and keeps GEMM1 static."""
    m, n, k, e, topk = 17, 1024, 1024, 4, 2
    dtype = torch.bfloat16

    set_random_seed(11)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        hidden_states = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        row_amplitudes = torch.logspace(
            -1, 1, m, device="cuda", dtype=torch.float32
        ).to(dtype)
        hidden_states.mul_(row_amplitudes.unsqueeze(1))

        w1_q, w2_q, reference_quant_config = make_test_quant_config(
            e,
            n,
            k,
            in_dtype=dtype,
            quant_dtype="nvfp4",
            per_act_token_quant=False,
            make_gate=True,
            is_scale_swizzled=False,
        )
        assert reference_quant_config.a1_gscale is not None
        assert reference_quant_config.a2_gscale is not None
        assert reference_quant_config.g1_alphas is not None
        assert reference_quant_config.g2_alphas is not None
        assert reference_quant_config.w1_scale is not None
        assert reference_quant_config.w2_scale is not None
        reference_quant_config.a1_gscale.fill_(8.0)
        # Deliberately unusable for these activations. The dynamic path must
        # neither fold nor consume this checkpoint-static GEMM2 input scale.
        reference_quant_config.a2_gscale.fill_(1.0e-6)

        w1_scale_linear = torch.stack(
            [
                convert_swizzled_to_linear(
                    reference_quant_config.w1_scale[expert_idx], 2 * n, k, 16
                )
                for expert_idx in range(e)
            ]
        )
        w2_scale_linear = torch.stack(
            [
                convert_swizzled_to_linear(
                    reference_quant_config.w2_scale[expert_idx], k, n, 16
                )
                for expert_idx in range(e)
            ]
        )
        w1_kernel, w1_scale_linear = reorder_w1w3_to_w3w1(
            w1_q.clone(), w1_scale_linear, dim=1
        )
        (
            w1_kernel,
            w1_scale_kernel,
            w2_kernel,
            w2_scale_kernel,
        ) = prepare_static_weights_for_trtllm_fp4_moe(
            w1_kernel,
            w2_q.clone(),
            w1_scale_linear,
            w2_scale_linear,
            hidden_size=k,
            intermediate_size=n,
            num_experts=e,
            is_gated_activation=True,
        )
        quant_config = nvfp4_moe_quant_config(
            g1_alphas=reference_quant_config.g1_alphas,
            g2_alphas=reference_quant_config.g2_alphas,
            a1_gscale=reference_quant_config.a1_gscale,
            a2_gscale=reference_quant_config.a2_gscale,
            w1_scale=w1_scale_kernel,
            w2_scale=w2_scale_kernel,
            is_scale_swizzled=False,
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, score, topk, renormalize=False
        )
        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=MoEActivation.SILU,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=next_power_of_2(m),
        )

        experts_cls = (
            TrtLlmNvFp4ExpertsMonolithic
            if use_monolithic
            else TrtLlmNvFp4ExpertsModular
        )
        trtllm_inner = experts_cls(
            moe_config=moe_config,
            quant_config=quant_config,
            dynamic_gemm2=True,
        )
        fake_layer = torch.nn.Module()
        fake_layer.w13_weight_scale_2 = quant_config.g1_alphas
        fake_layer.w2_weight_scale_2 = quant_config.g2_alphas
        fake_layer.w13_input_scale = 1.0 / quant_config.a1_gscale
        fake_layer.w2_input_scale = 1.0 / quant_config.a2_gscale
        trtllm_inner.process_weights_after_loading(fake_layer)

        trtllm_experts = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=use_monolithic,
            ),
            trtllm_inner,
        )
        if use_monolithic:
            actual = trtllm_experts.apply_monolithic(
                hidden_states=hidden_states,
                w1=w1_kernel,
                w2=w2_kernel,
                router_logits=torch.softmax(score.float(), dim=-1),
                activation=MoEActivation.SILU,
                global_num_experts=e,
                expert_map=None,
                apply_router_weight_on_input=False,
            )
        else:
            actual = trtllm_experts.apply(
                hidden_states=hidden_states,
                w1=w1_kernel,
                w2=w2_kernel,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=MoEActivation.SILU,
                global_num_experts=e,
                expert_map=None,
                apply_router_weight_on_input=False,
            )

        w1_dequant = torch.empty((e, 2 * n, k), device="cuda", dtype=dtype)
        w2_dequant = torch.empty((e, k, n), device="cuda", dtype=dtype)
        for expert_idx in range(e):
            w1_dequant[expert_idx] = dequantize_nvfp4_to_dtype(
                w1_q[expert_idx],
                reference_quant_config.w1_scale[expert_idx],
                1.0 / reference_quant_config.g1_alphas[expert_idx],
                dtype=dtype,
                device=w1_q.device,
            )
            w2_dequant[expert_idx] = dequantize_nvfp4_to_dtype(
                w2_q[expert_idx],
                reference_quant_config.w2_scale[expert_idx],
                1.0 / reference_quant_config.g2_alphas[expert_idx],
                dtype=dtype,
                device=w2_q.device,
            )

        expected = _dynamic_gemm2_reference(
            hidden_states,
            w1_dequant,
            w2_dequant,
            topk_weights,
            topk_ids,
            reference_quant_config.a1_gscale[0],
        )
        relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float())
        relative_l2 /= torch.linalg.vector_norm(expected.float())
        assert relative_l2 < 0.05
        torch.testing.assert_close(actual, expected, atol=5e-1, rtol=2e-1)


if __name__ == "__main__":
    test_trtllm_fp4_moe_no_graph(
        64,
        704,
        4096,
        128,
        8,
        torch.bfloat16,
        MoEActivation.GELU,
        MoEActivation.GELU,
        None,
        None,
    )
