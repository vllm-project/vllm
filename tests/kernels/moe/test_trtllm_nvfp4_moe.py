# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the FlashInfer TRTLLM NvFP4 MoE backend
(`TrtLlmNvFp4ExpertsModular`).

Covers the activations the wrapper claims to support — SiLU, RELU^2 (non-gated),
and GELU — including a Gemma4-shaped case (128 experts, top-k 8,
intermediate_size 704) that exercises the non-256-aligned padding path.
"""

from dataclasses import replace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import check_deferred_moe_finalize, make_test_quant_config
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
    FusedMoEQuantConfig,
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
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed

if pytest and (
    not has_flashinfer_trtllm_fused_moe()
    or not current_platform.is_device_capability_family(100)
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


def _make_trtllm_fp4_moe_kernel(
    moe_config: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig,
    per_token_activation: bool = False,
) -> mk.FusedMoEKernel:
    experts = TrtLlmNvFp4ExpertsModular(
        moe_config=moe_config,
        quant_config=quant_config,
        per_token_activation=per_token_activation,
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
    experts.process_weights_after_loading(fake_layer)
    return mk.FusedMoEKernel(
        maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        experts,
    )


ACTIVATION_CASES = [
    pytest.param(MoEActivation.SILU, MoEActivation.SILU, None, False, id="silu"),
    pytest.param(
        MoEActivation.SILU,
        MoEActivation.SILU,
        None,
        True,
        id="silu_per_token",
    ),
    pytest.param(
        MoEActivation.SILU,
        SILU_WITH_CLAMP,
        _SWIGLU_LIMIT,
        False,
        id="silu_clamp",
    ),
    pytest.param(MoEActivation.SITU, SITU, None, False, id="situ"),
    pytest.param(
        MoEActivation.RELU2_NO_MUL,
        MoEActivation.RELU2_NO_MUL,
        None,
        False,
        id="relu2_no_mul",
    ),
    pytest.param(
        MoEActivation.RELU2_NO_MUL,
        MoEActivation.RELU2_NO_MUL,
        None,
        True,
        id="relu2_no_mul_per_token",
    ),
    pytest.param(MoEActivation.GELU, MoEActivation.GELU, None, False, id="gelu"),
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
@pytest.mark.parametrize(
    "activation,torch_activation,swiglu_limit,per_token_activation",
    ACTIVATION_CASES,
)
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
    per_token_activation: bool,
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

        w1_q_linear = w1_q
        w2_q_linear = w2_q
        w1_scale_reference = quant_config.w1_scale
        w2_scale_reference = quant_config.w2_scale
        assert w1_scale_reference is not None
        assert w2_scale_reference is not None
        w1_scale_linear = torch.stack(
            [
                convert_swizzled_to_linear(scale, w1_q.size(1), k, quant_blocksize)
                for scale in w1_scale_reference
            ]
        )
        w2_scale_linear = torch.stack(
            [
                convert_swizzled_to_linear(scale, w2_q.size(1), n, quant_blocksize)
                for scale in w2_scale_reference
            ]
        )
        w1_q_for_kernel = w1_q
        if is_gated_act:
            w1_q_for_kernel, w1_scale_linear = reorder_w1w3_to_w3w1(
                w1_q.clone(), w1_scale_linear
            )
        w1_q, w1_scale, w2_q, w2_scale = prepare_static_weights_for_trtllm_fp4_moe(
            w1_q_for_kernel,
            w2_q,
            w1_scale_linear,
            w2_scale_linear,
            hidden_size=k,
            intermediate_size=n,
            num_experts=e,
            is_gated_activation=is_gated_act,
        )
        quant_config._w1.scale = w1_scale
        quant_config._w2.scale = w2_scale

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

        trtllm_experts = _make_trtllm_fp4_moe_kernel(
            moe_config,
            quant_config,
            per_token_activation=per_token_activation,
        )
        if activation == MoEActivation.SITU:
            assert isinstance(trtllm_experts.fused_experts, TrtLlmNvFp4ExpertsModular)
            torch.testing.assert_close(
                trtllm_experts.fused_experts.g1_scale_c, quant_config.a2_gscale
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
        assert isinstance(trtllm_output, torch.Tensor)

        # Reference: round-trip activations and weights through FP4
        # quant/dequant so the comparison isolates kernel/activation behavior
        # from quantization error.
        if per_token_activation:
            from flashinfer import SfLayout, nvfp4_quantize

            a_fp4, a_scale_interleaved, per_token_scale = nvfp4_quantize(
                a,
                NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE,
                sfLayout=SfLayout.layout_linear,
                per_token_activation=True,
            )
            a_global_scale = 1.0 / per_token_scale.unsqueeze(1)
        else:
            assert quant_config.a1_gscale is not None
            a_global_scale = quant_config.a1_gscale[0]
            a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)
        a_in_dtype = dequantize_nvfp4_to_dtype(
            a_fp4,
            a_scale_interleaved,
            a_global_scale,
            dtype=a.dtype,
            device=a.device,
            block_size=quant_blocksize,
            is_sf_linear_layout=per_token_activation,
        )

        w1_d = torch.empty(
            (e, (2 if is_gated_act else 1) * n, k), device="cuda", dtype=dtype
        )
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)
        assert quant_config.g1_alphas is not None
        assert quant_config.g2_alphas is not None
        for idx in range(e):
            w1_d[idx] = dequantize_nvfp4_to_dtype(
                w1_q_linear[idx],
                w1_scale_reference[idx],
                (1 / quant_config.g1_alphas[idx]),
                dtype=dtype,
                device=w1_q.device,
                block_size=quant_blocksize,
            )
            w2_d[idx] = dequantize_nvfp4_to_dtype(
                w2_q_linear[idx],
                w2_scale_reference[idx],
                (1 / quant_config.g2_alphas[idx]),
                dtype=dtype,
                device=w2_q.device,
                block_size=quant_blocksize,
            )

        torch_output = torch_moe(
            a_in_dtype, w1_d, w2_d, score, topk, activation=torch_activation
        )

        if swiglu_limit is None:
            cosine = torch.nn.functional.cosine_similarity(
                torch_output.float().flatten(),
                trtllm_output.float().flatten(),
                dim=0,
            )
            assert cosine > 0.99
        torch.testing.assert_close(torch_output, trtllm_output, atol=2e-1, rtol=2e-1)


def _run_fp4_padded(
    kernel: mk.FusedMoEKernel,
    hidden_states: torch.Tensor,
    a1q_scale: torch.Tensor | None,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Drive `TrtLlmNvFp4ExpertsModular.apply` on a padded layout.

    `topk_ids` already carries the padding tail (rows whose ids are -1).
    FlashInfer manages the workspaces, so pass the (0,)-shaped stand-ins that
    `workspace_shapes` advertises.
    """
    experts = kernel.fused_experts
    assert isinstance(experts, TrtLlmNvFp4ExpertsModular)
    output = torch.empty(
        (hidden_states.shape[0], experts.hidden_dim_unpadded),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    empty_ws = torch.empty((0,), device=hidden_states.device, dtype=torch.bfloat16)
    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=experts.moe_config.num_experts,
        expert_map=None,
        a1q_scale=a1q_scale,
        a2_scale=None,
        workspace13=empty_ws,
        workspace2=empty_ws,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    return output


@pytest.mark.parametrize("per_token_activation", [False, True])
@pytest.mark.parametrize("m,n,k", [(64, 1024, 1024)])
@pytest.mark.parametrize("e", [128])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_trtllm_fp4_moe_padding_robust(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    per_token_activation: bool,
    workspace_init,
):
    """Padding rows (topk_ids == -1) must be skipped, never fed into results.

    DeepEP v2 hands these experts a PaddedStandard buffer whose tail rows are
    inactive. Construct that layout directly, poison the padding rows with
    NaN, and check the valid outputs are finite, bit-identical to a
    zero-padding run, and bit-identical to the unpadded kernel run.
    """
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        pad = 64
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1_q, w2_q, quant_config = make_test_quant_config(
            e,
            n,
            k,
            in_dtype=dtype,
            quant_dtype="nvfp4",
            block_shape=None,
            per_act_token_quant=False,
            make_gate=True,
            is_scale_swizzled=False,
        )
        assert quant_config.w1_scale is not None
        assert quant_config.w2_scale is not None
        w1_scale_linear = torch.stack(
            [
                convert_swizzled_to_linear(scale, w1_q.size(1), k, 16)
                for scale in quant_config.w1_scale
            ]
        )
        w2_scale_linear = torch.stack(
            [
                convert_swizzled_to_linear(scale, w2_q.size(1), n, 16)
                for scale in quant_config.w2_scale
            ]
        )
        w1_q_kernel, w1_scale_linear = reorder_w1w3_to_w3w1(
            w1_q.clone(), w1_scale_linear
        )
        w1_q, w1_scale, w2_q, w2_scale = prepare_static_weights_for_trtllm_fp4_moe(
            w1_q_kernel,
            w2_q,
            w1_scale_linear,
            w2_scale_linear,
            hidden_size=k,
            intermediate_size=n,
            num_experts=e,
            is_gated_activation=True,
        )
        quant_config._w1.scale = w1_scale
        quant_config._w2.scale = w2_scale

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        scores = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(scores, topk)
        topk_ids = topk_ids.to(torch.int32)

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
            max_num_tokens=next_power_of_2(m + pad),
        )
        kernel = _make_trtllm_fp4_moe_kernel(
            moe_config,
            quant_config,
            per_token_activation=per_token_activation,
        )

        topk_ids_padded = torch.full(
            (m + pad, topk), -1, device="cuda", dtype=torch.int32
        )
        topk_ids_padded[:m] = topk_ids
        topk_weights_padded = torch.zeros(
            (m + pad, topk), device="cuda", dtype=topk_weights.dtype
        )
        topk_weights_padded[:m] = topk_weights

        a_clean = torch.zeros((m + pad, k), device="cuda", dtype=dtype)
        a_clean[:m] = a
        a_poison = a_clean.clone()
        a_poison[m:] = float("nan")

        def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
            if per_token_activation:
                return x, None
            assert quant_config.a1_gscale is not None
            return ops.scaled_fp4_quant(
                x, quant_config.a1_gscale[0], is_sf_swizzled_layout=False
            )

        hs_clean, scale_clean = quantize(a_clean)
        hs_poison, scale_poison = quantize(a_poison)
        hs_unpadded, scale_unpadded = quantize(a)

        out_clean = _run_fp4_padded(
            kernel,
            hs_clean,
            scale_clean,
            w1_q,
            w2_q,
            topk_weights_padded,
            topk_ids_padded,
        )
        out_poison = _run_fp4_padded(
            kernel,
            hs_poison,
            scale_poison,
            w1_q,
            w2_q,
            topk_weights_padded,
            topk_ids_padded,
        )
        out_unpadded = _run_fp4_padded(
            kernel,
            hs_unpadded,
            scale_unpadded,
            w1_q,
            w2_q,
            topk_weights,
            topk_ids,
        )

        assert torch.isfinite(out_poison[:m]).all(), (
            "padding garbage leaked into valid outputs"
        )
        torch.testing.assert_close(out_poison[:m], out_clean[:m], rtol=0, atol=0)
        torch.testing.assert_close(out_clean[:m], out_unpadded, rtol=0, atol=0)


@torch.inference_mode()
def test_trtllm_fp4_moe_reprocess_does_not_refold_swiglu_params():
    """Weight reloads rerun post-processing on the same experts object; the
    g1_alphas fold of clamp/beta must start from the unfolded values."""
    e, n, k = 8, 256, 256
    clamp, beta = 7.0, 1.0
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        _, _, quant_config = make_test_quant_config(
            e,
            n,
            k,
            in_dtype=torch.bfloat16,
            quant_dtype="nvfp4",
            block_shape=None,
            per_act_token_quant=False,
            is_scale_swizzled=False,
        )
        quant_config.gemm1_clamp_limit = clamp
        quant_config.gemm1_beta = beta
        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=2,
            hidden_dim=k,
            intermediate_size=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=MoEActivation.SILU,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=torch.bfloat16,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=16,
        )
        experts = TrtLlmNvFp4ExpertsModular(
            moe_config=moe_config, quant_config=quant_config
        )
        g1_alphas = quant_config.g1_alphas
        assert g1_alphas is not None
        layer = torch.nn.Module()
        layer.w13_weight_scale_2 = g1_alphas
        layer.w2_weight_scale_2 = quant_config.g2_alphas
        layer.w13_input_scale = torch.ones_like(g1_alphas)
        layer.w2_input_scale = torch.ones_like(g1_alphas)

        for new_alphas in (0.5, 0.25):
            g1_alphas.fill_(new_alphas)
            experts.process_weights_after_loading(layer)
            expected = torch.full_like(g1_alphas, 1.0 / new_alphas)
            torch.testing.assert_close(experts.gemm1_clamp_limit, clamp * expected)
            torch.testing.assert_close(experts.gemm1_beta, beta * expected)


@pytest.mark.parametrize("m,chunk_size", [(1, None), (16, None), (16, 5)])
@torch.inference_mode()
def test_trtllm_fp4_moe_deferred_finalize(
    m: int,
    chunk_size: int | None,
    monkeypatch: pytest.MonkeyPatch,
    workspace_init,
):
    """TRTLLM-Gen NvFP4 modular experts can leave the top-k finalize to the caller."""
    e, topk, n, k = 128, 8, 1024, 1024
    dtype = torch.bfloat16
    if chunk_size is not None:
        monkeypatch.setattr(
            TrtLlmNvFp4ExpertsModular, "_get_chunk_size", lambda self: chunk_size
        )

    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1_q, w2_q, quant_config = make_test_quant_config(
            e, n, k, in_dtype=dtype, quant_dtype="nvfp4", is_scale_swizzled=False
        )
        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        # One rank of a TP group, which deferral needs.
        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=2 * n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=MoEActivation.SILU,
            device="cuda",
            moe_parallel_config=replace(
                FusedMoEParallelConfig.make_no_parallel(), tp_size=2
            ),
            in_dtype=dtype,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=next_power_of_2(m),
        )
        kernel = _make_trtllm_fp4_moe_kernel(moe_config, quant_config)

        check_deferred_moe_finalize(
            moe_config,
            lambda: kernel.apply(
                hidden_states=a,
                w1=w1_q,
                w2=w2_q,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=MoEActivation.SILU,
                global_num_experts=e,
                expert_map=None,
                apply_router_weight_on_input=False,
            ),
            router_weights=topk_weights,
            chunked=chunk_size is not None,
        )


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
        assert quant_config.a1_gscale is not None
        assert quant_config.a2_gscale is not None
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
                routing_sink=None,
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

        assert isinstance(actual, torch.Tensor)

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
        False,
        None,
    )
