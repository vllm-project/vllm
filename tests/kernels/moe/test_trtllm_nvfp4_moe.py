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
    dequantize_nvfp4_to_dtype,
)
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.custom_op import CustomOp, op_registry
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsModular,
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


SILU_WITH_CLAMP = op_registry[_CLAMP_OP_NAME]


ACTIVATION_CASES = [
    pytest.param(MoEActivation.SILU, MoEActivation.SILU, None, id="silu"),
    pytest.param(MoEActivation.SILU, SILU_WITH_CLAMP, _SWIGLU_LIMIT, id="silu_clamp"),
    pytest.param(
        MoEActivation.RELU2_NO_MUL,
        MoEActivation.RELU2_NO_MUL,
        None,
        id="relu2_no_mul",
    ),
    pytest.param(MoEActivation.GELU, MoEActivation.GELU, None, id="gelu"),
]


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
    torch_activation: MoEActivation | type[SiluAndMulWithClamp],
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
            intermediate_size_per_partition=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            is_act_and_mul=is_gated_act,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=next_power_of_2(m),
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

        trtllm_experts = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            trtllm_inner,
            inplace=False,
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
