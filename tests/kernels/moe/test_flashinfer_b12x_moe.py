# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(120):
    pytest.skip(
        reason="FlashInfer CuteDSL SM12x MoE requires SM120 "
        "(RTX Pro 6000 / DGX Spark).",
        allow_module_level=True,
    )

from vllm.utils.flashinfer import has_flashinfer_b12x_moe

if not has_flashinfer_b12x_moe():
    pytest.skip(
        reason=(
            "FlashInfer cute_dsl_fused_moe_nvfp4 / convert_sf_to_mma_layout "
            "not available in installed FlashInfer (needs PRs #3051 and #3066)."
        ),
        allow_module_level=True,
    )

# Import fp4_quantize after the skip guard — FlashInfer must be installed.
from flashinfer.fp4_quantization import fp4_quantize

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.utils import torch_moe
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import (
    FlashInferB12xExperts,
)
from vllm.utils.flashinfer import flashinfer_convert_sf_to_mma_layout
from vllm.utils.torch_utils import set_random_seed

# Dimensions chosen to satisfy FP4 alignment requirements (k multiple of 256,
# n multiple of 128) while keeping tests fast.
MNK_FACTORS = [
    (2, 128, 256),
    (2, 256, 512),
    (16, 128, 256),
    (64, 256, 512),
]


def _reorder_gate_up_to_up_gate(
    w: torch.Tensor,
    w_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Swap gate and up-projection halves along dim=1 to [up, gate] order.

    The SM12x kernel expects weights in [up (w3), gate (w1)] order while the
    BF16 reference uses [gate (w1), up (w3)].  This replicates the reordering
    done at model-load time by ``prepare_nvfp4_moe_layer_for_fi_or_cutlass``.
    """
    n = w.shape[1] // 2
    return (
        torch.cat([w[:, n:, :], w[:, :n, :]], dim=1),
        torch.cat([w_s[:, n:, :], w_s[:, :n, :]], dim=1),
    )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [8, 16])
@pytest.mark.parametrize("topk", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_flashinfer_b12x_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    workspace_init,
):
    """Test FlashInferB12xExperts against a BF16 torch reference.

    The SM12x kernel takes BF16 hidden states directly and fuses token
    dispatch, W1 GEMM, SwiGLU, and W2 GEMM into one call.  We verify
    correctness against ``torch_moe`` using generous tolerances to account
    for the internal FP4 quantization of activations and weights.

    Scale convention
    ----------------
    The SM12x kernel uses ``w1_alpha`` as *both* the activation-quantisation
    global scale and the weight dequantisation factor.  These two roles are
    conflated into a single parameter in ``launch_sm120_moe``, so they must
    equal the same value.  We use ``global_scale = 1.0`` for
    ``fp4_quantize`` so that ``w1_alpha = ones`` satisfies both roles
    simultaneously.  The alternative — vLLM's convention of baking a large
    ``w_gs`` into block-scale values and compensating with
    ``g1_alphas = 1/w_gs`` — is incompatible with this kernel.
    """
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        # Generate BF16 reference weights in [gate, up] order.
        # Shape: w1=(e, 2n, k), w2=(e, k, n).
        w1_bf16 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 15
        w2_bf16 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 15

        # ------------------------------------------------------------------ #
        # Quantise weights for the SM12x kernel using FlashInfer's convention:
        #   global_scale = 1.0   →   block_scale = max_abs_block / fp4_max
        #   w1_alpha = 1.0       (no extra global factor to compensate)
        #
        # The scale factors returned by fp4_quantize(..., is_sf_swizzled_layout=True)
        # are already in the swizzled 2D layout expected by convert_sf_to_mma_layout.
        # No additional swizzle_blockscale() call is needed.
        # ------------------------------------------------------------------ #
        gs = torch.ones(1, device="cuda", dtype=torch.float32)
        sf_vec_size = 16

        # W1: reorder BF16 from [gate, up] → [up, gate], then quantise.
        w1_reordered = torch.cat(
            [w1_bf16[:, n:, :], w1_bf16[:, :n, :]], dim=1
        )  # shape (e, 2n, k), [up, gate]
        w1_flat = w1_reordered.reshape(e * 2 * n, k)
        w1_q_flat, w1_sf_flat = fp4_quantize(
            w1_flat,
            global_scale=gs,
            sf_vec_size=sf_vec_size,
            is_sf_swizzled_layout=True,
        )
        w1_q = w1_q_flat.view(e, 2 * n, k // 2)  # uint8, packed FP4
        w1_blockscale = w1_sf_flat.view(e, 2 * n, w1_sf_flat.shape[1])  # float8

        # W2: no row reordering needed for the down-projection.
        w2_flat = w2_bf16.reshape(e * k, n)
        w2_q_flat, w2_sf_flat = fp4_quantize(
            w2_flat,
            global_scale=gs,
            sf_vec_size=sf_vec_size,
            is_sf_swizzled_layout=True,
        )
        w2_q = w2_q_flat.view(e, k, n // 2)  # uint8, packed FP4
        w2_blockscale = w2_sf_flat.view(e, k, w2_sf_flat.shape[1])  # float8

        # All per-expert alphas are 1.0 (global_scale = 1.0, no compensation).
        ones_e = torch.ones(e, device="cuda", dtype=torch.float32)

        quant_config = nvfp4_moe_quant_config(
            g1_alphas=ones_e,
            g2_alphas=ones_e,
            a1_gscale=ones_e,
            a2_gscale=ones_e,
            w1_scale=w1_blockscale,
            w2_scale=w2_blockscale,
        )

        moe_config = make_dummy_moe_config(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size_per_partition=n,
            in_dtype=dtype,
        )

        experts = FlashInferB12xExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        # In production, process_weights_after_loading computes these after
        # normalizing block scales. In the test the scales are already in final
        # form (global_scale=1.0), so we compute the MMA layouts directly.
        num_experts_w1, m1, k1_sf = w1_blockscale.shape
        experts.w1_sf_mma = flashinfer_convert_sf_to_mma_layout(
            w1_blockscale.reshape(num_experts_w1 * m1, k1_sf),
            m=m1,
            k=k1_sf * 16,
            num_groups=num_experts_w1,
        )
        num_experts_w2, m2, k2_sf = w2_blockscale.shape
        experts.w2_sf_mma = flashinfer_convert_sf_to_mma_layout(
            w2_blockscale.reshape(num_experts_w2 * m2, k2_sf),
            m=m2,
            k=k2_sf * 16,
            num_groups=num_experts_w2,
        )

        kernel = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            experts,
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        sm12x_output = kernel.apply(
            hidden_states=a,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
        )

        # Reference: BF16 torch MoE using original [gate, up] BF16 weights.
        # torch_moe's SiluAndMul expects [gate, up] order, matching w1_bf16.
        torch_output = torch_moe(a, w1_bf16, w2_bf16, score, topk)

        torch.testing.assert_close(sm12x_output, torch_output, atol=2e-1, rtol=2e-1)


if __name__ == "__main__":
    test_flashinfer_b12x_moe(16, 128, 256, 8, 2, torch.bfloat16)
