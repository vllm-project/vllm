# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(120):
    pytest.skip(
        reason="FlashInfer CuteDSL SM12x MoE requires SM120 (Blackwell GeForce).",
        allow_module_level=True,
    )

from vllm.utils.flashinfer import has_flashinfer_cutedsl_sm12x_moe

if not has_flashinfer_cutedsl_sm12x_moe():
    pytest.skip(
        reason=(
            "FlashInfer cute_dsl_fused_moe_nvfp4 / convert_sf_to_mma_layout "
            "not available in installed FlashInfer (needs PRs #3051 and #3066)."
        ),
        allow_module_level=True,
    )

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config, make_test_weights
from tests.kernels.utils import torch_moe
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (
    FlashInferCuteDSLSM12xExperts,
)
from vllm.utils.torch_utils import set_random_seed

# Dimensions chosen to satisfy FP4 alignment requirements (k multiple of 256,
# n multiple of 128) while keeping tests fast.
MNK_FACTORS = [
    (2, 128, 256),
    (2, 256, 512),
    (16, 128, 256),
    (64, 256, 512),
]


def _reorder_w1w3_to_w3w1(
    w: torch.Tensor,
    w_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Swap gate and up-projection halves along dim=1.

    Replicates the reordering done at model-load time for the SM12X backend
    inside ``prepare_nvfp4_moe_layer_for_fi_or_cutlass``.  The kernel expects
    weights in [up (w3), gate (w1)] order while vLLM stores them as
    [gate (w1), up (w3)].
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
def test_flashinfer_cutedsl_sm12x_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    """Test FlashInferCuteDSLSM12xExperts against a BF16 torch reference.

    The SM12x kernel takes BF16 hidden states directly and fuses token
    dispatch, W1 GEMM, SwiGLU, and W2 GEMM into one call.  We verify
    correctness against ``torch_moe`` using generous tolerances to account
    for the internal FP4 quantization of activations.
    """
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        (w1_bf16, w1_q, w1_blockscale, w1_gs), (
            w2_bf16,
            w2_q,
            w2_blockscale,
            w2_gs,
        ) = make_test_weights(e, n, k, in_dtype=dtype, quant_dtype="nvfp4")

        assert w1_gs is not None and w2_gs is not None
        assert w1_blockscale is not None and w2_blockscale is not None

        # Simulate the w1/w3 → w3/w1 weight reorder that happens at
        # model-load time for the SM12X backend.  The reference (torch_moe)
        # uses the original [gate, up] BF16 weights, so only the quantized
        # tensors are reordered here.
        w1_q, w1_blockscale = _reorder_w1w3_to_w3w1(w1_q, w1_blockscale)

        a1_gs = torch.ones((e,), device="cuda", dtype=torch.float32)
        a2_gs = torch.ones((e,), device="cuda", dtype=torch.float32)

        quant_config = nvfp4_moe_quant_config(
            g1_alphas=(1 / w1_gs),
            g2_alphas=(1 / w2_gs),
            a1_gscale=a1_gs,
            a2_gscale=a2_gs,
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

        kernel = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            FlashInferCuteDSLSM12xExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
            inplace=False,
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

        # Reference: BF16 torch MoE using original (unswapped) weights.
        # torch_moe's SiluAndMul expects [gate, up] order, matching w1_bf16.
        torch_output = torch_moe(a, w1_bf16, w2_bf16, score, topk)

        torch.testing.assert_close(sm12x_output, torch_output, atol=2e-1, rtol=2e-1)


if __name__ == "__main__":
    test_flashinfer_cutedsl_sm12x_moe(16, 128, 256, 8, 2, torch.bfloat16)
