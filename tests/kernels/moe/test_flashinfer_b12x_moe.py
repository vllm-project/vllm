# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(120):
    pytest.skip(
        reason="FlashInfer CuteDSL SM12x MoE requires SM120 "
        "(RTX Pro 6000 / DGX Spark).",
        allow_module_level=True,
    )

from vllm.utils.flashinfer import (
    has_flashinfer_b12x_moe,
    has_flashinfer_b12x_moe_activation,
)

if not has_flashinfer_b12x_moe():
    pytest.skip(
        reason=(
            "b12x_fused_moe / convert_sf_to_mma_layout not available in "
            "installed FlashInfer."
        ),
        allow_module_level=True,
    )

# Import fp4_quantize after the skip guard; FlashInfer must be installed.
from flashinfer.fp4_quantization import fp4_quantize

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.utils import torch_experts
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import (
    FlashInferB12xExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    prepare_nvfp4_moe_layer_for_fi_or_cutlass,
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

# MiniMax-M3 SwiGLU-OAI parameters.
SWIGLU_ALPHA = 1.702
SWIGLU_BETA = 1.0
SWIGLU_LIMIT = 7.0


def _quantize_nvfp4_linear_sf(w_bf16: torch.Tensor):
    """Quantize per expert in checkpoint layout: linear block scales,
    global_scale=1.0."""
    e, rows, cols = w_bf16.shape
    gs = torch.ones(1, device="cuda", dtype=torch.float32)
    q, sf = fp4_quantize(
        w_bf16.reshape(e * rows, cols),
        global_scale=gs,
        sf_vec_size=16,
        is_sf_swizzled_layout=False,
    )
    sf = sf.view(torch.float8_e4m3fn)
    return q.view(e, rows, cols // 2), sf.view(e, rows, cols // 16)


def _torch_ref(
    a, w13_bf16, w2_bf16, topk_weights, topk_ids, activation, alpha=SWIGLU_ALPHA
):
    """BF16 reference on the original [gate, up] weights."""
    if activation == MoEActivation.SILU:
        return torch_experts(
            a, w13_bf16, w2_bf16, topk_weights, topk_ids, activation=activation
        )
    assert activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
    act = SiluAndMulWithClamp(SWIGLU_LIMIT, alpha, SWIGLU_BETA)
    m, k = a.shape
    topk = topk_ids.shape[1]
    a_rep = a.view(m, 1, k).repeat(1, topk, 1).reshape(-1, k)
    ids_flat = topk_ids.view(-1)
    out = torch.zeros(m * topk, k, dtype=a.dtype, device=a.device)
    for i in range(w13_bf16.shape[0]):
        mask = ids_flat == i
        if mask.sum():
            h = a_rep[mask] @ w13_bf16[i].t()
            out[mask] = act(h) @ w2_bf16[i].t()
    return (
        (out.view(m, topk, k).float() * topk_weights.view(m, topk, 1))
        .sum(dim=1)
        .to(a.dtype)
    )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [8, 16])
@pytest.mark.parametrize("topk", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "activation",
    [MoEActivation.SILU, MoEActivation.SWIGLUOAI_UNINTERLEAVE],
)
@torch.inference_mode()
def test_flashinfer_b12x_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    activation: MoEActivation,
    workspace_init,
):
    """Test FlashInferB12xExperts against a BF16 torch reference.

    Weights start in checkpoint layout (fused FC1 as [gate, up], linear
    block scales) and go through the production weight pipeline
    (``prepare_nvfp4_moe_layer_for_fi_or_cutlass`` +
    ``process_weights_after_loading``), so the test catches a broken
    [gate, up] -> [up, gate] reorder or unplumbed activation params (the
    garbled MiniMax-M3 swigluoai output).

    Scale convention: the SM12x kernel uses ``w1_alpha`` as *both* the
    activation-quantisation global scale and the weight dequantisation
    factor, so we quantise with ``global_scale = 1.0`` and ``w1_alpha =
    ones``; vLLM's bake-a-large-``w_gs``-into-block-scales convention is
    incompatible with this kernel.
    """
    is_swigluoai = activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
    if is_swigluoai and not has_flashinfer_b12x_moe_activation():
        pytest.skip("Installed FlashInfer b12x_fused_moe lacks swigluoai support.")

    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        # O(1)-magnitude outputs; smaller scalings make any comparison
        # metric pass regardless of kernel correctness.
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 2

        # BF16 reference weights in checkpoint [gate, up] order.
        w13_bf16 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 8
        w2_bf16 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 8

        w13_q, w13_sf = _quantize_nvfp4_linear_sf(w13_bf16)
        w2_q, w2_sf = _quantize_nvfp4_linear_sf(w2_bf16)

        # All per-expert alphas are 1.0 (global_scale = 1.0, no compensation).
        ones_e = torch.ones(e, device="cuda", dtype=torch.float32)

        layer = torch.nn.Module()
        layer.activation = activation
        (w13_q, w13_sf, _, _, w2_q, w2_sf, _, _) = (
            prepare_nvfp4_moe_layer_for_fi_or_cutlass(
                backend=NvFp4MoeBackend.FLASHINFER_B12X,
                layer=layer,
                w13=w13_q,
                w13_scale=w13_sf,
                w13_scale_2=ones_e.clone(),
                a13_scale=torch.ones(e, 2, device="cuda", dtype=torch.float32),
                w2=w2_q,
                w2_scale=w2_sf,
                w2_scale_2=ones_e.clone(),
                a2_scale=torch.ones(e, 1, device="cuda", dtype=torch.float32),
                is_act_and_mul=True,
            )
        )
        layer.w13_weight = w13_q
        layer.w13_weight_scale = w13_sf
        layer.w13_weight_scale_2 = ones_e.clone()
        layer.w2_weight = w2_q
        layer.w2_weight_scale = w2_sf
        layer.w2_weight_scale_2 = ones_e.clone()

        quant_config = nvfp4_moe_quant_config(
            g1_alphas=ones_e.clone(),
            g2_alphas=ones_e.clone(),
            a1_gscale=ones_e.clone(),
            a2_gscale=ones_e.clone(),
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )

        moe_config = dataclasses.replace(
            make_dummy_moe_config(
                num_experts=e,
                experts_per_token=topk,
                hidden_dim=k,
                intermediate_size=n,
                in_dtype=dtype,
            ),
            activation=activation,
            swiglu_alpha=SWIGLU_ALPHA if is_swigluoai else None,
            swiglu_beta=SWIGLU_BETA if is_swigluoai else None,
            swiglu_limit=SWIGLU_LIMIT if is_swigluoai else None,
        )

        experts = FlashInferB12xExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        experts.process_weights_after_loading(layer)

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
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
            activation=activation,
            apply_router_weight_on_input=False,
            expert_map=None,
        )

        torch_output = _torch_ref(
            a, w13_bf16, w2_bf16, topk_weights, topk_ids, activation
        )

        # NVFP4 error is proportional to the signal, so an elementwise
        # atol/rtol check cannot separate quantisation noise from a swapped
        # gate/up ordering or a wrong activation; relative Frobenius error can.
        diff = sm12x_output.float() - torch_output.float()
        rel_fro = (diff.norm() / torch_output.float().norm()).item()
        assert rel_fro < 0.45, f"rel_fro={rel_fro:.3f} (expected < 0.45)"

        if is_swigluoai:
            # An alpha regression shifts outputs too little for the threshold
            # above. Compare against an alpha=1.0 reference instead: the
            # quantisation noise is common to both, so whichever reference
            # matches the kernel's actual alpha wins.
            wrong_alpha_ref = _torch_ref(
                a, w13_bf16, w2_bf16, topk_weights, topk_ids, activation, alpha=1.0
            )
            wrong_diff = sm12x_output.float() - wrong_alpha_ref.float()
            wrong_rel_fro = (wrong_diff.norm() / wrong_alpha_ref.float().norm()).item()
            assert rel_fro < wrong_rel_fro, (
                f"output matches alpha=1.0 reference better than "
                f"alpha={SWIGLU_ALPHA} ({wrong_rel_fro:.3f} <= {rel_fro:.3f}); "
                f"swiglu_alpha is likely not reaching the kernel"
            )
