# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(120):
    pytest.skip(
        reason="FlashInfer B12x MoE requires SM120 (RTX Pro 6000 / DGX Spark).",
        allow_module_level=True,
    )

from vllm.utils.flashinfer import has_flashinfer_b12x_moe

if not has_flashinfer_b12x_moe():
    pytest.skip(
        reason=(
            "FlashInfer B12xMoEWrapper not available in installed "
            "FlashInfer (needs PR #3080)."
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

    The B12x kernel expects weights in [up (w3), gate (w1)] order while the
    BF16 reference uses [gate (w1), up (w3)].  This replicates the reordering
    done at model-load time by ``prepare_nvfp4_moe_layer_for_fi_or_cutlass``.
    """
    n = w.shape[1] // 2
    return (
        torch.cat([w[:, n:, :], w[:, :n, :]], dim=1),
        torch.cat([w_s[:, n:, :], w_s[:, :n, :]], dim=1),
    )


def _process_b12x_weights(
    experts: FlashInferB12xExperts,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_scale_2: torch.Tensor,
    w2_scale_2: torch.Tensor,
    w1_input_scale: torch.Tensor | None = None,
    w2_input_scale: torch.Tensor | None = None,
) -> None:
    if w1_input_scale is None:
        w1_input_scale = torch.ones((), device=w1_scale.device, dtype=torch.float32)
    if w2_input_scale is None:
        w2_input_scale = torch.ones((), device=w2_scale.device, dtype=torch.float32)
    layer = SimpleNamespace(
        w13_weight=torch.empty(0, device=w1_scale.device),
        w13_weight_scale=w1_scale,
        w13_weight_scale_2=w1_scale_2,
        w13_input_scale=w1_input_scale,
        w2_weight_scale=w2_scale,
        w2_weight_scale_2=w2_scale_2,
        w2_input_scale=w2_input_scale,
    )
    experts.process_weights_after_loading(layer)


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
            intermediate_size=n,
            in_dtype=dtype,
        )

        experts = FlashInferB12xExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        _process_b12x_weights(
            experts,
            w1_blockscale,
            w2_blockscale,
            ones_e,
            ones_e,
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


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [8, 16])
@pytest.mark.parametrize("topk", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_flashinfer_b12x_moe_relu2(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    workspace_init,
):
    """Test FlashInferB12xExperts with ReLU2 (non-gated) activation.

    ReLU2 is used by Nemotron-H style models.  Unlike the gated SiLU
    path, w1 has shape [E, N, K] (not [E, 2N, K]) and the activation
    is relu(x)^2 without a gate/up split.
    """
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        # Non-gated: w1 shape is (e, n, k), not (e, 2n, k).
        w1_bf16 = torch.randn((e, n, k), device="cuda", dtype=dtype) / 15
        w2_bf16 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 15

        gs = torch.ones(1, device="cuda", dtype=torch.float32)
        sf_vec_size = 16

        # W1: no gate/up reordering for non-gated.
        w1_flat = w1_bf16.reshape(e * n, k)
        w1_q_flat, w1_sf_flat = fp4_quantize(
            w1_flat,
            global_scale=gs,
            sf_vec_size=sf_vec_size,
            is_sf_swizzled_layout=True,
        )
        w1_q = w1_q_flat.view(e, n, k // 2)
        w1_blockscale = w1_sf_flat.view(e, n, w1_sf_flat.shape[1])

        w2_flat = w2_bf16.reshape(e * k, n)
        w2_q_flat, w2_sf_flat = fp4_quantize(
            w2_flat,
            global_scale=gs,
            sf_vec_size=sf_vec_size,
            is_sf_swizzled_layout=True,
        )
        w2_q = w2_q_flat.view(e, k, n // 2)
        w2_blockscale = w2_sf_flat.view(e, k, w2_sf_flat.shape[1])

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
            intermediate_size=n,
            in_dtype=dtype,
            activation=MoEActivation.RELU2_NO_MUL,
        )

        experts = FlashInferB12xExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        _process_b12x_weights(
            experts,
            w1_blockscale,
            w2_blockscale,
            ones_e,
            ones_e,
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

        b12x_output = kernel.apply(
            hidden_states=a,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
            activation=MoEActivation.RELU2_NO_MUL,
            apply_router_weight_on_input=False,
            expert_map=None,
        )

        torch_output = torch_moe(
            a,
            w1_bf16,
            w2_bf16,
            score,
            topk,
            activation=MoEActivation.RELU2_NO_MUL,
        )

        torch.testing.assert_close(
            b12x_output,
            torch_output,
            atol=2e-1,
            rtol=2e-1,
        )


def _assert_kernel_parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    same_backend: bool,
) -> None:
    actual_float = actual.float()
    expected_float = expected.float()
    reference_rms = expected_float.square().mean().sqrt().clamp_min(1e-6)
    nrmse = (
        (actual_float - expected_float).square().mean().sqrt() / reference_rms
    ).item()
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(), expected_float.flatten(), dim=0
    ).item()
    if same_backend:
        assert nrmse < 0.02
        assert cosine > 0.9998
    else:
        assert nrmse < 0.05
        assert cosine > 0.999


@pytest.mark.parametrize("m", [31, 32, 33])
@torch.inference_mode()
def test_flashinfer_b12x_cutlass_hybrid_scale_contract(
    m: int,
    monkeypatch: pytest.MonkeyPatch,
    workspace_init,
):
    """Hybrid dispatch preserves checkpoint numerics across its threshold.

    Use non-unit activation and weight-global scales so this fails if vLLM
    folds an activation scale into the B12x block SF, replaces a B12x alpha
    with one, or builds CUTLASS's dequant multipliers from mutated tensors.
    """
    from flashinfer.fused_moe import B12xMoEWrapper

    if (
        "cutlass_prefill_threshold"
        not in inspect.signature(B12xMoEWrapper.__init__).parameters
    ):
        pytest.skip("FlashInfer B12x/CUTLASS hybrid dispatch is unavailable")

    set_random_seed(11)
    e, n, k, topk = 8, 128, 256, 2
    device = torch.device("cuda")
    dtype = torch.bfloat16

    hidden_states = torch.randn((m, k), device=device, dtype=dtype) / 10
    w1_bf16 = torch.randn((e, n, k), device=device, dtype=dtype) / 15
    w2_bf16 = torch.randn((e, k, n), device=device, dtype=dtype) / 15
    unit_global_scale = torch.ones((), device=device, dtype=torch.float32)
    w1_q_flat, logical_w1_sf_flat = fp4_quantize(
        w1_bf16.reshape(e * n, k),
        global_scale=unit_global_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )
    w2_q_flat, logical_w2_sf_flat = fp4_quantize(
        w2_bf16.reshape(e * k, n),
        global_scale=unit_global_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )
    w1_q = w1_q_flat.view(e, n, k // 2)
    w2_q = w2_q_flat.view(e, k, n // 2)
    logical_w1_sf = logical_w1_sf_flat.view(torch.float8_e4m3fn).view(e, n, -1)
    logical_w2_sf = logical_w2_sf_flat.view(torch.float8_e4m3fn).view(e, k, -1)

    # ModelOpt checkpoint convention: normalized SF plus a separate
    # per-expert weight multiplier and a small global activation scale.
    w1_weight_alpha = torch.tensor(
        [0.5, 0.25], device=device, dtype=torch.float32
    ).repeat(e // 2)
    w2_weight_alpha = torch.tensor(
        [0.25, 0.5], device=device, dtype=torch.float32
    ).repeat(e // 2)
    w1_checkpoint_sf = (logical_w1_sf.float() / w1_weight_alpha.view(e, 1, 1)).to(
        torch.float8_e4m3fn
    )
    w2_checkpoint_sf = (logical_w2_sf.float() / w2_weight_alpha.view(e, 1, 1)).to(
        torch.float8_e4m3fn
    )
    w1_input_scale = torch.tensor(1.0 / 16.0, device=device)
    w2_input_scale = torch.tensor(1.0 / 8.0, device=device)

    moe_config = make_dummy_moe_config(
        num_experts=e,
        experts_per_token=topk,
        hidden_dim=k,
        intermediate_size=n,
        in_dtype=dtype,
        activation=MoEActivation.RELU2_NO_MUL,
    )

    def build_experts(threshold: int) -> FlashInferB12xExperts:
        monkeypatch.setenv(
            "VLLM_FLASHINFER_B12X_CUTLASS_PREFILL_THRESHOLD", str(threshold)
        )
        w1_scale = w1_checkpoint_sf.clone()
        w2_scale = w2_checkpoint_sf.clone()
        w1_scale_2 = w1_weight_alpha.clone()
        w2_scale_2 = w2_weight_alpha.clone()
        quant_config = nvfp4_moe_quant_config(
            g1_alphas=w1_scale_2,
            g2_alphas=w2_scale_2,
            a1_gscale=1.0 / w1_input_scale,
            a2_gscale=1.0 / w2_input_scale,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )
        experts = FlashInferB12xExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        _process_b12x_weights(
            experts,
            w1_scale,
            w2_scale,
            w1_scale_2,
            w2_scale_2,
            w1_input_scale,
            w2_input_scale,
        )
        return experts

    pure_b12x = build_experts(0)
    hybrid = build_experts(32)

    # The live B12x tensors recover the unnormalized logical SF and use the
    # checkpoint's small activation scales as kernel alphas.
    torch.testing.assert_close(pure_b12x.w1_scale, logical_w1_sf, rtol=0, atol=0)
    torch.testing.assert_close(pure_b12x.w2_scale, logical_w2_sf, rtol=0, atol=0)
    torch.testing.assert_close(
        pure_b12x.g1_alphas,
        w1_input_scale.expand_as(w1_weight_alpha),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        pure_b12x.g2_alphas,
        w2_input_scale.expand_as(w2_weight_alpha),
        rtol=0,
        atol=0,
    )
    assert pure_b12x._fc2_input_scale is w2_input_scale

    # CUTLASS keeps the original normalized SF and combines the independent
    # weight and activation multipliers in quant_scales[2]/[5].
    assert hybrid._cutlass_quant_scales is not None
    cutlass_scales = hybrid._cutlass_quant_scales
    torch.testing.assert_close(cutlass_scales[0], 1.0 / w1_input_scale)
    torch.testing.assert_close(cutlass_scales[3], 1.0 / w2_input_scale)
    assert torch.equal(
        cutlass_scales[1].view(torch.uint8).flatten(),
        w1_checkpoint_sf.view(torch.uint8).flatten(),
    )
    assert torch.equal(
        cutlass_scales[4].view(torch.uint8).flatten(),
        w2_checkpoint_sf.view(torch.uint8).flatten(),
    )
    torch.testing.assert_close(cutlass_scales[2], w1_weight_alpha * w1_input_scale)
    torch.testing.assert_close(cutlass_scales[5], w2_weight_alpha * w2_input_scale)

    topk_ids = torch.randint(0, e, (m, topk), device=device, dtype=torch.int32)
    topk_weights = torch.rand((m, topk), device=device, dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    def run(experts: FlashInferB12xExperts) -> torch.Tensor:
        output = torch.empty((m, k), device=device, dtype=dtype)
        experts.apply(
            output=output,
            hidden_states=hidden_states,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.RELU2_NO_MUL,
            global_num_experts=e,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
        return output.clone()

    pure_output = run(pure_b12x)
    hybrid_output = run(hybrid)
    assert hybrid._wrapper is not None
    assert hybrid._wrapper._should_route_to_cutlass(m) is (m >= 32)
    _assert_kernel_parity(
        hybrid_output,
        pure_output,
        same_backend=m < 32,
    )


if __name__ == "__main__":
    test_flashinfer_b12x_moe(16, 128, 256, 8, 2, torch.bfloat16)
