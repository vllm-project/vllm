# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.utils.flashinfer import (
    flashinfer_cutedsl_grouped_gemm_nt_masked,
    has_flashinfer_cutedsl_grouped_gemm_nt_masked,
    nvfp4_batched_quantize,
    silu_and_mul,
)

logger = init_logger(__name__)


def is_valid_flashinfer_cutedsl_fused_moe(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> bool:
    """
    Check if the given problem size is supported by the FlashInfer CuteDSL MoE
    kernel.
    """
    if not has_flashinfer_cutedsl_grouped_gemm_nt_masked():
        logger.debug_once(
            "FlashInferCuteDSLExperts disabled: "
            "flashinfer_cutedsl_fused_moe not available."
        )
        return False
    # Data type checks
    if (
        w1.dtype != torch.uint8
        or w2.dtype != torch.uint8
        or hidden_states.dtype not in [torch.float32, torch.float16, torch.bfloat16]
    ):
        logger.debug_once(
            "FlashInferCuteDSLExperts disabled: w1/w2 must be torch.uint8 "
            f"(got w1={w1.dtype}, w2={w2.dtype}), hidden_states must be "
            f"float32, float16, or bfloat16 (got {hidden_states.dtype})."
        )
        return False
    return True


class FlashInferCuteDSLExperts(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(
        self,
        out_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(quant_config)
        assert quant_config.quant_dtype == "nvfp4", (
            "Only nvfp4 quantization are currently supported."
        )
        self.out_dtype = out_dtype

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.BatchedExperts,
            mk.FusedMoEActivationFormat.BatchedExperts,
        )

    def supports_expert_map(self) -> bool:
        return False

    def supports_chunking(self) -> bool:
        # This refers to TP chunking; DP chunking is handled separately.
        # TODO(shuw@nvidia.com): Set to False to be consistent with
        # batched_deep_gemm_moe
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # We use global_num_experts due to how moe_align_block_size handles
        # expert_maps.
        """
        Compute the shapes for the temporary and final outputs of the two gemms
        and activation in the fused expert function.  Since the gemms are
        independent, the workspace for the first gemm can be shared with the
        workspace for the last gemm.

        Returns a tuple of:
        - workspace13 shape tuple: must be large enough to hold the
          result of either expert gemm.
        - workspace2 shape tuple: must be large enough to hold the
          result of the activation function.
        - output shape tuple: must be exact size of the final gemm output.
        - Workspace type: The dtype to use for the workspace tensors.
        - Note: in order for activation chunking to work, the first dimension
          of each tuple must be the number of tokens.
        """
        # assert a.dim() == 2
        # assert aq.dim() == 3
        # output_shape = aq.shape
        # workspace_dtype = a.dtype
        # E = aq.size(0)
        # workspace2 = (E, M, N)
        # workspace1 = output_shape
        output_shape = (local_num_experts, M, K)
        workspace2 = (local_num_experts, M, N)
        workspace1 = output_shape
        # The workspace is determined by `aq`, since it comes after any
        # potential communication op and is involved in the expert computation.
        return (workspace1, workspace2, output_shape)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],  # Not used
        workspace13: Optional[torch.Tensor],
        workspace2: Optional[torch.Tensor],
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: Optional[bool],
    ):
        assert self.quant_dtype == "nvfp4", (
            "Only nvfp4 quantization are currently supported."
        )
        # Ensure w1_scale and w2_scale are not None before calling view
        assert self.w1_scale is not None and self.w2_scale is not None, (
            "w1_scale and w2_scale must not be None for FlashInferExperts"
        )
        assert expert_tokens_meta is not None
        expert_num_tokens = expert_tokens_meta.expert_num_tokens
        assert hidden_states.ndim == 3
        assert self.w1_scale.ndim == 3
        assert self.w2_scale.ndim == 3

        flashinfer_cutedsl_moe_masked(
            hidden_states=hidden_states,
            input_global_scale=self.a1_gscale,
            w1=w1,
            w1_blockscale=self.w1_scale,
            w1_alpha=self.g1_alphas,
            w2=w2,
            a2_global_scale=self.a2_gscale,
            w2_blockscale=self.w2_scale,
            w2_alpha=self.g2_alphas,
            masked_m=expert_num_tokens,
            workspace=workspace2,
            out=output,
        )


def get_cute_dtype(input: torch.Tensor) -> str:
    if input.dtype == torch.bfloat16:
        return "bfloat16"
    elif input.dtype == torch.float16:
        return "float16"
    elif input.dtype == torch.float32:
        return "float32"
    else:
        raise ValueError(f"Unsupported cute dtype {input.dtype}")


def scaled_fp4_grouped_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Wrapper around nvfp4_batched_quantize

    Args:
        input_tensor (Tensor):
            - Shape (l, m, k)
        input_global_scale (Tensor): Shape (l,)
        mask (Tensor): Mask tensor, broadcastable

    Returns:
        output (Tensor): Quantized tensor, logical shape (m, k//2, l)
        output_scales (Tensor): Blockscale tensor, logical shape
        (32, 4, rm, 4, rk, l)
    """
    num_experts, m, k = input_tensor.shape

    sf_vec_size = 16
    assert k % sf_vec_size == 0, f"k must be multiple of 16, but got {k}."

    scale_k = k // sf_vec_size
    padded_k = (scale_k + (4 - 1)) // 4 * 4
    padded_m = (m + (128 - 1)) // 128 * 128

    aq, aq_sf = nvfp4_batched_quantize(
        input_tensor,
        input_global_scale,
        mask=mask,
    )

    # --- re-layout quantized tensor ---
    # physical (l, m, k//2) -> logical (m, k//2, l)
    output = aq.permute(1, 2, 0)

    # --- re-layout blockscales ---
    # physical (l, rm, rk, 32, 4, 4) -> logical (32, 4, rm, 4, rk, l)
    output_scales = aq_sf.view(torch.float8_e4m3fn).view(
        num_experts, padded_m // 128, padded_k // 4, 32, 4, 4
    )
    output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)

    return output, output_scales


def flashinfer_cutedsl_moe_masked(
    hidden_states: torch.Tensor,
    input_global_scale: torch.Tensor,
    w1: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alpha,
    w2: torch.Tensor,
    a2_global_scale: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alpha,
    masked_m: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
):
    """
    Perform masked Mixture-of-Experts computation with FlashInfer's CuteDSL
    kernels.

    Args:
        hidden_states (torch.Tensor): [num_experts, m, k], bf16
        input_global_scale (torch.Tensor): (l,)
        w1 (torch.Tensor): fp4 weights, [l, 2 * n, k // 2], uint8
        w1_blockscale (torch.Tensor): blockscale factors, e4m3,
        w1_alpha (torch.Tensor): (l,)
        w2 (torch.Tensor): fp4 weights, [l, k, n // 2], uint8
        a2_global_scale (torch.Tensor): (l,)
        w2_blockscale (torch.Tensor): blockscale factors, e4m3,
        w2_alpha (torch.Tensor): (l,)
        masked_m (torch.Tensor): Masked dimension indices
        workspace (torch.Tensor): For gateup_output

    Notes:
        - Assumes max(masked_m) <= m.
    """

    # === Assertions on dtypes ===
    assert input_global_scale.dtype == torch.float32, (
        f"input_global_scale must be float32, got {input_global_scale.dtype}"
    )
    assert w1.dtype == torch.uint8, f"w1 must be uint8, got {w1.dtype}"
    assert w1_blockscale.dtype == torch.float8_e4m3fn, (
        f"w1_blockscale must be float8_e4m3fn, got {w1_blockscale.dtype}"
    )
    assert w1_alpha.dtype == torch.float32, (
        f"w1_alpha must be float32, got {w1_alpha.dtype}"
    )
    assert w2.dtype == torch.uint8, f"w2 must be uint8, got {w2.dtype}"
    assert a2_global_scale.dtype == torch.float32, (
        f"a2_global_scale must be float32, got {a2_global_scale.dtype}"
    )
    assert w2_blockscale.dtype == torch.float8_e4m3fn, (
        f"w2_blockscale must be float8_e4m3fn, got {w2_blockscale.dtype}"
    )
    assert w2_alpha.dtype == torch.float32, (
        f"w2_alpha must be float32, got {w2_alpha.dtype}"
    )

    # === Assertions on shapes ===
    n = w2.shape[-1] * 2  # intermediate dimension
    num_experts, m, k = hidden_states.shape

    assert w1.shape[-2] == 2 * n, f"w1 last-2 dim must be 2*n, got {w1.shape}"
    assert w1.shape[-1] * 2 == k, (
        f"w1 last dim * 2 must equal k, got {w1.shape[-1]} vs k={k}"
    )
    assert w2.shape[-2:] == (
        k,
        n // 2,
    ), f"w2 shape mismatch, got {w2.shape[-2:]}, expected {(k, n // 2)}"

    assert input_global_scale.shape == (num_experts,), (
        f"input_global_scale must be (l,), got {input_global_scale.shape}"
    )
    assert w1_alpha.shape == (num_experts,), (
        f"w1_alpha must be (l,), got {w1_alpha.shape}"
    )
    assert a2_global_scale.shape == (num_experts,), (
        f"a2_global_scale must be (l,), got {a2_global_scale.shape}"
    )
    assert w2_alpha.shape == (num_experts,), (
        f"w2_alpha must be (l,), got {w2_alpha.shape}"
    )

    aq, aq_sf = scaled_fp4_grouped_quant(
        hidden_states,
        input_global_scale,
        masked_m,
    )

    workspace = workspace.permute(1, 2, 0)  # requirement of kernel
    sf_vec_size = 16
    assert aq_sf.dtype == torch.float8_e4m3fn
    assert aq.dtype == torch.uint8
    ab_dtype = "float4_e2m1fn"
    sf_dtype = "float8_e4m3fn"

    c_dtype = get_cute_dtype(hidden_states)

    # Gemm1
    flashinfer_cutedsl_grouped_gemm_nt_masked(
        (aq, aq_sf),
        (w1.permute(1, 2, 0), w1_blockscale),
        workspace,
        masked_m,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        alpha=w1_alpha.view(1, 1, num_experts),
        alpha_dtype=get_cute_dtype(w1_alpha),
    )  # in logical [m, n, l]

    # SILU and quantization

    diq, diq_sf = scaled_fp4_grouped_quant(
        silu_and_mul(workspace.permute(2, 0, 1)),
        a2_global_scale,
        masked_m,
    )

    # Gemm2
    out = out.permute(1, 2, 0)  # requirement of kernel
    flashinfer_cutedsl_grouped_gemm_nt_masked(
        (diq, diq_sf),
        (w2.permute(1, 2, 0), w2_blockscale),
        out,
        masked_m,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        alpha=w2_alpha.view(1, 1, num_experts),
        alpha_dtype=get_cute_dtype(w2_alpha),
    )  # in logical [m, k, l]
    out = out.permute(2, 0, 1)
