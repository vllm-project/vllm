# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_cutedsl_grouped_gemm_nt_masked,
    scaled_fp4_grouped_quantize,
    silu_and_mul_scaled_nvfp4_experts_quantize,
)

logger = init_logger(__name__)


class FlashInferCuteDSLExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        assert quant_config.quant_dtype == "nvfp4", (
            "Only nvfp4 quantization are currently supported."
        )
        self.out_dtype = moe_config.in_dtype

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_device_capability_family(100)

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kNvfp4Static, kNvfp4Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: str) -> bool:
        return activation in ["silu"]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

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
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: str,
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
        K_dim = K * 2 if envs.VLLM_DEEPEPLL_NVFP4_DISPATCH else K
        output_shape = (local_num_experts, M, K_dim)
        workspace2 = (local_num_experts, M, N)
        workspace1 = output_shape
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
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,  # Not used
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
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

        input_global_scale = (
            None if envs.VLLM_DEEPEPLL_NVFP4_DISPATCH else self.a1_gscale
        )
        flashinfer_hidden_states = (
            (hidden_states, a1q_scale)
            if envs.VLLM_DEEPEPLL_NVFP4_DISPATCH
            else hidden_states
        )
        flashinfer_cutedsl_moe_masked(
            hidden_states=flashinfer_hidden_states,
            input_global_scale=input_global_scale,
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


def flashinfer_cutedsl_moe_masked(
    hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
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
        hidden_states: Either of the following case
            * torch.Tensor: [num_experts, m, k], bf16
            * tuple[torch.Tensor, torch.Tensor]: [num_experts, m, k // 2],
                  uint8, [num_experts, m, k // 16], float8_e4m3fn
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
    if isinstance(hidden_states, tuple):
        assert input_global_scale is None, (
            "input_global_scale is needed when input needs quant"
        )

        aq = hidden_states[0].view(torch.uint8)
        aq_sf = hidden_states[1].view(torch.float8_e4m3fn)
        # m, k_by_2, num_experts = aq.shape
        num_experts, m, k_by_2 = aq.shape
        k = k_by_2 * 2
        aq = aq.permute(1, 2, 0)
    else:
        num_experts, m, k = hidden_states.shape

        assert input_global_scale.dtype == torch.float32, (
            f"input_global_scale must be float32, got {input_global_scale.dtype}"
        )
        assert input_global_scale.shape == (num_experts,), (
            f"input_global_scale must be (l,), got {input_global_scale.shape}"
        )

        aq, aq_sf = scaled_fp4_grouped_quantize(
            hidden_states,
            masked_m,
            input_global_scale,
        )

    assert w1.shape[-2] == 2 * n, f"w1 last-2 dim must be 2*n, got {w1.shape}"
    assert w1.shape[-1] * 2 == k, (
        f"w1 last dim * 2 must equal k, got {w1.shape[-1]} vs k={k}"
    )
    assert w2.shape[-2:] == (
        k,
        n // 2,
    ), f"w2 shape mismatch, got {w2.shape[-2:]}, expected {(k, n // 2)}"

    assert w1_alpha.shape == (num_experts,), (
        f"w1_alpha must be (l,), got {w1_alpha.shape}"
    )
    assert a2_global_scale.shape == (num_experts,), (
        f"a2_global_scale must be (l,), got {a2_global_scale.shape}"
    )
    assert w2_alpha.shape == (num_experts,), (
        f"w2_alpha must be (l,), got {w2_alpha.shape}"
    )

    workspace = workspace.permute(1, 2, 0)  # requirement of kernel
    sf_vec_size = 16
    assert aq_sf.dtype == torch.float8_e4m3fn
    assert aq.dtype == torch.uint8
    ab_dtype = "float4_e2m1fn"
    sf_dtype = "float8_e4m3fn"

    if isinstance(hidden_states, tuple):
        c_dtype = "bfloat16"
    else:
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
    diq, diq_sf = silu_and_mul_scaled_nvfp4_experts_quantize(
        workspace.permute(2, 0, 1),
        masked_m,
        a2_global_scale,
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
