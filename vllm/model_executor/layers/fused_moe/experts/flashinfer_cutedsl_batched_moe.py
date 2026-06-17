# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
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
    flashinfer_cutedsl_compile_grouped_gemm_nt_masked,
    flashinfer_cutedsl_get_cutlass_dtype,
    flashinfer_cutedsl_grouped_gemm_nt_masked,
    has_flashinfer_cutedsl_grouped_gemm_nt_masked,
    scaled_fp4_grouped_quantize,
    silu_and_mul_scaled_nvfp4_experts_quantize,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class _FlashInferGroupedGemmCompileSpec:
    # Cache key is the MaskedBatchedMatmulCuteDSL specialization:
    # (m, n, k, l, a_major, b_major, c_major, ab_dtype, sf_dtype, c_dtype,
    #  alpha_dtype, sf_vec_size, mma_tiler_mn, cluster_shape_mn, sm_count,
    #  sm_version, num_ranks, enable_dst_signals, enable_barrier_flag,
    #  is_combine_fusion, is_swap_ab). Token-size iteration is intentional
    # because m is part of the compile key.
    m: int
    n: int
    k: int
    l: int
    c_dtype: str
    sm_count: int
    sm_version: str
    has_alpha: bool = True

    def compile(self) -> None:
        _compile_flashinfer_cutedsl_grouped_gemm_nt_masked(
            m=self.m,
            n=self.n,
            k=self.k,
            l=self.l,
            c_dtype=self.c_dtype,
            sm_count=self.sm_count,
            sm_version=self.sm_version,
            has_alpha=self.has_alpha,
        )


class FlashInferCuteDSLBatchedExperts(mk.FusedMoEExpertsModular):
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

    def get_cutedsl_warmup_plan(self, runner: object) -> object:
        from vllm.model_executor.warmup.cutedsl_warmup import (
            CuTeDSLCompileUnit,
            CuTeDSLWarmupPlan,
            get_cutedsl_warmup_token_sizes,
        )

        specs = tuple(
            self._iter_cutedsl_compile_specs(
                runner,
                get_cutedsl_warmup_token_sizes(runner),
            )
        )

        return CuTeDSLWarmupPlan(
            provider="flashinfer_cutedsl_batched_moe",
            compile_units=tuple(
                CuTeDSLCompileUnit(
                    name="flashinfer_cutedsl_batched_moe",
                    key=spec,
                    compile=spec.compile,
                )
                for spec in specs
            ),
        )

    def _iter_cutedsl_compile_specs(
        self,
        runner: object,
        token_sizes: Sequence[int],
    ) -> Sequence[_FlashInferGroupedGemmCompileSpec]:
        device = getattr(runner, "device", torch.device("cuda"))
        major, minor = torch.cuda.get_device_capability(device)
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        sm_version = f"sm_{major}{minor}"
        hidden_dim = self.moe_config.hidden_dim
        intermediate_dim = self.moe_config.intermediate_size_per_partition
        local_num_experts = self.moe_config.num_local_experts
        c_dtype = get_cute_dtype_from_torch(self.moe_config.in_dtype)

        specs: list[_FlashInferGroupedGemmCompileSpec] = []
        for num_tokens in token_sizes:
            m = max(1, int(num_tokens))
            specs.append(
                _FlashInferGroupedGemmCompileSpec(
                    m=m,
                    n=2 * intermediate_dim,
                    k=hidden_dim,
                    l=local_num_experts,
                    c_dtype=c_dtype,
                    sm_count=sm_count,
                    sm_version=sm_version,
                )
            )
            specs.append(
                _FlashInferGroupedGemmCompileSpec(
                    m=m,
                    n=hidden_dim,
                    k=intermediate_dim,
                    l=local_num_experts,
                    c_dtype=c_dtype,
                    sm_count=sm_count,
                    sm_version=sm_version,
                )
            )
        return specs

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)
        layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(100)
            and has_flashinfer_cutedsl_grouped_gemm_nt_masked()
        )

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
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

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
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
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

        # We use global_num_experts due to how moe_align_block_size handles
        # expert_maps.
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
        activation: MoEActivation,
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


def get_cute_dtype_from_torch(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unsupported cute dtype {dtype}")


def _compile_flashinfer_cutedsl_grouped_gemm_nt_masked(
    *,
    m: int,
    n: int,
    k: int,
    l: int,
    c_dtype: str,
    sm_count: int,
    sm_version: str,
    has_alpha: bool,
) -> None:
    flashinfer_cutedsl_compile_grouped_gemm_nt_masked(
        m=m,
        n=n,
        k=k,
        l=l,
        a_major="k",
        b_major="k",
        c_major="n",
        ab_dtype=flashinfer_cutedsl_get_cutlass_dtype("float4_e2m1fn"),
        sf_dtype=flashinfer_cutedsl_get_cutlass_dtype("float8_e4m3fn"),
        c_dtype=flashinfer_cutedsl_get_cutlass_dtype(c_dtype),
        alpha_dtype=(
            flashinfer_cutedsl_get_cutlass_dtype("float32") if has_alpha else None
        ),
        sf_vec_size=16,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        sm_count=sm_count,
        sm_version=sm_version,
        num_ranks=0,
        enable_dst_signals=False,
    )


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
