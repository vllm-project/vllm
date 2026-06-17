# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_b12x_compile_dynamic_moe,
    flashinfer_b12x_compile_static_moe,
    flashinfer_b12x_fused_moe,
    flashinfer_b12x_select_moe_backend,
    flashinfer_convert_sf_to_mma_layout,
    has_flashinfer_b12x_moe,
)


@dataclass(frozen=True)
class _FlashInferB12xCompileSpec:
    # Cache key is selected by backend:
    # static=("static", activation_precision, state_E, weight_E, m, k, n,
    #         num_topk, max_rows, mac, mma_tiler_mn, topk_ids_dtype,
    #         input_scales_are_reciprocal, fast_math, activation)
    # dynamic=("dynamic", activation_precision, E, k, n, num_topk, mac,
    #          mma_tiler_mn, topk_ids_dtype, input_scales_are_reciprocal,
    #          fast_math, activation, share_input_across_experts)
    # Token-size iteration is needed only for static because m/max_rows are in
    # the static cache key. Dynamic excludes m/max_rows and is emitted once if
    # any configured warmup size selects the dynamic backend.
    backend: str
    num_local_experts: int
    num_global_experts: int
    num_tokens: int
    hidden_dim: int
    intermediate_dim: int
    topk: int
    max_rows: int

    def compile(self) -> None:
        if self.backend == "static":
            flashinfer_b12x_compile_static_moe(
                self.num_local_experts,
                self.num_global_experts,
                self.num_tokens,
                self.hidden_dim,
                self.intermediate_dim,
                self.topk,
                self.max_rows,
                topk_ids_dtype=torch.int32,
                activation="silu",
                activation_precision="fp4",
            )
        elif self.backend == "dynamic":
            flashinfer_b12x_compile_dynamic_moe(
                self.num_local_experts,
                self.num_tokens,
                self.hidden_dim,
                self.intermediate_dim,
                self.topk,
                self.max_rows,
                topk_ids_dtype=torch.int32,
                activation="silu",
                activation_precision="fp4",
            )


class FlashInferB12xExperts(mk.FusedMoEExpertsModular):
    """FlashInfer CuteDSL fused MoE expert for SM12x (SM120/SM121,
    RTX Pro 6000 / DGX Spark).

    Uses ``b12x_fused_moe`` from FlashInfer PR #3080 which fuses token
    dispatch, two GEMMs, SwiGLU activation, and topk-weight reduction into a
    single kernel call.  Input quantization (BF16→FP4) is performed inside the
    kernel so BF16 hidden states are passed directly.

    Weight scale factors are converted to the MMA layout produced by
    ``convert_sf_to_mma_layout`` once during ``process_weights_after_loading``
    and cached as ``w1_sf_mma`` / ``w2_sf_mma``.

    Only NVFP4 (kNvfp4Static/kNvfp4Dynamic) quantization is supported.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.quant_dtype == "nvfp4", (
            "FlashInferB12xExperts only supports nvfp4 quantization."
        )
        self.out_dtype = moe_config.in_dtype
        self.num_local_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        # FC2 input scale tensor bound in process_weights_after_loading: the
        # calibrated (now-zeroed) a2_gscale for static-quant checkpoints, or
        # a synthesized uniform-1.0 tensor for W4A16 checkpoints that lack
        # one. Holding it on the instance keeps apply() alloc-free.
        self._fc2_input_scale: torch.Tensor | None = None

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
            provider="flashinfer_b12x_moe",
            compile_units=tuple(
                CuTeDSLCompileUnit(
                    name="flashinfer_b12x_moe",
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
    ) -> list[_FlashInferB12xCompileSpec]:
        del runner

        hidden_dim = self.moe_config.hidden_dim
        intermediate_dim = self.moe_config.intermediate_size_per_partition
        topk = self.moe_config.experts_per_token
        max_rows_per_token = max(1, topk)

        static_specs: list[_FlashInferB12xCompileSpec] = []
        dynamic_num_tokens: int | None = None
        for raw_num_tokens in token_sizes:
            num_tokens = max(1, int(raw_num_tokens))
            backend = flashinfer_b12x_select_moe_backend(
                num_tokens=num_tokens,
                num_topk=topk,
                activation_precision="fp4",
            )
            if backend == "dynamic":
                dynamic_num_tokens = dynamic_num_tokens or num_tokens
                continue

            static_specs.append(
                _FlashInferB12xCompileSpec(
                    backend=backend,
                    num_local_experts=self.num_local_experts,
                    num_global_experts=self.moe_config.num_experts,
                    num_tokens=num_tokens,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    topk=topk,
                    max_rows=max(1, num_tokens * max_rows_per_token),
                )
            )

        if dynamic_num_tokens is None:
            return static_specs

        return [
            *static_specs,
            _FlashInferB12xCompileSpec(
                backend="dynamic",
                num_local_experts=self.num_local_experts,
                num_global_experts=self.moe_config.num_experts,
                num_tokens=dynamic_num_tokens,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                topk=topk,
                max_rows=max(1, dynamic_num_tokens * max_rows_per_token),
            ),
        ]

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Normalise block scales to absorb the per-expert weight global scale
        # (w_gs).  vLLM's NVFP4 convention stores:
        #   block_scale = max_abs * w_gs / fp4_max,  g1_alphas = 1/w_gs
        # The SM12x kernel treats w1_alpha (= g1_alphas) as a per-expert weight
        # dequant multiplier separate from input_gs (activation scale).  We bake
        # w_gs into the block scales so that w1_alpha = 1.0 and the kernel sees
        # the simpler form:
        #   block_scale = max_abs / fp4_max,  w1_alpha = 1.0
        # The FP4-packed values and dequantised results are identical in both
        # representations.  We set scale_2 = 1.0 to signal that the bake-in is
        # already done.
        layer.w13_weight_scale.data = (
            layer.w13_weight_scale.float() * layer.w13_weight_scale_2.view(-1, 1, 1)
        ).to(layer.w13_weight_scale.dtype)
        layer.w13_weight_scale_2.data.fill_(1.0)

        layer.w2_weight_scale.data = (
            layer.w2_weight_scale.float() * layer.w2_weight_scale_2.view(-1, 1, 1)
        ).to(layer.w2_weight_scale.dtype)
        layer.w2_weight_scale_2.data.fill_(1.0)

        # The SM12x kernel uses dynamic per-block quantization for FC2 input
        # activations (the SwiGLU output before the down projection).  The
        # calibrated a2_gscale from the modelopt checkpoint (~tens to hundreds)
        # is intended for static-quantisation backends (TRTLLM/CUTLASS) and
        # causes every intermediate activation to saturate at max FP4 when
        # multiplied by values that large.  Force to 1.0 so the kernel uses
        # its own per-block dynamic scale.
        if self.a2_gscale is not None:
            self.a2_gscale.fill_(1.0)
            self._fc2_input_scale = self.a2_gscale
        else:
            # W4A16 NVFP4 checkpoints have no calibrated a2_gscale; b12x
            # performs dynamic per-block FC2-input quantization, so a uniform
            # 1.0 scale per expert is equivalent to the bake-in above for
            # static-quant checkpoints. Allocate once here so apply() stays
            # alloc-free.
            self._fc2_input_scale = torch.ones(
                self.num_local_experts,
                device=layer.w13_weight.device,
                dtype=torch.float32,
            )

        # Precompute MMA-layout views of the weight scale factors once here
        # rather than recomputing on every forward pass.
        assert self.w1_scale is not None
        num_experts_w1, m1, k1_sf = self.w1_scale.shape
        k1 = k1_sf * 16
        self.w1_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w1_scale.reshape(num_experts_w1 * m1, k1_sf),
            m=m1,
            k=k1,
            num_groups=num_experts_w1,
        )

        assert self.w2_scale is not None
        num_experts_w2, m2, k2_sf = self.w2_scale.shape
        k2 = k2_sf * 16
        self.w2_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w2_scale.reshape(num_experts_w2 * m2, k2_sf),
            m=m2,
            k=k2,
            num_groups=num_experts_w2,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(120)
            and has_flashinfer_b12x_moe()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # b12x performs in-kernel BF16->FP4 activation quant, so W4A16
        # NVFP4 checkpoints (activation_key=None, e.g. mixed-precision
        # compressed-tensors layouts) are runtime-compatible.
        return (weight_key, activation_key) in (
            (kNvfp4Static, kNvfp4Dynamic),
            (kNvfp4Static, None),
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # b12x_fused_moe applies topk weights internally.
        return TopKWeightAndReduceNoOP()

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
        # b12x_fused_moe manages its own internal workspace.
        workspace1 = (1,)
        workspace2 = (0,)
        output_shape = (M, K)
        return (workspace1, workspace2, output_shape)

    @property
    def expects_unquantized_inputs(self) -> bool:
        # b12x_fused_moe expects BF16 hidden states and performs its own FP4
        # quantization internally.  Returning True prevents the modular kernel
        # from pre-quantizing activations, which would produce an FP4-packed
        # tensor with size(-1)=k//2 and break the scale-factor conversion that
        # expects size(-1)=k.
        return True

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
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert self.w1_scale is not None and self.w2_scale is not None, (
            "w1_scale and w2_scale must not be None for FlashInferB12xExperts"
        )
        assert self.g1_alphas is not None and self.g2_alphas is not None, (
            "g1_alphas and g2_alphas must not be None for FlashInferB12xExperts"
        )
        assert self._fc2_input_scale is not None, (
            "_fc2_input_scale must be set by process_weights_after_loading"
        )

        top_k = topk_ids.shape[1]

        flashinfer_b12x_fused_moe(
            x=hidden_states,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
            w1_weight=w1,
            w1_weight_sf=self.w1_sf_mma,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=self._fc2_input_scale,
            w2_weight=w2,
            w2_weight_sf=self.w2_sf_mma,
            w2_alpha=self.g2_alphas,
            num_experts=global_num_experts,
            top_k=top_k,
            num_local_experts=self.num_local_experts,
            output_dtype=self.out_dtype,
            output=output,
        )
