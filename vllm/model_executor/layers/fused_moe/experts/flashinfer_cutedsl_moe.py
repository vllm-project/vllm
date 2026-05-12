# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
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
    kNvfp4StaticGroupScale,
    kStaticTensorScale,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_convert_sf_to_mma_layout,
    flashinfer_cute_dsl_fused_moe_nvfp4,
    has_flashinfer_b12x_moe,
    has_flashinfer_cutedsl_moe_nvfp4,
)


class FlashInferCuteDSLExperts(mk.FusedMoEExpertsModular):
    """
    CuteDSL NvFP4 MoE experts using the FlashInfer functional API.

    Uses Standard activation format (non-batched). The kernel handles
    routing, expert computation, and reduction internally.
    Supports expert parallelism natively.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        assert quant_config.quant_dtype == "nvfp4", (
            "Only nvfp4 quantization is currently supported."
        )
        self.out_dtype = moe_config.in_dtype
        self.hidden_dim = moe_config.hidden_dim
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.topk = moe_config.experts_per_token
        self.local_num_experts = moe_config.num_local_experts
        self.global_num_experts = moe_config.num_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self.local_expert_offset = self.ep_rank * self.local_num_experts

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)
        layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(100)
            and has_flashinfer_cutedsl_moe_nvfp4()
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
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
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
        workspace1 = (0,)
        workspace2 = (0,)
        # K is packed (K//2 for uint8), so output uses hidden_dim.
        assert self.hidden_dim == K * 2
        output = (M, self.hidden_dim)
        return (workspace1, workspace2, output)

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
        assert self.quant_dtype == "nvfp4"
        assert a1q_scale is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        # a1q_scale is (M, K//16) float8_e4m3fn from fp4_quantize.
        # The functional API expects x_sf with trailing dim: (M, K//16, 1).
        x_sf = a1q_scale.unsqueeze(-1)

        from vllm.utils.flashinfer import _is_fi_autotuning, autotune

        with autotune(_is_fi_autotuning):
            flashinfer_cute_dsl_fused_moe_nvfp4(
                x=hidden_states,
                x_sf=x_sf,
                token_selected_experts=topk_ids.to(torch.int32),
                token_final_scales=topk_weights.float(),
                w1_weight=w1,
                w1_weight_sf=self.w1_scale,
                w1_alpha=self.g1_alphas,
                fc2_input_scale=self.a2_gscale,
                w2_weight=w2,
                w2_weight_sf=self.w2_scale,
                w2_alpha=self.g2_alphas,
                num_experts=self.global_num_experts,
                top_k=self.topk,
                num_local_experts=self.local_num_experts,
                local_expert_offset=self.local_expert_offset,
                moe_output=output,
            )


class FlashInferB12xExperts(mk.FusedMoEExpertsModular):
    """FlashInfer B12x fused MoE expert for SM12x (SM120/SM121).

    Targets RTX Pro 6000 / DGX Spark (Blackwell GeForce).

    Uses ``B12xMoEWrapper`` from FlashInfer with pre-allocated workspace
    and CUDA graph support.  The wrapper caches weight views (MMA layout
    conversion) internally and supports both SiLU (gated) and ReLU2
    (non-gated, e.g. Nemotron-H) activations via the ``activation``
    parameter.

    Input quantization (BF16 -> FP4) is performed inside the kernel so
    BF16 hidden states are passed directly.

    Only NVFP4 (kNvfp4Static/kNvfp4Dynamic) quantization is supported.
    """

    _ACTIVATION_MAP: dict[MoEActivation, str] = {
        MoEActivation.SILU: "silu",
        MoEActivation.RELU2_NO_MUL: "relu2",
    }

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

        # Shape params for B12xMoEWrapper construction.
        self.global_num_experts = moe_config.num_experts
        self.topk = moe_config.experts_per_token
        self.hidden_dim = moe_config.hidden_dim
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.max_num_tokens = moe_config.max_num_tokens
        self.local_expert_offset = self.ep_rank * self.num_local_experts

        activation = moe_config.activation
        if activation not in self._ACTIVATION_MAP:
            raise ValueError(
                f"FlashInferB12xExperts does not support "
                f"activation {activation!r}. "
                f"Supported: {list(self._ACTIVATION_MAP.keys())}"
            )
        self._activation_str = self._ACTIVATION_MAP[activation]

        # Hybrid CUTLASS-prefill / B12x-decode dispatch (flashinfer PR
        # #b12x_decode_cutlass_prefill).  When > 0, the wrapper routes
        # batches with num_tokens >= threshold through cutlass_fused_moe;
        # see register_cutlass_prefill_weights() below.
        self.cutlass_prefill_threshold = (
            envs.VLLM_FLASHINFER_B12X_CUTLASS_PREFILL_THRESHOLD
        )

        # Intermediate activation precision: "fp4" (W4A4, default) or
        # "bf16" (W4A16). Same FP4 weight bytes for both modes; the
        # kernel internally chooses whether to re-quantize the
        # post-SwiGLU/ReLU2 activations or keep them in BF16.
        self.activation_precision = envs.VLLM_FLASHINFER_B12X_ACTIVATION_PRECISION

        # Lazily created on first apply() call.
        self._wrapper: object | None = None
        self._cutlass_registered: bool = False

        # Saved CUTLASS-format scales (cloned before B12x's in-place rewrite).
        # Only populated when cutlass_prefill_threshold > 0 to avoid the
        # 2x scale-memory cost when hybrid dispatch is disabled.
        self._cutlass_w13_scale: torch.Tensor | None = None
        self._cutlass_w2_scale: torch.Tensor | None = None
        self._cutlass_a1_gscale: torch.Tensor | None = None
        self._cutlass_a2_gscale: torch.Tensor | None = None
        self._cutlass_g1_alphas: torch.Tensor | None = None
        self._cutlass_g2_alphas: torch.Tensor | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # When hybrid CUTLASS prefill is enabled, save copies of the
        # CUTLASS-format scales BEFORE the in-place B12x rewrite below
        # destroys them.  The FP4 weight bytes themselves are reusable —
        # prepare_nvfp4_moe_layer_for_fi_or_cutlass produces the same
        # [w3, w1] reorder + swizzled SF for both FLASHINFER_CUTLASS and
        # FLASHINFER_B12X — so we only need to clone the scales.
        #
        # The arithmetic for g_alphas: B12x intentionally leaves g1_alphas
        # = 1/w_gs (does NOT fold a_input_scale).  CUTLASS wants
        # 1/(a_gs * w_gs) = (1/w_gs) / a_gs, hence the division.
        #
        # The clones are registered as nn.Parameter on the layer below so
        # FusedMoE.get_expert_weights picks them up and EPLB rearranges
        # them in lockstep with the live b12x scales.
        if self.cutlass_prefill_threshold > 0:
            assert layer.w13_weight_scale.dtype == torch.float8_e4m3fn, (
                "Expected swizzled FP8 SF before B12x rewrite, got "
                f"{layer.w13_weight_scale.dtype}"
            )
            cutlass_w13_scale = layer.w13_weight_scale.clone()
            cutlass_w2_scale = layer.w2_weight_scale.clone()
            cutlass_a1_gscale = self.a1_gscale.clone()
            cutlass_a2_gscale = self.a2_gscale.clone()
            cutlass_g1_alphas = (
                self.g1_alphas.float() / self.a1_gscale
            ).contiguous()
            cutlass_g2_alphas = (
                self.g2_alphas.float() / self.a2_gscale
            ).contiguous()

            # Register on the layer so EPLB's get_expert_weights() picks them
            # up via named_parameters() and rearrange_expert_weights_inplace
            # permutes them in-place alongside the live b12x scales.  Storage
            # is shared, so the cached `quant_scales` list passed to
            # register_cutlass_prefill_weights() sees the permuted values on
            # subsequent CUTLASS prefill calls without re-registering.
            # All tensors have first-dim == num_local_experts, are contiguous,
            # and have names that don't collide with NON_EXPERT_WEIGHTS or
            # the runner.* prefixes in FusedMoE.get_expert_weights.
            layer.register_parameter(
                "w13_cutlass_weight_scale",
                torch.nn.Parameter(cutlass_w13_scale, requires_grad=False),
            )
            layer.register_parameter(
                "w2_cutlass_weight_scale",
                torch.nn.Parameter(cutlass_w2_scale, requires_grad=False),
            )
            layer.register_parameter(
                "w13_cutlass_a_gscale",
                torch.nn.Parameter(cutlass_a1_gscale, requires_grad=False),
            )
            layer.register_parameter(
                "w2_cutlass_a_gscale",
                torch.nn.Parameter(cutlass_a2_gscale, requires_grad=False),
            )
            layer.register_parameter(
                "w13_cutlass_g_alphas",
                torch.nn.Parameter(cutlass_g1_alphas, requires_grad=False),
            )
            layer.register_parameter(
                "w2_cutlass_g_alphas",
                torch.nn.Parameter(cutlass_g2_alphas, requires_grad=False),
            )

            # Hold references on the experts class so _ensure_wrapper can
            # build the quant_scales list without re-fetching from layer.
            # These alias the registered Parameters' storage, so EPLB
            # rearrangement of the parameters is observed here too.
            self._cutlass_w13_scale = layer.w13_cutlass_weight_scale.data
            self._cutlass_w2_scale = layer.w2_cutlass_weight_scale.data
            self._cutlass_a1_gscale = layer.w13_cutlass_a_gscale.data
            self._cutlass_a2_gscale = layer.w2_cutlass_a_gscale.data
            self._cutlass_g1_alphas = layer.w13_cutlass_g_alphas.data
            self._cutlass_g2_alphas = layer.w2_cutlass_g_alphas.data

        # The SM12x kernel uses w1_alpha as *both* the activation input_gs and
        # the weight dequant factor (they are conflated in launch_sm120_moe).
        # vLLM's NVFP4 convention stores block_scale = max_abs * w_gs / fp4_max
        # and g1_alphas = 1/w_gs.  We need block_scale = max_abs / fp4_max and
        # g1_alphas = 1.0 so both conflated roles equal 1.0.
        #
        # The FP4-packed values are identical in both conventions — only the
        # block scale representation changes.  Multiply float8 block scales by
        # 1/w_gs (= w13_weight_scale_2) to normalise, then set scale_2 = 1.0.
        #
        # We intentionally do NOT bake w13_input_scale into w13_weight_scale_2
        # here (unlike other backends) because that would make g1_alphas ≠ 1.0,
        # re-introducing the conflation bug for the activation-quantisation role.
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
        # its own per-block dynamic scale — matching the unit-test convention.
        if self.a2_gscale is not None:
            self.a2_gscale.fill_(1.0)

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
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Original W4A4 NVFP4 (modelopt format).
        if (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic):
            return True
        # W4A16 NVFP4 compressed-tensors `nvfp4-pack-quantized`: weights
        # are packed as uint8 (two FP4 elements per byte) with the same
        # NVFP4 group + per-tensor scales; no activation quant. The b12x
        # kernel reads the same FP4 byte payload, so this is supported.
        if (
            weight_key is not None
            and weight_key.dtype == torch.uint8
            and weight_key.scale == kNvfp4StaticGroupScale
            and weight_key.scale2 == kStaticTensorScale
            and weight_key.symmetric
            and activation_key is None
        ):
            return True
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.RELU2_NO_MUL)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # B12xMoEWrapper applies topk weights internally.
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
        # B12xMoEWrapper manages its own internal workspace.
        workspace1 = (1,)
        workspace2 = (0,)
        output_shape = (M, K)
        return (workspace1, workspace2, output_shape)

    @property
    def expects_unquantized_inputs(self) -> bool:
        # B12xMoEWrapper expects BF16 hidden states and performs its own FP4
        # quantization internally.  Returning True prevents the modular kernel
        # from pre-quantizing activations.
        return True

    def _ensure_wrapper(self, w1: torch.Tensor, w2: torch.Tensor) -> None:
        """Lazily create B12xMoEWrapper on first use.

        Also registers CUTLASS-format prefill weights when hybrid dispatch
        is enabled; the FP4 byte tensors are shared with the b12x decode
        path (only scales differ — saved in process_weights_after_loading).
        """
        if self._wrapper is None:
            from flashinfer.fused_moe import B12xMoEWrapper
            import inspect as _inspect

            # activation_precision: "fp4" = W4A4, "bf16" = W4A16.
            # Same NVFP4 weight bytes for both modes; the kernel
            # picks the intermediate-activation handling internally.
            b12x_kwargs = dict(
                num_experts=self.global_num_experts,
                top_k=self.topk,
                hidden_size=self.hidden_dim,
                intermediate_size=self.intermediate_size_per_partition,
                use_cuda_graph=True,
                max_num_tokens=self.max_num_tokens,
                num_local_experts=self.num_local_experts,
                activation=self._activation_str,
                activation_precision=self.activation_precision,
            )
            # cutlass_prefill_threshold is gated on a FlashInfer build with
            # PR #b12x_decode_cutlass_prefill. Skip the kwarg silently if the
            # installed FlashInfer lacks it (and threshold is 0); error
            # cleanly if the user is asking for the hybrid path.
            if "cutlass_prefill_threshold" in _inspect.signature(
                B12xMoEWrapper.__init__
            ).parameters:
                b12x_kwargs["cutlass_prefill_threshold"] = (
                    self.cutlass_prefill_threshold
                )
            elif self.cutlass_prefill_threshold > 0:
                raise RuntimeError(
                    "VLLM_FLASHINFER_B12X_CUTLASS_PREFILL_THRESHOLD > 0 "
                    "requires a FlashInfer build with PR "
                    "#b12x_decode_cutlass_prefill (e.g. "
                    "askliar/flashinfer:b12x_micro_kernel_merged); "
                    "current FlashInfer does not expose the kwarg."
                )
            self._wrapper = B12xMoEWrapper(**b12x_kwargs)

        if (
            self.cutlass_prefill_threshold > 0
            and not self._cutlass_registered
        ):
            assert self._cutlass_w13_scale is not None, (
                "cutlass_prefill_threshold > 0 but CUTLASS scales were "
                "not saved in process_weights_after_loading"
            )
            # quant_scales order matches FlashInferExperts (NVFP4 mode):
            # [a1_gs, w1_blockscale_int32, 1/(a1_gs*w1_gs),
            #  a2_gs, w2_blockscale_int32, 1/(a2_gs*w2_gs)].
            # register_cutlass_prefill_weights does .contiguous().view(long)
            # on w*_q internally — pass uint8 directly.
            self._wrapper.register_cutlass_prefill_weights(
                w1_q=w1,
                w2_q=w2,
                quant_scales=[
                    self._cutlass_a1_gscale,
                    self._cutlass_w13_scale.view(torch.int32),
                    self._cutlass_g1_alphas,
                    self._cutlass_a2_gscale,
                    self._cutlass_w2_scale.view(torch.int32),
                    self._cutlass_g2_alphas,
                ],
            )
            self._cutlass_registered = True

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
        assert self.a2_gscale is not None, (
            "a2_gscale must not be None for FlashInferB12xExperts"
        )

        self._ensure_wrapper(w1, w2)

        # Convert swizzled 3D scale factors [E, M, K_sf] to 6D MMA layout
        # expected by the SM12x kernel's _get_weight_views().
        sf_vec_size = 16
        E_w1, M_w1, K_sf_w1 = self.w1_scale.shape
        w1_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w1_scale.reshape(E_w1 * M_w1, K_sf_w1),
            m=M_w1,
            k=K_sf_w1 * sf_vec_size,
            num_groups=E_w1,
            sf_vec_size=sf_vec_size,
        )
        E_w2, M_w2, K_sf_w2 = self.w2_scale.shape
        w2_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w2_scale.reshape(E_w2 * M_w2, K_sf_w2),
            m=M_w2,
            k=K_sf_w2 * sf_vec_size,
            num_groups=E_w2,
            sf_vec_size=sf_vec_size,
        )

        result = self._wrapper.run(
            x=hidden_states,
            w1_weight=w1,
            w1_weight_sf=w1_sf_mma,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=self.a2_gscale,
            w2_weight=w2,
            w2_weight_sf=w2_sf_mma,
            w2_alpha=self.g2_alphas,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
        )
        output.copy_(result)
