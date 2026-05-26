# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from typing import Any

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
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_convert_sf_to_mma_layout,
    has_flashinfer_b12x_moe,
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
        # FC2 input scale tensor bound in process_weights_after_loading: the
        # calibrated (now-zeroed) a2_gscale for static-quant checkpoints, or
        # a synthesized uniform-1.0 tensor for W4A16 checkpoints that lack
        # one. Holding it on the instance keeps apply() alloc-free.
        self._fc2_input_scale: torch.Tensor | None = None

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

        # Hybrid CUTLASS-prefill / B12x-decode dispatch. When > 0, the
        # wrapper routes batches with num_tokens >= threshold through
        # cutlass_fused_moe; see register_cutlass_prefill_weights() in
        # _ensure_wrapper. Requires a FlashInfer build exposing the
        # `cutlass_prefill_threshold` kwarg on B12xMoEWrapper.
        self.cutlass_prefill_threshold = (
            envs.VLLM_FLASHINFER_B12X_CUTLASS_PREFILL_THRESHOLD
        )

        # Lazily created on first apply() call.
        self._wrapper: Any | None = None
        self._cutlass_registered: bool = False
        self.w1_sf_mma: torch.Tensor | None = None
        self.w2_sf_mma: torch.Tensor | None = None

        # CUTLASS-format scales saved before the in-place B12x rewrite in
        # process_weights_after_loading. Only populated when
        # cutlass_prefill_threshold > 0.
        self._cutlass_w13_scale: torch.Tensor | None = None
        self._cutlass_w2_scale: torch.Tensor | None = None
        self._cutlass_a1_gscale: torch.Tensor | None = None
        self._cutlass_a2_gscale: torch.Tensor | None = None
        self._cutlass_g1_alphas: torch.Tensor | None = None
        self._cutlass_g2_alphas: torch.Tensor | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # When hybrid CUTLASS prefill is enabled, save copies of the
        # CUTLASS-format scales BEFORE the in-place B12x rewrite below
        # destroys them. The FP4 weight bytes themselves are reusable —
        # prepare_nvfp4_moe_layer_for_fi_or_cutlass produces the same
        # [w3, w1] reorder + swizzled SF for both FLASHINFER_CUTLASS and
        # FLASHINFER_B12X — so we only need to clone the scales.
        #
        # g_alphas: B12x leaves g1_alphas = 1/w_gs (does NOT fold
        # a_input_scale). CUTLASS wants 1/(a_gs * w_gs) = (1/w_gs) / a_gs,
        # hence the division.
        #
        # The clones are registered as nn.Parameter on the layer so
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
        return True

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
        return activation in (MoEActivation.SILU, MoEActivation.RELU2_NO_MUL)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # B12xMoEWrapper does not yet support expert parallelism: its local
        # expert count must equal the global expert count.
        return not moe_parallel_config.use_ep

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

            b12x_kwargs = dict(
                num_experts=self.global_num_experts,
                top_k=self.topk,
                hidden_size=self.hidden_dim,
                intermediate_size=self.intermediate_size_per_partition,
                use_cuda_graph=True,
                max_num_tokens=self.max_num_tokens,
                num_local_experts=self.num_local_experts,
                activation=self._activation_str,
            )
            # cutlass_prefill_threshold is gated on a FlashInfer build that
            # exposes the kwarg. Skip silently if absent and threshold is 0;
            # error cleanly if the user is asking for the hybrid path.
            if "cutlass_prefill_threshold" in inspect.signature(
                B12xMoEWrapper.__init__
            ).parameters:
                b12x_kwargs["cutlass_prefill_threshold"] = (
                    self.cutlass_prefill_threshold
                )
            elif self.cutlass_prefill_threshold > 0:
                raise RuntimeError(
                    "VLLM_FLASHINFER_B12X_CUTLASS_PREFILL_THRESHOLD > 0 "
                    "requires a FlashInfer build that exposes the "
                    "`cutlass_prefill_threshold` kwarg on B12xMoEWrapper; "
                    "current FlashInfer does not."
                )
            self._wrapper = B12xMoEWrapper(**b12x_kwargs)

        if self.cutlass_prefill_threshold > 0 and not self._cutlass_registered:
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
        assert self._fc2_input_scale is not None, (
            "_fc2_input_scale must be set by process_weights_after_loading"
        )
        assert self.w1_sf_mma is not None and self.w2_sf_mma is not None, (
            "process_weights_after_loading must run before FlashInferB12xExperts.apply"
        )

        self._ensure_wrapper(w1, w2)
        wrapper = self._wrapper
        assert wrapper is not None

        wrapper_output = wrapper.run(
            x=hidden_states,
            w1_weight=w1,
            w1_weight_sf=self.w1_sf_mma,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=self._fc2_input_scale,
            w2_weight=w2,
            w2_weight_sf=self.w2_sf_mma,
            w2_alpha=self.g2_alphas,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
        )
        output.copy_(wrapper_output)
