# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LoRA-aware FlashInfer TRT-LLM MoE experts (BF16 / MXFP8 / MXINT4xBF16).

Reuses the routed API + ``gemm1_lora_delta`` path from FlashInfer PR #3153:

  * The W13 (gate_up) LoRA delta is passed directly as ``gemm1_lora_delta`` to
    the routed kernel, which fuses it into FC1 before SwiGLU (BiasType::Mn).
  * The W2 (down) LoRA cannot be fused -- we take the FC1 activation output
    returned by the kernel (``gemm1_activation_output``, permuted) together with
    ``expanded_idx_to_permuted_idx``, unpermute it, compute the W2 delta out of
    kernel via punica, and add it to the already-finalized output.

Constraints (matching the PR support matrix; final gating lives in the oracle):
  * SM100+ (Blackwell), gated SwiGLU, shuffled weights only;
  * BF16 / MXFP8 / MXINT4xBF16 base only;
  * routing must be computed outside the MoE (the Modular path satisfies this).

NOTE: this module requires a FlashInfer build that includes PR #3153 (routed API
with ``gemm1_lora_delta``). Points marked ``TODO(verify)`` must be re-checked on a
PR-enabled build with real weights; use
``z_test/proto_bf16_w13_lora_delta.py`` (A1-F / plumbing checks) to localize.
"""

from abc import abstractmethod

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.lora_experts_mixin import (
    LoRAExpertsMixin,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    trtllm_moe_pack_topk_ids_weights,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe


# --------------------------------------------------------------------------- #
#  Shared base: common apply() flow for W13 (fused) + W2 (post-processing)
# --------------------------------------------------------------------------- #
class _TrtLlmLoRAExpertsBase(LoRAExpertsMixin, mk.FusedMoEExpertsModular):
    """Shared logic for LoRA-aware trtllm MoE experts. Subclasses only implement:

    * ``_invoke_routed_moe(...)`` -- call the dtype-specific routed flashinfer
      API and return its ``Array<Tensor>`` (do_finalize + gemm1_lora_delta
      semantics described in the PR).
    * the ``_supports_*`` capability gates.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.routing_method_type = moe_config.routing_method
        self.topk = moe_config.experts_per_token
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.hidden_dim = moe_config.hidden_dim
        self.local_num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank

    # ---- Modular interface ----------------------------------------------
    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(100)
            and has_flashinfer_trtllm_fused_moe()
        )

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        # flashinfer manages its own workspace; only declare output (M, K)
        return (0,), (0,), (M, K)

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        """Override the base 3D-weight assumption.

        FusedMoEKernel._fused_experts calls moe_problem_size before apply(),
        but the base impl asserts ``len(w1.shape) == 3``. The flashinfer
        trtllm path stores shuffled weights in 4D BlockMajorK layout, so we
        derive the (E, M, N, K, topk) tuple from config + inputs instead.
        The N/K here only feed workspace sizing, which we zero out in
        workspace_shapes(); the real shapes are handled inside flashinfer.
        """
        E = self.local_num_experts
        N = 2 * self.intermediate_size_per_partition
        K = self.hidden_dim
        M = a1.size(0) if a1.dim() == 2 else a1.size(1)
        topk = topk_ids.size(1)
        return E, M, N, K, topk

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # do_finalize=True: the kernel already does moe_sum, so this is a No-Op
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_all2all_kernels
            or moe_parallel_config.use_ag_rs_all2all_kernels
        ) and not moe_parallel_config.enable_eplb

    @staticmethod
    def _supports_router_logits_dtype(router_logits_dtype, routing_method) -> bool:
        return True

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False  # gated only

    @property
    def expects_unquantized_inputs(self) -> bool:
        # BF16 / MXINT4 inputs are unquantized; the MXFP8 subclass overrides.
        return True

    # ---- Subclasses must implement --------------------------------------
    @abstractmethod
    def _invoke_routed_moe(
        self,
        *,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        packed_topk_ids: torch.Tensor,
        gemm1_lora_delta: torch.Tensor | None,
        global_num_experts: int,
        a1q_scale: torch.Tensor | None,
        output: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Call the dtype-specific trtllm_*_routed_moe and return list[Tensor].

        Return contract (do_finalize=True):
          gemm1_lora_delta is None -> [output]
          otherwise                -> [output, expanded_idx_to_permuted_idx,
                                       gemm1_activation_output(permuted)]
        """
        raise NotImplementedError

    # ---- Core apply ------------------------------------------------------
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
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        lora_context = self._lora_context
        num_tokens = hidden_states.size(0)
        top_k = self.topk
        intermediate_size = self.intermediate_size_per_partition
        K = output.size(1)

        # The LoRA tile-config heuristic (try_get_optimal_moe_config) unpacks
        # w1/w2 as standard 3D MoE weights, but flashinfer stores shuffled
        # 4D BlockMajorK weights. add_lora_w13/add_lora_w2 only read .shape
        # from w1/w2 (the actual GEMM uses lora_a/b_stacked), so pass
        # zero-storage meta tensors carrying the logical 3D shapes:
        #   w1: (E, 2I, H)  w2: (E, H, I)
        w1_cfg = torch.empty(
            (self.local_num_experts, 2 * intermediate_size, K),
            device="meta",
            dtype=torch.bfloat16,
        )
        w2_cfg = torch.empty(
            (self.local_num_experts, K, intermediate_size),
            device="meta",
            dtype=torch.bfloat16,
        )

        # Routing is computed outside the MoE; pack it into the
        # (eid<<16)|w.bf16 format the routed API expects.
        packed_topk_ids = trtllm_moe_pack_topk_ids_weights(topk_ids, topk_weights)

        # ---- 1) W13 LoRA delta -> gemm1_lora_delta (bf16, [T, top_k, 2I]) ----
        # TODO(verify): the vLLM w13 slice order (gate=w1 first, up=w3 second)
        #   must match FlashInfer's expected [gate_0..gate_{I-1}, up_0..up_{I-1}].
        gemm1_lora_delta = None
        w13_meta = (None, None, None, None)
        if lora_context is not None:
            gemm1_lora_delta = torch.zeros(
                num_tokens,
                top_k,
                2 * intermediate_size,
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
            # The LoRA shrink needs unquantized, correct-magnitude activations.
            # On the quantized subclasses (MXFP8 / FP8 block-scale) the
            # ``hidden_states`` passed here is already the quantized ``a1q``
            # produced by the modular prepare step, so fall back to the
            # unquantized stash the modular kernel keeps on the context.
            lora_x = hidden_states
            if not self.expects_unquantized_inputs:
                orig = lora_context.original_hidden_states
                assert orig is not None and orig.shape[0] == hidden_states.shape[0], (
                    "quantized trtllm LoRA path requires original_hidden_states"
                )
                lora_x = orig
            # add_inputs=False: write the pure delta only (the base is fused in
            # by the kernel) and do NOT multiply by the routing weight (it is a
            # pre-SwiGLU bias).
            w13_meta = self.apply_w13_lora(
                lora_context,
                y=gemm1_lora_delta,
                x=lora_x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                expert_map=expert_map,
                w1=w1_cfg,
                w2=w2_cfg,
                num_tokens=num_tokens,
                top_k_num=top_k,
                add_inputs=False,
            )

        # ---- 2) Call the routed flashinfer kernel ----
        ret = self._invoke_routed_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            packed_topk_ids=packed_topk_ids,
            gemm1_lora_delta=gemm1_lora_delta,
            global_num_experts=global_num_experts,
            a1q_scale=a1q_scale,
            output=output,
        )

        if lora_context is None:
            return  # no LoRA: the kernel already wrote the result into output

        # ---- 3) W2 LoRA (computed out of kernel) ----
        expanded_idx_to_permuted_idx = self._as_tensor(ret[1])
        gemm1_act_permuted = self._as_tensor(ret[2])  # [max_padded, I], post-act

        act = self._unpermute_activation(
            gemm1_act_permuted,
            expanded_idx_to_permuted_idx,
            num_tokens,
            top_k,
            intermediate_size,
        )  # (T*top_k, I) -- same layout as the triton path's intermediate_cache2

        (
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            token_lora_mapping,
        ) = w13_meta

        w2_delta = torch.zeros(
            num_tokens,
            top_k,
            K,
            dtype=output.dtype,
            device=output.device,
        )
        self.apply_w2_lora(
            lora_context,
            y=w2_delta,
            x=act,
            topk_weights=topk_weights,
            sorted_token_ids_lora=sorted_token_ids_lora,
            expert_ids_lora=expert_ids_lora,
            num_tokens_post_padded_lora=num_tokens_post_padded_lora,
            token_lora_mapping=token_lora_mapping,
            num_tokens=num_tokens,
            w1=w1_cfg,
            w2=w2_cfg,
            top_k_num=top_k,
            add_inputs=False,
        )
        # The base output is already finalized (routing-weighted + summed over
        # top_k); the W2 delta is likewise already routing-weighted, so sum it
        # over top_k and add.
        # TODO(verify): if routed_scaling_factor is not None, scale to match.
        output.add_(w2_delta.sum(dim=1))

    # ---- Helpers ---------------------------------------------------------
    @staticmethod
    def _as_tensor(t) -> torch.Tensor:
        return t if torch.is_tensor(t) else torch.from_dlpack(t)

    @staticmethod
    def _unpermute_activation(
        act_permuted: torch.Tensor,
        idx_map: torch.Tensor,
        num_tokens: int,
        top_k: int,
        intermediate_size: int,
    ) -> torch.Tensor:
        """Permuted FC1 activation -> (num_tokens*top_k, I).

        expanded_idx = token*top_k + k; idx_map[expanded_idx] = permuted_idx or -1.
        TODO optimize these operations
        """

        valid = idx_map >= 0
        safe_idx = idx_map.clamp_min(0).long()
        gathered = act_permuted[safe_idx]
        return gathered * valid.unsqueeze(1).to(act_permuted.dtype)


# --------------------------------------------------------------------------- #
#  BF16
# --------------------------------------------------------------------------- #
class TrtLlmBf16LoRAExperts(_TrtLlmLoRAExpertsBase):
    """BF16 unquantized trtllm MoE + LoRA."""

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU]

    @staticmethod
    def _supports_routing_method(routing_method, weight_key, activation_key) -> bool:
        return routing_method in [
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.Llama4,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    def _invoke_routed_moe(
        self,
        *,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        packed_topk_ids: torch.Tensor,
        gemm1_lora_delta: torch.Tensor | None,
        global_num_experts: int,
        a1q_scale: torch.Tensor | None,
        output: torch.Tensor,
    ) -> list[torch.Tensor]:
        import flashinfer

        # Unlike the fp8/mxint4 routed APIs, trtllm_bf16_routed_moe has no
        # `output=` kwarg: it returns the finalized tensor (or a list whose
        # [0] is it when gemm1_lora_delta is set). Copy it into the caller's
        # buffer so the modular-kernel output plumbing sees the result.
        ret = flashinfer.fused_moe.trtllm_bf16_routed_moe(
            topk_ids=packed_topk_ids,
            hidden_states=hidden_states,
            gemm1_weights=w1,
            gemm2_weights=w2,
            gemm1_lora_delta=gemm1_lora_delta,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=self.routing_method_type,
            do_finalize=True,
        )
        if isinstance(ret, (list, tuple)):
            output.copy_(self._as_tensor(ret[0]))
            return list(ret)
        output.copy_(self._as_tensor(ret))
        return [output]


# --------------------------------------------------------------------------- #
#  MXFP8
# --------------------------------------------------------------------------- #
class TrtLlmMxfp8LoRAExperts(_TrtLlmLoRAExpertsBase):
    """MXFP8 block-scale trtllm MoE + LoRA."""

    @property
    def expects_unquantized_inputs(self) -> bool:
        return False  # needs mxfp8-quantized inputs + scale

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kMxfp8Dynamic,
            kMxfp8Static,
        )

        return weight_key in (kMxfp8Static, kMxfp8Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU]

    @staticmethod
    def _supports_routing_method(routing_method, weight_key, activation_key) -> bool:
        return routing_method in [
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.Llama4,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    def _invoke_routed_moe(
        self,
        *,
        hidden_states,
        w1,
        w2,
        packed_topk_ids,
        gemm1_lora_delta,
        global_num_experts,
        a1q_scale,
        output,
    ):
        import flashinfer
        from flashinfer.fused_moe import Fp8QuantizationType, WeightLayout

        assert a1q_scale is not None, "MXFP8 requires hidden_states_scale"
        return flashinfer.fused_moe.trtllm_fp8_block_scale_routed_moe(
            topk_ids=packed_topk_ids,
            routing_bias=None,
            hidden_states=hidden_states,
            hidden_states_scale=a1q_scale,
            gemm1_weights=w1,
            gemm1_weights_scale=self.w1_scale,
            gemm1_lora_delta=gemm1_lora_delta,
            gemm2_weights=w2,
            gemm2_weights_scale=self.w2_scale,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=int(self.routing_method_type),
            use_shuffled_weight=True,
            weight_layout=int(WeightLayout.MajorK),
            fp8_quantization_type=Fp8QuantizationType.MxFp8,
            do_finalize=True,
            output=output,
        )


# --------------------------------------------------------------------------- #
#  FP8 block-scale (DeepSeekFp8, 128x128)
# --------------------------------------------------------------------------- #
class TrtLlmFp8LoRAExperts(_TrtLlmLoRAExpertsBase):
    """FP8 128x128 block-scale (DeepSeekFp8) trtllm MoE + LoRA.

    Mirrors the DeepSeekFp8 branch of the non-LoRA ``TrtLlmFp8ExpertsModular``
    (``hidden_states_scale`` transposed, ``BlockMajorK`` layout) and adds the
    ``gemm1_lora_delta`` W13 fusion; W2 LoRA is computed out of kernel by the
    shared base ``apply``.
    """

    @property
    def expects_unquantized_inputs(self) -> bool:
        return False  # needs fp8-quantized inputs + block scale

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8Dynamic128Sym,
            kFp8Static128BlockSym,
        )

        return (weight_key, activation_key) == (
            kFp8Static128BlockSym,
            kFp8Dynamic128Sym,
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU]

    @staticmethod
    def _supports_routing_method(routing_method, weight_key, activation_key) -> bool:
        return routing_method in [
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.Llama4,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    def _invoke_routed_moe(
        self,
        *,
        hidden_states,
        w1,
        w2,
        packed_topk_ids,
        gemm1_lora_delta,
        global_num_experts,
        a1q_scale,
        output,
    ):
        import flashinfer
        from flashinfer.fused_moe import Fp8QuantizationType, WeightLayout

        assert a1q_scale is not None, "FP8 block scale requires hidden_states_scale"
        assert self.w1_scale is not None and self.w2_scale is not None
        return flashinfer.fused_moe.trtllm_fp8_block_scale_routed_moe(
            topk_ids=packed_topk_ids,
            routing_bias=None,
            hidden_states=hidden_states,
            # DeepSeekFp8 wants the transposed per-block scale (matches the
            # non-LoRA modular expert).
            hidden_states_scale=a1q_scale.t().contiguous(),
            gemm1_weights=w1,
            gemm1_weights_scale=self.w1_scale,
            gemm1_lora_delta=gemm1_lora_delta,
            gemm2_weights=w2,
            gemm2_weights_scale=self.w2_scale,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=int(self.routing_method_type),
            use_shuffled_weight=True,
            weight_layout=int(WeightLayout.BlockMajorK),
            fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
            do_finalize=True,
            output=output,
        )


# --------------------------------------------------------------------------- #
#  MXINT4 x BF16
# --------------------------------------------------------------------------- #
class TrtLlmMxint4LoRAExperts(_TrtLlmLoRAExpertsBase):
    """MXINT4 x BF16 block-scale trtllm MoE + LoRA."""

    @staticmethod
    def _supports_current_device() -> bool:
        from vllm.model_executor.layers.quantization.utils.flashinfer_mxint4_moe import (  # noqa: E501
            is_flashinfer_mxint4_moe_available,
        )

        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(100)
            and is_flashinfer_mxint4_moe_available()
        )

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kInt4Static32,
        )

        return (weight_key, activation_key) == (kInt4Static32, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # MxInt4 uses a fused SwiGLU (SWIGLUOAI)
        return activation == MoEActivation.SWIGLUOAI

    @staticmethod
    def _supports_routing_method(routing_method, weight_key, activation_key) -> bool:
        return routing_method in [
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.Llama4,
        ]

    def supports_expert_map(self) -> bool:
        return False

    def _invoke_routed_moe(
        self,
        *,
        hidden_states,
        w1,
        w2,
        packed_topk_ids,
        gemm1_lora_delta,
        global_num_experts,
        a1q_scale,
        output,
    ):
        import flashinfer

        assert self.w1_scale is not None and self.w2_scale is not None
        return flashinfer.fused_moe.trtllm_mxint4_block_scale_routed_moe(
            topk_ids=packed_topk_ids,
            hidden_states=hidden_states,
            gemm1_weights=w1,
            gemm1_weights_scale=self.w1_scale,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm1_lora_delta=gemm1_lora_delta,
            gemm2_weights=w2,
            gemm2_weights_scale=self.w2_scale,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=int(self.routing_method_type),
            do_finalize=True,
            output=output,
        )
