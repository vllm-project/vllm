# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LoRA-aware FlashInfer TRT-LLM MoE experts (BF16).

Reuses the routed API + ``gemm1_lora_delta`` path from FlashInfer PR #3153:

  * The W13 (gate_up) LoRA delta is passed directly as ``gemm1_lora_delta`` to
    the routed kernel, which fuses it into FC1 before SwiGLU (BiasType::Mn).
  * The W2 (down) LoRA cannot be fused -- we take the FC1 activation output
    returned by the kernel (``gemm1_activation_output``, permuted) together with
    ``expanded_idx_to_permuted_idx``, unpermute it, compute the W2 delta out of
    kernel via punica, and add it to the already-finalized output.

Constraints (matching the PR support matrix; final gating lives in the oracle):
  * SM100+ (Blackwell), gated SwiGLU, shuffled weights only;
  * BF16 only;
  * routing must be computed outside the MoE (the Modular path satisfies this).

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
from vllm.model_executor.layers.fused_moe.experts.lora_context import MoELoRAContext
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
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe


@triton.jit
def _unpermute_activation_kernel(
    act_ptr,  # act_permuted: (num_permuted, num_cols)
    idx_ptr,  # idx_map: (num_rows,), values in [0, num_permuted) or -1
    out_ptr,  # out: (num_rows, num_cols)
    num_cols,
    stride_ar,
    stride_or,
    BLOCK_I: tl.constexpr,
):
    row = tl.program_id(0)
    col_offs = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    col_mask = col_offs < num_cols

    idx = tl.load(idx_ptr + row)
    out_ptrs = out_ptr + row * stride_or + col_offs
    if idx >= 0:
        vals = tl.load(act_ptr + idx * stride_ar + col_offs, mask=col_mask, other=0.0)
        tl.store(out_ptrs, vals, mask=col_mask)
    else:
        zeros = tl.zeros((BLOCK_I,), dtype=out_ptr.dtype.element_ty)
        tl.store(out_ptrs, zeros, mask=col_mask)


@triton.jit
def _finalize_lora_kernel(
    gemm2_ptr,  # (num_permuted, K) base FC2 output, permuted, unweighted
    weight_ptr,  # (num_tokens * top_k,) routing weights (expanded order)
    idx_ptr,  # (num_tokens * top_k,) expanded_idx -> permuted_idx or -1
    delta_ptr,  # (num_tokens, top_k, K) W2 LoRA delta, already routing-weighted
    out_ptr,  # (num_tokens, K)
    K,
    stride_g0,
    stride_d0,
    stride_d1,
    stride_o0,
    scale,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    col = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = col < K

    acc_base = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc_delta = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        eid = token * TOP_K + k
        pidx = tl.load(idx_ptr + eid)
        if pidx >= 0:
            w = tl.load(weight_ptr + eid).to(tl.float32)
            base = tl.load(gemm2_ptr + pidx * stride_g0 + col, mask=mask, other=0.0).to(
                tl.float32
            )
            acc_base += w * base
        acc_delta += tl.load(
            delta_ptr + token * stride_d0 + k * stride_d1 + col, mask=mask, other=0.0
        ).to(tl.float32)

    out = acc_base * scale + acc_delta
    tl.store(
        out_ptr + token * stride_o0 + col, out.to(out_ptr.dtype.element_ty), mask=mask
    )


class _TrtLlmLoRAExpertsBase(LoRAExpertsMixin, mk.FusedMoEExpertsModular):
    """LoRA-aware trtllm MoE experts"""

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
        # apply() writes the fully finalized result into `output` (fused base
        # finalize + W2 LoRA reduction), so this is a No-Op.
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
        return True

    @abstractmethod
    def invoke_routed_moe(
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

        The LoRA path always sets gemm1_lora_delta and runs with
        do_finalize=False so the base finalize can be fused with the W2 LoRA
        reduction (see _finalize_with_w2_lora). Return contract:
          gemm1_lora_delta is None -> [output]  (do_finalize=True)
          otherwise                -> [gemm2_output(permuted, unweighted),
                                       expert_weights,
                                       expanded_idx_to_permuted_idx,
                                       gemm1_activation_output(permuted)]
        """
        raise NotImplementedError

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
        assert lora_context is not None, "LoRA context must be set"
        num_tokens = hidden_states.size(0)
        top_k = self.topk
        intermediate_size = self.intermediate_size_per_partition
        K = output.size(1)

        # Routing is computed outside the MoE; pack it into the
        # (eid<<16)|w.bf16 format the routed API expects.
        packed_topk_ids = trtllm_moe_pack_topk_ids_weights(topk_ids, topk_weights)

        # ---- Base-model fast path ----
        # When no token in the batch selects a LoRA adapter, skip the LoRA machinery
        # and run the plain base MoE with do_finalize=True, which writes the finalized
        # result straight into `output`.
        if self._batch_has_no_lora(lora_context):
            self.invoke_routed_moe(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                packed_topk_ids=packed_topk_ids,
                gemm1_lora_delta=None,  # without LoRA, no delta
                global_num_experts=global_num_experts,
                a1q_scale=a1q_scale,
                output=output,
            )
            return

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

        # ---- 1) W13 LoRA delta -> gemm1_lora_delta (bf16, [T, top_k, 2I]) ----
        gemm1_lora_delta = None
        w13_meta = (None, None, None, None)

        # zeros (not empty): under EP the punica expand kernel only writes
        # slots whose expert is local to this rank; non-local (token, top_k)
        # slots must stay 0 so they contribute no bias when fed to flashinfer.
        gemm1_lora_delta = torch.zeros(
            num_tokens,
            top_k,
            2 * intermediate_size,
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )

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
        # swap_w13_slices=True: apply_w13_lora writes in vLLM's w13 order
        # (gate=w1 first, up=w3 second), but FlashInfer's gemm1_lora_delta
        # expects [up, gate]; reversing the slices emits that order directly,
        # avoiding an out-of-place concat swap.
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
            swap_w13_slices=True,
        )

        # ---- 2) Call the routed flashinfer kernel ----
        ret = self.invoke_routed_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            packed_topk_ids=packed_topk_ids,
            gemm1_lora_delta=gemm1_lora_delta,
            global_num_experts=global_num_experts,
            a1q_scale=a1q_scale,
            output=output,
        )
        # ---- 3) W2 LoRA (computed out of kernel) ----
        # do_finalize=False: flashinfer returns the *unfinalized* base output.
        #   ret = [gemm2_output(permuted, unweighted),
        #          expert_weights, expanded_idx_to_permuted_idx,
        #          gemm1_activation_output(permuted)]
        gemm2_permuted = ret[0]
        expert_weights = ret[1]
        expanded_idx_to_permuted_idx = ret[2]
        gemm1_act_permuted = ret[3]  # [max_padded, I], post-act
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
        # ---- 4) Fused finalize: reduce the base path over top_k (with routing
        # weights) and add the already-weighted W2 delta, in a single kernel.
        # This replaces flashinfer's internal finalize launch plus a separate
        # w2_delta.sum(dim=1) + add_.
        self._finalize_with_w2_lora(
            output,
            gemm2_permuted,
            expert_weights,
            expanded_idx_to_permuted_idx,
            w2_delta,
            num_tokens,
            top_k,
            scale=1.0,
        )

    @staticmethod
    def _batch_has_no_lora(lora_context: MoELoRAContext) -> bool:
        """True when no token in the batch selects a LoRA adapter.

        Mirrors the no-lora fast path in
        ``PunicaWrapperGPU.add_lora_fused_moe``: the punica kernel metadata
        carries a CPU ``no_lora_flag`` computed once per forward from the
        token->LoRA mapping. Reading it is a host-only check (no device sync),
        and under CUDA graphs the branch is frozen at capture time against the
        graph's ``has_lora`` dispatch key, so it stays correct on replay.
        """
        meta = getattr(lora_context.punica_wrapper, "token_mapping_meta", None)
        if meta is None:
            return False
        flag = meta.no_lora_flag_cpu
        return bool(flag.numel() == 1 and flag.item())

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
        Fused gather + drop-masking: each output row copies the matching
        permuted row, or is zeroed when idx_map < 0.
        """
        num_rows = num_tokens * top_k
        out = torch.empty(
            (num_rows, intermediate_size),
            dtype=act_permuted.dtype,
            device=act_permuted.device,
        )
        BLOCK_I = 1024
        grid = (num_rows, triton.cdiv(intermediate_size, BLOCK_I))
        _unpermute_activation_kernel[grid](
            act_permuted,
            idx_map,
            out,
            intermediate_size,
            act_permuted.stride(0),
            out.stride(0),
            BLOCK_I=BLOCK_I,
        )
        return out

    @staticmethod
    def _finalize_with_w2_lora(
        output: torch.Tensor,
        gemm2_permuted: torch.Tensor,
        expert_weights: torch.Tensor,
        idx_map: torch.Tensor,
        w2_delta: torch.Tensor,
        num_tokens: int,
        top_k: int,
        scale: float = 1.0,
    ) -> None:
        """Fused base finalize + W2 LoRA reduction, written into ``output``.

        For each token: sum the routing-weighted permuted base rows over top_k
        (``expert_weights`` in expanded order, ``idx_map < 0`` dropped), scale by
        ``scale``, and add the already-weighted ``w2_delta`` reduced over top_k.
        """
        K = gemm2_permuted.size(1)
        BLOCK_K = 512
        grid = (num_tokens, triton.cdiv(K, BLOCK_K))
        _finalize_lora_kernel[grid](
            gemm2_permuted,
            expert_weights.reshape(-1),
            idx_map,
            w2_delta,
            output,
            K,
            gemm2_permuted.stride(0),
            w2_delta.stride(0),
            w2_delta.stride(1),
            output.stride(0),
            scale,
            TOP_K=top_k,
            BLOCK_K=BLOCK_K,
        )


# BF16 unquantized trtllm MoE + LoRA
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

    def invoke_routed_moe(
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

        # With gemm1_lora_delta set (the LoRA path) run do_finalize=False and
        # return the unfinalized permuted base output so apply() can fuse the
        # finalize with the W2 LoRA reduction (see _finalize_with_w2_lora).
        # Without a delta (base path), run do_finalize=True and hand flashinfer
        # the caller's buffer via output= so it finalizes in place -- no copy.
        do_finalize = gemm1_lora_delta is None
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
            do_finalize=do_finalize,
            output=output if do_finalize else None,
        )
        if not do_finalize:
            # [gemm2_output, expert_weights, expanded_idx, gemm1_activation]
            return list(ret)
        # do_finalize=True finalized directly into `output`.
        return [output]
