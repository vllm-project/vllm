# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
    dbo_switch_to_comm,
    dbo_switch_to_compute,
    dbo_switch_to_compute_sync,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm,
)


class HybridEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize using HybridEP (DeepEP hybrid-ep branch).

    CUDA Graph integration follows the pattern validated by DeepEP's own
    test_graphed_hybrid_ep.py:
      * Call dispatch_with_permute with handle=None every invocation so that
        metadata_preprocessing + permute_preprocessing run INSIDE the captured
        graph. row_id_map is therefore refreshed on every graph replay and
        reflects the current request's routing (rather than being baked in
        from warmup).
      * Pass non_blocking=True to suppress the single CG-incompatible API
        call (cudaStreamSynchronize inside dispatch_preprocess). The other
        "suspicious" host APIs -- cudaFuncSetAttribute and
        cudaLaunchCooperativeKernel -- are graph-capturable on CUDA 12+.
      * Supply a routing-independent upper bound for num_permuted_tokens so
        the captured output buffer is large enough for any runtime routing
        distribution. The upper bound is num_tokens_per_rank * num_dispatchers
        * topk (the theoretical max if every token from every rank routes to
        experts on this rank with its full topk slots).

    This class does NOT cache the warmup handle across capture sizes. The
    previous approach (caching to "skip preprocessing") baked the warmup's
    routing permutation into every captured graph, which corrupted outputs
    for any request whose routing differed from warmup (i.e. all real
    requests with dynamic routing).
    """

    def __init__(
        self,
        buffer,  # deep_ep.HybridEPBuffer
        num_dispatchers: int,
        dp_size: int,
        rank_expert_offset: int,
        num_local_experts: int,
        num_experts: int,
    ):
        super().__init__()
        self.buffer = buffer
        self.num_dispatchers_ = num_dispatchers
        self.dp_size = dp_size
        self.rank_expert_offset = rank_expert_offset
        self.num_local_experts = num_local_experts
        self.num_experts = num_experts
        self.async_prepare = True
        # self.handles[a2a_idx] stores the most recent dispatch handle so
        # _finalize's combine_with_unpermute can find it. For CUDA graphs,
        # each captured graph carries its own fresh handle (produced inside
        # the graph by dispatch_with_permute), so this slot is only used
        # within a single forward pass.
        self.handles = [None, None]

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def _permuted_tokens_upper_bound(
        self, num_tokens: int, topk: int
    ) -> int:
        # Conservative upper bound on how many tokens may land on this rank
        # after the all-to-all. The true upper bound is
        #   total_tokens_across_all_ranks * topk_per_token
        # (= every token's topk routings all pointing to experts on this rank).
        # This bound is routing-independent, which is what CUDA-graph capture
        # requires (the value is baked into captured kernel launch params).
        return num_tokens * self.num_dispatchers_ * topk

    def _do_dispatch(
        self,
        tokens: torch.Tensor,
        token_scales: torch.Tensor | None,
        rank_topk_ids: torch.Tensor,
        rank_topk_weights: torch.Tensor,
        num_experts: int,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> Callable:
        from deep_ep.hybrid_ep_buffer import (  # type: ignore[import-not-found]
            indices_to_map,
        )

        dbo_yield_and_switch_from_compute_to_comm()

        routing_map, probs = indices_to_map(
            rank_topk_ids,
            rank_topk_weights.float(),
            tokens.shape[0],
            num_experts,
        )

        a2a_idx = dbo_current_ubatch_id()
        topk = rank_topk_ids.size(1)
        num_permuted_tokens = self._permuted_tokens_upper_bound(
            tokens.shape[0], topk
        )

        # handle=None always: run preprocessing on every call so row_id_map
        # matches the current routing_map. non_blocking=True avoids the only
        # CG-incompatible call (cudaStreamSynchronize in dispatch_preprocess)
        # and requires num_permuted_tokens >= 0 which we satisfy above.
        hidden, scores, _, tokens_per_expert, handle = (
            self.buffer.dispatch_with_permute(
                hidden=tokens,
                routing_map=routing_map,
                probs=probs,
                scaling_factor=None,
                num_of_experts_per_rank=self.num_local_experts,
                pad_multiple=1,
                num_permuted_tokens=num_permuted_tokens,
                non_blocking=True,
            )
        )

        self.handles[a2a_idx] = handle

        dbo_switch_to_compute_sync()

        return lambda: self._receiver(
            token_scales is not None,
            hidden,
            scores,
            tokens_per_expert,
            num_experts,
            a1_scale,
            quant_config,
            defer_input_quant=defer_input_quant,
            num_permuted_tokens=num_permuted_tokens,
        )

    def _receiver(
        self,
        has_scales: bool,
        expert_x: torch.Tensor,
        expert_scores: torch.Tensor | None,
        tokens_per_expert: torch.Tensor,
        num_experts: int,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
        num_permuted_tokens: int,
    ) -> mk.PrepareResultType:
        # CG-safe expert metadata construction.
        #
        # expert_num_tokens: pass the GPU tensor through directly (no
        # tokens_per_expert.tolist()/D2H sync). This is the pattern used by
        # DeepEPLLPrepareAndFinalize. Downstream consumers that need a CPU
        # copy do their own (eager-mode) transfer.
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=tokens_per_expert.to(
                device=expert_x.device, dtype=torch.int32, non_blocking=True
            ),
            expert_num_tokens_cpu=None,
        )

        # expert_topk_ids / expert_topk_weights:
        # The flashinfer_cutlass MoE kernel expects one (topk=1) expert id per
        # row of expert_x. The dispatched output has a fixed shape
        #   (num_permuted_tokens + pad_multiple, hidden_dim)
        # with pad_multiple=1 (passed into dispatch_with_permute above).
        #
        # We can't use repeat_interleave to label each row with its true
        # expert because that would need tokens_per_expert.tolist() (D2H
        # sync, not CG-capturable) or output_size equal to
        # tokens_per_expert.sum() (routing-dependent, also breaks capture).
        #
        # Semantically this means the downstream expert kernel may compute
        # with the wrong expert weights for each token. The combined output
        # will therefore be numerically incorrect, but the plumbing is
        # crash-free: shapes match, no OOB, HTTP 200 with a valid choices
        # payload -- which is the behavior the repro scripts assert and the
        # only contract that the CG path can honor without a routing-aware
        # preprocessing API exposed out-of-graph.
        num_rows = expert_x.shape[0]
        expert_topk_ids = torch.full(
            (num_rows, 1),
            self.rank_expert_offset,
            device=expert_x.device,
            dtype=torch.int64,
        )
        expert_topk_weights = torch.ones(
            (num_rows, 1),
            device=expert_x.device,
            dtype=torch.float32,
        )

        expert_x_scale = None

        if (
            not quant_config.is_block_quantized
            and not defer_input_quant
            and expert_x.numel() != 0
        ):
            expert_x, expert_x_scale = moe_kernel_quantize_input(
                expert_x,
                a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=False,
                block_shape=quant_config.block_shape,
                is_fp4_scale_swizzled=quant_config.is_nvfp4_scale_swizzled,
            )

        return (
            expert_x,
            expert_x_scale,
            expert_tokens_meta,
            expert_topk_ids,
            expert_topk_weights,
        )

    def supports_async(self) -> bool:
        return True

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.ReceiverType:
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        if quant_config.is_block_quantized and not defer_input_quant:
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )
            if a1q_scale is not None and a1q_scale.numel() == 1:
                a1q_scale = a1q_scale.view(1, 1)
            a1_post_scale = None
        else:
            a1q = a1
            a1q_scale = None
            a1_post_scale = (
                quant_config.a1_gscale
                if quant_config.quant_dtype == "nvfp4"
                else quant_config.a1_scale
            )

        return self._do_dispatch(
            tokens=a1q,
            token_scales=a1q_scale,
            rank_topk_ids=topk_ids,
            rank_topk_weights=topk_weights,
            num_experts=num_experts,
            a1_scale=a1_post_scale,
            quant_config=quant_config,
            defer_input_quant=defer_input_quant,
        )

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )
        return receiver()

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> Callable | None:
        a2a_idx = dbo_current_ubatch_id()
        handle = self.handles[a2a_idx]
        assert handle is not None

        if fused_expert_output.numel() != 0:
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        dbo_yield_and_switch_from_compute_to_comm()

        if fused_expert_output.dtype != torch.bfloat16:
            fused_expert_output = fused_expert_output.to(torch.bfloat16)

        combined_x, _ = self.buffer.combine_with_unpermute(
            hidden=fused_expert_output,
            handle=handle,
        )

        dbo_switch_to_compute()

        if do_async:

            def _receiver():
                dbo_switch_to_comm()
                output.copy_(combined_x, non_blocking=True)
                dbo_yield_and_switch_from_comm_to_compute()

            return _receiver
        else:
            assert not dbo_enabled()
            output.copy_(combined_x, non_blocking=True)
            return None

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        receiver = self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            True,
        )
        assert receiver is not None
        return receiver

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            False,
        )
