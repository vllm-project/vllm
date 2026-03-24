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


def _is_cuda_graph_capturing() -> bool:
    """Return True if the current CUDA stream is in graph capture mode."""
    return torch.cuda.is_current_stream_capturing()


class HybridEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize using HybridEP (DeepEP hybrid-ep branch).

    This class mirrors the structure of DeepEPHTPrepareAndFinalize but
    uses a different communication API:

    DeepEPHTPrepareAndFinalize (deep_ep.Buffer):
        get_dispatch_layout() -> dispatch() -> [experts] -> combine()
        - Three-step dispatch: compute layout, then send tokens, then receive.
        - Routing permutation is a separate step before/after communication.
        - CPU blocks on get_dispatch_layout to learn per-rank token counts.

    HybridEPPrepareAndFinalize (deep_ep.HybridEPBuffer):
        dispatch_with_permute() -> [experts] -> combine_with_unpermute()
        - Single fused kernel merges routing permutation + all-to-all.
        - Returns tokens already grouped by local expert (no separate permute).
        - Requires indices_to_map() to convert vLLM's topk format to a
          dense routing_map before dispatch.
        - After dispatch, topk_ids must be reconstructed from tokens_per_expert
          via repeat_interleave, since the fused kernel consumes them internally.

    CUDA Graph support:
        The C++ permute_preprocessing() path uses cudaLaunchCooperativeKernel,
        cudaFuncSetAttribute, and cudaStreamSynchronize -- all incompatible
        with CUDA graph capture. When a cached handle (containing row_id_map)
        is passed back to dispatch_with_permute(), the C++ code skips
        preprocessing entirely. During CG capture/replay, we reuse the
        handle from warmup and pass non_blocking=True to avoid all
        CG-incompatible operations.
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
        self.handles = [None, None]

        # Caches populated during warmup, reused during CG capture/replay.
        self._cached_num_permuted_tokens: list[int | None] = [None, None]
        self._cached_expert_num_tokens_list: list[list[int] | None] = [
            None,
            None,
        ]
        self._cached_expert_topk_ids: list[torch.Tensor | None] = [None, None]
        self._cached_expert_topk_weights: list[torch.Tensor | None] = [
            None,
            None,
        ]

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
        cached_handle = self.handles[a2a_idx]
        capturing = _is_cuda_graph_capturing()

        if capturing and cached_handle is not None:
            # CG capture/replay: reuse handle from warmup.
            # This passes row_id_map to C++, which skips
            # permute_preprocessing
            num_permuted_tokens = self._cached_num_permuted_tokens[a2a_idx]
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
                    handle=cached_handle,
                )
            )
        else:
            # Warmup / eager: full path with preprocessing.
            hidden, scores, _, tokens_per_expert, handle = (
                self.buffer.dispatch_with_permute(
                    hidden=tokens,
                    routing_map=routing_map,
                    probs=probs,
                    scaling_factor=None,
                    num_of_experts_per_rank=self.num_local_experts,
                    pad_multiple=1,
                    num_permuted_tokens=None,
                    non_blocking=False,
                )
            )

        self.handles[a2a_idx] = handle

        # Cache num_permuted_tokens for CG mode (required by non_blocking).
        if not capturing:
            self._cached_num_permuted_tokens[a2a_idx] = int(
                tokens_per_expert.sum().item()
            )

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
            capturing=capturing,
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
        capturing: bool = False,
    ) -> mk.PrepareResultType:
        a2a_idx = dbo_current_ubatch_id()

        if capturing and self._cached_expert_topk_ids[a2a_idx] is not None:
            # CG capture/replay: use cached metadata from warmup.
            # tokens_per_expert.tolist() is a GPU-to-CPU sync and
            # repeat_interleave produces data-dependent shapes --
            # both are incompatible with CG capture.
            expert_num_tokens_list = self._cached_expert_num_tokens_list[a2a_idx]
            expert_topk_ids = self._cached_expert_topk_ids[a2a_idx]
            expert_topk_weights = self._cached_expert_topk_weights[a2a_idx]
        else:
            # Warmup / eager: compute from live data, then cache.
            expert_num_tokens_list = tokens_per_expert.tolist()

            # dispatch_with_permute returns tokens grouped by local expert.
            # Reconstruct topk_ids so the expert kernel knows which expert
            # each token belongs to (in global expert space).
            expert_topk_ids = torch.repeat_interleave(
                torch.arange(
                    self.num_local_experts,
                    device=expert_x.device,
                    dtype=torch.int64,
                )
                + self.rank_expert_offset,
                tokens_per_expert.to(expert_x.device),
            ).unsqueeze(1)

            expert_topk_weights = torch.ones(
                expert_topk_ids.shape,
                device=expert_x.device,
                dtype=torch.float32,
            )

            # Cache for CG capture.
            self._cached_expert_num_tokens_list[a2a_idx] = expert_num_tokens_list
            self._cached_expert_topk_ids[a2a_idx] = expert_topk_ids
            self._cached_expert_topk_weights[a2a_idx] = expert_topk_weights

        expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list(
            expert_num_tokens_list, device=expert_x.device
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

        # Apply per-token topk weighting before the all2all combine.
        # output_is_reduced=True refers to the cross-rank reduction done
        # by combine_with_unpermute, not this local weighting step.
        # Same pattern as DeepEPHTPrepareAndFinalize._finalize.
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
