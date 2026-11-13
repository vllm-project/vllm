# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Sequence
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.v1.worker.ubatching import (
    dbo_comm_region,
    dbo_current_ubatch_id,
    dbo_maybe_run_recv_hook,
)

logger = init_logger(__name__)

# Receive-row bounds are rounded up to this many rows (as SGLang's MORI path does).
_RECV_ROW_ALIGN = 32

# Most rows a segmented receive may hand the experts (world_size * stride),
# relative to the rows all ranks send; uneven DP batches fall back to the
# compact receive past it. Matches SGLang's MORI path.
_SEGMENTED_MAX_ROW_INFLATION = 1.25


class MoriEPv2PrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Prepare/finalize over MORI EPv2 ops: one op, or one per DBO microbatch.

    An op's next call overwrites what the previous one left in its receive and
    staging buffers, so each DBO microbatch uses its own op.
    With `async_ll`, prepare_async/finalize_async return the receive as the
    hook, and the shared experts run between a combine's send and its receive.
    An AsyncLL op built with MORI's segmented receive layout lands each
    source's rows in its own segment, which the experts read in place.
    """

    def __init__(
        self,
        mori_ops: Sequence[Any],
        max_tokens_per_rank: int,
        num_dispatchers: int,
        num_local_experts: int,
        rank_expert_offset: int,
        async_ll: bool = False,
    ):
        super().__init__()
        self.mori_ops = list(mori_ops)
        self.async_ll = async_ll
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_local_experts = num_local_experts
        self.rank_expert_offset = rank_expert_offset
        self._padding_expert_ids: torch.Tensor | None = None
        # The routing each microbatch's dispatch returned, for its combine.
        self._routings: list[Any] = [None] * len(self.mori_ops)
        self._experts_read_recv_rows_only = False

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts):
        from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
            AiterExperts,
        )

        if fused_experts.moe_config.in_dtype != torch.bfloat16:
            raise ValueError(
                "MORI EPv2 only dispatches BF16 activations; got "
                f"{fused_experts.moe_config.in_dtype}. Set VLLM_MORI_EP_V2=0 to "
                "use MORI EPv1."
            )
        quant_dtype = fused_experts.quant_config.quant_dtype
        if quant_dtype is not None and not fused_experts.expects_unquantized_inputs:
            raise ValueError(
                "MORI EPv2 dispatches unquantized BF16 activations, but "
                f"{type(fused_experts).__name__} expects {quant_dtype} "
                "activations quantized before dispatch. Use AITER fused MoE experts "
                "(VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1), or set "
                "VLLM_MORI_EP_V2=0 for MORI EPv1, which also requires AITER fused MoE."
            )

        if quant_dtype is not None and fused_experts.quant_config.is_per_tensor:
            raise ValueError(
                "MORI EPv2 does not support per-tensor activation quantization: "
                "the experts would quantize the received rows, which AITER's "
                "per-tensor quant does not support. Set VLLM_MORI_EP_V2=0 to use "
                "MORI EPv1, which quantizes before dispatch."
            )

        # AITER computes only the first expert_num_tokens rows.
        self._experts_read_recv_rows_only = isinstance(fused_experts, AiterExperts)

    def allocate_fused_expert_output(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> torch.Tensor | None:
        # Experts that reduce top-k themselves write straight into the arena
        # rows combine sends, and combine skips its copy-in.
        if not isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP):
            return None
        buf = self.mori_ops[dbo_current_ubatch_id()].combine_in_view()
        if (
            len(shape) != 2
            or buf.dtype != dtype
            or buf.device != device
            or shape[1] != buf.shape[1]
            or shape[0] > buf.shape[0]
        ):
            return None
        return buf[: shape[0]]

    def output_is_reduced(self) -> bool:
        return True

    def num_dispatchers(self):
        return self.num_dispatchers_

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def supports_async(self) -> bool:
        return self.async_ll

    def supports_dbo(self) -> bool:
        # One op per microbatch; native ops overlap through dbo_comm_region.
        return len(self.mori_ops) > 1

    def _route_padding_locally(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # DP padding tokens carry expert id -1 (VLLM_MOE_SKIP_PADDING) and zero
        # weight. MORI EPv2 has no invalid-expert slot and its FlyDSL dispatch
        # faults on -1, so send them to this rank's own experts instead.
        if self._padding_expert_ids is None:
            slots = torch.arange(topk_ids.shape[1], device=topk_ids.device)
            self._padding_expert_ids = (
                self.rank_expert_offset + slots % self.num_local_experts
            ).to(topk_ids.dtype)
        return torch.where(topk_ids < 0, self._padding_expert_ids, topk_ids)

    def _recv_rows_bound(self, a1: torch.Tensor, capacity: int) -> int:
        """Host-side upper bound on the rows this rank can receive this step.

        MORI receives each source token at most once per rank, so the receive
        count is at most the tokens all dispatchers send: the sum of the DP
        batches, times the dispatchers per DP rank (each TP rank may send the
        same batch). Slicing the capacity-sized buffers (world_size *
        max_num_tokens rows) to this bound keeps the row masking, the experts
        and combine's copy-in off the unused tail. The bound is rounded up to a
        multiple of _RECV_ROW_ALIGN to limit the distinct expert shapes.
        """
        dp_meta = (
            get_forward_context().dp_metadata
            if is_forward_context_available()
            else None
        )
        if dp_meta is None:
            return capacity
        tokens = dp_meta.num_tokens_across_dp_cpu
        n_dp = tokens.numel()
        dp_rank = get_dp_group().rank_in_group
        if dp_rank >= n_dp or a1.shape[0] > int(tokens[dp_rank]):
            # Metadata that does not cover this rank's own batch is stale; a
            # bound derived from it could cut off received rows.
            logger.debug_once(
                "MORI EPv2: DP token counts do not cover this rank's batch; "
                "using the full receive buffer."
            )
            return capacity
        if n_dp > 0 and self.num_dispatchers_ % n_dp == 0:
            bound = self.num_dispatchers_ // n_dp * int(tokens.sum())
        else:
            bound = self.num_dispatchers_ * int(tokens.max())
        bound = max(bound, 1)
        bound = -(-bound // _RECV_ROW_ALIGN) * _RECV_ROW_ALIGN
        return min(capacity, bound)

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
        if self.async_ll:
            hook, receiver = self.prepare_async(
                a1,
                topk_weights,
                topk_ids,
                num_experts,
                expert_map,
                apply_router_weight_on_input,
                quant_config,
                defer_input_quant,
            )
            hook()
            return receiver()
        ubatch_id = dbo_current_ubatch_id()
        mori_op = self.mori_ops[ubatch_id]
        weights, ids = self._dispatch_args(
            a1, topk_weights, topk_ids, apply_router_weight_on_input
        )
        with dbo_comm_region():
            *received, self._routings[ubatch_id] = mori_op.dispatch(
                a1, weights, None, ids, return_routing=True
            )
        return self._finish_dispatch(a1, *received, quant_config=quant_config)

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
    ) -> tuple[Callable, mk.ReceiverType]:
        assert self.async_ll, "MORI EPv2 runs dispatch in two halves only on AsyncLL"
        ubatch_id = dbo_current_ubatch_id()
        mori_op = self.mori_ops[ubatch_id]
        weights, ids = self._dispatch_args(
            a1, topk_weights, topk_ids, apply_router_weight_on_input
        )
        stride = self._segment_stride(a1, mori_op)
        # Passed only when set, so MORI builds without the segmented layout work.
        segment = {"segment_stride": stride} if stride else {}
        # The returned views hold the received rows once dispatch_recv has run.
        *received, self._routings[ubatch_id] = mori_op.dispatch_send(
            a1, weights, None, ids, return_routing=True, **segment
        )
        if stride:
            return mori_op.dispatch_recv, lambda: self._segmented_result(*received)
        return mori_op.dispatch_recv, lambda: self._finish_dispatch(
            a1, *received, quant_config=quant_config
        )

    def _segment_stride(self, a1: torch.Tensor, mori_op: Any) -> int:
        """Rows per source of a segmented receive, or 0 for the compact one.

        MORI needs every rank to run the same stride, so it is derived only
        from what all ranks share: the op, graph capture and the DP token
        counts. The experts then get world_size * stride rows; the holes
        between the segments carry another rank's expert with weight 0, which
        only experts that read just the local experts (AITER) drop.
        """
        if not self._experts_read_recv_rows_only:
            return 0
        if not getattr(mori_op.cfg, "async_ll_segmented", False):
            return 0
        if torch.cuda.is_current_stream_capturing():
            return 0
        dp_meta = (
            get_forward_context().dp_metadata
            if is_forward_context_available()
            else None
        )
        if dp_meta is None:
            return 0
        tokens = dp_meta.num_tokens_across_dp_cpu
        n_dp = tokens.numel()
        if n_dp == 0 or self.num_dispatchers_ % n_dp != 0:
            return 0
        # Each dispatcher sends at most its DP rank's batch.
        stride = -(-max(int(tokens.max()), 1) // _RECV_ROW_ALIGN) * _RECV_ROW_ALIGN
        rows_sent = self.num_dispatchers_ // n_dp * int(tokens.sum())
        if (
            stride > mori_op.cfg.effective_max_recv_per_rank
            or self.num_dispatchers_ * stride > _SEGMENTED_MAX_ROW_INFLATION * rows_sent
        ):
            return 0
        if a1.shape[0] > stride:
            # Falling back on this rank alone would desynchronize the ranks.
            raise RuntimeError(
                f"MORI EPv2: {a1.shape[0]} tokens exceed the segment stride "
                f"{stride} derived from the DP token counts {tokens.tolist()}"
            )
        logger.info_once("MORI EPv2: AsyncLL dispatch uses the segmented receive")
        return stride

    @staticmethod
    def _segmented_result(
        dispatch_a1: torch.Tensor,
        dispatch_weights: torch.Tensor,
        dispatch_scale: torch.Tensor | None,
        dispatch_ids: torch.Tensor,
        dispatch_recv_token_num: torch.Tensor,
    ) -> mk.PrepareResultType:
        # Every one of the world_size * stride rows is an input and the holes
        # are already routed off this rank. The received-row count would only
        # repeat the row count, and AITER's MXFP4 path faults on small batches
        # given one, so the experts get none.
        return dispatch_a1, dispatch_scale, None, dispatch_ids, dispatch_weights

    def _dispatch_args(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The router's weights and expert ids in the dtypes MORI dispatch takes."""
        assert not apply_router_weight_on_input, (
            "MORI EPv2 does not support apply_router_weight_on_input=True"
        )
        if a1.dtype != torch.bfloat16:
            raise ValueError(f"MORI EPv2 requires BF16 dispatch input, got {a1.dtype}")
        weights = topk_weights.to(torch.float32)
        return weights, self._route_padding_locally(topk_ids.to(torch.int32))

    def _finish_dispatch(
        self,
        a1: torch.Tensor,
        dispatch_a1: torch.Tensor,
        dispatch_weights: torch.Tensor,
        dispatch_scale: torch.Tensor | None,
        dispatch_ids: torch.Tensor,
        dispatch_recv_token_num: torch.Tensor,
        *,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        rows = self._recv_rows_bound(a1, dispatch_a1.shape[0])
        if rows < dispatch_a1.shape[0]:
            dispatch_a1 = dispatch_a1[:rows]
            dispatch_weights = dispatch_weights[:rows]
            dispatch_ids = dispatch_ids[:rows]
            if dispatch_scale is not None:
                dispatch_scale = dispatch_scale[:rows]
        # MORI reuses capacity-sized buffers without clearing unused rows.
        unused = (
            torch.arange(dispatch_a1.shape[0], device=dispatch_a1.device)
            >= dispatch_recv_token_num
        ).unsqueeze(1)
        # Experts that read only the received rows can skip clearing the rest,
        # unless a dynamic per-tensor activation scale is taken over every row.
        if not self._experts_read_recv_rows_only or (
            quant_config.is_per_tensor and quant_config.a1_scale is None
        ):
            dispatch_a1.masked_fill_(unused, 0)
        dispatch_weights.masked_fill_(unused, 0)
        dispatch_ids.masked_fill_(unused, -1)
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_token_num,
            expert_num_tokens_cpu=None,
        )
        return (
            dispatch_a1,
            dispatch_scale,
            expert_tokens_meta,
            dispatch_ids,
            dispatch_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if self.async_ll:
            hook, receiver = self.finalize_async(
                output,
                fused_expert_output,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                weight_and_reduce_impl,
            )
            hook()
            receiver()
            return
        ubatch_id = dbo_current_ubatch_id()
        routing = self._combine_routing(ubatch_id)
        # combine takes its token mapping from the routing, not from indices.
        with dbo_comm_region():
            result = self.mori_ops[ubatch_id].combine(
                fused_expert_output, routing=routing
            )[0]
        self._finish_combine(ubatch_id, output, result)

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> tuple[Callable, Callable]:
        assert self.async_ll, "MORI EPv2 runs combine in two halves only on AsyncLL"
        # The modular kernel runs the peer's pending receive hook before
        # prepare_async but not before finalize_async, so run it here.
        dbo_maybe_run_recv_hook()
        ubatch_id = dbo_current_ubatch_id()
        mori_op = self.mori_ops[ubatch_id]
        # The returned view holds the combined rows once combine_recv has run.
        result = mori_op.combine_send(
            fused_expert_output, routing=self._combine_routing(ubatch_id)
        )[0]
        return mori_op.combine_recv, lambda: self._finish_combine(
            ubatch_id, output, result
        )

    def _combine_routing(self, ubatch_id: int) -> Any:
        routing = self._routings[ubatch_id]
        if routing is None:
            raise RuntimeError("MORI EPv2 combine called without a dispatch")
        return routing

    def _finish_combine(
        self, ubatch_id: int, output: torch.Tensor, result: torch.Tensor
    ) -> None:
        output.copy_(result[: output.shape[0]])
        if not torch.cuda.is_current_stream_capturing():
            self._routings[ubatch_id] = None
