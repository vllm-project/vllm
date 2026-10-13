# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

logger = init_logger(__name__)

# Receive-row bounds are rounded up to this many rows (as SGLang's MORI path does).
_RECV_ROW_ALIGN = 32


class MoriEPv2PrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    def __init__(
        self,
        mori_op: Any,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        num_local_experts: int,
        rank_expert_offset: int,
    ):
        super().__init__()
        self.mori_op = mori_op
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_local_experts = num_local_experts
        self.rank_expert_offset = rank_expert_offset
        self._padding_expert_ids: torch.Tensor | None = None
        self._routing = None
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
                f"{fused_experts.moe_config.in_dtype}"
            )
        quant_dtype = fused_experts.quant_config.quant_dtype
        if quant_dtype is not None and not fused_experts.expects_unquantized_inputs:
            raise ValueError(
                "MORI EPv2 dispatches unquantized BF16 activations, but "
                f"{type(fused_experts).__name__} expects {quant_dtype} "
                "activations quantized before dispatch"
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
        buf = self.mori_op.combine_in_view()
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
        return False

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
        del num_experts, expert_map, defer_input_quant
        assert not apply_router_weight_on_input, (
            "MORI EPv2 does not support apply_router_weight_on_input=True"
        )
        if a1.dtype != torch.bfloat16:
            raise ValueError(f"MORI EPv2 requires BF16 dispatch input, got {a1.dtype}")

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
            self._routing,
        ) = self.mori_op.dispatch(
            a1,
            topk_weights.to(torch.float32),
            None,
            self._route_padding_locally(topk_ids.to(torch.int32)),
            return_routing=True,
        )
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
        del topk_weights, topk_ids, apply_router_weight_on_input
        del weight_and_reduce_impl
        if self._routing is None:
            raise RuntimeError("MORI EPv2 combine called without a dispatch")
        # combine takes its token mapping from the routing, not from indices.
        result = self.mori_op.combine(fused_expert_output, routing=self._routing)[0]
        output.copy_(result[: output.shape[0]])
        if not torch.cuda.is_current_stream_capturing():
            self._routing = None
