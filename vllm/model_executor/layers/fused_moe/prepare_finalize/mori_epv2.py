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
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)

# Receive-row bounds are rounded up to this many rows (as SGLang's MORI path does).
_RECV_ROW_ALIGN = 32


@triton.jit
def _dispatch_args_kernel(
    weights_in,
    ids_in,
    weights_out,
    ids_out,
    numel,
    topk,
    rank_expert_offset,
    num_local_experts,
    BLOCK: tl.constexpr,
):
    # fp32 weights and int32 ids; padding ids (-1) go to this rank's experts.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    w = tl.load(weights_in + offs, mask=mask, other=0.0).to(tl.float32)
    e = tl.load(ids_in + offs, mask=mask, other=0).to(tl.int32)
    local = rank_expert_offset + (offs % topk) % num_local_experts
    e = tl.where(e < 0, local.to(tl.int32), e)
    tl.store(weights_out + offs, w, mask=mask)
    tl.store(ids_out + offs, e, mask=mask)


@triton.jit
def _clear_unused_rows_kernel(
    a1,
    ids,
    weights,
    recv_token_num,
    hidden,
    topk,
    a1_stride,
    topk_stride,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program per receive row; rows past the received count become inert.
    row = tl.program_id(0)
    if row >= tl.load(recv_token_num):
        offs_k = tl.arange(0, BLOCK_K)
        mask_k = offs_k < topk
        tl.store(
            ids + row * topk_stride + offs_k,
            tl.full([BLOCK_K], -1, tl.int32),
            mask=mask_k,
        )
        tl.store(
            weights + row * topk_stride + offs_k,
            tl.zeros([BLOCK_K], tl.float32),
            mask=mask_k,
        )
        zeros = tl.zeros([BLOCK_H], tl.float32).to(a1.dtype.element_ty)
        for h0 in range(0, hidden, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            tl.store(a1 + row * a1_stride + offs_h, zeros, mask=offs_h < hidden)


def _use_triton(*tensors: torch.Tensor) -> bool:
    return HAS_TRITON and all(t.is_cuda for t in tensors)


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

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts):
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

        # OCP MX (e.g. MXFP4) reports a per-tensor group shape but scales every
        # 1x32 block, so only FP8 / INT8 activations count as per tensor.
        quant_config = fused_experts.quant_config
        if quant_config.is_per_tensor and (
            quant_config.use_fp8_w8a8 or quant_config.use_int8_w8a8
        ):
            raise ValueError(
                "MORI EPv2 does not support per-tensor activation quantization: "
                "the experts would quantize the received rows, which AITER's "
                "per-tensor quant does not support. Set VLLM_MORI_EP_V2=0 to use "
                "MORI EPv1, which quantizes before dispatch."
            )

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

    def _dispatch_args(
        self, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The router's weights and ids in the dtypes MORI dispatch takes."""
        if (
            _use_triton(topk_weights, topk_ids)
            and topk_weights.is_contiguous()
            and topk_ids.is_contiguous()
            and topk_weights.shape == topk_ids.shape
        ):
            # One launch instead of the casts and the padding remap below.
            weights = torch.empty_like(topk_weights, dtype=torch.float32)
            ids = torch.empty_like(topk_ids, dtype=torch.int32)
            numel = topk_ids.numel()
            if numel:
                block = 1024
                _dispatch_args_kernel[(triton.cdiv(numel, block),)](
                    topk_weights,
                    topk_ids,
                    weights,
                    ids,
                    numel,
                    topk_ids.shape[-1],
                    self.rank_expert_offset,
                    self.num_local_experts,
                    BLOCK=block,
                )
            return weights, ids
        return (
            topk_weights.to(torch.float32),
            self._route_padding_locally(topk_ids.to(torch.int32)),
        )

    @staticmethod
    def _clear_unused_rows(
        dispatch_a1: torch.Tensor,
        dispatch_weights: torch.Tensor,
        dispatch_ids: torch.Tensor,
        dispatch_recv_token_num: torch.Tensor,
    ) -> None:
        """Zero the activations and weights of the rows past the received count
        and route them to expert -1."""
        rows = dispatch_a1.shape[0]
        if (
            _use_triton(dispatch_a1, dispatch_ids, dispatch_recv_token_num)
            and dispatch_a1.dim() == 2
            and dispatch_a1.stride(1) == 1
            and dispatch_ids.dim() == 2
            and dispatch_ids.stride(1) == 1
            and dispatch_ids.dtype == torch.int32
            and dispatch_weights.dtype == torch.float32
            and dispatch_weights.stride() == dispatch_ids.stride()
        ):
            # One launch instead of an arange, a compare and three masked fills.
            if rows:
                _clear_unused_rows_kernel[(rows,)](
                    dispatch_a1,
                    dispatch_ids,
                    dispatch_weights,
                    dispatch_recv_token_num,
                    dispatch_a1.shape[1],
                    dispatch_ids.shape[1],
                    dispatch_a1.stride(0),
                    dispatch_ids.stride(0),
                    BLOCK_H=2048,
                    BLOCK_K=triton.next_power_of_2(dispatch_ids.shape[1]),
                )
            return
        unused = (
            torch.arange(rows, device=dispatch_a1.device) >= dispatch_recv_token_num
        ).unsqueeze(1)
        dispatch_a1.masked_fill_(unused, 0)
        dispatch_weights.masked_fill_(unused, 0)
        dispatch_ids.masked_fill_(unused, -1)

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

        weights, ids = self._dispatch_args(topk_weights, topk_ids)
        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
            self._routing,
        ) = self.mori_op.dispatch(a1, weights, None, ids, return_routing=True)
        rows = self._recv_rows_bound(a1, dispatch_a1.shape[0])
        if rows < dispatch_a1.shape[0]:
            dispatch_a1 = dispatch_a1[:rows]
            dispatch_weights = dispatch_weights[:rows]
            dispatch_ids = dispatch_ids[:rows]
            if dispatch_scale is not None:
                dispatch_scale = dispatch_scale[:rows]
        # The experts get no received-row count and read every bounded row, and
        # MORI reuses its buffers, so make the unused rows inert: zero
        # activations and weights, routed to expert -1.
        self._clear_unused_rows(
            dispatch_a1, dispatch_weights, dispatch_ids, dispatch_recv_token_num
        )
        return (
            dispatch_a1,
            dispatch_scale,
            None,
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
