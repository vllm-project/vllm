# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import deep_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    swizzle_mxfp8_scale,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
)


def _quantize_before_dispatch(
    quant_config: FusedMoEQuantConfig, defer_input_quant: bool
) -> bool:
    """Do quantized dispatch for blockfp8 and mxfp8, unless the
    subsequent moe kernel requires bf16 inputs.
    """
    if defer_input_quant:
        return False
    return quant_config.is_block_quantized or quant_config.quant_dtype == "mxfp8"


def _pack_mxfp8_scale(scale: torch.Tensor) -> torch.Tensor:
    """Pack row-major [M, K/32] UE8M0 scales into [M, K/128] int32.

    DeepEP moves scale factors as opaque 4-byte packs (`sf_pack_t` is a
    float/UE8M0x4 union), so 1-byte UE8M0 scales must be packed 4-per-int32.
    """
    assert scale.dtype == torch.uint8 and scale.ndim == 2, (
        f"expected 2D uint8 mxfp8 scales, got {scale.shape} {scale.dtype}"
    )
    assert scale.size(1) % 4 == 0, (
        f"mxfp8 dispatch needs hidden_size % {MXFP8_BLOCK_SIZE * 4} == 0, "
        f"got {scale.size(1)} scale columns"
    )
    return scale.contiguous().view(torch.int32)


def _unpack_mxfp8_scale(
    scale: torch.Tensor, hidden_size: int, is_scale_swizzled: bool
) -> torch.Tensor:
    """Inverse of `_pack_mxfp8_scale`, restoring the expert kernel's layout.

    TRTLLM consumes the row-major [M, K/32] scales as-is; CUTLASS wants them
    swizzled into F8_128x4, which can only happen here because the swizzle
    interleaves scales across a 128-row tile (i.e. across tokens).
    """
    scale = scale.contiguous().view(torch.uint8)
    if is_scale_swizzled:
        scale = swizzle_mxfp8_scale(scale, M=scale.size(0), K=hidden_size)
    return scale


class DeepEPV2PrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Prepare/Finalize using DeepEP v2 ElasticBuffer (unified API).

    Uses non-expanded dispatch without CPU synchronization in every forward.
    The receive capacity is bounded by the DP-wide padded token count. Expert
    kernels consume routing IDs and GPU-side receive counts.

    Dispatch always uses async_with_compute_stream=False. finalize_async
    issues the combine with async_with_compute_stream=True (except under
    DBO) so the modular kernel can overlap the shared-expert FFN with the
    combine a2a; the returned receiver joins via a device-side event wait.
    """

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int, dtype: torch.dtype) -> int:
        hidden_size_bytes = hidden_size * dtype.itemsize
        xfer_atom_size = 512  # 32 * 16 (size(int4))
        if hidden_size_bytes % xfer_atom_size == 0:
            return hidden_size

        hidden_size_bytes = round_up(hidden_size_bytes, xfer_atom_size)
        return hidden_size_bytes // dtype.itemsize

    def __init__(
        self,
        buffer: deep_ep.ElasticBuffer,
        num_dispatchers: int,
        dp_size: int,
        rank_expert_offset: int,
        num_experts: int,
        num_topk: int,
        use_fp8_dispatch: bool = False,
        sp_size: int = 1,
    ):
        super().__init__()
        self.buffer = buffer
        self.num_dispatchers_ = num_dispatchers
        self.dp_size = dp_size
        self.rank_expert_offset = rank_expert_offset
        self.num_experts = num_experts
        self.num_topk = num_topk
        self.use_fp8_dispatch = use_fp8_dispatch
        self.sp_size = sp_size

        # DBO microbatching: one handle slot per micro-batch.
        self.handles: list[deep_ep.EPHandle | None] = [None, None]

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
        token_data = tokens
        if token_scales is not None:
            token_data = (tokens, token_scales)

        # Bound worst-case receive rows by the largest DP batch, sharded for SP.
        # Power-of-two buckets limit DeepEP's per-capacity JIT specializations.
        dp_meta = (
            get_forward_context().dp_metadata
            if is_forward_context_available()
            else None
        )
        if dp_meta is not None:
            n = int(dp_meta.num_tokens_across_dp_cpu.max())
            n = cdiv(n, self.sp_size)
        else:
            n = tokens.shape[0]
        num_max_tokens_per_rank = 1 << max(n - 1, 0).bit_length()

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            handle,
            event,
        ) = self.buffer.dispatch(
            x=token_data,
            topk_idx=rank_topk_ids,
            topk_weights=rank_topk_weights,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            do_expand=False,
            do_cpu_sync=False,
            async_with_compute_stream=False,
        )

        a2a_idx = dbo_current_ubatch_id()
        self.handles[a2a_idx] = handle

        return lambda: self._receiver(
            event,
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            handle.psum_num_recv_tokens_per_scaleup_rank,
            a1_scale,
            quant_config,
            defer_input_quant=defer_input_quant,
        )

    def _receiver(
        self,
        event: deep_ep.EventOverlap,
        recv_x: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        recv_topk_idx: torch.Tensor,
        recv_topk_weights: torch.Tensor | None,
        psum_recv_per_rank: torch.Tensor,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> mk.PrepareResultType:
        if event.event is not None:
            event.current_stream_wait()

        if isinstance(recv_x, tuple):
            expert_x, expert_x_scale = recv_x
        else:
            expert_x, expert_x_scale = recv_x, None

        # Dispatch leaves padding rows uninitialized. Convert local expert IDs
        # to global IDs and mask padding before expert kernels build routing.
        recv_topk_idx = _globalize_recv_topk_idx(
            recv_topk_idx,
            psum_recv_per_rank,
            self.rank_expert_offset,
            self.num_experts,
        )
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=None,
            expert_num_tokens_cpu=None,
        )
        expert_tokens_meta.psum_recv_per_rank = psum_recv_per_rank

        if _quantize_before_dispatch(quant_config, defer_input_quant):
            if quant_config.quant_dtype == "mxfp8" and expert_x_scale is not None:
                expert_x_scale = _unpack_mxfp8_scale(
                    expert_x_scale,
                    hidden_size=expert_x.size(-1),
                    is_scale_swizzled=quant_config.is_scale_swizzled,
                )
        elif not defer_input_quant:
            expert_x_scale = None
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = moe_kernel_quantize_input(
                    expert_x,
                    a1_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=False,
                    block_shape=quant_config.block_shape,
                    is_scale_swizzled=quant_config.is_scale_swizzled,
                )

        return (
            expert_x,
            expert_x_scale,
            expert_tokens_meta,
            recv_topk_idx,
            recv_topk_weights,
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

        if _quantize_before_dispatch(quant_config, defer_input_quant):
            # Scales must be row-major [num_tokens, ...] here so each token's
            # scales can be shuffled with it; any swizzling the expert kernel
            # wants is applied post-dispatch in `_receiver`.
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
                is_scale_swizzled=False,
                mx_alignment=quant_config.mx_alignment,
            )
            if a1q_scale is not None and a1q_scale.numel() == 1:
                a1q_scale = a1q_scale.view(1, 1)
            if quant_config.quant_dtype == "mxfp8":
                a1q_scale = _pack_mxfp8_scale(a1q_scale)
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

        if fused_expert_output.dtype != torch.bfloat16:
            raise ValueError(
                f"DeepEP v2 combine requires bfloat16 input, "
                f"got {fused_expert_output.dtype}"
            )

        # DBO drives its own hook/receiver schedule; keep the combine
        # synchronous there (the receiver then only performs the copy).
        combine_async = do_async and not dbo_enabled()
        combined_x, _, event = self.buffer.combine(
            x=fused_expert_output,
            handle=handle,
            topk_weights=None,
            async_with_compute_stream=combine_async,
            allocate_on_comm_stream=combine_async,
        )

        if do_async:
            # The combine ran on DeepEP's comm stream; the modular kernel
            # issues the shared-expert FFN before calling the receiver, which
            # joins via a device-side cudaStreamWaitEvent (no host sync, so
            # this is safe inside a captured region).
            def _receiver():
                if event.event is not None:
                    event.current_stream_wait()
                output.copy_(combined_x, non_blocking=True)

            return _receiver
        else:
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


@triton.jit
def _globalize_recv_topk_idx_kernel(
    topk_idx_ptr,  # [N*topk] local expert IDs (-1 = non-local), modified in place
    psum_ptr,  # [P] per-scaleup-rank recv prefix sum; num_recv = psum[P-1]
    P,
    rank_expert_offset,
    num_experts,
    n_elements,  # N * topk
    topk: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    # num_recv_tokens read on-device (no host sync) -> cudagraph-safe.
    num_recv = tl.load(psum_ptr + P - 1)
    val = tl.load(topk_idx_ptr + offs, mask=mask, other=-1)
    g = val + rank_expert_offset
    row = offs // topk
    # Keep a slot iff: it is a local expert (val >= 0), its global id is in
    # range, and its row is a real received token (< num_recv). Otherwise -1.
    valid = (val >= 0) & (g < num_experts) & (row < num_recv)
    tl.store(topk_idx_ptr + offs, tl.where(valid, g, -1), mask=mask)


def _globalize_recv_topk_idx(
    recv_topk_idx: torch.Tensor,  # [N, topk] local expert IDs, -1 = non-local
    psum_recv_per_rank: torch.Tensor,
    rank_expert_offset: int,
    num_experts: int,
) -> torch.Tensor:
    N, topk = recv_topk_idx.shape
    n = N * topk
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _globalize_recv_topk_idx_kernel[grid](
        recv_topk_idx,
        psum_recv_per_rank,
        psum_recv_per_rank.shape[0],
        rank_expert_offset,
        num_experts,
        n,
        topk=topk,
        BLOCK=BLOCK,
    )
    return recv_topk_idx
