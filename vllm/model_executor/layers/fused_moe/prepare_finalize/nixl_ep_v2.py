# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from nixl.runtime import ep

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import (
    expanded_recv_topk_ids,
    moe_kernel_quantize_input,
)
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import nvfp4_block_scale_interleave
from vllm.utils.math_utils import cdiv, next_power_of_2, round_up

logger = init_logger(__name__)


class NixlEPV2PrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Prepare/Finalize on the NIXL EP v2 (nixl.runtime.ep) ElasticBuffer.

    The receive layout is fixed per process:
      - use_cudagraph=True: every call takes the padded layout, with no CPU
        sync. The rows are the buffer's maximum world times a power-of-two
        token bucket, one per received token, so a captured graph survives a
        topology change. Rows past the received count are padding, with
        expert ID -1.
      - use_cudagraph=True and expand=True: the expanded layout with no CPU
        sync, sized for the worst case: the padded layout's rows times
        min(topk, local experts). Rows past the received count are padding,
        with expert ID -1.
      - use_cudagraph=False: every call takes the exact, expanded layout, one
        row per received (token, local expert) pair, grouped by expert.

    Activations are dispatched in BF16, or in block FP8 for block-FP8 layers.
    NVFP4 layers dispatch BF16 and quantize the received rows, or with
    nvfp4_dispatch quantize before the dispatch and send the NVFP4 codes and
    row-major scales, swizzled after receive if the experts want them so.
    Either way, layers whose experts quantize their own input dispatch BF16.
    DBO is not supported: there is one handle slot.

    Args:
        buffer: The process's ElasticBuffer, from the all2all manager.
        num_dispatchers: The EP group size.
        rank_expert_offset: The global ID of this rank's first expert.
        num_sms: The active SMs per dispatch: the launch grid unless
            VLLM_NIXL_EP_V2_NUM_SMS. Fixed per process, so only the first
            dispatch changes the topology descriptor.
        use_cudagraph: Dispatch with no CPU sync on every call, in the padded
            layout unless expand.
        sp_size: The sequence-parallel size; each EP rank holds a 1/sp_size
            shard of its DP rank's tokens.
        expand: Under use_cudagraph, take the expanded layout instead of the
            padded one.
        nvfp4_dispatch: Quantize NVFP4 layers' input before the dispatch
            instead of after it.

    """

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int) -> int:
        """Round the hidden size up to one the combine accepts.

        Args:
            hidden_size: The layer's hidden size.

        Returns:
            The hidden size rounded up to a multiple of 256: the combine
            kernel asserts it for its BF16 rows.

        """
        return round_up(hidden_size, 256)

    def __init__(
        self,
        buffer: ep.ElasticBuffer,
        num_dispatchers: int,
        rank_expert_offset: int,
        num_sms: int,
        use_cudagraph: bool,
        sp_size: int = 1,
        expand: bool = False,
        nvfp4_dispatch: bool = False,
    ):
        super().__init__()
        self.buffer = buffer
        self.num_dispatchers_ = num_dispatchers
        self.rank_expert_offset = rank_expert_offset
        self.num_sms = num_sms
        self.use_cudagraph = use_cudagraph
        self.sp_size = sp_size
        self.expand = expand
        self.nvfp4_dispatch = nvfp4_dispatch
        self.handle: ep.EPHandle | None = None
        # arange(num_local_experts) + rank_expert_offset, built on first use.
        self._expert_ids: torch.Tensor | None = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def topk_indices_dtype(self) -> torch.dtype | None:
        return ep.topk_idx_t

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    def supports_async(self) -> bool:
        return True

    def _token_bucket(self, num_tokens: int) -> int:
        # Every rank must pass the same size, and each size is a JIT variant:
        # take the next power of two of the largest rank's tokens.
        dp_metadata = get_forward_context().dp_metadata
        if dp_metadata is not None:
            num_tokens = cdiv(
                int(dp_metadata.num_tokens_across_dp_cpu.max()), self.sp_size
            )
        return min(next_power_of_2(num_tokens), self.buffer.num_max_tokens_per_rank)

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
            assert topk_ids.size(1) == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None] = a1
        quantize_after = False
        swizzle_scales = False
        wire = "bf16"
        if quant_config.quant_dtype == "nvfp4":
            if self.nvfp4_dispatch and not defer_input_quant:
                # One global scale for every rank: the oracle repeats one amax
                # per expert, and the experts GEMM folds the same a1_gscale in.
                codes, scales = moe_kernel_quantize_input(
                    a1,
                    quant_config.a1_gscale,
                    quant_dtype="nvfp4",
                    per_act_token_quant=False,
                    is_scale_swizzled=False,
                )
                assert scales is not None
                # Row-major [tokens, hidden // 16] e4m3: swizzling pads rows.
                x = (codes, scales.view(torch.float8_e4m3fn))
                swizzle_scales = quant_config.is_scale_swizzled
                wire = "nvfp4"
            else:
                # Otherwise NVFP4 is quantized after receive, as deepep_v2
                # does, unless the experts quantize their own input.
                quantize_after = not defer_input_quant
        elif quant_config.quant_dtype is not None:
            # Weight-only schemes have no quant_dtype and dispatch BF16.
            if (
                not (quant_config.use_fp8_w8a8 and quant_config.is_block_quantized)
                or defer_input_quant
            ):
                raise NotImplementedError(
                    "nixl_ep_v2 dispatches BF16, block-FP8 or NVFP4 activations "
                    f"only, got quant_dtype={quant_config.quant_dtype}, "
                    f"block_shape={quant_config.block_shape}, "
                    f"defer_input_quant={defer_input_quant}"
                )
            x = moe_kernel_quantize_input(
                a1,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=False,
                block_shape=quant_config.block_shape,
                is_scale_swizzled=False,
            )
            wire = "fp8"
        if self.nvfp4_dispatch and wire != "nvfp4":
            logger.info_once(
                "VLLM_NIXL_EP_V2_NVFP4_DISPATCH has no effect on layers with "
                "quant_dtype=%s, defer_input_quant=%s",
                quant_config.quant_dtype,
                defer_input_quant,
            )
        logger.info_once("nixl_ep_v2 dispatch wire: %s", wire)

        recv_x, recv_topk_idx, recv_topk_weights, handle, _ = self.buffer.dispatch(
            x=x,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            num_max_tokens_per_rank=self._token_bucket(len(a1))
            if self.use_cudagraph
            else None,
            num_sms=self.num_sms,
            do_cpu_sync=not self.use_cudagraph,
            do_expand=self.expand or not self.use_cudagraph,
        )
        self.handle = handle
        post_quant_config = quant_config if quantize_after else None
        return lambda: self._receiver(
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            handle,
            post_quant_config,
            swizzle_scales,
        )

    def _receiver(
        self,
        recv_x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        recv_topk_idx: torch.Tensor | None,
        recv_topk_weights: torch.Tensor,
        handle: ep.EPHandle,
        post_quant_config: FusedMoEQuantConfig | None,
        swizzle_scales: bool,
    ) -> mk.PrepareResultType:
        expert_x, expert_x_scale = (
            recv_x if isinstance(recv_x, tuple) else (recv_x, None)
        )

        if recv_topk_idx is None and handle.recv_count_is_exact:
            # Expanded: rows grouped by local expert, counts known on the host.
            counts = handle.num_recv_tokens_per_expert_list
            meta = mk.ExpertTokensMetadata.make_from_list(counts, expert_x.device)
            if self._expert_ids is None:
                self._expert_ids = (
                    torch.arange(
                        len(counts), dtype=ep.topk_idx_t, device=expert_x.device
                    )
                    + self.rank_expert_offset
                )
            topk_ids = torch.repeat_interleave(
                self._expert_ids, meta.expert_num_tokens, output_size=len(expert_x)
            ).unsqueeze(1)
            topk_weights = recv_topk_weights.unsqueeze(1)
        elif recv_topk_idx is None:
            # Expanded for the worst case: the counts stay on the device. The
            # last element of the per-expert prefix sum is the received rows.
            psum = handle.psum_num_recv_tokens_per_expert
            meta = mk.ExpertTokensMetadata(None, None, psum_recv_per_rank=psum)
            topk_ids = expanded_recv_topk_ids(
                psum, len(expert_x), self.rank_expert_offset, ep.topk_idx_t
            )
            topk_weights = recv_topk_weights.unsqueeze(1)
        else:
            # Padded: the counts stay on the device. The last element of the
            # per-rank prefix sum is the received total.
            psum = handle.psum_num_recv_tokens_per_scaleup_rank
            meta = mk.ExpertTokensMetadata(None, None, psum_recv_per_rank=psum)
            topk_ids = _globalize_recv_topk_idx(
                recv_topk_idx, psum[-1:], self.rank_expert_offset
            )
            topk_weights = recv_topk_weights

        if post_quant_config is not None and expert_x.numel() != 0:
            expert_x, expert_x_scale = moe_kernel_quantize_input(
                expert_x,
                post_quant_config.a1_gscale,
                quant_dtype=post_quant_config.quant_dtype,
                per_act_token_quant=False,
                is_scale_swizzled=post_quant_config.is_scale_swizzled,
            )
        elif swizzle_scales and expert_x.numel() != 0:
            # Every layout's padding rows get scales too, ignored like their
            # codes.
            assert expert_x_scale is not None
            expert_x_scale = nvfp4_block_scale_interleave(
                expert_x_scale.view(torch.uint8)
            )

        return expert_x, expert_x_scale, meta, topk_ids, topk_weights

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
        return self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )()

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> Callable[[], None]:
        # Drop the slot's reference, so the handle is freed with the combine.
        handle, self.handle = self.handle, None
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
                "nixl_ep_v2 combine requires bfloat16 input, got "
                f"{fused_expert_output.dtype}"
            )

        # The combine sums rows without weights: the experts applied them.
        combined_x, _, event = self.buffer.combine(
            x=fused_expert_output,
            handle=handle,
            async_with_compute_stream=do_async,
            allocate_on_comm_stream=do_async,
        )

        def receiver() -> None:
            if event.event is not None:
                event.current_stream_wait()
            output.copy_(combined_x)

        return receiver

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        return self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=True,
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
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=False,
        )()


@triton.jit
def _globalize_recv_topk_idx_kernel(
    topk_idx_ptr,  # [rows * topk] local expert IDs, -1 = not local; in place
    num_recv_ptr,  # [1] received rows, read on the device (graph-safe)
    rank_expert_offset,
    n_elements,
    topk: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    num_recv = tl.load(num_recv_ptr)
    local = tl.load(topk_idx_ptr + offs, mask=mask, other=-1)
    # Rows past num_recv are uninitialized and could alias valid IDs.
    valid = (local >= 0) & (offs // topk < num_recv)
    global_id = tl.where(valid, local + rank_expert_offset, -1)
    tl.store(topk_idx_ptr + offs, global_id, mask=mask)


def _globalize_recv_topk_idx(
    topk_idx: torch.Tensor, num_recv: torch.Tensor, rank_expert_offset: int
) -> torch.Tensor:
    """Map the padded layout's local expert IDs to global IDs, and every
    non-local slot and padding row to -1, in place."""
    n_elements = topk_idx.numel()
    block = 1024
    _globalize_recv_topk_idx_kernel[(triton.cdiv(n_elements, block),)](
        topk_idx,
        num_recv,
        rank_expert_offset,
        n_elements,
        topk=topk_idx.size(1),
        BLOCK=block,
    )
    return topk_idx
