# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mori
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform

logger = init_logger(__name__)


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Prepare/Finalize using MoRI kernels."""

    def __init__(
        self,
        mori_op: mori.ops.EpDispatchCombineOp,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
        use_fp4_dispatch: bool = False,
    ):
        super().__init__()
        assert not (use_fp8_dispatch and use_fp4_dispatch)
        self.mori_op = mori_op
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        self.use_fp4_dispatch = use_fp4_dispatch
        self._dispatch_topk_ids: torch.Tensor | None = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

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

    def _num_live_dispatch_rows(self, num_local_tokens: int, num_rows: int) -> int:
        """Rows of the dispatch output that can hold a received token.

        dispatch() returns MoRI's whole receive buffer, sized
        num_dispatchers * max_tokens_per_rank, and the modular kernel sizes
        the expert GEMMs from its row count. Each source rank sends this rank
        at most one copy of each of its tokens, and received tokens are packed
        from row 0, so only num_dispatchers * (max tokens on any rank) rows
        can be live. The bound has to cover every source rank, so it comes
        from the DP token counts rather than this rank's batch alone.
        """
        if not is_forward_context_available():
            return num_rows
        dp_metadata = get_forward_context().dp_metadata
        if dp_metadata is None:
            return num_rows
        max_tokens = max(
            int(dp_metadata.num_tokens_across_dp_cpu.max()), num_local_tokens
        )
        return min(self.num_dispatchers_ * max_tokens, num_rows)

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
        """Returns a tuple of:
        - quantized + dispatched a.
        - Optional quantized + dispatched a1_scales.
        - Optional ExpertTokensMetadata containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weight
        """
        assert not apply_router_weight_on_input, (
            "mori does not support apply_router_weight_on_input=True now."
        )
        scale = None
        # When defer_input_quant is True, the expert kernel handles
        # quantization internally, so skip dispatch quantization.
        if self.use_fp8_dispatch and not defer_input_quant:
            from aiter import QuantType, get_hip_quant

            if quant_config.is_block_quantized:
                quant_func = get_hip_quant(QuantType.per_1x128)
                a1, scale = quant_func(a1, quant_dtype=current_platform.fp8_dtype())
            elif quant_config.is_per_act_token:
                quant_func = get_hip_quant(QuantType.per_Token)
                a1, scale = quant_func(a1, quant_dtype=current_platform.fp8_dtype())
            elif quant_config.is_per_tensor:
                quant_func = get_hip_quant(QuantType.per_Tensor)
                a1, scale = quant_func(
                    a1,
                    scale=quant_config.a1_scale,
                    quant_dtype=current_platform.fp8_dtype(),
                )
                # mori expects one scale slot per token; broadcast.
                scale = scale.expand(a1.shape[0], 1).contiguous()
        elif self.use_fp4_dispatch and not defer_input_quant:
            from aiter import QuantType, get_hip_quant

            # Packed MXFP4 [M, K // 2] with e8m0 scales [M, K // 32]. The
            # scales stay unshuffled: AITER fused_moe sorts them itself
            # when it receives pre-quantized FP4 input.
            quant_func = get_hip_quant(QuantType.per_1x32)
            a1, scale = quant_func(a1, shuffle=False)

        # mori's combine() reduces over this rank's own [num_tokens, topk]
        # routing. The modular kernel rebinds topk_ids to the dispatched ids
        # returned below before it calls finalize(), so hold a reference to
        # the pre-dispatch tensor (ROCm/mori#475).
        self._dispatch_topk_ids = topk_ids

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self.mori_op.dispatch(a1, topk_weights, scale, topk_ids)

        num_rows = self._num_live_dispatch_rows(a1.shape[0], dispatch_a1.shape[0])
        if num_rows < dispatch_a1.shape[0]:
            dispatch_a1 = dispatch_a1[:num_rows]
            dispatch_ids = dispatch_ids[:num_rows]
            if dispatch_weights is not None:
                dispatch_weights = dispatch_weights[:num_rows]
            if dispatch_scale is not None:
                dispatch_scale = dispatch_scale[:num_rows]

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_token_num, expert_num_tokens_cpu=None
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
        num_token = output.shape[0]
        assert self._dispatch_topk_ids is not None, (
            "finalize() called before prepare(); mori combine() needs the "
            "routing tensor that dispatch() was given"
        )
        result = self.mori_op.combine(
            fused_expert_output,
            None,
            self._dispatch_topk_ids,
        )[0]
        output.copy_(result[:num_token])
