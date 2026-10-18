# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import mori
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform

logger = init_logger(__name__)

# MoRI's intra-node dispatch/combine has a grid-wide barrier: every block
# waits for the last to arrive, so the launch grid is what sets the dispatch
# tail. DeepSeek-style speculative decode (num_speculative_tokens + 1 tokens
# verified per seq) inflates the real decode token count ~7x, so the decode
# batch here is well out of the single-token regime and wants a *wide* grid,
# not a narrow one. Tuned on MI355X (DEP8, DSpark K6): a 128-block x 16-warp
# decode grid is ~4% better TPOT than the handle's 80x16 default, while 192
# blocks regresses. block_num is capped at the CU count so the barrier stays
# resident. All grid dims are env-overridable for re-tuning per platform.
_MORI_WIDE_LAUNCH_MIN_TOKENS = int(os.getenv("VLLM_MORI_WIDE_LAUNCH_MIN_TOKENS", "4096"))
_MORI_DECODE_BLOCKS = int(os.getenv("VLLM_MORI_DECODE_BLOCKS", "128"))
_MORI_DECODE_WARPS = int(os.getenv("VLLM_MORI_DECODE_WARPS", "16"))
_MORI_PREFILL_BLOCKS = int(os.getenv("VLLM_MORI_PREFILL_BLOCKS", "128"))
_MORI_PREFILL_WARPS = int(os.getenv("VLLM_MORI_PREFILL_WARPS", "16"))


def _mori_launch_config(num_tokens: int) -> tuple[int, int]:
    """Return ``(block_num, warp_per_block)`` for one dispatch or combine.

    A narrow grid at decode width, the wide grid once the batch reaches
    prefill size. ``block_num`` is capped at the CU count so the grid-wide
    barrier can be resident. Both grids are overridable via env for tuning.
    """
    from vllm.utils.platform_utils import num_compute_units

    compute_units = num_compute_units()
    if num_tokens >= _MORI_WIDE_LAUNCH_MIN_TOKENS:
        return min(_MORI_PREFILL_BLOCKS, compute_units), _MORI_PREFILL_WARPS
    return min(_MORI_DECODE_BLOCKS, compute_units), _MORI_DECODE_WARPS


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
        block_num, warp_per_block = _mori_launch_config(a1.shape[0])

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self.mori_op.dispatch(
            a1,
            topk_weights,
            scale,
            topk_ids,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )

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
        block_num, warp_per_block = _mori_launch_config(num_token)
        result = self.mori_op.combine(
            fused_expert_output,
            None,
            self._dispatch_topk_ids,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )[0]
        output.copy_(result[:num_token])
