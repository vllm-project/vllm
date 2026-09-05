# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mori
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _is_stream_capturing() -> bool:
    try:
        return (
            current_platform.is_cuda_alike()
            and torch.cuda.is_current_stream_capturing()
        )
    except Exception:
        return False


def _trim_dispatch_output(
    dispatch_a1: torch.Tensor,
    dispatch_scale: torch.Tensor | None,
    dispatch_ids: torch.Tensor,
    dispatch_weights: torch.Tensor,
    dispatch_recv_token_num: torch.Tensor,
    ep_size: int,
    topk: int,
    input_num_tokens: int,
) -> tuple[
    torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor
]:
    orig_rows = dispatch_a1.shape[0]

    if _is_stream_capturing():
        valid_rows = input_num_tokens * topk * ep_size
    elif dispatch_recv_token_num is not None and dispatch_recv_token_num.numel() > 0:
        valid_rows = int(dispatch_recv_token_num.reshape(-1)[0].item())
    else:
        return dispatch_a1, dispatch_scale, dispatch_ids, dispatch_weights

    valid_rows = max(0, min(valid_rows, orig_rows))
    if valid_rows <= 0 or valid_rows >= orig_rows:
        return dispatch_a1, dispatch_scale, dispatch_ids, dispatch_weights

    dispatch_a1 = dispatch_a1[:valid_rows]
    dispatch_ids = dispatch_ids[:valid_rows]
    dispatch_weights = dispatch_weights[:valid_rows]
    if dispatch_scale is not None:
        dispatch_scale = dispatch_scale[:valid_rows]
    return dispatch_a1, dispatch_scale, dispatch_ids, dispatch_weights


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize using MoRI kernels.
    """

    def __init__(
        self,
        mori_op: mori.ops.EpDispatchCombineOp,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
    ):
        super().__init__()
        self.mori_op = mori_op
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch

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
        """
        Returns a tuple of:
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
        # quantization internally, so skip FP8 dispatch quantization.
        if self.use_fp8_dispatch and not defer_input_quant:
            from aiter import QuantType, get_hip_quant

            if quant_config.is_block_quantized:
                quant_func = get_hip_quant(QuantType.per_1x128)
                a1, scale = quant_func(a1, quant_dtype=current_platform.fp8_dtype())
            elif quant_config.is_per_act_token:
                quant_func = get_hip_quant(QuantType.per_Token)
                a1, scale = quant_func(a1, quant_dtype=current_platform.fp8_dtype())

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self.mori_op.dispatch(a1, topk_weights, scale, topk_ids)

        dispatch_a1, dispatch_scale, dispatch_ids, dispatch_weights = (
            _trim_dispatch_output(
                dispatch_a1,
                dispatch_scale,
                dispatch_ids,
                dispatch_weights,
                dispatch_recv_token_num,
                ep_size=self.num_dispatchers_,
                topk=topk_ids.shape[1],
                input_num_tokens=a1.shape[0],
            )
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
        result = self.mori_op.combine(
            fused_expert_output,
            None,
            topk_ids,
        )[0]
        output.copy_(result[:num_token])
