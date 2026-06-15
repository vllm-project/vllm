# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform


def _stage_combine_input(op, expert_out: torch.Tensor, combine_dtype: torch.dtype):
    if expert_out.dtype != combine_dtype:
        src = expert_out.to(combine_dtype)
    else:
        src = expert_out
    if getattr(getattr(op, "cfg", None), "zero_copy", False):
        cb = op.get_registered_combine_input_buffer(combine_dtype)
        n = src.shape[0]
        if n > 0:
            cb[:n].copy_(src)
        return cb
    return src if src.is_contiguous() else src.contiguous()


class FlydslEpPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Prepare/Finalize using FlyDSL intranode EP dispatch/combine."""

    def __init__(
        self,
        fly_op,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
    ):
        super().__init__()
        self.fly_op = fly_op
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        self._dispatch_indices: torch.Tensor | None = None
        self._cur_tok: int = 0
        self.combine_dtype = torch.bfloat16

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
        assert not apply_router_weight_on_input, (
            "flydsl_ep does not support apply_router_weight_on_input=True now."
        )
        scale = None
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
        ) = self.fly_op.dispatch(a1, topk_weights, scale, topk_ids)

        self._dispatch_indices = dispatch_ids
        self._cur_tok = a1.shape[0]

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
        assert self._dispatch_indices is not None
        comb_inp = _stage_combine_input(
            self.fly_op, fused_expert_output, self.combine_dtype
        )
        out_tok, _ = self.fly_op.combine(
            comb_inp,
            None,
            self._dispatch_indices,
            cur_tok=self._cur_tok,
        )
        output.copy_(out_tok[:num_token])
