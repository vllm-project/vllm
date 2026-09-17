# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

logger = init_logger(__name__)


class MoriEPv2PrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    def __init__(
        self,
        mori_op: Any,
        max_tokens_per_rank: int,
        num_dispatchers: int,
    ):
        super().__init__()
        self.mori_op = mori_op
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self._routing = None
        self._recv_topk_ids = None

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
        del num_experts, expert_map, quant_config, defer_input_quant
        assert not apply_router_weight_on_input, (
            "MORI EPv2 does not support apply_router_weight_on_input=True"
        )
        if a1.dtype != torch.bfloat16:
            raise ValueError(
                f"MORI EPv2 requires BF16 dispatch input, got {a1.dtype}"
            )

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
            topk_ids.to(torch.int32),
            return_routing=True,
            clone_routing=False,
        )
        self._recv_topk_ids = dispatch_ids
        logger.warning_once(
            "MORI EPv2 dispatch entered: "
            "mori.ops.dispatch_combine_v2.EpDispatchCombineOp.dispatch"
        )
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
        if self._routing is None or self._recv_topk_ids is None:
            raise RuntimeError("MORI EPv2 combine called without a dispatch")
        result = self.mori_op.combine(
            fused_expert_output,
            None,
            self._recv_topk_ids,
            routing=self._routing,
        )[0]
        logger.warning_once(
            "MORI EPv2 combine entered: "
            "mori.ops.dispatch_combine_v2.EpDispatchCombineOp.combine"
        )
        output.copy_(result[: output.shape[0]])
        if not torch.cuda.is_current_stream_capturing():
            self._routing = None
            self._recv_topk_ids = None
