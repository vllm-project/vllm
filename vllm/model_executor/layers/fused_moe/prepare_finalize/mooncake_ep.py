# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
    dbo_maybe_run_recv_hook,
)


MOONCAKE_EP_QUANT_BLOCK_SHAPE = [128, 128]


class MooncakeEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Prepare/Finalize using the Mooncake EP backend."""

    def __init__(
        self,
        buffer,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
    ):
        super().__init__()
        self.buffer = buffer
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_dispatchers_ = num_dispatchers
        self.use_fp8_dispatch = use_fp8_dispatch
        self.handles: list[tuple | None] = [None, None]

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def supports_async(self) -> bool:
        return True

    def _check_supported(
        self, a1: torch.Tensor, quant_config: FusedMoEQuantConfig
    ):
        if a1.dtype != torch.bfloat16:
            raise NotImplementedError(
                "Mooncake EP currently supports BF16 activations"
            )
        if quant_config.quant_dtype is not None and not self.use_fp8_dispatch:
            raise NotImplementedError(
                "Mooncake EP supports BF16 and blockwise FP8 dispatch only"
            )

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
    ) -> tuple[Callable | None, mk.ReceiverType]:
        if defer_input_quant:
            raise NotImplementedError(
                "Mooncake EP does not support deferred input quantization"
            )
        self._check_supported(a1, quant_config)
        if apply_router_weight_on_input:
            if topk_ids.size(1) != 1:
                raise NotImplementedError("Mooncake EP input weighting supports top-k=1")
            a1 = a1 * topk_weights.to(a1.dtype)

        a2a_idx = dbo_current_ubatch_id()
        active_ranks = torch.ones(
            self.num_dispatchers_, device=a1.device, dtype=torch.int32
        )
        expert_x, expert_num_tokens, handle, _, hook = self.buffer.dispatch(
            a1,
            topk_ids.to(dtype=torch.int64).contiguous(),
            active_ranks,
            self.max_tokens_per_rank,
            num_experts,
            timeout_us=-1,
            use_fp8=self.use_fp8_dispatch,
            async_finish=False,
            return_recv_hook=True,
        )
        self.handles[a2a_idx] = handle

        return hook, lambda: self._receiver(expert_x, expert_num_tokens)

    def _receiver(
        self,
        expert_x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        expert_num_tokens: torch.Tensor,
    ) -> mk.PrepareResultType:
        if self.use_fp8_dispatch:
            assert isinstance(expert_x, tuple)
            expert_x, expert_x_scale = expert_x
        else:
            assert isinstance(expert_x, torch.Tensor)
            expert_x_scale = None
        metadata = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
        )
        return expert_x, expert_x_scale, metadata, None, None

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
        if hook is not None:
            hook()
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
    ) -> tuple[Callable, Callable]:
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate)
        a2a_idx = dbo_current_ubatch_id()
        do_recv_hook = dbo_enabled() or do_async
        handle = self.handles[a2a_idx]
        assert handle is not None
        combine_weights = (
            torch.ones_like(topk_weights)
            if apply_router_weight_on_input
            else topk_weights
        )
        active_ranks = torch.ones(
            self.num_dispatchers_,
            device=fused_expert_output.device,
            dtype=torch.int32,
        )
        dbo_maybe_run_recv_hook()
        _, _, recv_hook = self.buffer.combine(
            fused_expert_output,
            topk_ids.to(dtype=torch.int64).contiguous(),
            combine_weights.contiguous(),
            active_ranks,
            timeout_us=-1,
            handle=handle,
            zero_copy=False,
            async_finish=False,
            return_recv_hook=do_recv_hook,
            out=output,
        )
        return recv_hook, lambda: None

    def finalize_async(
        self,
        output,
        fused_expert_output,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,
        weight_and_reduce_impl,
    ):
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
        output,
        fused_expert_output,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,
        weight_and_reduce_impl,
    ) -> None:
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=False,
        )
