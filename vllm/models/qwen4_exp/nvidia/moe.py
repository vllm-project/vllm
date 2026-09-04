# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp-specific MoE execution."""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
    TrtLlmBf16ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _unpack,
)
from vllm.platforms import current_platform

from .ops.moe import finalize_moe_with_shared

_MAX_DEFERRED_TOKENS = 128
logger = init_logger(__name__)


class Qwen4ExpMoERunner(MoERunner):
    """Fuse the BF16 MoE local finalize with the shared-expert add."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        experts_cls = getattr(self._quant_method, "experts_cls", None)
        self._defer_finalize = (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
            and experts_cls is TrtLlmBf16ExpertsMonolithic
            and self._shared_experts is not None
            and not self._fused_output_is_reduced
            and self.moe_config.dp_size == 1
            and self.moe_config.pcp_size == 1
            and not self.moe_config.is_sequence_parallel
        )
        logger.info_once(
            "Qwen4Exp BF16 deferred MoE tail: enabled=%s experts=%s "
            "shared=%s reduced=%s dp=%d ep=%d tp=%d pcp=%d sp=%s",
            self._defer_finalize,
            getattr(experts_cls, "__name__", experts_cls),
            self._shared_experts is not None,
            self._fused_output_is_reduced,
            self.moe_config.dp_size,
            self.moe_config.ep_size,
            self.moe_config.tp_size,
            self.moe_config.pcp_size,
            self.moe_config.is_sequence_parallel,
        )
        if self._defer_finalize:
            self.moe_config.defer_moe_finalize = True
            self.moe_config.defer_moe_finalize_allow_ep = True
            self.moe_config.defer_moe_finalize_max_num_tokens = _MAX_DEFERRED_TOKENS

        shared_layer = (
            self._shared_experts._layer if self._shared_experts is not None else None
        )
        self._defer_shared_gate = (
            self._defer_finalize
            and not self._shared_experts.enable_dbo
            and hasattr(shared_layer, "enable_deferred_gate")
        )
        if self._defer_shared_gate:
            shared_layer.enable_deferred_gate(_MAX_DEFERRED_TOKENS)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not (
            self._defer_finalize
            and self.moe_config.should_defer_moe_finalize(hidden_states.shape[0])
        ):
            return super().forward(
                hidden_states,
                router_logits,
                input_ids,
                shared_experts_input,
            )

        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )
        hidden_states, pre_trunc, post_trunc = self._maybe_pad_hidden_states(
            shared_experts_input,
            hidden_states,
        )
        if pre_trunc is not None or post_trunc is not None:
            return super().forward(
                hidden_states,
                router_logits,
                input_ids,
                shared_experts_input,
            )

        result = self._forward_impl(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        )
        shared_output, routed_output = _unpack(result)
        assert shared_output is not None
        assert isinstance(routed_output, UnfinalizedMoEOutput)
        shared_gate_logits = None
        if self._defer_shared_gate:
            shared_gate_logits = self._shared_experts._layer.deferred_gate_logits
            assert shared_gate_logits is not None
        result = finalize_moe_with_shared(
            routed_output,
            shared_output,
            shared_gate_logits,
        )
        result = self._maybe_reduce_final_output(
            result,
            trunc_size=None,
            output_is_reduced=False,
        )
        return self._maybe_add_zero_expert_output(result)


__all__ = ["Qwen4ExpMoERunner"]
