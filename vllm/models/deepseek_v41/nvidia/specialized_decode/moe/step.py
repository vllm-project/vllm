# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MoE of a specialized decode step.

Every MoE layer whose experts can stop after GEMM2 leaves its routed top-k
reduction, the shared-expert add and the all-reduce to the next boundary's
fused kernel, including the layers before an Engram layer and the last one,
whose boundaries take the kernel's residual-only variant:

- up to ``LOW_BATCH_MAX_ROWS`` rows, on the low-batch routed-expert GEMMs
  (``routed_expert_gemm``);
- above that, on the runner's own ``_forward_impl`` with the deferral raised to
  the step's rows for the call (``deferred_finalize``).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
    TrtLlmMxfp4ExpertsModular,
    TrtLlmMxfp4Mxfp8ExpertsModular,
)
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import _unpack
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExpertsOrder,
)

from .routed_expert_gemm import RoutedExpertGemm, front_applies

if TYPE_CHECKING:
    from ...model import DeepseekV4MoE
    from ..runtime.decode import DecodeStep


@contextmanager
def deferred_finalize(config: FusedMoEConfig, max_num_tokens: int) -> Iterator[None]:
    """Within the block, let the experts leave the top-k reduction of calls of
    up to ``max_num_tokens`` tokens to the caller; then restore the layer's own
    deferral, which the stock forward reads."""
    saved = config._defer_moe_finalize, config.defer_moe_finalize_max_num_tokens
    config._defer_moe_finalize = True
    config.defer_moe_finalize_max_num_tokens = max_num_tokens
    try:
        yield
    finally:
        config._defer_moe_finalize, config.defer_moe_finalize_max_num_tokens = saved


def defers_finalize(ffn: "DeepseekV4MoE", max_num_tokens: int) -> bool:
    """Whether ``ffn``'s experts can leave the top-k reduction to the fused
    all-reduce + mHC: the modular TRT-LLM MXFP4 experts without routed scaling,
    on a TP-only deployment."""
    runner = ffn.experts
    experts_cls = getattr(runner.routed_experts.quant_method, "experts_cls", None)
    if (
        experts_cls not in (TrtLlmMxfp4ExpertsModular, TrtLlmMxfp4Mxfp8ExpertsModular)
        or runner.routed_scaling_factor != 1.0
        or ffn.use_native_mega_moe
    ):
        return False
    with deferred_finalize(runner.moe_config, max_num_tokens):
        return runner.moe_config.should_defer_moe_finalize(max_num_tokens)


class MoEStep:
    """One MoE layer of the decode step."""

    def __init__(
        self, ffn: "DeepseekV4MoE", gemm: RoutedExpertGemm | None, max_rows: int
    ) -> None:
        self.ffn = ffn
        self.runner = ffn.experts
        self.deferrable = defers_finalize(ffn, max_rows)
        # The low-batch GEMMs leave the top-k reduction open too.
        self.gemm = gemm if self.deferrable else None

    def run(
        self, x: torch.Tensor, input_ids: torch.Tensor | None, step: "DecodeStep"
    ) -> torch.Tensor | MoEOutput:
        """The layer's TP-partial output, unfinalized where it can defer."""
        if not self.deferrable:
            return self.ffn(x, input_ids, None)
        gemm = self.gemm
        if step.low_batch and gemm is not None and gemm.accepts(x, self.runner):
            routed, shared_output = self._forward_low_batch(gemm, x, input_ids)
            return MoEOutput(routed=routed, shared_output=shared_output)
        # The runner's custom op returns tensors only, so run its body directly.
        with deferred_finalize(self.runner.moe_config, x.shape[0]):
            shared_output, routed = _unpack(
                self.runner._forward_impl(x, x, x, input_ids)
            )
        assert shared_output is not None
        if isinstance(routed, UnfinalizedMoEOutput):
            return MoEOutput(routed=routed, shared_output=shared_output)
        # Experts that split the call across launches finalize it.
        return shared_output + routed

    def _forward_low_batch(
        self,
        gemm: RoutedExpertGemm,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[UnfinalizedMoEOutput, torch.Tensor]:
        """The runner's ``_forward_impl`` with the routed experts on the
        low-batch GEMMs; the gate and the shared experts run as the runner runs
        them. Returns the unfinalized routed output and the shared output."""
        runner = self.runner
        shared = runner._shared_experts
        assert shared is not None
        runner.routed_experts._ensure_moe_quant_config_init()
        overlapping = shared.maybe_forward_async(hidden_states)
        router_logits, _ = runner.gate(hidden_states)
        runner._maybe_apply_shared_experts(hidden_states, SharedExpertsOrder.NO_OVERLAP)
        if front_applies(gemm, runner):
            routing = gemm.front(hidden_states, router_logits, input_ids)
        else:
            topk_weights, topk_ids = runner.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=runner._quant_method.topk_indices_dtype,
                input_ids=input_ids,
            )
            routing = gemm.route(hidden_states, topk_ids, topk_weights)
        routed = gemm.gemms(routing)
        if overlapping:
            shared.wait()
        # Read once: the property hands the output over and clears it.
        return routed, shared.output
