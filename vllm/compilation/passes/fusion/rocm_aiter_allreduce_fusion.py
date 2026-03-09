# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ROCm AITER AllReduce + RMSNorm fusion pass.

Replaces the pattern:
    allreduce(input) -> rmsnorm(result, weight)
with a single fused custom op that performs allreduce followed by
AITER RMSNorm in one graph node, eliminating the intermediate tensor.

This is the ROCm counterpart of the FlashInfer-based AllReduceFusionPass
used on NVIDIA GPUs. It supports two patterns:
  1. allreduce + rmsnorm (no residual) — first transformer block
  2. allreduce + fused_add_rmsnorm (with residual) — subsequent blocks

References:
  - GitHub Issue #35712: Enable AITER fused_allreduce_rmsnorm
  - GitHub Issue #35713: Enable AITER fused_allreduce_rmsnorm_quant
"""

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherFusedAddRMSNorm, MatcherRMSNorm

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Custom op: fused allreduce + rmsnorm for ROCm via AITER
# ---------------------------------------------------------------------------
# The op performs:
#   1. allreduce(allreduce_in) across the TP group
#   2. rmsnorm (with optional residual add) using AITER kernels
#
# Mutation semantics (mirroring FlashInfer's fused op):
#   - allreduce_in: overwritten with allreduce result (no-residual)
#                   or rmsnorm result (with-residual)
#   - residual:     overwritten with allreduce_out + old_residual
#                   (with-residual) or allreduce_out (no-residual)
#   - norm_out:     if not None, holds rmsnorm result (no-residual pattern)
# ---------------------------------------------------------------------------

_AITER_RMSNORM_OP = rocm_aiter_ops.get_rmsnorm_op()
_AITER_FUSED_ADD_RMSNORM_OP = rocm_aiter_ops.get_rmsnorm_fused_add_op()


def _call_aiter_fused_allreduce_rmsnorm(
    allreduce_in: torch.Tensor,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    norm_out: torch.Tensor | None = None,
) -> None:
    tp_group = get_tp_group()
    ar_result = tp_group.all_reduce(allreduce_in)

    if norm_out is not None:
        # No-residual pattern (first transformer block):
        #   norm_out = rmsnorm(allreduce_out, weight, eps)
        #   allreduce_in <- allreduce_out   (for consumers that read it)
        #   residual <- allreduce_out       (unused zero-init'd tensor)
        rmsnorm_result = _AITER_RMSNORM_OP(
            x=ar_result, weight=rms_gamma, variance_epsilon=rms_eps
        )
        norm_out.copy_(rmsnorm_result)
        allreduce_in.copy_(ar_result)
        residual.copy_(ar_result)
    else:
        # With-residual pattern (subsequent transformer blocks):
        #   new_residual = allreduce_out + old_residual
        #   allreduce_in <- rmsnorm(new_residual, weight, eps)
        #   residual <- new_residual
        rmsnorm_result, new_residual = _AITER_FUSED_ADD_RMSNORM_OP(
            x=ar_result,
            residual=residual,
            weight=rms_gamma,
            variance_epsilon=rms_eps,
        )
        allreduce_in.copy_(rmsnorm_result)
        residual.copy_(new_residual)


def _call_aiter_fused_allreduce_rmsnorm_fake(
    allreduce_in: torch.Tensor,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    norm_out: torch.Tensor | None = None,
) -> None:
    pass


direct_register_custom_op(
    op_name="rocm_aiter_fused_allreduce_rmsnorm",
    op_func=_call_aiter_fused_allreduce_rmsnorm,
    mutates_args=["allreduce_in", "residual", "norm_out"],
    fake_impl=_call_aiter_fused_allreduce_rmsnorm_fake,
)

_FUSED_AR_RMSNORM_OP = torch.ops.vllm.rocm_aiter_fused_allreduce_rmsnorm.default


# ---------------------------------------------------------------------------
# Pattern classes
# ---------------------------------------------------------------------------


class RocmAiterAllReduceRMSNormPattern:
    """
    Fuses allreduce + rmsnorm (without residual).
    Applies to the first Transformer block where there is no prior residual.
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        match_rocm_aiter: bool = True,
    ) -> None:
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device
        self.rmsnorm_matcher = MatcherRMSNorm(
            epsilon, match_rocm_aiter=match_rocm_aiter
        )

    def get_inputs(self) -> list[torch.Tensor]:
        input_t, weight = self.rmsnorm_matcher.inputs()
        return [input_t.to(self.dtype), weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms = self.rmsnorm_matcher(allreduce_output, weight)
            return rms, allreduce_output

        def replacement(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual = torch.zeros_like(input)
            rms_result = torch.empty_like(input)
            ar = auto_functionalized(
                _FUSED_AR_RMSNORM_OP,
                allreduce_in=input,
                residual=residual,
                norm_out=rms_result,
                rms_gamma=weight,
                rms_eps=self.epsilon,
            )
            # norm_out, allreduce_in
            return ar[3], ar[1]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class RocmAiterAllReduceFusedAddRMSNormPattern:
    """
    Fuses allreduce + fused_add_rmsnorm (with residual).
    Applies to o_proj + rmsnorm after attention and mlp + rmsnorm before
    attention in subsequent Transformer blocks.
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        match_rocm_aiter: bool = True,
    ) -> None:
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device
        self.rmsnorm_matcher = MatcherFusedAddRMSNorm(
            epsilon, match_rocm_aiter=match_rocm_aiter
        )

    def get_inputs(self) -> list[torch.Tensor]:
        input_t, residual, weight = self.rmsnorm_matcher.inputs()
        return [residual, input_t.to(self.dtype), weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms, residual = self.rmsnorm_matcher(allreduce_output, weight, residual)
            return rms, residual

        def replacement(
            residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            ar = auto_functionalized(
                _FUSED_AR_RMSNORM_OP,
                allreduce_in=input,
                residual=residual,
                norm_out=None,
                rms_gamma=weight,
                rms_eps=self.epsilon,
            )
            # allreduce_in (norm result), residual
            return ar[1], ar[2]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )

        # Same pattern but only returning the first element (end-of-graph
        # where residual is unused)
        first_return_only = lambda fn: lambda a, b, c: fn(a, b, c)[0]

        pm.register_replacement(
            first_return_only(pattern),  # type: ignore[no-untyped-call]
            first_return_only(replacement),  # type: ignore[no-untyped-call]
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
        )


# ---------------------------------------------------------------------------
# Fusion pass
# ---------------------------------------------------------------------------


class RocmAiterAllReduceFusionPass(VllmPatternMatcherPass):
    """
    Graph-level fusion of allreduce + RMSNorm on ROCm using AITER kernels.

    At the graph level this eliminates the intermediate tensor between
    allreduce and rmsnorm.  The actual compute uses vLLM's TP-group
    allreduce followed by AITER's CK-based RMSNorm kernel.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.disabled = True

        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size <= 1:
            logger.warning_once(
                "ROCm AITER AllReduce fusion pass disabled for tp_size <= 1."
            )
            return

        if not rocm_aiter_ops.is_enabled():
            logger.warning("AITER is not enabled; skipping ROCm allreduce fusion pass.")
            return

        if not rocm_aiter_ops.is_rmsnorm_enabled():
            logger.warning(
                "AITER RMSNorm is not enabled; skipping ROCm allreduce fusion pass."
            )
            return

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_allreduce_rmsnorm_fusion_pass"
        )

        self.register_patterns()
        self.dump_patterns(config, self.patterns)

    @enable_fake_mode
    def register_patterns(self) -> None:
        use_aiter_rmsnorm = rocm_aiter_ops.is_rmsnorm_enabled()
        for epsilon in [1e-5, 1e-6]:
            for match_aiter in [True, False] if use_aiter_rmsnorm else [False]:
                RocmAiterAllReduceRMSNormPattern(
                    epsilon,
                    self.model_dtype,
                    self.device,
                    match_rocm_aiter=match_aiter,
                ).register(self.patterns)
                RocmAiterAllReduceFusedAddRMSNormPattern(
                    epsilon,
                    self.model_dtype,
                    self.device,
                    match_rocm_aiter=match_aiter,
                ).register(self.patterns)

            torch._inductor.pattern_matcher._seen_patterns.clear()

        self.disabled = False

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        if self.disabled:
            logger.warning_once("ROCm AITER AllReduce fusion pass is disabled.")
            return False
        return True

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        if self.disabled:
            logger.debug("RocmAiterAllReduceFusionPass disabled")
            return

        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
        fusion_patterns = [
            RocmAiterAllReduceRMSNormPattern,
            RocmAiterAllReduceFusedAddRMSNormPattern,
        ]
        return self.hash_source(self, *fusion_patterns)
