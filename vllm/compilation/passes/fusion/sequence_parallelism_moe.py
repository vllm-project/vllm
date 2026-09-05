# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inductor graph rewrites for sequence-parallel MoE blocks."""

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir.ops
from vllm.compilation.passes.fusion.sequence_parallelism import (
    _SequenceParallelPatternHelper,
    get_first_out_wrapper,
)
from vllm.compilation.passes.vllm_inductor_pass import (
    VllmInductorPass,
    VllmPatternMatcherPass,
)
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import init_logger

from ..inductor_pass import enable_fake_mode
from ..utility.noop_elimination import NoOpEliminationPass

logger = init_logger(__name__)


def _sequence_parallel_chunk(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.sequence_parallel_chunk_impl(x)


class _MoEPatternHelper(_SequenceParallelPatternHelper):
    """Pattern helpers shared by the MoE sequence-parallel rewrites."""

    def get_inputs(self) -> list[torch.Tensor]:
        return [self.empty([8, 16]), self.empty([16])]


class AllReduceRMSNormChunkPattern(_MoEPatternHelper):
    """Remove the redundant all-reduce and chunk around an MoE router."""

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            all_reduce = self._all_reduce(input)
            rms_norm = vllm.ir.ops.rms_norm(all_reduce, weight, self.epsilon)
            return _sequence_parallel_chunk(rms_norm)

        def replacement(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            reduce_scatter = self._reduce_scatter(input)
            return vllm.ir.ops.rms_norm(reduce_scatter, weight, self.epsilon)

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllReduceFusedAddRMSNormChunkPattern(_MoEPatternHelper):
    """Rewrite the residual-carrying variant used by decoder blocks."""

    def get_inputs(self) -> list[torch.Tensor]:
        return [
            self.empty([8, 16]),  # residual
            self.empty([8, 16]),  # partial output of the row-parallel projection
            self.empty([16]),  # RMSNorm weight
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(input)
            rms_norm, residual_out = vllm.ir.ops.fused_add_rms_norm(
                all_reduce, residual, weight, self.epsilon
            )
            return _sequence_parallel_chunk(rms_norm), residual_out

        def replacement(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(input)
            local_len = reduce_scatter.size(0)
            residual = residual[
                self.tp_rank * local_len : self.tp_rank * local_len + local_len, ...
            ]
            return vllm.ir.ops.fused_add_rms_norm(
                reduce_scatter, residual, weight, self.epsilon
            )

        inputs = self.get_inputs()
        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only, pm_pass)
        pm.register_replacement(
            get_first_out_wrapper(pattern),
            get_first_out_wrapper(replacement),
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class AllGatherRMSNormPattern(_MoEPatternHelper):
    """Run RMSNorm on local tokens before gathering for the next projection."""

    def get_inputs(self) -> list[torch.Tensor]:
        return [self.empty([4, 16]), self.empty([16])]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            all_gather = self._all_gather(input)
            return vllm.ir.ops.rms_norm(all_gather, weight, self.epsilon)

        def replacement(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            rms_norm = vllm.ir.ops.rms_norm(input, weight, self.epsilon)
            return self._all_gather(rms_norm)

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherFusedAddRMSNormPattern(_MoEPatternHelper):
    """Move all-gather after a residual-carrying RMSNorm."""

    def get_inputs(self) -> list[torch.Tensor]:
        return [
            self.empty([4, 16]),  # local hidden states
            self.empty([8, 16]),  # full residual
            self.empty([16]),  # RMSNorm weight
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_gather = self._all_gather(input)
            return vllm.ir.ops.fused_add_rms_norm(
                all_gather, residual, weight, self.epsilon
            )

        def replacement(
            input: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            local_len = input.size(0)
            residual = residual[
                self.tp_rank * local_len : self.tp_rank * local_len + local_len, ...
            ]
            rms_norm, residual_out = vllm.ir.ops.fused_add_rms_norm(
                input, residual, weight, self.epsilon
            )
            return self._all_gather(rms_norm), residual_out

        inputs = self.get_inputs()
        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only, pm_pass)
        pm.register_replacement(
            get_first_out_wrapper(pattern),
            get_first_out_wrapper(replacement),
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class AllGatherChunkNoOpPattern(_MoEPatternHelper):
    """Eliminate all-gather followed by the inverse TP chunk."""

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(input: torch.Tensor) -> torch.Tensor:
            all_gather = self._all_gather(input)
            return _sequence_parallel_chunk(all_gather)

        def replacement(input: torch.Tensor) -> torch.Tensor:
            return input

        pm.register_replacement(
            pattern,
            replacement,
            [self.empty([4, 16])],
            pm.fwd_only,
            pm_pass,
        )


class SequenceParallelismMoEPass(VllmPatternMatcherPass):
    """Optimize collectives around tensor-parallel MoE routing.

    The pass is deliberately opt-in because it changes the sequence layout
    carried between decoder blocks.  It is safe to run after the regular
    ``SequenceParallelismPass``: in that case its all-gather/chunk rewrite
    removes the extra round trip introduced before the MoE router.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.min_token_num = config.compilation_config.pass_config.sp_min_token_num
        if (
            self.min_token_num is not None
            and config.scheduler_config.max_num_batched_tokens
        ):
            self.min_token_num = min(
                self.min_token_num, config.scheduler_config.max_num_batched_tokens
            )

        # Replacements involving fused residual RMSNorm temporarily introduce
        # slices whose shapes are stale until all adjacent SP rewrites have
        # been applied.  Remove those cleanup-only views before lowering.
        self.noop_cleanup = NoOpEliminationPass(config)
        self.noop_cleanup.pass_name = f"{self.pass_name}.{self.noop_cleanup.pass_name}"

        self.patterns = PatternMatcherPass(pass_name="sequence_parallelism_moe_pass")
        for epsilon in (1e-5, 1e-6):
            AllReduceRMSNormChunkPattern(
                epsilon, self.model_dtype, self.device
            ).register(self.patterns)
            AllReduceFusedAddRMSNormChunkPattern(
                epsilon, self.model_dtype, self.device
            ).register(self.patterns)
            AllGatherRMSNormPattern(epsilon, self.model_dtype, self.device).register(
                self.patterns
            )
            AllGatherFusedAddRMSNormPattern(
                epsilon, self.model_dtype, self.device
            ).register(self.patterns)
            AllGatherChunkNoOpPattern(epsilon, self.model_dtype, self.device).register(
                self.patterns
            )

        self.dump_patterns(config, self.patterns)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """Only rewrite ranges for which sequence parallelism is enabled."""
        assert (
            self.compilation_config.use_inductor_graph_partition
            or not self.compilation_config.splitting_ops
        ), "SequenceParallelismMoEPass requires full-graph compilation"

        return self.min_token_num is not None and (
            compile_range.start >= self.min_token_num
        )

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
        self.noop_cleanup(graph)
