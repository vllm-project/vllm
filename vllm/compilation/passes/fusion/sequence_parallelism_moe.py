# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.logger import init_logger

from ..inductor_pass import enable_fake_mode
from ..utility.noop_elimination import NoOpEliminationPass
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherFusedAddRMSNorm, MatcherRMSNorm

logger = init_logger(__name__)


class _SequenceParallelismMoEPatternHelper:
    """Helper for sequence-parallel MoE communication patterns."""

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
    ) -> None:
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device
        self.tp_group = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return tensor_model_parallel_all_reduce(x)

    def _sequence_parallel_chunk(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.sequence_parallel_chunk_impl.default(x)

    def _reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.reduce_scatter.default(
            x,
            dim=0,
            world_size=self.tp_size,
            group_name=self.tp_group.unique_name,
        )


class AllReduceRMSNormSequenceParallelChunkPattern(
    _SequenceParallelismMoEPatternHelper
):
    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
    ) -> None:
        super().__init__(epsilon, dtype, device)
        self.rmsnorm_matcher = MatcherRMSNorm(epsilon)
        if self.rmsnorm_matcher.model_dtype is None:
            self.rmsnorm_matcher.model_dtype = dtype

    def get_inputs(
        self,
    ) -> list[torch.Tensor]:
        input_ = torch.empty([8, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4], device=self.device, dtype=self.dtype)
        return [input_, weight]

    def register(
        self,
        pm_pass: PatternMatcherPass,
    ) -> None:
        def pattern(
            input_: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            all_reduce = self._all_reduce(input_)
            rmsnorm = self.rmsnorm_matcher(all_reduce, weight)
            return self._sequence_parallel_chunk(rmsnorm)

        def replacement(
            input_: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            reduce_scatter = self._reduce_scatter(input_)
            rmsnorm = self.rmsnorm_matcher(reduce_scatter, weight)
            return rmsnorm

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            skip_duplicates=True,
        )


class AllReduceFusedAddRMSNormSequenceParallelChunkPattern(
    _SequenceParallelismMoEPatternHelper
):
    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
    ) -> None:
        super().__init__(epsilon, dtype, device)
        self.rmsnorm_matcher = MatcherFusedAddRMSNorm(epsilon)
        if self.rmsnorm_matcher.model_dtype is None:
            self.rmsnorm_matcher.model_dtype = dtype

    def get_inputs(self) -> list[torch.Tensor]:
        residual = torch.empty([8, 4], device=self.device, dtype=self.dtype)
        input_ = torch.empty([8, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4], device=self.device, dtype=self.dtype)
        return [residual, input_, weight]

    def register(
        self,
        pm_pass: PatternMatcherPass,
    ) -> None:
        def pattern(
            residual: torch.Tensor,
            input_: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(input_)
            rmsnorm = self.rmsnorm_matcher(all_reduce, weight, residual)
            return self._sequence_parallel_chunk(rmsnorm[0]), rmsnorm[1]

        def replacement(
            residual: torch.Tensor,
            input_: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(input_)
            # Pattern matcher replaces from top-to-bottom, so residual can still
            # be full-size while this node already runs on RS-sharded inputs.
            # Use a temporary prefix slice to align shapes; this becomes a no-op
            # once upstream replacements shard residual on the same range.
            residual_chunk = residual[0 : reduce_scatter.size(0), ...]
            rmsnorm = self.rmsnorm_matcher(
                reduce_scatter,
                weight,
                residual_chunk,
            )
            return rmsnorm[0], rmsnorm[1]

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            skip_duplicates=True,
        )


class SequenceParallelismMoEPass(VllmPatternMatcherPass):
    """
    Replace `all_reduce + (fused) rmsnorm + sequence_parallel_chunk` with
    local-rank computation based on `reduce_scatter`.

    Similar to `SequenceParallelismPass`, fused-add+rmsnorm replacements may
    insert temporary residual slices while pattern matching proceeds through
    the graph. A trailing `NoOpEliminationPass` cleans up any slices that
    become shape-noops after all matches are applied.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        # Clean up temporary no-op slices introduced by replacement ordering.
        self.noop_cleanup = NoOpEliminationPass(config)
        self.noop_cleanup.pass_name = f"{self.pass_name}.{self.noop_cleanup.pass_name}"

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="sequence_parallelism_moe_pass"
        )

        pattern_dtype = self.model_dtype or torch.get_default_dtype()
        # Register model eps first to reduce pattern-matching search space.
        eps_candidates: set[float] = {1e-5, 1e-6}
        model_config = getattr(config, "model_config", None)
        hf_text_config = getattr(model_config, "hf_text_config", None)
        model_eps = getattr(hf_text_config, "rms_norm_eps", None)
        if isinstance(model_eps, float):
            eps_candidates = {model_eps, *eps_candidates}
        for epsilon in eps_candidates:
            AllReduceRMSNormSequenceParallelChunkPattern(
                epsilon, pattern_dtype, self.device
            ).register(self.patterns)
            AllReduceFusedAddRMSNormSequenceParallelChunkPattern(
                epsilon, pattern_dtype, self.device
            ).register(self.patterns)
        self.dump_patterns(config, self.patterns)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        # Piecewise mode only supports concrete compile sizes.
        if (
            not self.compilation_config.use_inductor_graph_partition
            and self.compilation_config.splitting_ops
        ):
            tp_size = get_tensor_model_parallel_world_size()
            return bool(
                compile_range.is_single_size() and compile_range.end % tp_size == 0
            )
        return True

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
        self.noop_cleanup(graph)
