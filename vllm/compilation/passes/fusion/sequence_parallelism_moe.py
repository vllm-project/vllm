# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Callable, Sequence
from typing import Any

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


def get_first_out_wrapper(
    fn: Callable[..., Sequence[torch.Tensor]],
) -> Callable[..., torch.Tensor]:
    """Expose tuple replacements to sites that consume only the first output."""

    @functools.wraps(fn)
    def wrapper(*args: Any) -> torch.Tensor:
        return fn(*args)[0]

    return wrapper


class _SequenceParallelismMoEPatternHelper:
    """Shared TP collectives for MoE sequence-parallel rewrites."""

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
        """Match the replicated pre-rewrite form emitted by the model."""
        return tensor_model_parallel_all_reduce(x)

    def _sequence_parallel_chunk(self, x: torch.Tensor) -> torch.Tensor:
        """Match the existing post-norm local chunk op."""
        return torch.ops.vllm.sequence_parallel_chunk_impl.default(x)

    def _reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        """Materialize the local token shard directly in the replacement."""
        return torch.ops.vllm.reduce_scatter.default(
            x,
            dim=0,
            world_size=self.tp_size,
            group_name=self.tp_group.unique_name,
        )

    def _all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """Rebuild the full token sequence only for post-MoE consumers."""
        return torch.ops.vllm.all_gather.default(
            x,
            dim=0,
            world_size=self.tp_size,
            group_name=self.tp_group.unique_name,
        )


class AllReduceRMSNormSequenceParallelChunkPattern(
    _SequenceParallelismMoEPatternHelper
):
    """Replace all-reduce + RMSNorm + chunk with local reduce-scatter + RMSNorm."""

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
        """Use a small shape that still exercises the token-sharding path."""
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
            # RMSNorm is token-local, so once each rank owns its shard the
            # full sequence no longer needs to be materialized.
            reduce_scatter = self._reduce_scatter(input_)
            return self.rmsnorm_matcher(reduce_scatter, weight)

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
    """Shard post-attention fused add+RMSNorm so the MoE path stays local."""

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
        """Trace the common case where the residual is still full-sequence."""
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
            return (
                self._sequence_parallel_chunk(rmsnorm[0]),
                self._sequence_parallel_chunk(rmsnorm[1]),
            )

        def replacement(
            residual: torch.Tensor,
            input_: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(input_)
            residual_chunk = self._sequence_parallel_chunk(residual)
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
        # Some sites consume only the normalized activation.
        pm.register_replacement(
            get_first_out_wrapper(pattern),
            get_first_out_wrapper(replacement),
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            skip_duplicates=True,
        )


class AllGatherFusedAddRMSNormPattern(_SequenceParallelismMoEPatternHelper):
    """
    Move post-MoE fused add+RMSNorm before all-gather.

      all_gather(x) -> depad -> fused_add_rms_norm(..., residual)
      =>
      fused_add_rms_norm(x, ..., residual) -> all_gather(norm_out) -> depad

    This keeps the residual sharded across the MoE block and gathers only the
    normalized activation.
    """

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

    def get_inputs(self) -> list[Any]:
        """Keep `all_gather -> slice` in the traced pattern by using a depadded size."""
        local_tokens = 4
        global_tokens = local_tokens * self.tp_size - 1
        residual = torch.empty(
            [global_tokens, 4],
            device=self.device,
            dtype=self.dtype,
        )
        input_ = torch.empty([local_tokens, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4], device=self.device, dtype=self.dtype)
        return [residual, input_, weight, global_tokens]

    def register(
        self,
        pm_pass: PatternMatcherPass,
    ) -> None:
        def pattern(
            residual: torch.Tensor,
            input_: torch.Tensor,
            weight: torch.Tensor,
            num_tokens: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_gather = self._all_gather(input_)
            depad = all_gather[0:num_tokens, ...]
            rmsnorm = self.rmsnorm_matcher(depad, weight, residual)
            gathered_norm = torch.ops.aten.slice_scatter.default(
                all_gather,
                rmsnorm[0],
                0,
                0,
                num_tokens,
            )
            return (
                gathered_norm[0:num_tokens, ...],
                self._sequence_parallel_chunk(rmsnorm[1]),
            )

        def replacement(
            residual: torch.Tensor,
            input_: torch.Tensor,
            weight: torch.Tensor,
            num_tokens: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual_chunk = self._sequence_parallel_chunk(residual)
            rmsnorm = self.rmsnorm_matcher(input_, weight, residual_chunk)
            all_gather = self._all_gather(rmsnorm[0])
            depad = all_gather[0:num_tokens, ...]
            return depad, rmsnorm[1]

        # Use the same depadded size as `get_inputs()` so tracing keeps the
        # explicit `all_gather -> slice` structure.
        scalar_workaround = {
            "num_tokens": 4 * self.tp_size - 1,
        }
        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            scalar_workaround=scalar_workaround,
            skip_duplicates=True,
        )
        # Some sites ignore the residual output.
        pm.register_replacement(
            get_first_out_wrapper(pattern),
            get_first_out_wrapper(replacement),
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            scalar_workaround=scalar_workaround,
            skip_duplicates=True,
        )


class SequenceParallelismMoEPass(VllmPatternMatcherPass):
    """
    Apply the MoE sequence-parallel rewrites.

    The pass keeps residuals sharded across the MoE block and gathers only the
    normalized activation on the exit path. Temporary slices are expected while
    individual replacements fire; `NoOpEliminationPass` removes the ones that
    become redundant afterward.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        # Replacement order is not stable, so cleanup happens after matching.
        self.noop_cleanup = NoOpEliminationPass(config)
        self.noop_cleanup.pass_name = f"{self.pass_name}.{self.noop_cleanup.pass_name}"

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="sequence_parallelism_moe_pass"
        )

        pattern_dtype = self.model_dtype or torch.get_default_dtype()
        # Epsilon is part of the matched computation, so prefer the model value
        # and keep common fallbacks for configs that do not expose it cleanly.
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
            AllGatherFusedAddRMSNormPattern(
                epsilon, pattern_dtype, self.device
            ).register(self.patterns)
        self.dump_patterns(config, self.patterns)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """Require concrete, TP-divisible token counts in piecewise mode."""

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
        """Apply rewrites, then drop cleanup-only slice nodes."""
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
        self.noop_cleanup(graph)
