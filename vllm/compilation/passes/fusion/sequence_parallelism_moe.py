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
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


class _SequenceParallelismMoEPatternHelper:
    """Helper for sequence-parallel MoE communication patterns."""

    def __init__(self) -> None:
        self.tp_group = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return tensor_model_parallel_all_reduce(x)

    def _sequence_parallel_chunk(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.sequence_parallel_chunk_impl.default(x)

    def _reduce_scatter_with_padding(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.reduce_scatter_with_padding.default(
            x,
            dim=0,
            world_size=self.tp_size,
            group_name=self.tp_group.unique_name,
        )


class AllReduceSequenceParallelChunkPattern(_SequenceParallelismMoEPatternHelper):
    def get_inputs(
        self,
        dtype: torch.dtype,
        device: str | None,
    ) -> list[torch.Tensor]:
        input_ = torch.empty([8, 4], device=device or "cuda", dtype=dtype)
        return [input_]

    def register(
        self,
        pm_pass: PatternMatcherPass,
        dtype: torch.dtype,
        device: str | None,
    ) -> None:
        def pattern(input_: torch.Tensor) -> torch.Tensor:
            all_reduce = self._all_reduce(input_)
            return self._sequence_parallel_chunk(all_reduce)

        def replacement(input_: torch.Tensor) -> torch.Tensor:
            return self._reduce_scatter_with_padding(input_)

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(dtype, device),
            pm.fwd_only,
            pm_pass,
        )


class SequenceParallelismMoEPass(VllmPatternMatcherPass):
    """
    Replace `all_reduce + sequence_parallel_chunk` with
    `reduce_scatter_with_padding`.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="sequence_parallelism_moe_pass"
        )

        # Register once to avoid duplicate pattern graph errors in Inductor.
        pattern_dtype = self.model_dtype or torch.float16
        AllReduceSequenceParallelChunkPattern().register(
            self.patterns, pattern_dtype, self.device
        )
        self.dump_patterns(config, self.patterns)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
