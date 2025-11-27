# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
import torch.nn as nn
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.logger import init_logger

from .inductor_pass import enable_fake_mode
from .noop_elimination import NoOpEliminationPass
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


class _SequenceParallelMOEPatternHelper:
    """Helper for sequence parallelism patterns."""

    def __init__(
        self,
        dtype: torch.dtype,
        device: str,
    ):
        self.dtype = dtype
        self.device = device
        self.tp_group = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return tensor_model_parallel_all_reduce(x)

    def _reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.reduce_scatter.default(
            x, dim=0, world_size=self.tp_size, group_name=self.tp_group.unique_name
        )

    def _sequence_parallel_chunk(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.sequence_parallel_chunk_impl.default(x)


class AllReduceMoePattern(_SequenceParallelMOEPatternHelper):
    def __init__(self, dtype: torch.dtype, device: str):
        super().__init__(dtype, device)
        self.tp_size = get_tensor_model_parallel_world_size()

    def get_inputs(self):
        input = torch.empty([16, 16], device=self.device, dtype=self.dtype)
        return [input]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(input: torch.Tensor):
            all_reduce = self._all_reduce(input)
            chunks = self._sequence_parallel_chunk(all_reduce)

            return chunks

        def replacement(input: torch.Tensor):
            seq_len = input.size(0)
            remainder = seq_len % self.tp_size
            if remainder != 0:
                pad_len = self.tp_size - remainder
                y = nn.functional.pad(input, (0, 0, 0, pad_len))
            else:
                y = input
            reduce_scatter = self._reduce_scatter(y)

            return reduce_scatter

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class SequenceParallelismMoEPass(VllmPatternMatcherPass):
    """
    The general transformation is:
    Input -> AllReduce -> Chunk -> Output
    becomes
    Input -> ReduceScatter -> Output
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.noop_cleanup = NoOpEliminationPass(config)
        self.noop_cleanup.pass_name = f"{self.pass_name}.{self.noop_cleanup.pass_name}"

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="sequence_parallelism_moe_pass",
        )

        AllReduceMoePattern(self.model_dtype, self.device).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    def is_applicable(self, shape: int | None) -> bool:
        # When sequence parallelism is enabled, the residual tensor from RMSNorm
        # needs to be split along the sequence dimension. However, this dimension
        # is symbolic during piecewise compilation, and splitting symbolic shapes
        # is not supported.
        #
        # This pass is therefore only applied when the sequence dimension is
        # concrete:
        # 1. In full-graph compilation mode (no Dynamo splitting ops are used).
        #   For this case we always pad num_tokens to be a multiple of
        #   tensor_parallel_size, so there's no need to check shape % tp_size == 0.
        # 2. For specific shape provided during compilation (e.g., from
        #    `compile_sizes`), which must be divisible by the tensor-parallel
        #    size.
        if (
            not self.compilation_config.splitting_ops
            or self.compilation_config.use_inductor_graph_partition
        ):
            return True
        tp_size = get_tensor_model_parallel_world_size()
        return shape is not None and shape % tp_size == 0

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph):
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
        self.noop_cleanup(graph)
