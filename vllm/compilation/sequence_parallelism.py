# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Tuple

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.logger import init_logger

from .vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


class AllReduceRMSNormPattern:

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str):
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device


class EmbeddingAllReduceRMSNormPattern(AllReduceRMSNormPattern):

    def get_inputs(self):
        arg2_1 = torch.empty([16, 4], device=self.device, dtype=self.dtype)
        mul_6 = torch.tensor([[3, 7, 1, 4, 9, 2, 5, 0]],
                             device=self.device,
                             dtype=torch.long)
        unsqueeze = torch.rand([1, 8, 1], device=self.device, \
            dtype=self.dtype) > 0.5
        full_default = torch.zeros([1, 8, 4], device=self.device, \
            dtype=self.dtype)
        permute = torch.empty([1, 8, 4], device=self.device, dtype=self.dtype)
        arg3_1 = torch.empty([4], device=self.device, dtype=self.dtype)

        return [arg2_1, mul_6, unsqueeze, full_default, permute, arg3_1]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            arg2_1: torch.Tensor,
            mul_6: torch.Tensor,
            unsqueeze: torch.Tensor,
            full_default: torch.Tensor,
            permute: torch.Tensor,
            arg3_1: torch.Tensor,
        ):
            embedding = torch.ops.aten.embedding.default(arg2_1, mul_6)
            where = torch.ops.aten.where.self(unsqueeze, full_default,
                                              embedding)
            all_reduce = tensor_model_parallel_all_reduce(where)
            rmsnorm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.rms_norm.default,
                result=permute,
                input=all_reduce,
                weight=arg3_1,
                epsilon=self.epsilon,
            )

            return rmsnorm[1], all_reduce

        def replacement(
            arg2_1: torch.Tensor,
            mul_6: torch.Tensor,
            unsqueeze: torch.Tensor,
            full_default: torch.Tensor,
            permute: torch.Tensor,
            arg3_1: torch.Tensor,
        ):
            embedding = torch.ops.aten.embedding.default(arg2_1, mul_6)
            where = torch.ops.aten.where.self(unsqueeze, full_default,
                                              embedding)

            tp = get_tp_group()
            tp_size = get_tensor_model_parallel_world_size()
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                where, dim=0, world_size=tp_size, group_name=tp.unique_name)

            rmsnorm_result = torch.empty_like(reduce_scatter)
            rmsnorm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.rms_norm.default,
                result=rmsnorm_result,
                input=reduce_scatter,
                weight=arg3_1,
                epsilon=self.epsilon,
            )

            all_gather = torch.ops.vllm.all_gather.default(
                rmsnorm[1],
                dim=0,
                world_size=tp_size,
                group_name=tp.unique_name)

            return all_gather, reduce_scatter

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class MiddleAllReduceRMSNormPattern(AllReduceRMSNormPattern):

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4],
                                       device=self.device,
                                       dtype=self.dtype)

        return [
            residual,
            mm_1,
            rms_norm_weights,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            all_reduce = tensor_model_parallel_all_reduce(mm_1)

            rmsnorm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.fused_add_rms_norm.default,
                input=all_reduce,
                residual=residual,
                weight=rms_norm_weights,
                epsilon=self.epsilon,
            )

            return rmsnorm[1], rmsnorm[2]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            tp = get_tp_group()
            tp_size = get_tensor_model_parallel_world_size()
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                mm_1, dim=0, world_size=tp_size, group_name=tp.unique_name)

            # TODO is it possible to extract epsilon from somewhere
            rmsnorm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.fused_add_rms_norm.default,
                input=reduce_scatter,
                residual=residual,
                weight=rms_norm_weights,
                epsilon=self.epsilon,
            )

            all_gather = torch.ops.vllm.all_gather.default(
                rmsnorm[1],
                dim=0,
                world_size=tp_size,
                group_name=tp.unique_name)
            return all_gather, rmsnorm[2]

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class LastAllReduceRMSNormPattern(AllReduceRMSNormPattern):

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4],
                                       device=self.device,
                                       dtype=self.dtype)

        return [
            residual,
            mm_1,
            rms_norm_weights,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            all_reduce = tensor_model_parallel_all_reduce(mm_1)

            rmsnorm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.fused_add_rms_norm.default,
                input=all_reduce,
                residual=residual,
                weight=rms_norm_weights,
                epsilon=self.epsilon,
            )

            return rmsnorm[1]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            tp = get_tp_group()
            tp_size = get_tensor_model_parallel_world_size()
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                mm_1, dim=0, world_size=tp_size, group_name=tp.unique_name)

            # TODO is it possible to extract epsilon from somewhere
            rmsnorm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.fused_add_rms_norm.default,
                input=reduce_scatter,
                residual=residual,
                weight=rms_norm_weights,
                epsilon=self.epsilon,
            )

            normalized = torch.ops.vllm.all_gather.default(
                rmsnorm[1],
                dim=0,
                world_size=tp_size,
                group_name=tp.unique_name)

            return normalized

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class SequenceParallelismPass(VllmInductorPass):

    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="sequence_parallelism_pass")
        for epsilon in [1e-5, 1e-6]:
            EmbeddingAllReduceRMSNormPattern(
                epsilon, self.dtype, self.device).register(self.patterns)

            MiddleAllReduceRMSNormPattern(epsilon, self.dtype,
                                          self.device).register(self.patterns)

            LastAllReduceRMSNormPattern(epsilon, self.dtype,
                                        self.device).register(self.patterns)
            # WARNING: This is a hack to clear the pattern matcher cache
            # and allow multiple values of epsilon.
            torch._inductor.pattern_matcher._seen_patterns.clear()

    def is_applicable_for_shape(self, shape: Optional[int]) -> bool:
        # only do replace for specific shapes
        tp_size = get_tensor_model_parallel_world_size()
        return shape is not None and shape % tp_size == 0

    def __call__(self, graph: fx.Graph):
        self.dump_graph(graph, "before_sequence_parallelism_pass")
        count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", count)
        self.dump_graph(graph, "after_sequence_parallelism_pass")
