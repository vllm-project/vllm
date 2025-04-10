# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

import torch
import torch.fx as fx
from torch._inductor.pattern_matcher import (Match, PatternMatcherPass,
                                             fwd_only, register_replacement)

from vllm.config import CompilationConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.logger import init_logger

from .inductor_pass import get_pass_context
from .vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


def get_world_name() -> str:
    return torch.distributed.group.WORLD.group_name


def residual_slice_shape(residual: torch.Tensor) -> int:
    n_slices = get_tensor_model_parallel_world_size()
    assert residual.size(0) % n_slices == 0
    return residual.size(0) // n_slices


def search_embedding_all_reduce_rmsnorm(
    arg2_1: torch.Tensor,
    mul_6: torch.Tensor,
    unsqueeze: torch.Tensor,
    full_default: torch.Tensor,
    permute: torch.Tensor,
    arg3_1: torch.Tensor,
):
    embedding = torch.ops.aten.embedding.default(arg2_1, mul_6)
    where = torch.ops.aten.where.self(unsqueeze, full_default, embedding)
    all_reduce = tensor_model_parallel_all_reduce(where)
    rmsnorm = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.rms_norm.default,
        result=permute,
        input=all_reduce,
        weight=arg3_1,
        epsilon=1e-5,
    )

    return rmsnorm[1], all_reduce


def replace_with_embedding_reduce_scatter_rmsnorm(
    arg2_1: torch.Tensor,
    mul_6: torch.Tensor,
    unsqueeze: torch.Tensor,
    full_default: torch.Tensor,
    permute: torch.Tensor,
    arg3_1: torch.Tensor,
):
    embedding = torch.ops.aten.embedding.default(arg2_1, mul_6)
    where = torch.ops.aten.where.self(unsqueeze, full_default, embedding)

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
        epsilon=1e-5,
    )

    all_gather = torch.ops.vllm.all_gather.default(rmsnorm[1],
                                                   dim=0,
                                                   world_size=tp_size,
                                                   group_name=tp.unique_name)

    return all_gather, reduce_scatter


def search_gemm_allreduce_rmsnorm(
    residual: torch.Tensor,
    gemm_1_weights: torch.Tensor,
    gemm_1_activations: torch.Tensor,
    rms_norm_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gemm_1_w_perm = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, gemm_1_w_perm)
    all_reduce = tensor_model_parallel_all_reduce(mm_1)

    rmsnorm = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input=all_reduce,
        residual=residual,
        weight=rms_norm_weights,
        epsilon=1e-5,
    )

    return rmsnorm[1], rmsnorm[2]


def replace_with_gemm_rs_ag_rmsnorm(
    residual: torch.Tensor,
    gemm_1_weights: torch.Tensor,
    gemm_1_activations: torch.Tensor,
    rms_norm_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tp = get_tp_group()
    tp_size = get_tensor_model_parallel_world_size()
    gemm_1_w_perm = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, gemm_1_w_perm)
    reduce_scatter = torch.ops.vllm.reduce_scatter.default(
        mm_1, dim=0, world_size=tp_size, group_name=tp.unique_name)

    # TODO is it possible to extract epsilon from somewhere
    rmsnorm = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input=reduce_scatter,
        residual=residual,
        weight=rms_norm_weights,
        epsilon=1e-5,
    )

    all_gather = torch.ops.vllm.all_gather.default(rmsnorm[1],
                                                   dim=0,
                                                   world_size=tp_size,
                                                   group_name=tp.unique_name)
    return all_gather, rmsnorm[2]


def search_last_gemm_allreduce_rmsnorm(
    residual: torch.Tensor,
    gemm_1_weights: torch.Tensor,
    gemm_1_activations: torch.Tensor,
    rms_norm_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gemm_1_w_perm = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, gemm_1_w_perm)
    all_reduce = tensor_model_parallel_all_reduce(mm_1)

    rmsnorm = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input=all_reduce,
        residual=residual,
        weight=rms_norm_weights,
        epsilon=1e-5,
    )

    return rmsnorm[1]


def replace_with_last_gemm_rs_ag_rmsnorm(
    residual: torch.Tensor,
    gemm_1_weights: torch.Tensor,
    gemm_1_activations: torch.Tensor,
    rms_norm_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tp = get_tp_group()
    tp_size = get_tensor_model_parallel_world_size()
    gemm_1_w_perm = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, gemm_1_w_perm)
    reduce_scatter = torch.ops.vllm.reduce_scatter.default(
        mm_1, dim=0, world_size=tp_size, group_name=tp.unique_name)

    # TODO is it possible to extract epsilon from somewhere
    rmsnorm = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input=reduce_scatter,
        residual=residual,
        weight=rms_norm_weights,
        epsilon=1e-5,
    )

    normalized = torch.ops.vllm.all_gather.default(rmsnorm[1],
                                                   dim=0,
                                                   world_size=tp_size,
                                                   group_name=tp.unique_name)

    return normalized


def generate_inputs_for_embedding_ar_rmsnorm():
    arg2_1 = torch.rand([16, 4], device="cuda", dtype=torch.float16)
    # mul_6: token indices (batch_size x seq_len)
    mul_6 = torch.tensor([[3, 7, 1, 4, 9, 2, 5, 0]],
                         device="cuda",
                         dtype=torch.long)
    unsqueeze = torch.rand([1, 8, 1], device="cuda", dtype=torch.float16) > 0.5
    full_default = torch.zeros([1, 8, 4], device="cuda", dtype=torch.float16)
    permute = torch.rand([1, 8, 4], device="cuda", dtype=torch.float16)
    arg3_1 = torch.rand([4], device="cuda", dtype=torch.float16)
    return [arg2_1, mul_6, unsqueeze, full_default, permute, arg3_1]


class CollectiveFusionPass(VllmInductorPass):
    _instance: "Optional[CollectiveFusionPass]" = None

    @classmethod
    def instance(cls, config: CompilationConfig) -> "CollectiveFusionPass":
        """
        Get the singleton instance of the CollectiveFusionPass.
        If the instance exists, the config is updated but
        initialization is not repeated.
        """
        if cls._instance is None:
            cls._instance = CollectiveFusionPass(config)
        else:
            cls._instance.config = config
        return cls._instance

    def __init__(self, config: CompilationConfig):
        assert self.__class__._instance is None, (
            "CollectiveFusionPass singleton instance already exists")
        super().__init__(config)

        self.embedding_ag_rmsnorm_pattern = PatternMatcherPass()
        self.gemm_rs_ag_gemm_pattern = PatternMatcherPass()
        self.final_ar_rmsnorm_pattern = PatternMatcherPass()
        self.matches: List[Match] = []

        embedding_rmsnorm_inputs = generate_inputs_for_embedding_ar_rmsnorm()
        register_replacement(
            search_embedding_all_reduce_rmsnorm,
            replace_with_embedding_reduce_scatter_rmsnorm,
            embedding_rmsnorm_inputs,
            fwd_only,
            [self.embedding_ag_rmsnorm_pattern],
            extra_check=lambda m: self.record_match(m),
        )

        gemm_1_weights = torch.empty([4, 4],
                                     device="cuda",
                                     dtype=torch.float16)
        gemm_1_activations = torch.empty([4, 4],
                                         device="cuda",
                                         dtype=torch.float16)
        residual = torch.empty([4, 4], device="cuda", dtype=torch.float16)
        rms_norm_weights = torch.empty([4, 4],
                                       device="cuda",
                                       dtype=torch.float16)

        inputs = [
            residual,
            gemm_1_weights,
            gemm_1_activations,
            rms_norm_weights,
        ]
        register_replacement(
            search_gemm_allreduce_rmsnorm,
            replace_with_gemm_rs_ag_rmsnorm,
            inputs,
            fwd_only,
            [self.gemm_rs_ag_gemm_pattern],
            extra_check=lambda m: self.record_match(m),
        )

        register_replacement(
            search_last_gemm_allreduce_rmsnorm,
            replace_with_last_gemm_rs_ag_rmsnorm,
            inputs,
            fwd_only,
            [self.final_ar_rmsnorm_pattern],
            extra_check=lambda m: self.record_match(m),
        )

    def record_match(self, match: Match) -> bool:
        self.matches.append(match)
        # only do replace for specific shapes
        if get_pass_context().runtime_shape is not None:
            return bool(match)
        else:
            return False

    def __call__(self, graph: fx.Graph):
        import torch.distributed as dist

        rank = dist.get_rank()

        self.dump_graph(graph, "before_collective_fusion")
        embedding_match_cnt = self.embedding_ag_rmsnorm_pattern.apply(graph)
        gemm_ar_rmsnorm_match_cnt = self.gemm_rs_ag_gemm_pattern.apply(graph)

        if embedding_match_cnt > 0 or gemm_ar_rmsnorm_match_cnt > 0:
            final_match_cnt = self.final_ar_rmsnorm_pattern.apply(graph)
            logger.debug(
                "all matches = %d, embedding matches = %d, \
                gemm_ar_rmsnorm matches = %d, \
                final ar rmsnorm matches = %d",
                len(self.matches),
                embedding_match_cnt,
                gemm_ar_rmsnorm_match_cnt,
                final_match_cnt,
            )
        else:
            logger.debug(
                "all matches = %d, embedding matches = %d, \
                gemm_ar_rmsnorm matches = %d",
                len(self.matches),
                embedding_match_cnt,
                gemm_ar_rmsnorm_match_cnt,
            )

        if rank == 0:
            print(f"after graph {graph}")
        self.dump_graph(graph, "after_collective_fusion")
        self.matches.clear()
