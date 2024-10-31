import operator
from typing import Callable, Dict, List, Optional, Tuple, Union, Iterable

import torch
import torch.fx as fx

from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement, fwd_only, joint_fwd_bwd, Match

from vllm.compilation.inductor_pass import InductorPass
from vllm.compilation.utils import find_fn, find_auto_fn, find_getitem, last_node_in_match
from vllm.distributed.parallel_state import get_tp_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.linear import should_slice, slice_residual

from vllm.logger import init_logger

logger = init_logger(__name__)


def match_gemm_rs_ag_gemm(
        residual: torch.Tensor,
        gemm_1_weights: torch.Tensor,
        gemm_1_activations: torch.Tensor,
        rms_norm_weight: torch.Tensor,
        gemm_2_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gemm_1_w_perm = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, gemm_1_w_perm)

    all_reduce = torch.ops.higher_order.auto_functionalized(
        torch.ops.vllm.inplace_all_reduce.default,
        tensor = mm_1,
        group_name = 'tp:0'  # how to deal with groupname?
    )
    all_reduce = all_reduce[1]

    norm_res = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input = all_reduce,
        residual = residual,
        weight = rms_norm_weight,
        epsilon = 1e-05
    )
    normalized = norm_res[1]
    new_residual = norm_res[2]

    gemm_2_w_perm = torch.ops.aten.permute.default(gemm_2_weights, [1, 0])
    mm_2 = torch.ops.aten.mm.default(normalized, gemm_2_w_perm)

    return mm_2, new_residual


@torch.library.custom_op("vllm::gemm_rs_ag_gemm", mutates_args=())
def gemm_rs_ag_gemm(residual: torch.Tensor,
                    my_residual: torch.Tensor,
                    gemm_1_weights: torch.Tensor,
                    gemm_1_activations: torch.Tensor,
                    rms_norm_weight: torch.Tensor,
                    gemm_2_weights: torch.Tensor,
                    first_layer: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if should_slice(residual.shape) and first_layer:
        res_slices = slice_residual(residual)
        slice_size = res_slices[get_tensor_model_parallel_rank()].shape[0]
        split_1 = torch.ops.aten.split.Tensor(residual, slice_size)
        getitem_26 = split_1[0];  split_1 = None
    else:
        getitem_26 = residual
        slice_size = residual.shape[0]

    if not should_slice(residual.shape):
        permute_3 = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
        output = torch.matmul(gemm_1_activations, permute_3)
        output = tensor_model_parallel_all_reduce(output)

        auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input=output, residual=getitem_26, weight=rms_norm_weight, epsilon=1e-05)
        getitem_29 = auto_functionalized_4[1]
        getitem_30 = auto_functionalized_4[2]

        permute_5 = torch.ops.aten.permute.default(gemm_2_weights, [1, 0])
        getitem_35 = torch.matmul(getitem_29, permute_5)
        getitem_30a = getitem_30.clone()
        return getitem_35, getitem_30, getitem_30a
    else:
        group_name = torch.distributed.group.WORLD.group_name # TODO: factor out to setup
        permute_3 = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
        clone = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format)
        output = torch.ops.symm_mem.fused_matmul_reduce_scatter.default(gemm_1_activations, clone, 'avg', 0, group_name)
        auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input=output, residual=getitem_26, weight=rms_norm_weight, epsilon=1e-05)
        getitem_29 = auto_functionalized_4[1]
        getitem_30 = auto_functionalized_4[2]
        residual_1 = residual if first_layer else my_residual
        slice_scatter_2 = torch.ops.aten.slice_scatter.default(residual_1, getitem_30, 0, 0, slice_size)
        split_2 = torch.ops.aten.split.Tensor(slice_scatter_2, slice_size)
        getitem_31 = split_2[0]
        permute_5 = torch.ops.aten.permute.default(gemm_2_weights, [1, 0])
        clone_1 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format)
        fused_all_gather_matmul = torch.ops.symm_mem.fused_all_gather_matmul.default(getitem_29, [clone_1], 0, group_name)
        getitem_34 = fused_all_gather_matmul[1]
        getitem_35 = getitem_34[0]

        # TODO: can we avoid clone here?
        return getitem_35, getitem_31.clone(), slice_scatter_2


@torch.library.register_fake("vllm::gemm_rs_ag_gemm")
def gemm_rs_ag_gemm_fake(residual: torch.Tensor,
                         my_residual: torch.Tensor,
                         gemm_1_weights: torch.Tensor,
                         gemm_1_activations: torch.Tensor,
                         rms_norm_weight: torch.Tensor,
                         gemm_2_weights: torch.Tensor,
                         first_layer: bool,
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if should_slice(gemm_1_activations.shape) and first_layer:
        res_slices = slice_residual(residual)
        slice_size = res_slices[get_tensor_model_parallel_rank()].shape[0]
        split_1 = torch.ops.aten.split.Tensor(residual, slice_size)
        my_residual = split_1[0]
    else:
        my_residual = residual

    # verify the type is always correct
    mm_res = torch.empty(
        (gemm_1_activations.shape[0], gemm_2_weights.shape[0]),
        device=gemm_1_activations.device,
        dtype=gemm_1_activations.dtype
    )

    return (mm_res, my_residual, residual)


def match_final(my_residual: torch.Tensor,
                gemm_1_weights: torch.Tensor,
                gemm_1_activations: torch.Tensor,
                rms_norm_weights: torch.Tensor) -> torch.Tensor:
    gemm_1_w_perm = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, gemm_1_w_perm)

    all_reduce = torch.ops.higher_order.auto_functionalized(
        torch.ops.vllm.inplace_all_reduce.default,
        tensor = mm_1,
        group_name = 'tp:0' # TODO: not same as group name
    )
    all_reduce = all_reduce[1]

    norm_res = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input = all_reduce,
        residual = my_residual,
        weight = rms_norm_weights,
        epsilon = 1e-05
    )
    normalized = norm_res[1]

    return normalized


# TODO: wrap in custom op to prevent infinite recursion in inductor logging statement?
def replace_final(my_residual: torch.Tensor,
                  gemm_1_weights: torch.Tensor,
                  gemm_1_activations: torch.Tensor,
                  rms_norm_weights: torch.Tensor) -> torch.Tensor:
    tp_group_name = "tp:0" # f"tp:{group_name}" # TODO: not same as group name

    permute_254 = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, permute_254)
    auto_functionalized_161 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_1, group_name = tp_group_name)
    getitem_1217 = auto_functionalized_161[1]

    # is this the right thing to call it on?
    if should_slice(gemm_1_activations.shape):
        group_name = torch.distributed.group.WORLD.group_name
        world_size = get_tensor_model_parallel_world_size()
        all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor.default(my_residual, world_size, group_name)
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor)
    else:
        wait_tensor = my_residual

    auto_functionalized_162 = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input = getitem_1217,
        residual = wait_tensor,
        weight = rms_norm_weights,
        epsilon = 1e-05)

    getitem_1219 = auto_functionalized_162[1]
    return getitem_1219


class CollectiveFusionPass(InductorPass):
    def __init__(self):
        self.my_patterns = PatternMatcherPass()
        self.my_patterns2 = PatternMatcherPass()
        self.matches: List[Match] = []

        x = torch.empty([4,4], device='cuda')
        w = torch.empty([4,4], device='cuda')
        resid = torch.empty([4,4], device='cuda')
        resid_w = torch.empty([4,4], device='cuda')
        x2 = torch.empty([4,4], device='cuda')
        inputs = [resid, x, w, resid_w, x2]

        register_replacement(match_gemm_rs_ag_gemm,
                             match_gemm_rs_ag_gemm,
                             inputs,
                             fwd_only,
                             [self.my_patterns],
                             extra_check=lambda m: self.record_match(m))

        final_inputs = [x, w, resid, resid_w]
        register_replacement(match_final,
                             replace_final,
                             final_inputs,
                             fwd_only,
                             [self.my_patterns2])

    def record_match(self, match: Match) -> bool:
        # Hijack the extra_check to record the match and
        # save it for post-processing.
        self.matches.append(match)

        # Return False to prevent automatic replacement.
        return False

    def process_matches(self, graph: fx.Graph):
        nodes = list(graph.nodes)
        first_match = None

        def find_min_index(match) -> int:
            return min(match.nodes, key=lambda x: nodes.index(x))

        # "sort" matches in topo order.
        matches = sorted(self.matches, key=lambda x: find_min_index(x))

        res_replacements = []
        my_res_replacements = []

        for match in matches:
            last_node = last_node_in_match(match)

            with graph.inserting_after(last_node):
                kwargs = match.kwargs
                kwargs["first_layer"] = match == matches[0]
                kwargs["residual"] = res_replacements[-1] if len(res_replacements) > 0 else match.kwargs["residual"]
                kwargs["my_residual"] = my_res_replacements[-1] if len(my_res_replacements) > 0 else match.kwargs["residual"]
                fused_node = graph.call_function(torch.ops.vllm.gemm_rs_ag_gemm.default, kwargs=kwargs)

                graph.inserting_after(fused_node)
                result_node_new = graph.call_function(operator.getitem, (fused_node, 0))
                residual_node_new = graph.call_function(operator.getitem, (fused_node, 1))
                my_residual_node_new = graph.call_function(operator.getitem, (fused_node, 2))
                res_replacements.append(residual_node_new)
                my_res_replacements.append(my_residual_node_new)

            rms_node = find_auto_fn(reversed(match.nodes), torch.ops._C.fused_add_rms_norm.default)
            gemm_node = find_fn(reversed(match.nodes), torch.ops.aten.mm.default)
            assert rms_node is not None
            assert gemm_node is not None

            assert len(rms_node.users) == 2
            assert len(gemm_node.users) == 1 or len(gemm_node.users) == 2

            find_getitem(rms_node, 2).replace_all_uses_with(residual_node_new)
            gemm_node.replace_all_uses_with(result_node_new)

        # Finally, remove matched nodes
        graph.eliminate_dead_code()
        assert all(node not in graph.nodes for match in matches for node in match.nodes)

    def __call__(self, graph: fx.Graph):
        self.dump_graph(graph, "before_collective_fusion")
        count = self.my_patterns.apply(graph)
        logger.info(f"fused gemm match count = {len(self.matches)}")

        # Don't apply final pattern unless we've matched and replaced the
        # gemm+collective ops.
        if len(self.matches) > 0:
            count =self. my_patterns2.apply(graph)
            logger.info(f"final match count = {count}")
            self.process_matches(graph)

        self.dump_graph(graph, "after_collective_fusion")
        self.matches.clear()
