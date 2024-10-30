import operator
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.fx as fx

from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement, fwd_only, joint_fwd_bwd, Match

from vllm.distributed.parallel_state import get_tp_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.distributed import tensor_model_parallel_all_reduce

from vllm.logger import init_logger

logger = init_logger(__name__)


def pprint(x):
    #print(x)
    pass


# This check is a hack, copied from linear.py
def should_slice(shape) -> bool:
    n_slices = get_tensor_model_parallel_world_size()
    return (shape[0] % n_slices == 0 and shape[0] >= 128)


def match_gemm_rs_ag_gemm(residual,
                          gemm_1_weights,
                          gemm_1_activations,
                          rms_norm_weight,
                          gemm_2_weights,
                          ):
    permute_2 = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, permute_2)
    auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_1, group_name = 'tp:0')  # how to deal with groupname?
    getitem_25 = auto_functionalized_4[1]
    auto_functionalized_5 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = getitem_25, residual = residual, weight = rms_norm_weight, epsilon = 1e-05)
    getitem_27 = auto_functionalized_5[1]
    getitem_28 = auto_functionalized_5[2]  # new residual
    permute_3 = torch.ops.aten.permute.default(gemm_2_weights, [1, 0])
    mm_2 = torch.ops.aten.mm.default(getitem_27, permute_3)
    return mm_2, getitem_28


def slices(residual) -> List[torch.Tensor]:
    n_slices = get_tensor_model_parallel_world_size()
    residual_slices = torch.chunk(residual, n_slices, dim=0)
    #pprint(f"SLICES {[r.shape for r in residual_slices]}")
    return residual_slices


@torch.library.custom_op("vllm::gemm_rs_ag_gemm", mutates_args=())
def gemm_rs_ag_gemm(residual: torch.Tensor,
                    my_residual: torch.Tensor,
                    gemm_1_weights: torch.Tensor,
                    gemm_1_activations: torch.Tensor,
                    rms_norm_weight: torch.Tensor,
                    gemm_2_weights: torch.Tensor,
                    first_layer: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    print(f"CUSTOM {residual.shape}({my_residual.shape}), should_slice={should_slice(residual.shape)}, first={first_layer}")

    ####
    # this is terrible
    res_slices = slices(residual)
    slice_size = res_slices[get_tensor_model_parallel_rank()].shape[0]
    ####
    print(f"SLICE_SIZE = {slice_size}, orig_shape={residual.shape}, slice_shapes=[{[x.shape for x in res_slices]}]")

    if should_slice(residual.shape) and first_layer:
        print(f"FIRST! rank={get_tensor_model_parallel_rank()}")
        split_1 = torch.ops.aten.split.Tensor(residual, slice_size)
        getitem_26 = split_1[0];  split_1 = None
    else:
        #getitem_26 = my_residual
        getitem_26 = residual
        slice_size = residual.shape[0]

    if not should_slice(residual.shape):
        # this branch probably broken
        print("NAIVE")
        permute_3 = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
        output = torch.matmul(gemm_1_activations, permute_3)

        output = tensor_model_parallel_all_reduce(output)  ###

        auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input=output, residual=getitem_26, weight=rms_norm_weight, epsilon=1e-05)
        getitem_29 = auto_functionalized_4[1]
        getitem_30 = auto_functionalized_4[2]

        permute_5 = torch.ops.aten.permute.default(gemm_2_weights, [1, 0])
        getitem_35 = torch.matmul(getitem_29, permute_5)
        getitem_30a = getitem_30.clone()
        print(f"DONE CUSTOM NAIVE {getitem_35.shape}, {getitem_30.shape}, {getitem_30a.shape} {first_layer}")
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

        print(f"DONE CUSTOM {getitem_35.shape}, {getitem_31.shape}, {slice_scatter_2.shape} {first_layer}")
        return getitem_35, getitem_31.clone(), slice_scatter_2   # check if clones are needed


@torch.library.register_fake("vllm::gemm_rs_ag_gemm")
def gemm_rs_ag_gemm_fake(residual: torch.Tensor,
                         my_residual: torch.Tensor,
                         gemm_1_weights: torch.Tensor,
                         gemm_1_activations: torch.Tensor,
                         rms_norm_weight: torch.Tensor,
                         gemm_2_weights: torch.Tensor,
                         first_layer: bool,
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #####
    # this is terrible
    res_slices = slices(residual)
    slice_size = res_slices[get_tensor_model_parallel_rank()].shape[0]  # can we use rank 0 instead?
    #####

#    if should_slice(residual.shape) and first_layer:
    if should_slice(gemm_1_activations.shape) and first_layer:
        print(f"FIRST! rank={get_tensor_model_parallel_rank()}")
        split_1 = torch.ops.aten.split.Tensor(residual, slice_size)
        my_residual = split_1[0];  split_1 = None
    else:
        #residual = my_residual
        #slice_size = residual.shape[0]
        my_residual = residual

    # verify the type is always correct
    mm_res = torch.empty((gemm_1_activations.shape[0], gemm_2_weights.shape[0]), device=gemm_1_activations.device, dtype=gemm_1_activations.dtype)

    print(f"DONE FAKE = {mm_res.shape}, {my_residual.shape}, {residual.shape} {first_layer}")

    return (mm_res, my_residual, residual)


# implementation doesn't matter, only needed for signature
def replace_gemm_rs_ag_gemm(residual, gemm_1_weights, gemm_1_activations, rms_norm_weight, gemm_2_weights):
    results = torch.ops.vllm.gemm_rs_ag_gemm(residual, residual, gemm_1_weights, gemm_1_activations, rms_norm_weight, gemm_2_weights)
    getitem_34 = results[0]
    getitem_35 = results[1]
    return getitem_34, getitem_35


def match_final(my_residual, gemm_1_weights, gemm_1_activations, rms_norm_weights):
    permute_128 = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_127 = torch.ops.aten.mm.default(gemm_1_activations, permute_128)

    auto_functionalized_224 = torch.ops.higher_order.auto_functionalized(
        torch.ops.vllm.inplace_all_reduce.default,
        tensor = mm_127,
        group_name = 'tp:0' # TODO: not same as group name
    )
    getitem_1024 = auto_functionalized_224[1]

    auto_functionalized_225 = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input = getitem_1024,
        residual = my_residual,
        weight = rms_norm_weights,
        epsilon = 1e-05
    )
    getitem_1026 = auto_functionalized_225[1]

    return getitem_1026


# TODO: wrap in custom op to prevent infinite recursion in inductor logging statement?
def replace_final(my_residual, gemm_1_weights, gemm_1_activations, rms_norm_weights):
    tp_group_name = "tp:0" # f"tp:{group_name}" # TODO: not same as group name

    permute_254 = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, permute_254)
    auto_functionalized_161 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_1, group_name = tp_group_name)
    getitem_1217 = auto_functionalized_161[1]

    # is this the right thing to call it on?
    if should_slice(gemm_1_activations.shape):
        group_name = torch.distributed.group.WORLD.group_name # TODO: factor out
        world_size = get_tp_group().world_size                # TODO: factor out
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


my_patterns: Optional[PatternMatcherPass] = None
my_patterns2: Optional[PatternMatcherPass] = None
matches: List[Match] = []

def get_matches():
    global my_patterns, my_patterns2, matches

    def record_match_fn(match: Match):
        matches.append(match)
        return False

    if not my_patterns:
        my_patterns = PatternMatcherPass()
        my_patterns2 = PatternMatcherPass()

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
                             [my_patterns],
                             extra_check=record_match_fn)

        final_inputs = [x, w, resid, resid_w]
        register_replacement(match_final,
                             replace_final,
                             final_inputs,
                             fwd_only,
                             [my_patterns2])



# find the output and the residual
def find_fn(nodes, op):
    for node in reversed(nodes):
        if node.op == "call_function" and node.target == op:
            return node
    return None

def find_auto_fn(nodes, op):
    for node in reversed(nodes):
        if node.op == "call_function" and node.target == auto_functionalized and node.args[0] == op:
            return node
    return None

def find_getitem(node, idx):
    for user in reversed(node.users):
        if user.op == "call_function" and user.target == operator.getitem and user.args[1] == idx:
            return user
    return None

def process_matches(graph: fx.Graph, matches):
    print(f"len = {len(matches)}")

    nodes = list(graph.nodes)
    first_match = None

    def find_min_index(match) -> int:
        return min(match.nodes, key=lambda x: nodes.index(x))

    # "sort" matches in topo order
    matches = sorted(matches, key=lambda x: find_min_index(x))

    # this is pretty hacky since the order doesn't necessarily encode the dependency.
    res_replacements = []
    my_res_replacements = []

    for match in matches:
        last_node_in_match = match.nodes[-1] #max(match.nodes, key=lambda x: nodes.index(x))

        with graph.inserting_after(last_node_in_match):
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

        rms_node = find_auto_fn(match.nodes, torch.ops._C.fused_add_rms_norm.default)
        gemm_node = find_fn(match.nodes, torch.ops.aten.mm.default)
        if gemm_node is None:
            gemm_node = find_fn(match.nodes, torch.ops.symm_mem.fused_all_gather_matmul.default)
        assert rms_node is not None
        assert gemm_node is not None

        #assert len(rms_node.users) == 2
        #assert len(gemm_node.users) == 1

        # meta["val"] is used by de-functionalization
        rms_val = rms_node.meta["val"]
        gemm_val = gemm_node.meta["val"]
        fused_node.meta["val"] = (gemm_val, rms_val[2])

        find_getitem(rms_node, 2).replace_all_uses_with(residual_node_new)
        gemm_node.replace_all_uses_with(result_node_new)

    # Finally, remove matched nodes
    graph.eliminate_dead_code()
    assert all(node not in graph.nodes for match in matches for node in match.nodes)


def collective_fusion(graph: fx.Graph):
    global matches
    rank = get_tensor_model_parallel_rank()
    get_matches()
    matches.clear()

    count = my_patterns.apply(graph)
    print(f"fused gemm match count = {len(matches)} {id(matches)}")

    # a bit hacky
    if len(matches) > 0:
        count = my_patterns2.apply(graph)
        print(f"final match count = {count}")
        process_matches(graph, matches)

    return graph

