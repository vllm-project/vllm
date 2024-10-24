import copy
import operator
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.fx as fx
from typing import Tuple, List, Optional

from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement, fwd_only, joint_fwd_bwd, Match

from vllm.distributed.parallel_state import get_tp_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank

from vllm.logger import init_logger

from .compile_context import get_compile_context
from .levels import CompilationLevel

logger = init_logger(__name__)

aten = torch.ops.aten

FILENO=0


def pprint(x):
    #print(x)
    pass


# This check is a hack, copied from linear.py
def should_slice(shape) -> bool:
    n_slices = get_tensor_model_parallel_world_size()
    return (shape[0] % n_slices == 0 and shape[0] >= 128)


# getitem_1 = residual (mutation)
# arg7_1 = first gemm weights
# getitem_24 = first gemm activation
# arg8_1 = rms norm weights
# getitem_1a = residual
# arg9_1 = second gemm weights
def match_gemm_rs_ag_gemm(getitem_1,   # residual
                          arg7_1,      # first gemm weight
                          getitem_22,  # first gemm activation
                          arg8_1,      # rms norm weight
                          arg9_1,      # second gemm weight
                          ):
    permute_2 = torch.ops.aten.permute.default(arg7_1, [1, 0])
    mm_1 = torch.ops.aten.mm.default(getitem_22, permute_2)
    auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_1, group_name = 'tp:0')  # how to deal with groupname?
    getitem_25 = auto_functionalized_4[1]
    auto_functionalized_5 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = getitem_25, residual = getitem_1, weight = arg8_1, epsilon = 1e-05)
    getitem_27 = auto_functionalized_5[1]
    getitem_28 = auto_functionalized_5[2]  # new residual
    permute_3 = torch.ops.aten.permute.default(arg9_1, [1, 0])
    mm_2 = torch.ops.aten.mm.default(getitem_27, permute_3)
    return mm_2, getitem_28


def slices(residual) -> List[torch.Tensor]:
    n_slices = get_tensor_model_parallel_world_size()
    residual_slices = torch.chunk(residual, n_slices, dim=0)
    #pprint(f"SLICES {[r.shape for r in residual_slices]}")
    return residual_slices


@torch.library.custom_op("vllm::gemm_rs_ag_gemm_orig", mutates_args=())
def gemm_rs_ag_gemm(getitem_1: torch.Tensor,    # residual
                    getitem_28: torch.Tensor,   # my residual
                    arg7_1: torch.Tensor,       # first gemm weights
                    getitem_22: torch.Tensor,   # first gemm activation
                    arg8_1: torch.Tensor,       # rms norm weights
                    arg9_1: torch.Tensor,       # second gemm weights
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    group_name = torch.distributed.group.WORLD.group_name # TODO: factor out to setup
    slice_size = 2048
    split_1 = torch.ops.aten.split.Tensor(getitem_1, slice_size)
    getitem_26 = split_1[0];  split_1 = None
    permute_3 = torch.ops.aten.permute.default(arg7_1, [1, 0])
    clone = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format)
    fused_matmul_reduce_scatter = torch.ops.symm_mem.fused_matmul_reduce_scatter.default(getitem_22, clone, 'avg', 0, group_name)
    auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = fused_matmul_reduce_scatter, residual = getitem_26, weight = arg8_1, epsilon = 1e-05)
    getitem_29 = auto_functionalized_4[1]
    getitem_30 = auto_functionalized_4[2]
    slice_scatter_2 = torch.ops.aten.slice_scatter.default(getitem_1, getitem_30, 0, 0, slice_size)
    split_2 = torch.ops.aten.split.Tensor(slice_scatter_2, slice_size)
    getitem_31 = split_2[0]
    permute_5 = torch.ops.aten.permute.default(arg9_1, [1, 0])
    clone_1 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format)
    fused_all_gather_matmul = torch.ops.symm_mem.fused_all_gather_matmul.default(getitem_29, [clone_1], 0, group_name) # XXXXXXXXXXX
    getitem_34 = fused_all_gather_matmul[1]
    getitem_35 = getitem_34[0]
    return getitem_35, getitem_31


# First split only on first occurrence!
# need to introduce splits since original graph does not have them.

@torch.library.custom_op("vllm::gemm_rs_ag_gemm", mutates_args=())
def gemm_rs_ag_gemm(getitem_1: torch.Tensor,
                    getitem_28: torch.Tensor,
                    arg7_1: torch.Tensor,
                    getitem_22: torch.Tensor,
                    arg8_1: torch.Tensor,
                    arg9_1: torch.Tensor,
                    first_layer: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    print(f"CUSTOM {getitem_1.shape}, should_slice={should_slice(getitem_1.shape)}, first={first_layer}")

    # this is terrible
    if True:
        res_slices = slices(getitem_1)
        slice_size = res_slices[get_tensor_model_parallel_rank()].shape[0]
    else:
        slice_size = 2048
    print(f"SLICE_SIZE = {slice_size}, orig_shape={getitem_1.shape}, slice_shapes=[{[x.shape for x in res_slices]}]")

    if should_slice(getitem_1.shape) and first_layer:
        print(f"FIRST! rank={get_tensor_model_parallel_rank}")
        split_1 = torch.ops.aten.split.Tensor(getitem_1, slice_size)
        getitem_26 = split_1[0];  split_1 = None
    else:
        getitem_26 = getitem_1

    if not should_slice(getitem_1.shape):
        print("NAIVE")
        permute_3 = torch.ops.aten.permute.default(arg7_1, [1, 0])
        output = torch.matmul(getitem_22, permute_3)
        # all reduce?
        auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input=output, residual=getitem_26, weight=arg8_1, epsilon=1e-05)
        getitem_29 = auto_functionalized_4[1]
        getitem_30 = auto_functionalized_4[2]
        permute_5 = torch.ops.aten.permute.default(arg9_1, [1, 0])
        getitem_35 = torch.matmul(getitem_29, permute_5)
        getitem_1 = getitem_26

        print(f"DONE CUSTOM NAIVE {getitem_30.shape}")
        return getitem_35, getitem_30
    else:
        group_name = torch.distributed.group.WORLD.group_name # TODO: factor out to setup
        permute_3 = torch.ops.aten.permute.default(arg7_1, [1, 0])
        clone = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format)
        output = torch.ops.symm_mem.fused_matmul_reduce_scatter.default(getitem_22, clone, 'avg', 0, group_name)
        auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input=output, residual=getitem_26, weight=arg8_1, epsilon=1e-05)
        getitem_29 = auto_functionalized_4[1]
        getitem_30 = auto_functionalized_4[2]
        slice_scatter_2 = torch.ops.aten.slice_scatter.default(getitem_28, getitem_30, 0, 0, slice_size)
        split_2 = torch.ops.aten.split.Tensor(slice_scatter_2, slice_size)
        getitem_31 = split_2[0]
        getitem_1 = getitem_31
        permute_5 = torch.ops.aten.permute.default(arg9_1, [1, 0])
        clone_1 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format)
        fused_all_gather_matmul = torch.ops.symm_mem.fused_all_gather_matmul.default(getitem_29, [clone_1], 0, group_name)
        getitem_34 = fused_all_gather_matmul[1]
        getitem_35 = getitem_34[0]

        print(f"DONE CUSTOM {getitem_31.shape}")
        return getitem_35, getitem_31   # matmul, residual  #####


# this is wrong?  do we need it?
@torch.library.register_fake("vllm::gemm_rs_ag_gemm")
def gemm_rs_ag_gemm_fake(getitem_1: torch.Tensor,
                         getitem_28: torch.Tensor,
                         arg7_1: torch.Tensor,
                         getitem_22: torch.Tensor,
                         arg8_1: torch.Tensor,
                         arg9_1: torch.Tensor,
                         first_layer: bool,
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    mm_res = torch.empty((getitem_22.shape[0], arg9_1.shape[0]), device=getitem_22.device, dtype=getitem_22.dtype)  #???
    resid = torch.empty_like(getitem_1)
    return (mm_res, resid)


# doesn't matter, only needed for signature
def replace_gemm_rs_ag_gemm(getitem_1, arg7_1, getitem_22, arg8_1, arg9_1):
    results = torch.ops.vllm.gemm_rs_ag_gemm(getitem_1, getitem_1, arg7_1, getitem_22, arg8_1, arg9_1)
    getitem_34 = results[0]
    getitem_35 = results[1]
    return getitem_34, getitem_35


def match_final(arg227_1, getitem_1022, getitem_1020, arg228_1):
    permute_128 = torch.ops.aten.permute.default(arg227_1, [1, 0])
    mm_127 = torch.ops.aten.mm.default(getitem_1022, permute_128)
    auto_functionalized_224 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_127, group_name = 'tp:0') # TODO: not same as group name
    getitem_1024 = auto_functionalized_224[1]
    auto_functionalized_225 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = getitem_1024, residual = getitem_1020, weight = arg228_1, epsilon = 1e-05)
    getitem_1026 = auto_functionalized_225[1]
    return getitem_1026


def replace_final(arg227_1, getitem_1215, getitem_1209, arg228_1):
    tp_group_name = "tp:0" # f"tp:{group_name}" # TODO: not same as group name

    permute_254 = torch.ops.aten.permute.default(arg227_1, [1, 0])
    mm_1 = torch.ops.aten.mm.default(getitem_1215, permute_254)
    auto_functionalized_161 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_1, group_name = tp_group_name)
    getitem_1217 = auto_functionalized_161[1]

    if should_slice(getitem_1209.shape):
        group_name = torch.distributed.group.WORLD.group_name # factor out?
        all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_1209, 2, group_name)
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor)
    else:
        wait_tensor = getitem_1209

    auto_functionalized_162 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = getitem_1217, residual = wait_tensor, weight = arg228_1, epsilon = 1e-05)
    getitem_1219 = auto_functionalized_162[1]
    return getitem_1219


my_patterns: Optional[PatternMatcherPass] = None
my_patterns2: Optional[PatternMatcherPass] = None

def get_matches():
    global my_patterns, my_patterns2
    matches = []

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
        inputs = [resid, resid, x, w, resid_w, x2]
        inputs_small = inputs[0:2]
        inputs_med = inputs[0:4]

        register_replacement(match_gemm_rs_ag_gemm,
                             replace_gemm_rs_ag_gemm,
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

    return matches


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
    first_layer = True  # hacky

    nodes = list(graph.nodes)
    first_match = None
    min_node = None
    for match in matches:
        first_node_in_match = min(match.nodes, key=lambda x: nodes.index(x))
        if not min_node or nodes.index(first_node_in_match) < min_node:
            min_node = nodes.index(first_node_in_match)
            first_match = match

    for match in matches:
        last_node_in_match = match.nodes[-1] #max(match.nodes, key=lambda x: nodes.index(x))

        with graph.inserting_after(last_node_in_match):
            kwargs = match.kwargs
            kwargs["first_layer"] = match == first_match
            kwargs["getitem_28"] = find_auto_fn(match.nodes, torch.ops._C.fused_add_rms_norm.default)
            first_layer = False
            fused_node = graph.call_function(torch.ops.vllm.gemm_rs_ag_gemm.default, kwargs=kwargs)

            graph.inserting_after(fused_node)
            result_node_new = graph.call_function(operator.getitem, (fused_node, 0))
            residual_node_new = graph.call_function(operator.getitem, (fused_node, 1))

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


def dump_graph(graph: torch.fx.Graph, stage: str):
    logger.info("Printing graph to %s", f"{stage}.py")
    with open(f"{stage}.py", "w") as f:
        print(graph.python_code(root_module="self", verbose=True).src, file=f)


def async_rewrite(graph: fx.Graph):
    matches = get_matches()
    matches.clear()

    count = my_patterns.apply(graph)
    print(f"fused gemm match count = {len(matches)}")

    # a bit hacky
    if len(matches) > 0:
        print("FINAL MATCH")
        count = my_patterns2.apply(graph)
        print(f"final match count = {count}")
        print("FINAL MATCH DONE")
        process_matches(graph, matches)

    return graph


def fix_functionalization(graph: fx.Graph):
    """
    Rewrite the graph module to replace the pattern involving
    torch._higher_order_ops.auto_functionalize.auto_functionalized
    with a direct call to the inplace custom op.

    # TODO: check if PyTorch nightly has fixed this issue
    """

    # debug code, if we want to see the graph before the transformation
    # with open("before.py", "w") as f:
    #     print(graph.python_code(root_module="self", verbose=True).src, file=f)

    nodes_to_remove = []

    for node in graph.nodes:
        # Identify the auto_functionalized node
        if node.op == 'call_function' and node.target == torch._higher_order_ops.auto_functionalize.auto_functionalized:  # noqa
            if node.args[0] == torch.ops._C.rotary_embedding.default:
                # manual replace for rotary_embedding

                # Now, collect the arguments
                kwargs = node.kwargs

                query = kwargs['query']
                mm_node = query.args[0].args[0]

                # Create a new call to torch.ops._C.rotary_embedding.default
                with graph.inserting_before(node):
                    # just insert the call to the custom op
                    # NOTE: don't run dead code elimination,
                    # otherwise this op will be removed
                    graph.call_function(torch.ops._C.rotary_embedding.default,
                                        kwargs=kwargs)

                # Remove the auto_functionalized node
                # Since the node may have outputs, we need to handle its users
                # Replace uses of the outputs (getitem nodes) with mm_node
                for user in list(node.users):
                    if user.op == 'call_function' and user.target == operator.getitem:  # noqa
                        # Remove the getitem node
                        for getitem_user in list(user.users):
                            if (getitem_user.op == 'call_function'
                                    and getitem_user.target
                                    == torch.ops.aten.slice_scatter.default):
                                # Replace the uses of slice_scatter node
                                # with mm_node
                                getitem_user.replace_all_uses_with(mm_node)
                                nodes_to_remove.append(getitem_user)
                        nodes_to_remove.append(user)
                nodes_to_remove.append(node)

            elif node.args[0] == torch.ops._C.fused_add_rms_norm.default:
                # manual replace for fused_add_rms_norm
                # this is the most effective optimization for llama
                # failing to do this will result in many unnecessary copies

                kwargs = node.kwargs

                input = kwargs['input']
                residual = kwargs['residual']

                # Create a new call to torch.ops._C.rotary_embedding.default
                with graph.inserting_before(node):
                    # just insert the call to the custom op
                    # NOTE: don't run dead code elimination,
                    # otherwise this op will be removed
                    graph.call_function(
                        torch.ops._C.fused_add_rms_norm.default, kwargs=kwargs)

                for user in list(node.users):
                    if user.op == 'call_function' and user.target == operator.getitem:  # noqa
                        # Remove the getitem node
                        if user.args[1] == 1:
                            replace_node = input
                        elif user.args[1] == 2:
                            replace_node = residual
                        user.replace_all_uses_with(replace_node)
                        nodes_to_remove.append(user)
                nodes_to_remove.append(node)

            elif node.args[0] == torch.ops._C.rms_norm.default:
                # manual replace for rms_norm

                kwargs = node.kwargs

                input = kwargs['input']
                out = kwargs['out']
                weight = kwargs['weight']
                epsilon = kwargs['epsilon']
                # Create a new call to torch.ops._C.rotary_embedding.default
                # cannot use kwargs, because we have an `out`, see https://github.com/pytorch/pytorch/blob/a00faf440888ffb724bad413f329a49e2b6388e7/torch/_inductor/lowering.py#L351 # noqa
                with graph.inserting_before(node):
                    # just insert the call to the custom op
                    # NOTE: don't run dead code elimination,
                    # otherwise this op will be removed
                    graph.call_function(
                        torch.ops._C.rms_norm.default,
                        args=(out, input, weight, epsilon),
                    )

                replace_node = out

                for user in list(node.users):
                    if user.op == 'call_function' and user.target == operator.getitem:  # noqa
                        user.replace_all_uses_with(replace_node)
                        nodes_to_remove.append(user)
                nodes_to_remove.append(node)

            elif node.args[0] == torch.ops._C.silu_and_mul.default:
                # manual replace for silu_and_mul

                kwargs = node.kwargs

                input = kwargs['input']
                out = kwargs['out']

                # Create a new call to torch.ops._C.rotary_embedding.default
                # cannot use kwargs, because we have an `out`, see https://github.com/pytorch/pytorch/blob/a00faf440888ffb724bad413f329a49e2b6388e7/torch/_inductor/lowering.py#L351 # noqa
                with graph.inserting_before(node):
                    # just insert the call to the custom op
                    # NOTE: don't run dead code elimination,
                    # otherwise this op will be removed
                    graph.call_function(
                        torch.ops._C.silu_and_mul.default,
                        args=(out, input),
                    )
                replace_node = out

                for user in list(node.users):
                    if user.op == 'call_function' and user.target == operator.getitem:  # noqa
                        user.replace_all_uses_with(replace_node)
                        nodes_to_remove.append(user)
                nodes_to_remove.append(node)

    # Remove the nodes all at once
    for node in nodes_to_remove:
        graph.erase_node(node)

    # debug code, if we want to see the graph after the transformation
    # with open("after.py", "w") as f:
    #     print(graph.python_code(root_module="self", verbose=True).src, file=f)


def wrap_inductor(graph, example_inputs, additional_inductor_config):
    from torch._inductor import config
    torch._inductor.config._micro_pipeline_tp = True
    current_config = config.shallow_copy_dict()
    from torch._inductor.compile_fx import compile_fx

    if additional_inductor_config is not None:
        current_config.update(additional_inductor_config)
    if current_config['post_grad_custom_post_pass'] is not None:
        logger.warning(
            "post_grad_custom_post_pass is already set in the config. "
            "Overwriting it with the fix_functionalization")
    current_config['post_grad_custom_post_pass'] = fix_functionalization
    return compile_fx(graph, example_inputs, config_patches=current_config)


def vllm_backend(
        graph,
        example_inputs,
        additional_inductor_config: Optional[Dict] = None) -> Callable:

    context = get_compile_context()
    context = copy.deepcopy(context) if context is not None else []
    sizes_to_specialize: List[int] = context

    # flags for all the seen shapes, whether we need to specialize
    runtime_shapes_to_compile_flags: Dict[Tuple[int, ...], bool] = {}

    # if we need to specialize, the compiled graph for that shape
    runtime_shapes_to_compiled_graph: Dict[Tuple[int, ...], Callable] = {}

    # this is the first compilation, we will compile a graph with
    # dynamic shape, as the caller will mark first dimension as dynamic
    logger.info("Compiling a graph for general shapes")
    graph_for_symbolic_shape = wrap_inductor(graph, example_inputs,
                                             additional_inductor_config)

    # TODO: Dynamo does not pass all dynamic shapes.
    # Need to investigate why. It works now because all the dynamic
    # shapes have the same value, and either of them can be used.
    sym_shape_indices = [
        i for i, x in enumerate(example_inputs) if isinstance(x, torch.SymInt)
    ]

    first_run = True

    # this is the function we return to Dynamo to run finally
    def compiled_graph_wrapper(*args):

        runtime_shapes: Tuple[int,
                              ...] = tuple(args[i] for i in sym_shape_indices)

        nonlocal first_run
        nonlocal runtime_shapes_to_compile_flags
        nonlocal runtime_shapes_to_compiled_graph

        if first_run:
            # the first compilation is for profiling, we directly run it
            first_run = False
            return graph_for_symbolic_shape(*args)

        if runtime_shapes not in runtime_shapes_to_compile_flags:
            # we haven't seen this shape before
            # query if we need to specialize for this shape
            # we only specialize for the first dimension.
            # TODO: investigate if any model needs to specialize
            # beyond the first dimension
            runtime_shapes_to_compile_flags[runtime_shapes] = runtime_shapes[
                0] in sizes_to_specialize

        if not runtime_shapes_to_compile_flags[runtime_shapes]:
            # we don't need to specialize for this shape
            return graph_for_symbolic_shape(*args)

        if runtime_shapes not in runtime_shapes_to_compiled_graph:
            # we need to specialize for this shape, and we haven't compiled
            # compile the graph for this shape
            logger.info("Compiling a graph for shapes %s", runtime_shapes)
            runtime_shapes_to_compiled_graph[runtime_shapes] = wrap_inductor(
                graph, args, additional_inductor_config)

        return runtime_shapes_to_compiled_graph[runtime_shapes](*args)

    return compiled_graph_wrapper


def select_default_backend(level: int) -> Union[str, Callable]:
    if level in [CompilationLevel.DYNAMO_AS_IS, CompilationLevel.DYNAMO_ONCE]:
        backend_str = "eager"
        return backend_str
    assert level in [
        CompilationLevel.INDUCTOR, CompilationLevel.INDUCTOR_MAX_AUTOTUNE
    ], f"Invalid level {level}"

    from vllm.compilation.backends import vllm_backend
    from vllm.plugins import get_inductor_additional_configs
    additional_configs = get_inductor_additional_configs()

    if level == CompilationLevel.INDUCTOR_MAX_AUTOTUNE:
        if "max_autotune" in additional_configs and not additional_configs[
                "max_autotune"]:
            logger.warning(
                "max_autotune is disabled, but is overridden by level %s",
                CompilationLevel.INDUCTOR_MAX_AUTOTUNE)
        additional_configs['max_autotune'] = True

    from functools import partial
    backend = partial(vllm_backend,
                      additional_inductor_config=additional_configs)

    return backend
