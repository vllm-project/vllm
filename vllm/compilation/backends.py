import copy
import operator
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.fx as fx

from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement, fwd_only, joint_fwd_bwd

from vllm.distributed.parallel_state import get_tp_group

from vllm.logger import init_logger

from .compile_context import get_compile_context
from .levels import CompilationLevel

logger = init_logger(__name__)

aten = torch.ops.aten

FILENO=0

def match_gemm_rs_ag_gemm_orig():
    permute_2 = torch.ops.aten.permute(arg7_1, [1, 0])
    mm_1 = torch.ops.aten.mm(getitem_22, permute_2)
    auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce, tensor=mm_1, group_name='tp:0')
    getitem_25 = auto_functionalized_4[1]
    auto_functionalized_5 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input=getitem_25, residual=getitem_1, weight=arg8_1, epsilon=1e-05)
    getitem_27 = auto_functionalized_5[1]
    getitem_28 = auto_functionalized_5[2]
    permute_3 = torch.ops.aten.permute(arg9_1, [1, 0])
    mm_2 = torch.ops.aten.mm(getitem_27, permute_3)
    return mm_2


def match_gemm_rs_ag_gemm_small(arg7_1, getitem_22):
    permute_2 = torch.ops.aten.permute.default(arg7_1, [1, 0])
    mm_1 = torch.ops.aten.mm.default(getitem_22, permute_2)
    auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_1, group_name = 'tp:0')  # how to deal with groupname?
    getitem_25 = auto_functionalized_4[1]
    return getitem_25


def match_gemm_rs_ag_gemm_med(arg7_1, getitem_22, getitem_1, arg8_1):
    permute_2 = torch.ops.aten.permute.default(arg7_1, [1, 0])
    mm_1 = torch.ops.aten.mm.default(getitem_22, permute_2)
    auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_1, group_name = 'tp:0')  # how to deal with groupname?
    getitem_25 = auto_functionalized_4[1]
    auto_functionalized_5 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = getitem_25, residual = getitem_1, weight = arg8_1, epsilon = 1e-05)
    getitem_27 = auto_functionalized_5[1]
    getitem_28 = auto_functionalized_5[2]
    return getitem_27, getitem_28
    #permute_3 = torch.ops.aten.permute.default(arg9_1, [1, 0])
    #mm_2 = torch.ops.aten.mm.default(getitem_27, permute_3)
    #return mm_2


def match_gemm_rs_ag_gemm(arg7_1, getitem_22, arg8_1, getitem_1, arg9_1):
    permute_2 = torch.ops.aten.permute.default(arg7_1, [1, 0])
    mm_1 = torch.ops.aten.mm.default(getitem_22, permute_2)
    auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_1, group_name = 'tp:0')  # how to deal with groupname?
    getitem_25 = auto_functionalized_4[1]
    auto_functionalized_5 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = getitem_25, residual = getitem_1, weight = arg8_1, epsilon = 1e-05)
    getitem_27 = auto_functionalized_5[1]
    getitem_28 = auto_functionalized_5[2]
    permute_3 = torch.ops.aten.permute.default(arg9_1, [1, 0])
    mm_2 = torch.ops.aten.mm.default(getitem_27, permute_3)
    return mm_2, getitem_28


# getitem_1 full residual
def replace_gemm_rs_ag_gemm(arg7_1, getitem_24, arg8_1, getitem_1, arg9_1):
    split_1 = torch.ops.aten.split.Tensor(getitem_1, 2048)
    getitem_26 = split_1[0];  split_1 = None
    permute_3 = torch.ops.aten.permute.default(arg7_1, [1, 0])
    clone = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format)
    fused_matmul_reduce_scatter = torch.ops.symm_mem.fused_matmul_reduce_scatter.default(getitem_24, clone, 'avg', 0, '0')
    auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = fused_matmul_reduce_scatter, residual = getitem_26, weight = arg8_1, epsilon = 1e-05)
    getitem_29 = auto_functionalized_4[1]
    getitem_30 = auto_functionalized_4[2]
    slice_scatter_2 = torch.ops.aten.slice_scatter.default(getitem_1, getitem_30, 0, 0, 2048)
    split_2 = torch.ops.aten.split.Tensor(slice_scatter_2, 2048)
    getitem_31 = split_2[0]  # local residual
    permute_5 = torch.ops.aten.permute.default(arg9_1, [1, 0])
    clone_1 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format)
    fused_all_gather_matmul = torch.ops.symm_mem.fused_all_gather_matmul.default(getitem_29, [clone_1], 0, '0')
    getitem_34 = fused_all_gather_matmul[1]
    getitem_35 = getitem_34[0]
    return getitem_35, getitem_31


def match_final(arg227_1, getitem_1022, getitem_1020, arg228_1):
    permute_128 = torch.ops.aten.permute.default(arg227_1, [1, 0])
    mm_127 = torch.ops.aten.mm.default(getitem_1022, permute_128)
    auto_functionalized_224 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_127, group_name = 'tp:0')
    getitem_1024 = auto_functionalized_224[1]
    auto_functionalized_225 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = getitem_1024, residual = getitem_1020, weight = arg228_1, epsilon = 1e-05)
    getitem_1026 = auto_functionalized_225[1]
    return getitem_1026


def replace_final(arg227_1, getitem_1215, getitem_1209, arg228_1):
    permute_254 = torch.ops.aten.permute.default(arg227_1, [1, 0])
    mm_1 = torch.ops.aten.mm.default(getitem_1215, permute_254)
    auto_functionalized_161 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_1, group_name = 'tp:0')
    getitem_1217 = auto_functionalized_161[1]
    all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_1209, 2, '0')
    wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor)
    auto_functionalized_162 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = getitem_1217, residual = wait_tensor, weight = arg228_1, epsilon = 1e-05)
    getitem_1219 = auto_functionalized_162[1]
    return getitem_1219


my_patterns = PatternMatcherPass()
my_patterns2 = PatternMatcherPass()
x = torch.empty([4,4], device='cuda')
w = torch.empty([4,4], device='cuda')
resid = torch.empty([4,4], device='cuda')
resid_w = torch.empty([4,4], device='cuda')
x2 = torch.empty([4,4], device='cuda')
inputs = [x, w, resid, resid_w, x2]
inputs_small = inputs[0:2]
inputs_med = inputs[0:4]

register_replacement(match_gemm_rs_ag_gemm,
                     replace_gemm_rs_ag_gemm,
                     inputs,
                     fwd_only,
                     [my_patterns])

final_inputs = [x, w, resid, resid_w]
register_replacement(match_final,
                     replace_final,
                     final_inputs,
                     fwd_only,
                     [my_patterns2])

def async_rewrite(graph: fx.Graph):
    count = my_patterns.apply(graph)
    print(f"fused gemm match count = {count}")
    count = my_patterns2.apply(graph)
    print(f"final match count = {count}")
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
