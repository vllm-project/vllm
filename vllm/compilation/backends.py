import copy
import operator
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.fx as fx

from vllm.logger import init_logger

from .compile_context import get_compile_context
from .levels import CompilationLevel

logger = init_logger(__name__)


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
        backend = "eager"
        return backend
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
