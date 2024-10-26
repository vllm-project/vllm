import copy
import operator
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

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
    # inductor can inplace modify the graph, so we need to copy it
    # see https://github.com/pytorch/pytorch/issues/138980
    graph = copy.deepcopy(graph)
    return compile_fx(graph, example_inputs, config_patches=current_config)


def vllm_backend(
        graph,
        example_inputs,
        additional_inductor_config: Optional[Dict] = None) -> Callable:

    context = get_compile_context()
    context = copy.deepcopy(context) if context is not None else []
    sizes_to_specialize: List[int] = context

    from vllm.plugins import get_attention_ops
    attention_ops = get_attention_ops()

    from torch._dynamo.utils import lazy_format_graph_code

    # split graph by attention
    subgraph_id = 0
    node_to_subgraph_id = {}
    attention_graphs = []
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if node.op == 'call_function' and str(node.target) in attention_ops:
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            attention_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    graph_pool = torch.cuda.graph_pool_handle()
    final_graph: Callable = None  # type: ignore

    if subgraph_id != 0:
        # `keep_original_order` is important!
        # otherwise pytorch might reorder the nodes and
        # the semantics of the graph will change when we
        # have mutations in the graph
        split_gm = torch.fx.passes.split_module.split_module(
            graph,
            None,
            lambda node: node_to_subgraph_id[node],
            keep_original_order=True)

        for (name, module) in list(split_gm.named_modules()):
            if name == "":
                # stitching module
                logger.debug("%s",
                             lazy_format_graph_code("stiching module", module))
                continue
            if "." in name:
                # recursive child module
                continue
            graph_id = int(name.replace("submod_", ""))
            if graph_id not in attention_graphs:
                # cannot setattr to a module, so we need to set it to the dict
                split_gm.__dict__[name] = piecewise_backend(
                    module, sizes_to_specialize, additional_inductor_config,
                    graph_pool)

        final_graph = split_gm

    else:
        final_graph = piecewise_backend(graph, sizes_to_specialize,
                                        additional_inductor_config, graph_pool)

    # trigger the first compilation
    logger.info("Compiling a graph for general shapes")
    final_graph(*example_inputs)

    sym_shape_indices = [
        i for i, x in enumerate(example_inputs) if isinstance(x, torch.SymInt)
    ]

    seen_shapes: Set[Tuple[int, ...]] = set()

    def forward(*args):
        shape = tuple(args[i] for i in sym_shape_indices)
        if shape[0] in sizes_to_specialize and shape not in seen_shapes:
            logger.info("Compiling a graph for shapes %s", shape)
            seen_shapes.add(shape)

        return final_graph(*args)

    return forward


def piecewise_backend(graph, sizes_to_specialize, additional_inductor_config,
                      graph_pool):

    # flags for all the seen shapes, whether we need to specialize
    runtime_shapes_to_compile_flags: Dict[Tuple[int, ...], bool] = {}

    # if we need to specialize, the compiled graph for that shape
    runtime_shapes_to_compiled_graph: Dict[Tuple[int, ...], Tuple] = {}

    sym_shape_indices: List[int] = []

    compile_run = True
    first_run = True
    graph_for_symbolic_shape: Callable = None  # type: ignore

    # this is the function we return to run finally
    def compiled_graph_wrapper(*args):
        nonlocal first_run, compile_run
        nonlocal runtime_shapes_to_compile_flags
        nonlocal runtime_shapes_to_compiled_graph
        nonlocal graph_for_symbolic_shape
        nonlocal sym_shape_indices

        if compile_run:
            compile_run = False

            # this is the first compilation, we will compile a graph with
            # dynamic shape, as the caller will mark first dimension as dynamic

            graph_for_symbolic_shape = wrap_inductor(
                graph, args, additional_inductor_config)

            # TODO: Dynamo does not pass all dynamic shapes.
            # Need to investigate why. It works now because all the dynamic
            # shapes have the same value, and either of them can be used.
            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]

            return graph(*args)

        if first_run:
            first_run = False
            return graph_for_symbolic_shape(*args)

        runtime_shapes: Tuple[int,
                              ...] = tuple(args[i] for i in sym_shape_indices)

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
            compiled_graph = wrap_inductor(graph, args,
                                           additional_inductor_config)
            cudagraph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cudagraph, pool=graph_pool):
                output = compiled_graph(*args)
            runtime_shapes_to_compiled_graph[runtime_shapes] = (cudagraph,
                                                                output)

        (cudagraph, output) = runtime_shapes_to_compiled_graph[runtime_shapes]
        cudagraph.replay()
        return output

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
