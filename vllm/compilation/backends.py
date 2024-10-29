import copy
import dataclasses
import operator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.fx as fx

from vllm.logger import init_logger
from vllm.utils import weak_ref_tensors

from .config import CompilationConfig
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

    # inductor can inplace modify the graph, so we need to copy it
    # see https://github.com/pytorch/pytorch/issues/138980
    graph = copy.deepcopy(graph)
    return compile_fx(graph, example_inputs, config_patches=current_config)


def vllm_backend(graph, example_inputs) -> Callable:

    # config is read when we first compile the graph
    # (i.e. when this backend is first called)
    compilation_configs = CompilationConfig.default_config()

    from vllm.plugins import get_non_cudagraph_ops
    non_cudagraph_ops = get_non_cudagraph_ops()

    from torch._dynamo.utils import lazy_format_graph_code

    # split graph by non_cudagraph_ops
    subgraph_id = 0
    node_to_subgraph_id = {}
    attention_graphs = []
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if node.op == 'call_function' and str(
                node.target) in non_cudagraph_ops:
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

        logger.debug("%s", lazy_format_graph_code("stiching module", split_gm))

        # sort the names to make sure the order is deterministic
        names = [name for (name, module) in split_gm.named_modules()]
        names.sort()

        is_first_graph = True
        for name in names:
            if "." in name or name == "":
                # recursive child module or the root module
                continue

            module = getattr(split_gm, name)

            graph_id = int(name.replace("submod_", ""))
            if graph_id not in attention_graphs:
                # cannot setattr to a module, so we need to set it to the dict
                split_gm.__dict__[name] = piecewise_backend(
                    module, compilation_configs, graph_pool, is_first_graph)
                is_first_graph = False

        final_graph = split_gm

    else:
        final_graph = piecewise_backend(graph,
                                        compilation_configs,
                                        graph_pool,
                                        is_first_graph=True)

    # trigger the first compilation
    final_graph(*example_inputs)

    return final_graph


@dataclasses.dataclass
class Entry:
    runnable: Callable
    use_cudagraph: bool
    target_warmup_times: int
    current_warmup_times: int = 0
    cudagraph: Optional[torch.cuda.CUDAGraph] = None
    output: Optional[Any] = None


def piecewise_backend(graph,
                      compilation_configs: CompilationConfig,
                      graph_pool,
                      is_first_graph=False) -> Callable:

    # flags for all the seen shapes, whether we need to specialize
    runtime_shapes_to_compile_flags: Dict[Tuple[int, ...], bool] = {}

    # if we need to specialize, the compiled graph for that shape
    runtime_shapes_to_compiled_entry: Dict[Tuple[int, ...], Entry] = {}

    sym_shape_indices: List[int] = []

    compile_run = True
    first_run = True
    graph_for_symbolic_shape: Callable = None  # type: ignore

    # this is the function we return to run finally
    def compiled_graph_wrapper(*args):
        nonlocal first_run, compile_run
        nonlocal runtime_shapes_to_compile_flags
        nonlocal runtime_shapes_to_compiled_entry
        nonlocal graph_for_symbolic_shape
        nonlocal sym_shape_indices

        if compile_run:
            compile_run = False

            # this is the first compilation, we will compile a graph with
            # dynamic shape, as the caller will mark first dimension as dynamic

            if compilation_configs.use_inductor:
                if is_first_graph:
                    logger.info("Compiling a graph for general shape")
                graph_for_symbolic_shape = wrap_inductor(
                    graph, args, compilation_configs.inductor_compile_config)
            else:
                graph_for_symbolic_shape = graph

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
                0] in compilation_configs.compile_sizes or \
                runtime_shapes[0] in compilation_configs.capture_sizes

        if not runtime_shapes_to_compile_flags[runtime_shapes]:
            # we don't need to specialize for this shape
            return graph_for_symbolic_shape(*args)

        if runtime_shapes not in runtime_shapes_to_compiled_entry:
            # the first time we see this shape, we need to compile the graph
            if compilation_configs.use_inductor and runtime_shapes[
                    0] in compilation_configs.compile_sizes:
                if is_first_graph:
                    logger.info("Compiling a graph for shape %s",
                                runtime_shapes)
                runnable = wrap_inductor(
                    graph, args, compilation_configs.inductor_compile_config)
            else:
                runnable = graph
            use_cudagraph = compilation_configs.use_cudagraph and \
                runtime_shapes[0] in compilation_configs.capture_sizes # noqa
            entry = Entry(runnable=runnable,
                          use_cudagraph=use_cudagraph,
                          target_warmup_times=compilation_configs.
                          cudagraph_num_of_warmups)
            runtime_shapes_to_compiled_entry[runtime_shapes] = entry
        else:
            entry = runtime_shapes_to_compiled_entry[runtime_shapes]

        if not entry.use_cudagraph:
            return entry.runnable(*args)

        if entry.current_warmup_times < entry.target_warmup_times:
            entry.current_warmup_times += 1
            if is_first_graph:
                logger.debug("Warming up %s/%s for shape %s",
                             entry.current_warmup_times,
                             entry.target_warmup_times, runtime_shapes)
            return entry.runnable(*args)

        if entry.cudagraph is None:
            if is_first_graph:
                logger.info("Capturing a cudagraph for shape %s",
                            runtime_shapes)
            cudagraph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cudagraph, pool=graph_pool):
                entry.output = weak_ref_tensors(entry.runnable(*args))
            entry.cudagraph = cudagraph

        entry.cudagraph.replay()
        return entry.output

    return compiled_graph_wrapper


def select_default_backend(level: int) -> Union[str, Callable]:
    if level in [CompilationLevel.DYNAMO_AS_IS, CompilationLevel.DYNAMO_ONCE]:
        backend_str = "eager"
        return backend_str
    assert level == CompilationLevel.PIECEWISE

    return vllm_backend
