import copy
import dataclasses
import operator
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.fx as fx

from vllm.logger import init_logger
from vllm.utils import weak_ref_tensors

from .config import CompilationConfig
from .counter import compilation_counter
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


def wrap_inductor(graph,
                  example_inputs,
                  additional_inductor_config,
                  do_logging=False,
                  runtime_shape: Optional[int] = None,
                  use_inductor: bool = True):
    if not use_inductor:
        return graph

    compilation_counter.num_inductor_compilations += 1

    if do_logging:
        if runtime_shape is None:
            logger.info("Compiling a graph for general shape")
        else:
            logger.info("Compiling a graph for shape %s", runtime_shape)

    from torch._inductor import config
    current_config = config.shallow_copy_dict()
    from torch._inductor.compile_fx import compile_fx

    if additional_inductor_config is not None:
        current_config.update(additional_inductor_config)

    # inductor can inplace modify the graph, so we need to copy it
    # see https://github.com/pytorch/pytorch/issues/138980
    graph = copy.deepcopy(graph)
    return compile_fx(graph, example_inputs, config_patches=current_config)


@dataclasses.dataclass
class SplitItem:
    submod_name: str
    is_splitting_graph: bool
    graph: fx.GraphModule


def split_graph(graph: fx.GraphModule,
                ops: List[str]) -> Tuple[fx.GraphModule, List[SplitItem]]:
    # split graph by ops
    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if node.op == 'call_function' and str(node.target) in ops:
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    # `keep_original_order` is important!
    # otherwise pytorch might reorder the nodes and
    # the semantics of the graph will change when we
    # have mutations in the graph
    split_gm = torch.fx.passes.split_module.split_module(
        graph,
        None,
        lambda node: node_to_subgraph_id[node],
        keep_original_order=True)

    outputs = []

    # sort the names to make sure the order is deterministic
    names = [name for (name, module) in split_gm.named_modules()]
    names.sort()

    for name in names:
        if "." in name or name == "":
            # recursive child module or the root module
            continue

        module = getattr(split_gm, name)

        graph_id = int(name.replace("submod_", ""))
        outputs.append(SplitItem(name, graph_id in split_op_graphs, module))

    return split_gm, outputs


# we share the global graph pool among all the backends
global_graph_pool = None


class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    """Code adapted from `torch.fx.passes.shape_prop.ShapeProp`.
    It runs the given graph with fake inputs, and compile some
    submodules specified by `compile_submod_names` with the given
    compilation configs.
    """

    def __init__(self, module: torch.fx.GraphModule,
                 compile_submod_names: List[str],
                 compilation_configs: CompilationConfig, graph_pool):
        super().__init__(module)
        from torch._guards import detect_fake_mode
        self.fake_mode = detect_fake_mode()
        self.compile_submod_names = compile_submod_names
        self.compilation_configs = compilation_configs
        self.graph_pool = graph_pool
        self.have_seen_first_graph = False

    def run(self, *args):
        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        return super().run(*fake_args)

    def call_module(self, target: torch.fx.node.Target,
                    args: Tuple[torch.fx.node.Argument,
                                ...], kwargs: Dict[str, Any]) -> Any:
        assert isinstance(target, str)
        output = super().call_module(target, args, kwargs)

        if target in self.compile_submod_names:
            submod = self.fetch_attr(target)
            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]
            compiled_graph_for_general_shape = wrap_inductor(
                submod,
                args,
                self.compilation_configs.inductor_compile_config,
                runtime_shape=None,
                do_logging=not self.have_seen_first_graph,
                use_inductor=self.compilation_configs.use_inductor)

            self.module.__dict__[target] = PiecewiseBackend(
                submod, self.compilation_configs, self.graph_pool,
                not self.have_seen_first_graph, sym_shape_indices,
                compiled_graph_for_general_shape)

            self.have_seen_first_graph = True
            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        return output


class VllmBackend:
    """The compilation backend for `torch.compile` with VLLM.
    It is used for compilation level of `CompilationLevel.PIECEWISE`,
    where we customize the compilation.

    The major work of this backend is to split the graph into
    piecewise graphs, and pass them to the piecewise backend.
    """

    compilation_configs: CompilationConfig
    graph_pool: Any
    _called: bool = False
    # the graph we compiled
    graph: fx.GraphModule
    # the stiching graph module for all the piecewise graphs
    split_gm: fx.GraphModule
    piecewise_graphs: List[SplitItem]
    returned_callable: Callable

    def __init__(self, ):
        global global_graph_pool
        if global_graph_pool is None:
            global_graph_pool = torch.cuda.graph_pool_handle()

        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = global_graph_pool

        # `torch.compile` is JIT compiled, so we don't need to
        # do anything here

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:

        compilation_counter.num_graphs_seen += 1

        # we control the compilation process, each instance can only be
        # called once
        assert not self._called, "VllmBackend can only be called once"

        self.graph = graph
        # config is read now, because only here can
        # we get the sizes to capture for cudagraph
        # from compilation context
        self.compilation_configs = CompilationConfig.select_and_init_config()

        self.split_gm, self.piecewise_graphs = split_graph(
            graph, self.compilation_configs.non_cudagraph_ops)

        from torch._dynamo.utils import lazy_format_graph_code
        logger.debug("%s",
                     lazy_format_graph_code("stiching module", self.split_gm))

        compilation_counter.num_piecewise_graphs_seen += len(
            self.piecewise_graphs)
        submod_names_to_compile = [
            item.submod_name for item in self.piecewise_graphs
            if not item.is_splitting_graph
        ]

        # propagate the split graph to the piecewise backend,
        # compile submodules with symbolic shapes
        PiecewiseCompileInterpreter(self.split_gm, submod_names_to_compile,
                                    self.compilation_configs,
                                    self.graph_pool).run(*example_inputs)

        self._called = True

        return self.split_gm


@dataclasses.dataclass
class ConcreteSizeEntry:
    runtime_shape: int
    need_to_compile: bool  # the size is in compile_sizes
    use_cudagraph: bool  # the size is in capture_sizes

    compiled: bool = False
    runnable: Callable = None  # type: ignore
    num_finished_warmup: int = 0
    cudagraph: Optional[torch.cuda.CUDAGraph] = None
    output: Optional[Any] = None


class PiecewiseBackend:

    def __init__(self, graph: fx.GraphModule,
                 compilation_configs: CompilationConfig, graph_pool: Any,
                 is_first_graph: bool, sym_shape_indices: List[int],
                 compiled_graph_for_general_shape: Callable):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation and cudagraph capturing.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_configs.compile_sizes`.

        Independently, we will capture cudagraph for different shapes.

        If a shape needs both compilation and cudagraph, we will
        compile it first, and then capture cudagraph.
        """
        self.graph = graph
        self.compilation_configs = compilation_configs
        self.graph_pool = graph_pool
        self.is_first_graph = is_first_graph

        self.compile_sizes: Set[int] = set(
            self.compilation_configs.compile_sizes)
        self.capture_sizes: Set[int] = set(
            self.compilation_configs.capture_sizes
        ) if self.compilation_configs.use_cudagraph else set()

        self.first_run_finished = False

        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  # noqa

        self.sym_shape_indices = sym_shape_indices

        # the entries for different shapes that we need to either
        # compile or capture cudagraph
        self.concrete_size_entries: Dict[int, ConcreteSizeEntry] = {}
        for shape in self.compile_sizes.union(self.capture_sizes):
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_shape=shape,
                need_to_compile=shape in self.compile_sizes,
                use_cudagraph=shape in self.capture_sizes,
            )

    def __call__(self, *args) -> Any:
        if not self.first_run_finished:
            self.first_run_finished = True
            return self.compiled_graph_for_general_shape(*args)

        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            # we don't need to do anything for this shape
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape

        if entry.need_to_compile and not entry.compiled:
            entry.compiled = True
            # args are real arguments
            entry.runnable = wrap_inductor(
                self.graph,
                args,
                self.compilation_configs.inductor_compile_config,
                runtime_shape=runtime_shape,
                do_logging=self.is_first_graph,
                use_inductor=self.compilation_configs.use_inductor)

        if not entry.use_cudagraph:
            return entry.runnable(*args)

        if entry.cudagraph is None:
            if entry.num_finished_warmup < self.compilation_configs.cudagraph_num_of_warmups:  # noqa
                entry.num_finished_warmup += 1
                if self.is_first_graph:
                    logger.debug(
                        "Warming up %s/%s for shape %s",
                        entry.num_finished_warmup,
                        self.compilation_configs.cudagraph_num_of_warmups,
                        runtime_shape)
                return entry.runnable(*args)

            if self.is_first_graph:
                logger.info("Capturing a cudagraph for shape %s",
                            runtime_shape)

            cudagraph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cudagraph, pool=self.graph_pool):
                entry.output = weak_ref_tensors(entry.runnable(*args))

            compilation_counter.num_cudagraph_caputured += 1

            entry.cudagraph = cudagraph
            return entry.output

        entry.cudagraph.replay()
        return entry.output


def select_default_backend(level: int) -> Union[str, Callable]:
    if level in [CompilationLevel.DYNAMO_AS_IS, CompilationLevel.DYNAMO_ONCE]:
        backend_str = "eager"
        return backend_str
    assert level == CompilationLevel.PIECEWISE

    return VllmBackend()
