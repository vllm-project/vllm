import copy
import dataclasses
import operator
from contextlib import ExitStack
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from unittest.mock import patch

import torch
import torch.fx as fx

import vllm.envs as envs
from vllm.config import CompilationConfig
from vllm.logger import init_logger
from vllm.utils import combine_fx_passes, weak_ref_tensors

from .counter import compilation_counter
from .fusion import FusionPass
from .reshapes import RedundantReshapesPass

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
            elif (node.args[0] ==
                  torch.ops._C.fused_add_rms_norm_static_fp8_quant.default):
                # manual replace for fused_add_rms_norm_static_fp8_quant
                # this is the most effective optimization for llama
                # failing to do this will result in many unnecessary copies

                kwargs = node.kwargs

                result = kwargs['result']
                residual = kwargs['residual']

                # Create a new call to
                # torch.ops._C.fused_add_rms_norm_static_fp8_quant.default
                with graph.inserting_before(node):
                    # just insert the call to the custom op
                    # NOTE: don't run dead code elimination,
                    # otherwise this op will be removed
                    graph.call_function(
                        torch.ops._C.fused_add_rms_norm_static_fp8_quant.
                        default,
                        kwargs=kwargs)

                for user in list(node.users):
                    if user.op == 'call_function' and user.target == operator.getitem:  # noqa
                        # Remove the getitem node
                        if user.args[1] == 1:
                            replace_node = result
                        elif user.args[1] == 2:
                            replace_node = residual
                        user.replace_all_uses_with(replace_node)
                        nodes_to_remove.append(user)
                nodes_to_remove.append(node)

            elif node.args[0] == torch.ops._C.rms_norm.default:
                # manual replace for rms_norm

                kwargs = node.kwargs

                replace_node = kwargs['result']
                # Create a new call to torch.ops._C.rms_norm.default
                with graph.inserting_before(node):
                    # just insert the call to the custom op
                    # NOTE: don't run dead code elimination,
                    # otherwise this op will be removed
                    graph.call_function(torch.ops._C.rms_norm.default,
                                        kwargs=kwargs)

                for user in list(node.users):
                    if user.op == 'call_function' and user.target == operator.getitem:  # noqa
                        user.replace_all_uses_with(replace_node)
                        nodes_to_remove.append(user)
                nodes_to_remove.append(node)

            elif node.args[
                    0] == torch.ops._C.rms_norm_static_fp8_quant.default:  # noqa
                # manual replace for rms_norm_static_fp8_quant

                kwargs = node.kwargs

                replace_node = kwargs['result']
                # Create a new call to torch.ops._C.rms_norm_static_fp8_quant.default  # noqa
                with graph.inserting_before(node):
                    # just insert the call to the custom op
                    # NOTE: don't run dead code elimination,
                    # otherwise this op will be removed
                    graph.call_function(
                        torch.ops._C.rms_norm_static_fp8_quant.default,
                        kwargs=kwargs)

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

                # Create a new call to torch.ops._C.silu_and_mul.default
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
    graph_id: int
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

    names = [name for (name, module) in split_gm.named_modules()]

    for name in names:
        if "." in name or name == "":
            # recursive child module or the root module
            continue

        module = getattr(split_gm, name)

        graph_id = int(name.replace("submod_", ""))
        outputs.append(
            SplitItem(name, graph_id, (graph_id in split_op_graphs), module))

    # sort by intetger graph_id, rather than string name
    outputs.sort(key=lambda x: x.graph_id)

    return split_gm, outputs


# we share the global graph pool among all the backends
global_graph_pool = None


class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    """Code adapted from `torch.fx.passes.shape_prop.ShapeProp`.
    It runs the given graph with fake inputs, and compile some
    submodules specified by `compile_submod_names` with the given
    compilation configs.

    NOTE: the order in `compile_submod_names` matters, because
    it will be used to determine the order of the compiled piecewise
    graphs. The first graph will handle logging, and the last graph
    has some special cudagraph output handling.
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

    def run(self, *args):
        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        with self.fake_mode:
            return super().run(*fake_args)

    def call_module(self, target: torch.fx.node.Target,
                    args: Tuple[torch.fx.node.Argument,
                                ...], kwargs: Dict[str, Any]) -> Any:
        assert isinstance(target, str)
        output = super().call_module(target, args, kwargs)

        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)
            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]
            compiled_graph_for_general_shape = wrap_inductor(
                submod,
                args,
                self.compilation_configs.inductor_compile_config,
                runtime_shape=None,
                do_logging=index == 0,
                use_inductor=self.compilation_configs.use_inductor)

            self.module.__dict__[target] = PiecewiseBackend(
                submod, self.compilation_configs, self.graph_pool, index,
                len(self.compile_submod_names), sym_shape_indices,
                compiled_graph_for_general_shape)

            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        return output


class VllmBackend:
    """The compilation backend for `torch.compile` with VLLM.
    It is used for compilation level of `CompilationLevel.PIECEWISE`,
    where we customize the compilation.

    The major work of this backend is to split the graph into
    piecewise graphs, and pass them to the piecewise backend.

    This backend also handles custom passes and adds them to Inductor config.
    The order of the post-grad post-passes is:
    1. post_grad_passes (constructor parameter)
    2. config["post_grad_custom_post_pass"]
    3. fix_functionalization
    This way, all passes operate on a functionalized graph.
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
    # Inductor passes to run on the graph pre-defunctionalization
    post_grad_passes: Sequence[Callable]
    sym_tensor_indices: List[int]
    input_buffers: List[torch.Tensor]

    def __init__(
        self,
        compilation_configs: CompilationConfig,
    ):
        global global_graph_pool
        if global_graph_pool is None:
            global_graph_pool = torch.cuda.graph_pool_handle()

        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = global_graph_pool
        self.post_grad_passes = []

        self.sym_tensor_indices = []
        self.input_buffers = []

        self.compilation_configs = compilation_configs

        # `torch.compile` is JIT compiled, so we don't need to
        # do anything here

    def add_passes_to_config(self):
        config = self.compilation_configs
        passes = list(self.post_grad_passes)

        passes = passes + [RedundantReshapesPass(config)]

        if config.enable_fusion:
            passes = passes + [FusionPass.instance(config)]

        inductor_config = config.inductor_compile_config
        if "post_grad_custom_post_pass" in inductor_config:
            passes = passes + [inductor_config["post_grad_custom_post_pass"]]

        # add the fix_functionalization pass last, so that all other
        # passes operate on a functionalized graph
        passes = passes + [fix_functionalization]
        combined_pass = combine_fx_passes(passes)
        inductor_config["post_grad_custom_post_pass"] = combined_pass

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:

        compilation_counter.num_graphs_seen += 1

        # we control the compilation process, each instance can only be
        # called once
        assert not self._called, "VllmBackend can only be called once"

        self.graph = graph
        # config is updated now, because only here can
        # we get the sizes to capture for cudagraph
        # from compilation context
        self.compilation_configs.init_during_runtime()
        self.add_passes_to_config()

        self.split_gm, self.piecewise_graphs = split_graph(
            graph, self.compilation_configs.non_cudagraph_ops)

        from torch._dynamo.utils import lazy_format_graph_code
        logger.debug("%s", lazy_format_graph_code("before split", self.graph))
        logger.debug("%s", lazy_format_graph_code("after split",
                                                  self.split_gm))

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

        if not self.compilation_configs.use_cudagraph or \
            not self.compilation_configs.cudagraph_copy_inputs:
            return self.split_gm

        # if we need to copy input buffers for cudagraph
        from torch._guards import detect_fake_mode
        fake_mode = detect_fake_mode()
        fake_args = [
            fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in example_inputs
        ]

        # index of tensors that have symbolic shapes (batch size)
        self.sym_tensor_indices = [
            i for i, x in enumerate(fake_args)
            if isinstance(x, torch._subclasses.fake_tensor.FakeTensor)
        ]

        # compiler managed cudagraph input buffers
        # we assume the first run with symbolic shapes
        # has the maximum size among all the tensors
        self.input_buffers = [
            example_inputs[x].clone() for x in self.sym_tensor_indices
        ]

        def copy_and_call(*args):
            list_args = list(args)
            for i, index in enumerate(self.sym_tensor_indices):
                runtime_tensor = list_args[index]
                runtime_shape = runtime_tensor.shape[0]
                static_tensor = self.input_buffers[i][:runtime_shape]

                # copy the tensor to the static buffer
                static_tensor.copy_(runtime_tensor)

                # replace the tensor in the list_args to the static buffer
                list_args[index] = static_tensor
            return self.split_gm(*list_args)

        return copy_and_call


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

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[List[int]] = None


class PiecewiseBackend:

    def __init__(self, graph: fx.GraphModule,
                 compilation_configs: CompilationConfig, graph_pool: Any,
                 piecewise_compile_index: int, total_piecewise_compiles: int,
                 sym_shape_indices: List[int],
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
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = (
            piecewise_compile_index == total_piecewise_compiles - 1)

        self.compile_sizes: Set[int] = set(
            self.compilation_configs.compile_sizes)
        self.capture_sizes: Set[int] = set(
            self.compilation_configs.capture_sizes
        ) if self.compilation_configs.use_cudagraph else set()

        self.first_run_finished = False

        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  # noqa

        self.sym_shape_indices = sym_shape_indices

        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

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
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every shape.
                # We only log it in the debug mode.
                logger.debug("Capturing a cudagraph for shape %s",
                             runtime_shape)

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if not self.is_first_graph:
                    # during every model forward, we will capture
                    # many pieces of cudagraphs (roughly one per layer).
                    # running gc again and again across layers will
                    # make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(
                        patch("torch.cuda.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                with torch.cuda.graph(cudagraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's cudagraph pool
                    output = entry.runnable(*args)
                    if self.is_last_graph:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph, because the output of the last graph
                        # will not be used by any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_caputured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                "Input addresses for cudagraphs are different during replay."
                f" Expected {entry.input_addresses}, got {new_input_addresses}"
            )

        entry.cudagraph.replay()
        return entry.output
