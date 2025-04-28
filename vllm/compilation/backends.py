# SPDX-License-Identifier: Apache-2.0

import ast
import dataclasses
import os
import pprint
import time
from contextlib import ExitStack
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from unittest.mock import patch

import torch
import torch.fx as fx

import vllm.envs as envs
from vllm.config import CompilationConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils import weak_ref_tensors

from .compiler_interface import EagerAdaptor, InductorAdaptor
from .counter import compilation_counter
from .inductor_pass import InductorPass
from .monitor import end_monitoring_torch_compile
from .pass_manager import PostGradPassManager

logger = init_logger(__name__)


class CompilerManager:
    """
    A manager to manage the compilation process, including
    caching the compiled graph, loading the compiled graph,
    and compiling the graph.

    The cache is a dict mapping
    `(runtime_shape, graph_index, backend_name)`
    to `any_data` returned from the compiler.

    When serializing the cache, we save it to a Python file
    for readability. We don't use json here because json doesn't
    support int as key.
    """

    def __init__(self, use_inductor: bool):
        self.cache: Dict[Tuple[Optional[int], int, str], Any] = dict()
        cls = InductorAdaptor if use_inductor else EagerAdaptor
        self.compiler = cls()

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        return self.compiler.compute_hash(vllm_config)

    def initialize_cache(self, cache_dir: str, disable_cache: bool = False):
        self.disable_cache = disable_cache
        self.cache_dir = cache_dir
        self.cache_file_path = os.path.join(cache_dir, "vllm_compile_cache.py")

        if not disable_cache and os.path.exists(self.cache_file_path):
            # load the cache from the file
            with open(self.cache_file_path) as f:
                # we use ast.literal_eval to parse the data
                # because it is a safe way to parse Python literals.
                # do not use eval(), it is unsafe.
                self.cache = ast.literal_eval(f.read())

        self.compiler.initialize_cache(cache_dir=cache_dir,
                                       disable_cache=disable_cache)

    def save_to_file(self):
        if self.disable_cache:
            return
        with open(self.cache_file_path, "w") as f:
            printer = pprint.PrettyPrinter(indent=4)
            data = printer.pformat(self.cache)
            f.write(data)

    def load(self,
             graph: fx.GraphModule,
             example_inputs: List[Any],
             graph_index: int,
             runtime_shape: Optional[int] = None) -> Optional[Callable]:
        if (runtime_shape, graph_index, self.compiler.name) not in self.cache:
            return None
        handle = self.cache[(runtime_shape, graph_index, self.compiler.name)]
        compiled_graph = self.compiler.load(handle, graph, example_inputs,
                                            graph_index, runtime_shape)
        logger.debug(
            "Directly load the %s-th graph for shape %s from %s via "
            "handle %s", graph_index, str(runtime_shape), self.compiler.name,
            handle)
        return compiled_graph

    def compile(self,
                graph: fx.GraphModule,
                example_inputs,
                additional_inductor_config,
                compilation_config: CompilationConfig,
                graph_index: int = 0,
                num_graphs: int = 1,
                runtime_shape: Optional[int] = None) -> Any:
        if graph_index == 0:
            # before compiling the first graph, record the start time
            global compilation_start_time
            compilation_start_time = time.time()

        compilation_counter.num_backend_compilations += 1

        compiled_graph = None

        # try to load from the cache
        compiled_graph = self.load(graph, example_inputs, graph_index,
                                   runtime_shape)
        if compiled_graph is not None:
            if graph_index == num_graphs - 1:
                # after loading the last graph for this shape, record the time.
                # there can be multiple graphs due to piecewise compilation.
                now = time.time()
                elapsed = now - compilation_start_time
                logger.info(
                    "Directly load the compiled graph(s) for shape %s "
                    "from the cache, took %.3f s", str(runtime_shape), elapsed)
            return compiled_graph

        # no compiler cached the graph, or the cache is disabled,
        # we need to compile it
        compiled_graph, handle = self.compiler.compile(
            graph, example_inputs, additional_inductor_config, runtime_shape)

        assert compiled_graph is not None, "Failed to compile the graph"

        # store the artifact in the cache
        if handle is not None:
            self.cache[(runtime_shape, graph_index,
                        self.compiler.name)] = handle
            if graph_index == 0:
                # adds some info logging for the first graph
                logger.info("Cache the graph of shape %s for later use",
                            str(runtime_shape))
            logger.debug(
                "store the %s-th graph for shape %s from %s via handle %s",
                graph_index, str(runtime_shape), self.compiler.name, handle)

        # after compiling the last graph, record the end time
        if graph_index == num_graphs - 1:
            now = time.time()
            elapsed = now - compilation_start_time
            compilation_config.compilation_time += elapsed
            if runtime_shape is None:
                logger.info("Compiling a graph for general shape takes %.2f s",
                            elapsed)
            else:
                logger.info("Compiling a graph for shape %s takes %.2f s",
                            runtime_shape, elapsed)

        return compiled_graph


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

compilation_start_time = 0.0


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
                 compile_submod_names: List[str], vllm_config: VllmConfig,
                 graph_pool, vllm_backend: "VllmBackend"):
        super().__init__(module)
        from torch._guards import detect_fake_mode
        self.fake_mode = detect_fake_mode()
        self.compile_submod_names = compile_submod_names
        self.compilation_config = vllm_config.compilation_config
        self.graph_pool = graph_pool
        self.vllm_config = vllm_config
        self.vllm_backend = vllm_backend

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
            global compilation_start_time
            compiled_graph_for_general_shape = self.vllm_backend.\
                compiler_manager.compile(
                submod,
                args,
                self.compilation_config.inductor_compile_config,
                self.compilation_config,
                graph_index=index,
                num_graphs=len(self.compile_submod_names),
                runtime_shape=None)

            self.module.__dict__[target] = PiecewiseBackend(
                submod, self.vllm_config, self.graph_pool, index,
                len(self.compile_submod_names), sym_shape_indices,
                compiled_graph_for_general_shape, self.vllm_backend)

            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        return output


class VllmBackend:
    """The compilation backend for `torch.compile` with vLLM.
    It is used for compilation level of `CompilationLevel.PIECEWISE`,
    where we customize the compilation.

    The major work of this backend is to split the graph into
    piecewise graphs, and pass them to the piecewise backend.

    This backend also adds the PostGradPassManager to Inductor config,
    which handles the post-grad passes.
    """

    vllm_config: VllmConfig
    compilation_config: CompilationConfig
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
    compiler_manager: CompilerManager

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        global global_graph_pool
        if global_graph_pool is None:
            global_graph_pool = torch.cuda.graph_pool_handle()

        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = global_graph_pool

        # Passes to run on the graph post-grad.
        self.post_grad_pass_manager = PostGradPassManager()

        self.sym_tensor_indices = []
        self.input_buffers = []

        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config

        self.compiler_manager: CompilerManager = CompilerManager(
            self.compilation_config.use_inductor)

        # `torch.compile` is JIT compiled, so we don't need to
        # do anything here

    def configure_post_pass(self):
        config = self.compilation_config
        self.post_grad_pass_manager.configure(self.vllm_config)

        # Post-grad custom passes are run using the post_grad_custom_post_pass
        # hook. If a pass for that hook exists, add it to the pass manager.
        inductor_config = config.inductor_compile_config
        PASS_KEY = "post_grad_custom_post_pass"
        if PASS_KEY in inductor_config:
            # Config should automatically wrap all inductor passes
            assert isinstance(inductor_config[PASS_KEY], InductorPass)
            self.post_grad_pass_manager.add(inductor_config[PASS_KEY])
        inductor_config[PASS_KEY] = self.post_grad_pass_manager

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:

        vllm_config = self.vllm_config
        if not self.compilation_config.cache_dir:
            # no provided cache dir, generate one based on the known factors
            # that affects the compilation. if none of the factors change,
            # the cache dir will be the same so that we can reuse the compiled
            # graph.

            factors = []
            # 0. factors come from the env, for example, The values of
            # VLLM_PP_LAYER_PARTITION will affects the computation graph.
            env_hash = envs.compute_hash()
            factors.append(env_hash)

            # 1. factors come from the vllm_config (it mainly summarizes how the
            #    model is created)
            config_hash = vllm_config.compute_hash()
            factors.append(config_hash)

            # 2. factors come from the code files that are traced by Dynamo (
            #    it mainly summarizes how the model is used in forward pass)
            forward_code_files = list(
                sorted(self.compilation_config.traced_files))
            self.compilation_config.traced_files.clear()
            logger.debug(
                "Traced files (to be considered for compilation cache):\n%s",
                "\n".join(forward_code_files))
            hash_content = []
            for filepath in forward_code_files:
                hash_content.append(filepath)
                with open(filepath) as f:
                    hash_content.append(f.read())
            import hashlib
            code_hash = hashlib.md5("\n".join(hash_content).encode(),
                                    usedforsecurity=False).hexdigest()
            factors.append(code_hash)

            # 3. compiler hash
            compiler_hash = self.compiler_manager.compute_hash(vllm_config)
            factors.append(compiler_hash)

            # combine all factors to generate the cache dir
            hash_key = hashlib.md5(str(factors).encode(),
                                   usedforsecurity=False).hexdigest()[:10]

            cache_dir = os.path.join(
                envs.VLLM_CACHE_ROOT,
                "torch_compile_cache",
                hash_key,
            )
            self.compilation_config.cache_dir = cache_dir

        cache_dir = self.compilation_config.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        rank = vllm_config.parallel_config.rank
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        local_cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}")
        os.makedirs(local_cache_dir, exist_ok=True)
        self.compilation_config.local_cache_dir = local_cache_dir

        disable_cache = envs.VLLM_DISABLE_COMPILE_CACHE

        if disable_cache:
            logger.info("vLLM's torch.compile cache is disabled.")
        else:
            logger.info("Using cache directory: %s for vLLM's torch.compile",
                        local_cache_dir)

        self.compiler_manager.initialize_cache(local_cache_dir, disable_cache)

        # when dynamo calls the backend, it means the bytecode
        # transform and analysis are done
        compilation_counter.num_graphs_seen += 1
        from .monitor import torch_compile_start_time
        dynamo_time = time.time() - torch_compile_start_time
        logger.info("Dynamo bytecode transform time: %.2f s", dynamo_time)
        self.compilation_config.compilation_time += dynamo_time

        # we control the compilation process, each instance can only be
        # called once
        assert not self._called, "VllmBackend can only be called once"

        self.graph = graph
        self.configure_post_pass()

        self.split_gm, self.piecewise_graphs = split_graph(
            graph, self.compilation_config.splitting_ops)

        from torch._dynamo.utils import lazy_format_graph_code

        # depyf will hook lazy_format_graph_code and dump the graph
        # for debugging, no need to print the graph here
        lazy_format_graph_code("before split", self.graph)
        lazy_format_graph_code("after split", self.split_gm)

        compilation_counter.num_piecewise_graphs_seen += len(
            self.piecewise_graphs)
        submod_names_to_compile = [
            item.submod_name for item in self.piecewise_graphs
            if not item.is_splitting_graph
        ]

        # propagate the split graph to the piecewise backend,
        # compile submodules with symbolic shapes
        PiecewiseCompileInterpreter(self.split_gm, submod_names_to_compile,
                                    self.vllm_config, self.graph_pool,
                                    self).run(*example_inputs)

        graph_path = os.path.join(local_cache_dir, "computation_graph.py")
        if not os.path.exists(graph_path):
            # code adapted from https://github.com/thuml/depyf/blob/dab831108a752d1facc00acdd6d4243891845c37/depyf/explain/patched_lazy_format_graph_code.py#L30 # noqa
            # use `print_readable` because it can include submodules
            src = "from __future__ import annotations\nimport torch\n" + \
                self.split_gm.print_readable(print_output=False)
            src = src.replace("<lambda>", "GraphModule")
            with open(graph_path, "w") as f:
                f.write(src)

            logger.debug("Computation graph saved to %s", graph_path)

        self._called = True

        if not self.compilation_config.use_cudagraph or \
            not self.compilation_config.cudagraph_copy_inputs:
            return self.split_gm

        # if we need to copy input buffers for cudagraph
        from torch._guards import detect_fake_mode
        fake_mode = detect_fake_mode()
        fake_args = [
            fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in example_inputs
        ]

        # index of tensors that have symbolic shapes (batch size)
        # for weights and static buffers, they will have concrete shapes.
        # symbolic shape only happens for input tensors.
        from torch.fx.experimental.symbolic_shapes import is_symbolic
        self.sym_tensor_indices = [
            i for i, x in enumerate(fake_args)
            if isinstance(x, torch._subclasses.fake_tensor.FakeTensor) and \
                any(is_symbolic(d) for d in x.size())
        ]

        # compiler managed cudagraph input buffers
        # we assume the first run with symbolic shapes
        # has the maximum size among all the tensors
        self.input_buffers = [
            example_inputs[x].clone() for x in self.sym_tensor_indices
        ]

        # this is the callable we return to Dynamo to run
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
    use_cudagraph: bool  # the size is in cudagraph_capture_sizes

    compiled: bool = False
    runnable: Callable = None  # type: ignore
    num_finished_warmup: int = 0
    cudagraph: Optional[torch.cuda.CUDAGraph] = None
    output: Optional[Any] = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[List[int]] = None


class PiecewiseBackend:

    def __init__(self, graph: fx.GraphModule, vllm_config: VllmConfig,
                 graph_pool: Any, piecewise_compile_index: int,
                 total_piecewise_compiles: int, sym_shape_indices: List[int],
                 compiled_graph_for_general_shape: Callable,
                 vllm_backend: VllmBackend):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation and cudagraph capturing.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.

        Independently, we will capture cudagraph for different shapes.

        If a shape needs both compilation and cudagraph, we will
        compile it first, and then capture cudagraph.
        """
        self.graph = graph
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.graph_pool = graph_pool
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.vllm_backend = vllm_backend

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = (
            piecewise_compile_index == total_piecewise_compiles - 1)

        self.compile_sizes: Set[int] = set(
            self.compilation_config.compile_sizes)
        self.cudagraph_capture_sizes: Set[int] = set(
            self.compilation_config.cudagraph_capture_sizes
        ) if self.compilation_config.use_cudagraph else set()

        self.first_run_finished = False

        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  # noqa

        self.sym_shape_indices = sym_shape_indices

        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # the entries for different shapes that we need to either
        # compile or capture cudagraph
        self.concrete_size_entries: Dict[int, ConcreteSizeEntry] = {}

        # to_be_compiled_sizes tracks the remaining sizes to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_sizes: Set[int] = self.compile_sizes.copy()
        for shape in self.compile_sizes.union(self.cudagraph_capture_sizes):
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_shape=shape,
                need_to_compile=shape in self.compile_sizes,
                use_cudagraph=shape in self.cudagraph_capture_sizes,
            )

    def check_for_ending_compilation(self):
        if self.is_last_graph and not self.to_be_compiled_sizes:
            # no specific sizes to compile
            # save the hash of the inductor graph for the next run
            self.vllm_backend.compiler_manager.save_to_file()
            end_monitoring_torch_compile(self.vllm_config)

    def __call__(self, *args) -> Any:
        if not self.first_run_finished:
            self.first_run_finished = True
            self.check_for_ending_compilation()
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
            self.to_be_compiled_sizes.remove(runtime_shape)
            # args are real arguments
            entry.runnable = self.vllm_backend.compiler_manager.compile(
                self.graph,
                args,
                self.compilation_config.inductor_compile_config,
                self.compilation_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                runtime_shape=runtime_shape)

            # finished compilations for all required shapes
            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()

        if not entry.use_cudagraph:
            return entry.runnable(*args)

        if entry.cudagraph is None:
            if entry.num_finished_warmup < self.compilation_config.cudagraph_num_of_warmups:  # noqa
                entry.num_finished_warmup += 1
                if self.is_first_graph:
                    logger.debug(
                        "Warming up %s/%s for shape %s",
                        entry.num_finished_warmup,
                        self.compilation_config.cudagraph_num_of_warmups,
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
