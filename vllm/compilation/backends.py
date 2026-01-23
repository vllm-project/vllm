# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import contextvars
import dataclasses
import hashlib
import json
import operator
import os
import pprint
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from typing import Any

import torch
import torch.fx as fx
from torch._dispatch.python import enable_python_dispatcher
from torch._logging._internal import trace_structured

import vllm.envs as envs
from vllm.compilation.inductor_pass import pass_context
from vllm.compilation.partition_rules import (
    inductor_partition_rule_context,
    should_split,
)
from vllm.config import CompilationConfig, CUDAGraphMode, VllmConfig
from vllm.config.compilation import DynamicShapesType
from vllm.config.utils import Range, hash_factors
from vllm.logger import init_logger
from vllm.logging_utils import lazy
from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname

from .compiler_interface import (
    CompilerInterface,
    EagerAdaptor,
    InductorAdaptor,
    InductorStandaloneAdaptor,
    is_compile_cache_enabled,
)
from .counter import compilation_counter
from .inductor_pass import InductorPass
from .pass_manager import PostGradPassManager

logger = init_logger(__name__)

# A global flag to indicate if the current graph being compiled
# is the last one in a sequence of graphs (e.g., a sequence of blocks).
# This is a workaround to control CUDAGraph weak_ref_output behavior
# in **vit** piecewise compilation.
_is_last_graph_in_vit_sequence: bool = True


@contextmanager
def set_is_last_graph_in_vit_sequence(is_last: bool):
    """Context manager to indicate if the current graph being compiled
    is the last one in a sequence of graphs (e.g., a sequence of blocks).
    """
    global _is_last_graph_in_vit_sequence
    original_value = _is_last_graph_in_vit_sequence
    _is_last_graph_in_vit_sequence = is_last
    try:
        yield
    finally:
        _is_last_graph_in_vit_sequence = original_value


# A global flag to indicate if the current graph being compiled
# is the first one in a sequence of graphs (e.g., a sequence of blocks).
_is_first_graph_in_vit_sequence: bool = True


@contextmanager
def set_is_first_graph_in_vit_sequence(is_first: bool):
    """Context manager to indicate if the current graph being compiled
    is the first one in a sequence of graphs (e.g., a sequence of blocks).
    """
    global _is_first_graph_in_vit_sequence
    original_value = _is_first_graph_in_vit_sequence
    _is_first_graph_in_vit_sequence = is_first
    try:
        yield
    finally:
        _is_first_graph_in_vit_sequence = original_value

def make_copy_and_call(
    sym_tensor_indices: list[int],
    input_buffers: list[torch.Tensor | None],
    callable_fn: Callable[..., Any],
) -> Callable[..., Any]:
    """Create a wrapper that copies inputs to static buffers before calling.

    This is used for cudagraph input copying where we need to copy dynamic
    tensors to static buffers before invoking the compiled graph.

    Args:
        sym_tensor_indices: Indices of tensors with symbolic shapes
        input_buffers: List of static buffers (can contain None for lazy init)
        callable_fn: The compiled function to call

    Returns:
        A wrapper function that copies inputs and calls the compiled function
    """

    def copy_and_call(*args: Any) -> Any:
        list_args = list(args)
        for i, index in enumerate(sym_tensor_indices):
            runtime_tensor = list_args[index]
            runtime_shape = runtime_tensor.shape[0]

            # lazy initialization of buffer on first call
            if input_buffers[i] is None:
                input_buffers[i] = runtime_tensor.clone()

            static_tensor = input_buffers[i][:runtime_shape]  # type: ignore[index]
            static_tensor.copy_(runtime_tensor)
            list_args[index] = static_tensor
        return callable_fn(*list_args)

    return copy_and_call


def make_compiler(compilation_config: CompilationConfig) -> CompilerInterface:
    assert not envs.VLLM_USE_MEGA_AOT_ARTIFACT or envs.VLLM_USE_STANDALONE_COMPILE, (
        "VLLM_USE_MEGA_AOT_ARTIFACT=1 requires VLLM_USE_STANDALONE_COMPILE=1"
    )

    if compilation_config.backend == "inductor":
        # Use standalone compile only if requested, version is new enough,
        # and the symbol actually exists in this PyTorch build.
        if envs.VLLM_USE_STANDALONE_COMPILE and hasattr(
            torch._inductor, "standalone_compile"
        ):
            logger.debug("Using InductorStandaloneAdaptor")
            return InductorStandaloneAdaptor(
                compilation_config.compile_cache_save_format
            )
        else:
            logger.debug("Using InductorAdaptor")
            return InductorAdaptor()
    elif compilation_config.backend == "eager":
        logger.debug("Using EagerAdaptor")
        return EagerAdaptor()
    else:
        logger.debug("Using custom backend: %s", compilation_config.backend)
        compiler = resolve_obj_by_qualname(current_platform.get_compile_backend())()
        assert isinstance(compiler, CompilerInterface)
        return compiler


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

    def __init__(self, compilation_config: CompilationConfig) -> None:
        self.cache: dict[tuple[Range, int, str], Any] = dict()
        self.is_cache_updated = False
        self.compilation_config = compilation_config
        self.compiler = make_compiler(compilation_config)

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        return self.compiler.compute_hash(vllm_config)

    @contextmanager
    def compile_context(self, compile_range: Range) -> Generator[None, None, None]:
        """Provide compilation context for the duration of compilation to set
        any torch global properties we want to scope to a single Inductor
        compilation (e.g. partition rules, pass context)."""
        with pass_context(compile_range):
            if self.compilation_config.use_inductor_graph_partition:
                with inductor_partition_rule_context(
                    self.compilation_config.splitting_ops
                ):
                    yield
            else:
                yield

    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ) -> None:
        """
        Initialize the cache directory for the compiler.

        The organization of the cache directory is as follows:
        cache_dir=/path/to/hash_str/rank_i_j/prefix/
        inside cache_dir, there will be:
        - vllm_compile_cache.py
        - computation_graph.py
        - transformed_code.py

        for multiple prefixes, they can share the same
        base cache dir of /path/to/hash_str/rank_i_j/ ,
        to store some common compilation artifacts.
        """

        self.disable_cache = disable_cache
        self.cache_dir = cache_dir
        self.cache_file_path = os.path.join(cache_dir, "vllm_compile_cache.py")

        if not disable_cache and os.path.exists(self.cache_file_path):
            # load the cache from the file
            with open(self.cache_file_path) as f:
                # we use ast.literal_eval to parse the data
                # because it is a safe way to parse Python literals.
                # do not use eval(), it is unsafe.
                cache = ast.literal_eval(f.read())

            def check_type(value: Any, ty: type) -> None:
                if not isinstance(value, ty):
                    raise TypeError(f"Expected {ty} but got {type(value)} for {value}")

            def parse_key(key: Any) -> tuple[Range, int, str]:
                range_tuple, graph_index, compiler_name = key
                check_type(graph_index, int)
                check_type(compiler_name, str)
                if isinstance(range_tuple, tuple):
                    start, end = range_tuple
                    check_type(start, int)
                    check_type(end, int)
                    range_tuple = Range(start=start, end=end)
                check_type(range_tuple, Range)
                return range_tuple, graph_index, compiler_name

            self.cache = {parse_key(key): value for key, value in cache.items()}

        self.compiler.initialize_cache(
            cache_dir=cache_dir, disable_cache=disable_cache, prefix=prefix
        )

    def save_to_file(self) -> None:
        if self.disable_cache or not self.is_cache_updated:
            return
        printer = pprint.PrettyPrinter(indent=4)
        data = printer.pformat(self.cache)
        with open(self.cache_file_path, "w") as f:
            f.write(data)

    def load(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        compile_range: Range,
    ) -> Callable[..., Any] | None:
        if (compile_range, graph_index, self.compiler.name) not in self.cache:
            return None
        handle = self.cache[(compile_range, graph_index, self.compiler.name)]
        compiled_graph = self.compiler.load(
            handle, graph, example_inputs, graph_index, compile_range
        )
        logger.debug(
            "Directly load the %s-th graph for compile range %sfrom %s via handle %s",
            graph_index,
            str(compile_range),
            self.compiler.name,
            handle,
        )
        return compiled_graph

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        additional_inductor_config: dict[str, Any],
        compilation_config: CompilationConfig,
        compile_range: Range,
        graph_index: int = 0,
        num_graphs: int = 1,
    ) -> Any:
        if graph_index == 0:
            # before compiling the first graph, record the start time
            global compilation_start_time
            compilation_start_time = time.time()

        compilation_counter.num_backend_compilations += 1

        compiled_graph = None

        # try to load from the cache
        compiled_graph = self.load(graph, example_inputs, graph_index, compile_range)
        if compiled_graph is not None:
            if graph_index == num_graphs - 1:
                # after loading the last graph for this shape, record the time.
                # there can be multiple graphs due to piecewise compilation.
                now = time.time()
                elapsed = now - compilation_start_time
                compilation_config.compilation_time += elapsed
                logger.info(
                    "Directly load the compiled graph(s) for compile range %s "
                    "from the cache, took %.3f s",
                    str(compile_range),
                    elapsed,
                )
            return compiled_graph

        # no compiler cached the graph, or the cache is disabled,
        # we need to compile it
        if isinstance(self.compiler, InductorAdaptor):
            # Let compile_fx generate a key for us
            maybe_key = None
        else:
            maybe_key = "artifact_compile_range_"
            maybe_key += f"{compile_range.start}_{compile_range.end}"
            maybe_key += f"_subgraph_{graph_index}"
        with self.compile_context(compile_range):
            compiled_graph, handle = self.compiler.compile(
                graph,
                example_inputs,
                additional_inductor_config,
                compile_range,
                maybe_key,
            )

        assert compiled_graph is not None, "Failed to compile the graph"

        # store the artifact in the cache
        if is_compile_cache_enabled(additional_inductor_config) and handle is not None:
            self.cache[(compile_range, graph_index, self.compiler.name)] = handle
            compilation_counter.num_cache_entries_updated += 1
            self.is_cache_updated = True
            if graph_index == 0:
                # adds some info logging for the first graph
                logger.info_once(
                    "Cache the graph of compile range %s for later use",
                    str(compile_range),
                )
            logger.debug(
                "Store the %s-th graph for compile range%s from %s via handle %s",
                graph_index,
                str(compile_range),
                self.compiler.name,
                handle,
            )

        # after compiling the last graph, record the end time
        if graph_index == num_graphs - 1:
            now = time.time()
            elapsed = now - compilation_start_time
            compilation_config.compilation_time += elapsed
            logger.info_once(
                "Compiling a graph for compile range %s takes %.2f s",
                str(compile_range),
                elapsed,
                scope="local",
            )

        return compiled_graph


@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule


def split_graph(
    graph: fx.GraphModule, splitting_ops: list[str]
) -> tuple[fx.GraphModule, list[SplitItem]]:
    # split graph by ops
    subgraph_id = 0
    node_to_subgraph_id: dict[fx.Node, int] = {}
    split_op_graphs: list[int] = []
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue

        # Check if this is a getitem operation on a node from an earlier subgraph.
        # If so, assign it to the same subgraph as its input to avoid passing entire
        # tuple as input to submodules, which is against standalone_compile and
        # AoTAutograd input requirement.
        if node.op == "call_function" and node.target == operator.getitem:
            # Assign this getitem to the same subgraph as its input
            input_node = node.args[0]
            if input_node.op != "placeholder":
                assert input_node in node_to_subgraph_id
                node_to_subgraph_id[node] = node_to_subgraph_id[input_node]
                continue

        if should_split(node, splitting_ops):
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)

            # keep consecutive splitting ops together
            # (we know node.next exists because node isn't the last (output) node)
            if should_split(node.next, splitting_ops):
                # this will get incremented by the next node
                subgraph_id -= 1
            else:
                subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    # `keep_original_order` is important!
    # otherwise pytorch might reorder the nodes and
    # the semantics of the graph will change when we
    # have mutations in the graph
    split_gm = torch.fx.passes.split_module.split_module(
        graph, None, lambda node: node_to_subgraph_id[node], keep_original_order=True
    )

    outputs = []

    names = [name for (name, module) in split_gm.named_modules()]

    for name in names:
        if "." in name or name == "":
            # recursive child module or the root module
            continue

        module = getattr(split_gm, name)

        graph_id = int(name.replace("submod_", ""))
        outputs.append(SplitItem(name, graph_id, (graph_id in split_op_graphs), module))

    # sort by integer graph_id, rather than string name
    outputs.sort(key=lambda x: x.graph_id)

    return split_gm, outputs


compilation_start_time = 0.0


def wrap_with_cudagraph_if_needed(
    piecewise_backend: Any,
    vllm_config: VllmConfig,
    compilation_config: CompilationConfig,
    is_first_graph: bool,
    is_last_graph: bool,
) -> Any:
    """
    Wrap a piecewise backend with CUDA graph wrapper if needed.
    This function is shared between VllmBackend and
    construct_serializable_fn_from_inductor_cache.

    Args:
        piecewise_backend: The backend to wrap
        vllm_config: The vLLM configuration
        compilation_config: The compilation configuration
        is_first_graph: Whether this is the first graph in the sequence
        is_last_graph: Whether this is the last graph in the sequence

    Returns:
        The wrapped backend if CUDA graphs are enabled, otherwise the original backend
    """
    if (
        not compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
        or compilation_config.use_inductor_graph_partition
    ):
        return piecewise_backend

    # We're using Dynamo-based piecewise splitting, so we wrap
    # the whole subgraph with a static graph wrapper.
    from .cuda_graph import CUDAGraphOptions

    # resolve the static graph wrapper class (e.g. CUDAGraphWrapper
    # class) as platform dependent.
    static_graph_wrapper_class = resolve_obj_by_qualname(
        current_platform.get_static_graph_wrapper_cls()
    )

    # Always assign PIECEWISE runtime mode to the
    # CUDAGraphWrapper for piecewise_backend, to distinguish
    # it from the FULL cudagraph runtime mode, no matter it
    # is wrapped on a full or piecewise fx graph.
    return static_graph_wrapper_class(
        runnable=piecewise_backend,
        vllm_config=vllm_config,
        runtime_mode=CUDAGraphMode.PIECEWISE,
        cudagraph_options=CUDAGraphOptions(
            debug_log_enable=is_first_graph,
            gc_disable=not is_first_graph
                        or not _is_first_graph_in_vit_sequence,
            weak_ref_output=is_last_graph
                        and _is_last_graph_in_vit_sequence,
        ),
    )


class PiecewiseCompileInterpreter(torch.fx.Interpreter):  # type: ignore[misc]
    """Code adapted from `torch.fx.passes.shape_prop.ShapeProp`.
    It runs the given graph with fake inputs, and compile some
    submodules specified by `compile_submod_names` with the given
    compilation configs.

    NOTE: the order in `compile_submod_names` matters, because
    it will be used to determine the order of the compiled piecewise
    graphs. The first graph will handle logging, and the last graph
    has some special cudagraph output handling.

    Note: This class shares similar logic with
    reconstruct_serializable_fn_from_mega_artifact in caching.py.
    Both create PiecewiseBackend instances and wrap them with cudagraph.
    The key difference is:
    - reconstruct_serializable_fn_from_mega_artifact: PiecewiseBackend receives
      pre-compiled runnables (compiled_runnables is set, graph is None)
    - this class: PiecewiseBackend receives the FX graph to compile
      (graph is set, compiled_runnables is None)


    If modifying the backend creation/wrapping logic, consider updating both.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        compile_submod_names: list[str],
        vllm_config: VllmConfig,
        vllm_backend: "VllmBackend",
    ) -> None:
        super().__init__(module)
        from torch._guards import detect_fake_mode

        self.fake_mode = detect_fake_mode()
        self.compile_submod_names = compile_submod_names
        self.compilation_config = vllm_config.compilation_config
        self.vllm_config = vllm_config
        self.vllm_backend = vllm_backend
        # When True, it annoyingly dumps the torch.fx.Graph on errors.
        self.extra_traceback = False

    def run(self, *args: Any) -> Any:
        # maybe instead just assert inputs are fake?
        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        with self.fake_mode, enable_python_dispatcher():
            return super().run(*fake_args)

    def call_module(
        self,
        target: torch.fx.node.Target,
        args: tuple[torch.fx.node.Argument, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        assert isinstance(target, str)

        output = super().call_module(target, args, kwargs)

        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)

            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]

            # Lazy import here to avoid circular import
            from torch._inductor.compile_fx import graph_returns_tuple

            from .piecewise_backend import PiecewiseBackend

            piecewise_backend = PiecewiseBackend(
                submod,
                self.vllm_config,
                index,
                len(self.compile_submod_names),
                sym_shape_indices,
                self.vllm_backend,
                graph_returns_tuple(submod),
                submod_name=target,
            )

            self.module.__dict__[target] = wrap_with_cudagraph_if_needed(
                piecewise_backend,
                self.vllm_config,
                self.compilation_config,
                piecewise_backend.is_first_graph,
                piecewise_backend.is_last_graph,
            )

            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        return output


# the tag for the part of model being compiled,
# e.g. backbone/eagle_head
model_tag: str = "backbone"
model_is_encoder: bool = False

_on_compilation_complete_callback: contextvars.ContextVar[Callable[[], None] | None] = (
    contextvars.ContextVar("on_compilation_complete_callback", default=None)
)


@contextmanager
def set_on_compilation_complete(
    callback: Callable[[], None],
) -> Generator[None, None, None]:
    token = _on_compilation_complete_callback.set(callback)
    try:
        yield
    finally:
        _on_compilation_complete_callback.reset(token)


@contextmanager
def set_model_tag(tag: str, is_encoder: bool = False) -> Generator[None, None, None]:
    """Context manager to set the model tag."""
    global model_tag
    global model_is_encoder
    assert tag != model_tag, (
        f"Model tag {tag} is the same as the current tag {model_tag}."
    )
    old_tag = model_tag
    old_is_encoder = model_is_encoder

    model_tag = tag
    model_is_encoder = is_encoder
    try:
        yield
    finally:
        model_tag = old_tag
        model_is_encoder = old_is_encoder


class VllmBackend:
    """The compilation backend for `torch.compile` with vLLM.
    It is used for compilation mode of `CompilationMode.VLLM_COMPILE`,
    where we customize the compilation.

    The major work of this backend is to split the graph into
    piecewise graphs, and pass them to the piecewise backend.

    This backend also adds the PostGradPassManager to Inductor config,
    which handles the post-grad passes.
    """

    vllm_config: VllmConfig
    compilation_config: CompilationConfig
    _called: bool = False
    # the graph we compiled
    graph: fx.GraphModule
    # the stiching graph module for all the piecewise graphs
    split_gm: fx.GraphModule
    piecewise_graphs: list[SplitItem]
    returned_callable: Callable[..., Any]
    # Inductor passes to run on the graph pre-defunctionalization
    post_grad_passes: Sequence[Callable[..., Any]]
    compiler_manager: CompilerManager
    # Copy of CompilationConfig.inductor_compile_config +
    # an entry for PostGradPassManager
    inductor_config: dict[str, Any]

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        is_encoder: bool = False,
    ) -> None:
        # if the model is initialized with a non-empty prefix,
        # then usually it's enough to use that prefix,
        # e.g. language_model, vision_model, etc.
        # when multiple parts are initialized as independent
        # models, we need to use the model_tag to distinguish
        # them, e.g. backbone (default), eagle_head, etc.
        self.prefix = prefix or model_tag

        # Mark compilation for encoder.
        self.is_encoder = is_encoder or model_is_encoder

        # Passes to run on the graph post-grad.
        self.pass_manager = resolve_obj_by_qualname(
            current_platform.get_pass_manager_cls()
        )()
        self.pass_key = current_platform.pass_key

        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config

        self.compiler_manager: CompilerManager = CompilerManager(
            self.compilation_config
        )

        # Deepcopy the inductor config to detach the post-grad custom pass
        # from CompilationConfig.
        # We want to avoid PostGradPassManager in CompilationConfig because
        # in future we need PostGradPassManager.uuid() to be executed
        # only at compile time.
        self.inductor_config = deepcopy(self.compilation_config.inductor_compile_config)
        # `torch.compile` is JIT compiled, so we don't need to
        # do anything here

    def collect_standalone_compile_artifacts(
        self,
    ) -> tuple[Any, dict[str, list[int]] | None, dict[str, bool] | None]:
        """Collect inductor cache artifacts from all piecewise backends.

        Returns:
            tuple: (standalone_compile_artifacts, sym_shape_indices_map,
                    returns_tuple_map)
                - standalone_compile_artifacts: StandaloneCompiledArtifacts
                  with compiled artifacts
                - sym_shape_indices_map: dict mapping submod_name to
                  sym_shape_indices
                - returns_tuple_map: dict mapping submod_name to
                  returns_tuple
        """

        if not envs.VLLM_USE_MEGA_AOT_ARTIFACT:
            return None, None, None

        from .caching import StandaloneCompiledArtifacts
        from .piecewise_backend import PiecewiseBackend

        standalone_compile_artifacts = StandaloneCompiledArtifacts()
        sym_shape_indices_map = {}
        returns_tuple_map = {}

        for name, _ in self.split_gm.named_children():
            # get the actual attribute (shadowed by PiecewiseBackend in __dict__)
            child = getattr(self.split_gm, name)
            # unwrap the static graph wrapper class if applicable
            piecewise_backend = child.runnable if hasattr(child, "runnable") else child

            if not isinstance(piecewise_backend, PiecewiseBackend):
                continue

            submod_name = name
            sym_shape_indices_map[submod_name] = piecewise_backend.sym_shape_indices
            returns_tuple_map[submod_name] = piecewise_backend.returns_tuple

            for shape_str, bytes_data in piecewise_backend.to_bytes().items():
                standalone_compile_artifacts.insert(submod_name, shape_str, bytes_data)
                logger.debug(
                    "collected artifact for %s shape %s (%d bytes)",
                    submod_name,
                    shape_str,
                    len(bytes_data),
                )

        logger.info(
            "collected artifacts: %d entries, %d artifacts, %d bytes total",
            standalone_compile_artifacts.num_entries(),
            standalone_compile_artifacts.num_artifacts(),
            standalone_compile_artifacts.size_bytes(),
        )

        logger.debug(
            "standalone compile artifact keys: %s",
            list(standalone_compile_artifacts.submodule_bytes.keys()),
        )

        return standalone_compile_artifacts, sym_shape_indices_map, returns_tuple_map

    def configure_post_pass(self) -> None:
        self.pass_manager.configure(self.vllm_config)

        # Post-grad custom passes are run using the post_grad_custom_post_pass
        # hook. If a pass for that hook exists, add it to the pass manager.
        if self.pass_key in self.inductor_config:
            if isinstance(self.inductor_config[self.pass_key], PostGradPassManager):
                raise ValueError(
                    "PostGradPassManager can not be kept in CompilationConfig."
                )
            else:
                # Config should automatically wrap all inductor passes
                assert isinstance(
                    self.compilation_config.inductor_compile_config[self.pass_key],
                    InductorPass,
                )
                self.pass_manager.add(
                    self.compilation_config.inductor_compile_config[self.pass_key]
                )
        self.inductor_config[self.pass_key] = self.pass_manager

    def _log_compilation_config(self):
        """Log vLLM compilation config for TORCH_TRACE/tlparse."""
        cc = self.compilation_config
        pass_cfg = cc.pass_config

        # Helper to convert lists to comma-separated strings for tlparse display
        def list_to_str(lst: list | None) -> str:
            if lst is None:
                return ""
            return ", ".join(str(x) for x in lst)

        # Get enabled passes by introspecting dataclass fields
        enabled_passes = [
            f.name
            for f in dataclasses.fields(pass_cfg)
            if isinstance(getattr(pass_cfg, f.name), bool) and getattr(pass_cfg, f.name)
        ]

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "vllm_compilation_config",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(
                {
                    "model": self.vllm_config.model_config.model,
                    "prefix": self.prefix,
                    "mode": str(cc.mode),
                    "backend": cc.backend,
                    "custom_ops": list_to_str(cc.custom_ops),
                    "splitting_ops": list_to_str(cc.splitting_ops),
                    "cudagraph_mode": str(cc.cudagraph_mode),
                    "compile_sizes": list_to_str(cc.compile_sizes),
                    "compile_ranges_split_points": list_to_str(
                        cc.compile_ranges_split_points
                    ),
                    "use_inductor_graph_partition": cc.use_inductor_graph_partition,
                    "inductor_passes": list_to_str(list(cc.inductor_passes.keys())),
                    "enabled_passes": list_to_str(enabled_passes),
                    "dynamic_shapes_type": str(cc.dynamic_shapes_config.type),
                    "dynamic_shapes_evaluate_guards": cc.dynamic_shapes_config.evaluate_guards,  # noqa: E501
                }
            ),
        )

    def __call__(self, graph: fx.GraphModule, example_inputs: Sequence[Any]) -> Any:
        from .caching import (
            VllmSerializableFunction,
        )

        vllm_config = self.vllm_config

        self._log_compilation_config()

        # Minimal hashing here with existing utilities, reused below.

        env_factors = envs.compile_factors()
        env_hash = hash_factors(env_factors)
        # Compute config/compiler/code hashes once and reuse
        config_hash = vllm_config.compute_hash()
        compiler_hash = self.compiler_manager.compute_hash(vllm_config)
        forward_code_files = list(sorted(self.compilation_config.traced_files))

        logger.debug(
            "Traced files (to be considered for compilation cache):\n%s",
            lazy(lambda: "\n".join(forward_code_files)),
        )
        hash_content = []
        for filepath in forward_code_files:
            hash_content.append(filepath)
            if filepath == "<string>":
                # This means the function was dynamically generated, with
                # e.g. exec(). We can't actually check these.
                continue
            try:
                with open(filepath) as f:
                    hash_content.append(f.read())
            except (OSError, UnicodeDecodeError):
                logger.warning("Failed to read file %s", filepath)
                continue
        code_hash = hashlib.sha256("\n".join(hash_content).encode()).hexdigest()
        # Clear after consumption
        self.compilation_config.traced_files.clear()
        if not self.compilation_config.cache_dir:
            # no provided cache dir, generate one based on the known factors
            # that affects the compilation. if none of the factors change,
            # the cache dir will be the same so that we can reuse the compiled
            # graph.
            factors = [env_hash, config_hash, code_hash, compiler_hash]
            # Use SHA-256 for cache key hashing to be consistent across
            # compute_hash functions. Truncate for a short cache dir name.
            hash_key = hashlib.sha256(str(factors).encode()).hexdigest()[:10]
            cache_dir = os.path.join(
                envs.VLLM_CACHE_ROOT, "torch_compile_cache", hash_key
            )
            self.compilation_config.cache_dir = cache_dir

        cache_dir = self.compilation_config.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.compilation_config.cache_dir = cache_dir
        rank = vllm_config.parallel_config.rank
        dp_rank = vllm_config.parallel_config.data_parallel_index
        local_cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}", self.prefix)
        os.makedirs(local_cache_dir, exist_ok=True)
        self.compilation_config.local_cache_dir = local_cache_dir

        # Honors opt-outs such as CompilationMode.NONE or VLLM_DISABLE_COMPILE_CACHE.
        disable_cache = not is_compile_cache_enabled(self.inductor_config)

        if disable_cache:
            logger.info_once("vLLM's torch.compile cache is disabled.", scope="local")
        else:
            logger.info_once(
                "Using cache directory: %s for vLLM's torch.compile",
                local_cache_dir,
                scope="local",
            )

        self.compiler_manager.initialize_cache(
            local_cache_dir, disable_cache, self.prefix
        )

        # Reuses existing cache key

        logger.debug(
            "torch.compile cache factors: env=%s cfg=%s comp=%s code=%s dir=%s",
            env_hash,
            config_hash,
            compiler_hash,
            code_hash,
            local_cache_dir,
        )

        # Persist and log only hash-relevant factors together.
        try:
            logger.debug(
                "Compile env factors (raw):\n%s\nVllm config hash: %s",
                lazy(partial(pprint.pformat, env_factors, width=120)),
                config_hash,
            )
            meta_path = os.path.join(local_cache_dir, "cache_key_factors.json")
            if not os.path.exists(meta_path):
                with open(meta_path, "w") as f:
                    json.dump(
                        {
                            "env": env_factors,  # raw factors used for env_hash
                            "config_hash": config_hash,
                            "code_hash": code_hash,
                            "compiler_hash": compiler_hash,
                        },
                        f,
                        indent=2,
                        sort_keys=True,
                    )
        except Exception:
            # Best-effort only; metadata write failures are non-fatal.
            logger.warning(
                (
                    "Could not write compile cache metadata at %s; continuing without "
                    "metadata. Compiled cache remains valid; diagnostics may be "
                    "limited."
                ),
                local_cache_dir,
                exc_info=True,
            )

        # when dynamo calls the backend, it means the bytecode
        # transform and analysis are done
        compilation_counter.num_graphs_seen += 1
        from .monitor import torch_compile_start_time

        dynamo_time = time.time() - torch_compile_start_time
        logger.info_once(
            "Dynamo bytecode transform time: %.2f s", dynamo_time, scope="local"
        )
        self.compilation_config.compilation_time += dynamo_time

        # we control the compilation process, each instance can only be
        # called once
        assert not self._called, "VllmBackend can only be called once"

        self.graph = graph
        self.configure_post_pass()

        if self.compilation_config.use_inductor_graph_partition:
            # Let Inductor decide partitioning; avoid FX-level pre-splitting.
            fx_split_ops: list[str] = []
        else:
            fx_split_ops = self.compilation_config.splitting_ops or []

        self.split_gm, self.piecewise_graphs = split_graph(graph, fx_split_ops)

        # keep a split_gm copy from BEFORE the interpreter replaces
        # submodules with PiecewiseBackend -- used for serialization
        original_split_gm = None
        if envs.VLLM_USE_MEGA_AOT_ARTIFACT:
            original_split_gm = deepcopy(self.split_gm)

        from torch._dynamo.utils import lazy_format_graph_code

        # depyf will hook lazy_format_graph_code and dump the graph
        # for debugging, no need to print the graph here
        lazy_format_graph_code("before split", self.graph)
        lazy_format_graph_code("after split", self.split_gm)

        # Log the piecewise split graph for TORCH_TRACE/tlparse
        trace_structured(
            "graph_dump",
            metadata_fn=lambda: {"name": "vllm_piecewise_split_graph"},
            payload_fn=lambda: self.split_gm.print_readable(print_output=False),
        )

        compilation_counter.num_piecewise_graphs_seen += len(self.piecewise_graphs)
        submod_names_to_compile = [
            item.submod_name
            for item in self.piecewise_graphs
            if not item.is_splitting_graph
        ]

        # Extract fake values from the graph to use them when needed.
        all_fake_values = []
        for i in graph.graph.find_nodes(op="placeholder"):
            all_fake_values.append(i.meta["example_value"])

        fake_args = [
            all_fake_values[i] if isinstance(t, torch.Tensor) else t
            for i, t in enumerate(example_inputs)
        ]

        # propagate the split graph to the piecewise backend,
        # compile submodules with symbolic shapes
        PiecewiseCompileInterpreter(
            self.split_gm, submod_names_to_compile, self.vllm_config, self
        ).run(*fake_args)

        from torch._guards import detect_fake_mode

        fake_mode = detect_fake_mode()

        if (
            self.compilation_config.dynamic_shapes_config.evaluate_guards
            and self.compilation_config.dynamic_shapes_config.type
            == DynamicShapesType.BACKED
        ):
            from torch.utils._sympy.value_ranges import ValueRanges

            # Drop counter-0/1 specializations guards; for backed dynamic shapes,
            # torch.compile will specialize for 0/1 inputs or otherwise guards that
            # shape is >= 2. This is because it's really hard not to hit a check
            # against 0/1. When we evaluate shape guards, we exclude checking those
            # guards (We would fail always otherwise).

            # We avoid that by updating the ranges of backed sizes when the min is
            # 2 for any, we assume it's 0.
            for s, r in fake_mode.shape_env.var_to_range.items():
                if r.lower == 2:
                    fake_mode.shape_env.var_to_range[s] = ValueRanges(0, r.upper)

        graph_path = os.path.join(local_cache_dir, "computation_graph.py")
        if not os.path.exists(graph_path):
            # code adapted from
            # https://github.com/thuml/depyf/blob/dab831108a752d1facc00acdd6d4243891845c37/depyf/explain/patched_lazy_format_graph_code.py#L30
            # use `print_readable` because it can include submodules
            src = (
                "from __future__ import annotations\nimport torch\n"
                + self.split_gm.print_readable(print_output=False)
            )
            src = src.replace("<lambda>", "GraphModule")
            with open(graph_path, "w") as f:
                f.write(src)

            logger.debug_once(
                "Computation graph saved to %s", graph_path, scope="local"
            )

        self._called = True
        graph_to_serialize = (
            original_split_gm if envs.VLLM_USE_MEGA_AOT_ARTIFACT else self.graph
        )

        if (
            self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
            or not self.compilation_config.cudagraph_copy_inputs
        ):
            return VllmSerializableFunction(
                graph_to_serialize,
                example_inputs,
                self.prefix,
                self.split_gm,
                is_encoder=self.is_encoder,
                vllm_backend=self,
            )

        # index of tensors that have symbolic shapes (batch size)
        # for weights and static buffers, they will have concrete shapes.
        # symbolic shape only happens for input tensors.
        from torch.fx.experimental.symbolic_shapes import is_symbolic

        sym_tensor_indices = [
            i
            for i, x in enumerate(fake_args)
            if isinstance(x, torch._subclasses.fake_tensor.FakeTensor)
            and any(is_symbolic(d) for d in x.size())
        ]

        # compiler managed cudagraph input buffers
        # we assume the first run with symbolic shapes
        # has the maximum size among all the tensors
        copy_and_call = make_copy_and_call(
            sym_tensor_indices,
            [example_inputs[x].clone() for x in sym_tensor_indices],
            self.split_gm,
        )

        return VllmSerializableFunction(
            graph_to_serialize,
            example_inputs,
            self.prefix,
            copy_and_call,
            is_encoder=self.is_encoder,
            vllm_backend=self,
            sym_tensor_indices=sym_tensor_indices,
        )
