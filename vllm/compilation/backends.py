# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import dataclasses
import hashlib
import os
import pprint
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.fx as fx
from torch._dispatch.python import enable_python_dispatcher

import vllm.envs as envs
from vllm.compilation.inductor_pass import pass_context
from vllm.compilation.partition_rules import (
    inductor_partition_rule_context,
    resolve_defined_ops,
)
from vllm.config import CompilationConfig, CUDAGraphMode, VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.torch_utils import is_torch_equal_or_newer

from .compiler_interface import (
    CompilerInterface,
    EagerAdaptor,
    InductorAdaptor,
    InductorStandaloneAdaptor,
)
from .counter import compilation_counter
from .inductor_pass import InductorPass
from .pass_manager import PostGradPassManager

logger = init_logger(__name__)


def make_compiler(compilation_config: CompilationConfig) -> CompilerInterface:
    # Check if inductor should be used by looking at the backend field
    if compilation_config.backend == "inductor":
        # Use standalone compile if:
        # 1. Explicitly requested via VLLM_USE_STANDALONE_COMPILE, OR
        # 2. AOT compile is enabled (which requires serialization), OR
        # 3. Backend with inductor cache is enabled (requires serialization)
        # AND the PyTorch version is new enough and has standalone_compile
        should_use_standalone = (
            envs.VLLM_USE_STANDALONE_COMPILE
            or envs.VLLM_USE_AOT_COMPILE
            or envs.VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS
        )
        if (
            should_use_standalone
            and is_torch_equal_or_newer("2.8.0.dev")
            and hasattr(torch._inductor, "standalone_compile")
        ):
            if (
                envs.VLLM_USE_AOT_COMPILE
                or envs.VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS
            ) and not envs.VLLM_USE_STANDALONE_COMPILE:
                msg = (
                    "AOT compile or inductor cache backend is enabled, "
                    "automatically using InductorStandaloneAdaptor for "
                    "serialization support"
                )
                logger.info(msg)
            logger.debug("Using InductorStandaloneAdaptor")
            return InductorStandaloneAdaptor()
        else:
            if (
                envs.VLLM_USE_AOT_COMPILE
                or envs.VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS
            ):
                msg = (
                    "VLLM_USE_AOT_COMPILE or "
                    "VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS is "
                    "set but standalone compile is not available. These "
                    "features require PyTorch 2.8.0+ with standalone_compile "
                    "support. Falling back to InductorAdaptor without "
                    "serialization support."
                )
                logger.warning(msg)
            logger.debug("Using InductorAdaptor")
            return InductorAdaptor()
    else:
        logger.debug("Using EagerAdaptor")
        return EagerAdaptor()


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

    def __init__(self, compilation_config: CompilationConfig):
        self.cache: dict[tuple[int | None, int, str], Any] = dict()
        self.is_cache_updated = False
        self.compilation_config = compilation_config
        self.compiler = make_compiler(compilation_config)

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        return self.compiler.compute_hash(vllm_config)

    @contextmanager
    def compile_context(self, runtime_shape: int | None = None):
        """Provide compilation context for the duration of compilation to set
        any torch global properties we want to scope to a single Inductor
        compilation (e.g. partition rules, pass context)."""
        with pass_context(runtime_shape):
            if self.compilation_config.use_inductor_graph_partition:
                inductor_partition_ops = resolve_defined_ops(
                    self.compilation_config.splitting_ops
                )
                with inductor_partition_rule_context(inductor_partition_ops):
                    yield
            else:
                yield

    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ):
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
                self.cache = ast.literal_eval(f.read())

        self.compiler.initialize_cache(
            cache_dir=cache_dir, disable_cache=disable_cache, prefix=prefix
        )

    def save_to_file(self):
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
        runtime_shape: int | None = None,
    ) -> Callable | None:
        if (runtime_shape, graph_index, self.compiler.name) not in self.cache:
            return None
        handle = self.cache[(runtime_shape, graph_index, self.compiler.name)]
        compiled_graph = self.compiler.load(
            handle, graph, example_inputs, graph_index, runtime_shape
        )
        if runtime_shape is None:
            logger.debug(
                "Directly load the %s-th graph for dynamic shape from %s via handle %s",
                graph_index,
                self.compiler.name,
                handle,
            )
        else:
            logger.debug(
                "Directly load the %s-th graph for shape %s from %s via handle %s",
                graph_index,
                str(runtime_shape),
                self.compiler.name,
                handle,
            )
        return compiled_graph

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs,
        additional_inductor_config,
        compilation_config: CompilationConfig,
        graph_index: int = 0,
        num_graphs: int = 1,
        runtime_shape: int | None = None,
    ) -> Any:
        if graph_index == 0:
            # before compiling the first graph, record the start time
            global compilation_start_time
            compilation_start_time = time.time()

        compilation_counter.num_backend_compilations += 1

        compiled_graph = None

        # try to load from the cache
        compiled_graph = self.load(graph, example_inputs, graph_index, runtime_shape)
        if compiled_graph is not None:
            if graph_index == num_graphs - 1:
                # after loading the last graph for this shape, record the time.
                # there can be multiple graphs due to piecewise compilation.
                now = time.time()
                elapsed = now - compilation_start_time
                if runtime_shape is None:
                    logger.info(
                        "Directly load the compiled graph(s) for dynamic shape "
                        "from the cache, took %.3f s",
                        elapsed,
                    )
                else:
                    logger.info(
                        "Directly load the compiled graph(s) for shape %s "
                        "from the cache, took %.3f s",
                        str(runtime_shape),
                        elapsed,
                    )
            return compiled_graph

        # no compiler cached the graph, or the cache is disabled,
        # we need to compile it
        if isinstance(self.compiler, InductorAdaptor):
            # Let compile_fx generate a key for us
            maybe_key = None
        else:
            maybe_key = f"artifact_shape_{runtime_shape}_subgraph_{graph_index}"

        with self.compile_context(runtime_shape):
            compiled_graph, handle = self.compiler.compile(
                graph,
                example_inputs,
                additional_inductor_config,
                runtime_shape,
                maybe_key,
            )

        assert compiled_graph is not None, "Failed to compile the graph"

        # store the artifact in the cache
        if not envs.VLLM_DISABLE_COMPILE_CACHE and handle is not None:
            self.cache[(runtime_shape, graph_index, self.compiler.name)] = handle
            compilation_counter.num_cache_entries_updated += 1
            self.is_cache_updated = True
            if graph_index == 0:
                # adds some info logging for the first graph
                if runtime_shape is None:
                    logger.info_once(
                        "Cache the graph for dynamic shape for later use", scope="local"
                    )
                else:
                    logger.info_once(
                        "Cache the graph of shape %s for later use",
                        str(runtime_shape),
                        scope="local",
                    )
            if runtime_shape is None:
                logger.debug(
                    "Store the %s-th graph for dynamic shape from %s via handle %s",
                    graph_index,
                    self.compiler.name,
                    handle,
                )
            else:
                logger.debug(
                    "Store the %s-th graph for shape %s from %s via handle %s",
                    graph_index,
                    str(runtime_shape),
                    self.compiler.name,
                    handle,
                )

        # after compiling the last graph, record the end time
        if graph_index == num_graphs - 1:
            now = time.time()
            elapsed = now - compilation_start_time
            compilation_config.compilation_time += elapsed
            if runtime_shape is None:
                logger.info_once(
                    "Compiling a graph for dynamic shape takes %.2f s",
                    elapsed,
                    scope="local",
                )
            else:
                logger.info_once(
                    "Compiling a graph for shape %s takes %.2f s",
                    runtime_shape,
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
    graph: fx.GraphModule, resolved_ops: list[torch._ops.OpOverload]
) -> tuple[fx.GraphModule, list[SplitItem]]:
    # split graph by ops
    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        # Match node.target against resolved_ops
        # node.target can be OpOverloadPacket, need to check .default
        if node.op == "call_function" and (
            node.target in resolved_ops
            or (hasattr(node.target, "default") and node.target.default in resolved_ops)
        ):
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
    This function is shared between VllmBackend and VllmBackendWithCache.

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
        compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        and not compilation_config.use_inductor_graph_partition
    ):
        from .cuda_graph import CUDAGraphOptions

        static_graph_wrapper_class = resolve_obj_by_qualname(
            current_platform.get_static_graph_wrapper_cls()
        )

        return static_graph_wrapper_class(
            runnable=piecewise_backend,
            vllm_config=vllm_config,
            runtime_mode=CUDAGraphMode.PIECEWISE,
            cudagraph_options=CUDAGraphOptions(
                debug_log_enable=is_first_graph,
                gc_disable=not is_first_graph,
                weak_ref_output=is_last_graph,
            ),
        )
    else:
        return piecewise_backend


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

    def __init__(
        self,
        module: torch.fx.GraphModule,
        compile_submod_names: list[str],
        vllm_config: VllmConfig,
        vllm_backend: "VllmBackend",
    ):
        super().__init__(module)
        from torch._guards import detect_fake_mode

        self.fake_mode = detect_fake_mode()
        self.compile_submod_names = compile_submod_names
        self.compilation_config = vllm_config.compilation_config
        self.vllm_config = vllm_config
        self.vllm_backend = vllm_backend
        # When True, it annoyingly dumps the torch.fx.Graph on errors.
        self.extra_traceback = False

    def run(self, *args):
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
            global compilation_start_time

            with torch._functorch.config.patch("bundled_autograd_cache", True):
                compiled_graph_for_dynamic_shape = (
                    self.vllm_backend.compiler_manager.compile(
                        submod,
                        args,
                        self.compilation_config.inductor_compile_config,
                        self.compilation_config,
                        graph_index=index,
                        num_graphs=len(self.compile_submod_names),
                        runtime_shape=None,
                    )
                )

            # Lazy import here to avoid circular import
            from .piecewise_backend import PiecewiseBackend

            piecewise_backend = PiecewiseBackend(
                submod,
                self.vllm_config,
                index,
                len(self.compile_submod_names),
                sym_shape_indices,
                compiled_graph_for_dynamic_shape,
                self.vllm_backend,
            )

            # Use the shared cudagraph wrapper function
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


@contextmanager
def set_model_tag(tag: str):
    """Context manager to set the model tag."""
    global model_tag
    assert tag != model_tag, (
        f"Model tag {tag} is the same as the current tag {model_tag}."
    )
    old_tag = model_tag
    model_tag = tag
    try:
        yield
    finally:
        model_tag = old_tag


class VllmBackendWithCache:
    """A backend that reconstructs the compiled model from cached inductor
    artifacts.

    This backend takes the inductor cache directly and constructs a split_gm
    object from scratch, without relying on VllmBackend's existing logic.
    This avoids the overhead of saving the dynamo graph module and
    re-splitting the graph module.

    The workflow is:
    1. Take the inductor cache with all compiled graph pieces
    2. Construct a callable that dispatches to the right compiled graphs
    3. Wrap with cudagraph if needed
    """

    def __init__(
        self,
        inductor_compiled_artifacts: Any,
        vllm_config: VllmConfig,
        prefix: str = "",
        submod_names: list[str] | None = None,
        sym_shape_indices_map: dict[str, list[int]] | None = None,
        returns_tuple_map: dict[str, bool] | None = None,
    ):
        """
        Initialize the backend with an inductor cache.

        Args:
            inductor_compiled_artifacts: The inductor cache containing
                compiled artifacts
            vllm_config: The vLLM configuration
            prefix: The prefix for this backend (e.g., model_tag)
            submod_names: List of submodule names in compilation order
            sym_shape_indices_map: Mapping from submod_name to sym_shape_indices
            returns_tuple_map: Mapping from submod_name to returns_tuple
        """
        self.inductor_compiled_artifacts = inductor_compiled_artifacts
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.prefix = prefix
        self.submod_names = submod_names or []
        self.sym_shape_indices_map = sym_shape_indices_map or {}
        self.returns_tuple_map = returns_tuple_map or {}

        # Create a VllmBackend instance for PiecewiseBackend to use
        # This is needed for compiler_manager access
        self.vllm_backend = VllmBackend(vllm_config, prefix)

        # Initialize the compiler manager with cache disabled since
        # we're loading from cache. We don't need to compile anything
        # new, just need the save_to_file method to work.
        # Use a dummy cache directory since we won't actually write.
        dummy_cache_dir = os.path.join(envs.VLLM_CACHE_ROOT, "dummy_cache")
        os.makedirs(dummy_cache_dir, exist_ok=True)
        self.vllm_backend.compiler_manager.initialize_cache(
            cache_dir=dummy_cache_dir,
            disable_cache=True,
            prefix=prefix,
        )

        # Load all artifacts from cache
        self.inductor_compiled_artifacts.load_all()

        # Build the dispatch callable
        self.build_dispatch_callable()

    def build_dispatch_callable(self):
        """Build a callable that dispatches to the right compiled graphs."""
        # Store compiled callables for each submodule and shape
        self.compiled_callables: dict[str, dict[str, Callable]] = {}

        for submod_name in self.submod_names:
            self.compiled_callables[submod_name] = {}

            # For each submodule, we need to load the compiled graph
            # for each shape. The cache stores entries as
            # "{submod_name}_{shape}". We need to extract the general
            # shape (None) and any specific shapes
            for cache_key in self.inductor_compiled_artifacts.submodule_bytes:
                if cache_key.startswith(f"{submod_name}_"):
                    shape_str = cache_key[len(submod_name) + 1 :]
                    compiled_fn = self.inductor_compiled_artifacts.get_loaded(
                        submod_name, shape_str
                    )
                    self.compiled_callables[submod_name][shape_str] = compiled_fn

    def create_piecewise_backend_from_cache(
        self,
        submod_name: str,
        index: int,
    ):
        """Create a piecewise backend from cached artifacts for a
        specific submodule."""
        from .piecewise_backend import PiecewiseBackend

        # Get the compiled callable for the general shape
        general_shape_fn = self.compiled_callables[submod_name].get("None")
        if general_shape_fn is None:
            raise ValueError(
                f"No general shape compiled function found for {submod_name}"
            )

        # Create a lightweight piecewise backend that uses the cached artifacts
        # We need to create a minimal graph module as a placeholder
        # since PiecewiseBackend expects one
        dummy_graph = fx.GraphModule({}, fx.Graph())

        # Determine which shapes are available for this submodule
        available_shapes = [
            shape_str
            for shape_str in self.compiled_callables[submod_name]
            if shape_str != "None"
        ]

        def get_compiled_graph_for_size(shape_str: str):
            """Get the compiled graph for a specific shape from cache."""
            return self.compiled_callables[submod_name].get(shape_str)

        # Get sym_shape_indices from the map. Default to [0] (batch dimension)
        # if not found, which is the typical case in vLLM where the first
        # argument represents the batch size (a symbolic shape).
        sym_shape_indices = self.sym_shape_indices_map.get(submod_name, [0])

        # Get returns_tuple from the map
        returns_tuple = self.returns_tuple_map.get(submod_name)

        piecewise_backend = PiecewiseBackend(
            graph=dummy_graph,
            vllm_config=self.vllm_config,
            piecewise_compile_index=index,
            total_piecewise_compiles=len(self.submod_names),
            sym_shape_indices=sym_shape_indices,
            compiled_graph_for_general_shape=general_shape_fn,
            vllm_backend=self.vllm_backend,
            get_compiled_graph_for_size=(
                get_compiled_graph_for_size if available_shapes else None
            ),
            returns_tuple=returns_tuple,
        )

        return piecewise_backend

    def create_split_gm_from_cache(self, split_gm: fx.GraphModule) -> fx.GraphModule:
        """Replace the submodules in split_gm with piecewise backends
        loaded from cache.

        This allows us to reuse the graph structure from split_gm while
        loading the compiled artifacts from cache.

        Args:
            split_gm: The split graph module from deserialization. This
                contains the structure of how submodules are chained, but
                the submodules themselves need to be replaced with piecewise
                backends loaded from cache.

        Returns:
            The modified split_gm with submodules replaced by piecewise
            backends from cache.
        """
        for i, submod_name in enumerate(self.submod_names):
            # Create piecewise backend from cache
            piecewise_backend = self.create_piecewise_backend_from_cache(submod_name, i)

            # Wrap with cudagraph if needed
            is_first = i == 0
            is_last = i == len(self.submod_names) - 1
            wrapped_backend = wrap_with_cudagraph_if_needed(
                piecewise_backend,
                self.vllm_config,
                self.compilation_config,
                is_first,
                is_last,
            )

            # Replace the submodule in split_gm
            setattr(split_gm, submod_name, wrapped_backend)
            logger.debug(
                "Replaced submodule %s with piecewise backend from cache",
                submod_name,
            )

        return split_gm


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
    _called: bool = False
    # the graph we compiled
    graph: fx.GraphModule
    # the stiching graph module for all the piecewise graphs
    split_gm: fx.GraphModule
    piecewise_graphs: list[SplitItem]
    returned_callable: Callable
    # Inductor passes to run on the graph pre-defunctionalization
    post_grad_passes: Sequence[Callable]
    sym_tensor_indices: list[int]
    input_buffers: list[torch.Tensor]
    compiler_manager: CompilerManager

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        # if the model is initialized with a non-empty prefix,
        # then usually it's enough to use that prefix,
        # e.g. language_model, vision_model, etc.
        # when multiple parts are initialized as independent
        # models, we need to use the model_tag to distinguish
        # them, e.g. backbone (default), eagle_head, etc.
        self.prefix = prefix or model_tag

        # Passes to run on the graph post-grad.
        self.post_grad_pass_manager = PostGradPassManager()

        self.sym_tensor_indices = []
        self.input_buffers = []

        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config

        self.compiler_manager: CompilerManager = CompilerManager(
            self.compilation_config
        )

        # `torch.compile` is JIT compiled, so we don't need to
        # do anything here

    def _collect_inductor_compiled_artifacts(
        self, submod_names: list[str]
    ) -> tuple[Any, dict[str, list[int]] | None, dict[str, bool] | None]:
        """Collect inductor cache artifacts from all piecewise backends.

        Returns:
            tuple: (inductor_compiled_artifacts, sym_shape_indices_map,
                    returns_tuple_map)
                - inductor_compiled_artifacts: InductorCompiledArtifacts
                  with compiled artifacts
                - sym_shape_indices_map: dict mapping submod_name to
                  sym_shape_indices
                - returns_tuple_map: dict mapping submod_name to
                  returns_tuple
        """

        if not envs.VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS:
            return None, None, None

        from torch._inductor.compile_fx import graph_returns_tuple

        from .caching import VllmSerializableFunction

        inductor_compiled_artifacts = (
            VllmSerializableFunction.InductorCompiledArtifacts()
        )
        sym_shape_indices_map = {}
        returns_tuple_map = {}

        for submod_name in submod_names:
            # Get the piecewise backend from the split_gm
            if not hasattr(self.split_gm, submod_name):
                logger.warning(
                    "Submodule %s not found in split_gm, skipping cache collection",
                    submod_name,
                )
                continue

            piecewise_backend = getattr(self.split_gm, submod_name)

            # If it's wrapped in a CUDA graph wrapper, unwrap it
            if hasattr(piecewise_backend, "runnable"):
                piecewise_backend = piecewise_backend.runnable

            # Collect sym_shape_indices from the piecewise backend
            if hasattr(piecewise_backend, "sym_shape_indices"):
                sym_shape_indices_map[submod_name] = piecewise_backend.sym_shape_indices
                logger.debug(
                    "Collected sym_shape_indices for %s: %s",
                    submod_name,
                    piecewise_backend.sym_shape_indices,
                )

            # Collect returns_tuple information
            if hasattr(piecewise_backend, "graph"):
                returns_tuple = graph_returns_tuple(piecewise_backend.graph)
                returns_tuple_map[submod_name] = returns_tuple
                logger.debug(
                    "Collected returns_tuple for %s: %s",
                    submod_name,
                    returns_tuple,
                )

            has_serialize = hasattr(
                piecewise_backend.compiled_graph_for_general_shape, "serialize"
            )
            logger.debug(
                "Piecewise backend for %s: has serialize=%s",
                submod_name,
                has_serialize,
            )

            if has_serialize:
                bytes_dict = piecewise_backend.to_bytes()
                if bytes_dict:
                    for shape_str, bytes_data in bytes_dict.items():
                        inductor_compiled_artifacts.insert(
                            submod_name, shape_str, bytes_data
                        )
                        logger.debug(
                            "Collected inductor cache for %s with shape %s (%d bytes)",
                            submod_name,
                            shape_str,
                            len(bytes_data),
                        )
                else:
                    logger.warning(
                        "Piecewise backend for %s returned empty to_bytes() - "
                        "bundled_autograd_cache may not have been enabled",
                        submod_name,
                    )
            else:
                logger.debug(
                    "Compiled graph for %s does not support serialization "
                    "(missing 'serialize' method). aot "
                    "was not enabled during compilation. Check that "
                    "VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS=1",
                    submod_name,
                )

        logger.info(
            "Collected inductor cache: %d entries, %d artifacts, %d bytes total",
            inductor_compiled_artifacts.num_entries(),
            inductor_compiled_artifacts.num_artifacts(),
            inductor_compiled_artifacts.size_bytes(),
        )

        logger.info(
            "Inductor cache keys: %s",
            list(inductor_compiled_artifacts.submodule_bytes.keys()),
        )

        return inductor_compiled_artifacts, sym_shape_indices_map, returns_tuple_map

    def configure_post_pass(self):
        config = self.compilation_config
        self.post_grad_pass_manager.configure(self.vllm_config)

        # Post-grad custom passes are run using the post_grad_custom_post_pass
        # hook. If a pass for that hook exists, add it to the pass manager.
        inductor_config = config.inductor_compile_config
        PASS_KEY = "post_grad_custom_post_pass"
        if PASS_KEY in inductor_config:
            if isinstance(inductor_config[PASS_KEY], PostGradPassManager):
                # PassManager already added to config, make sure it's correct
                assert (
                    inductor_config[PASS_KEY].uuid()
                    == self.post_grad_pass_manager.uuid()
                )
            else:
                # Config should automatically wrap all inductor passes
                assert isinstance(inductor_config[PASS_KEY], InductorPass)
                self.post_grad_pass_manager.add(inductor_config[PASS_KEY])
        inductor_config[PASS_KEY] = self.post_grad_pass_manager

    def __call__(self, graph: fx.GraphModule, example_inputs):
        from .caching import (
            VllmSerializableFunction,
            _compute_code_hash,
            compilation_config_hash_factors,
        )

        vllm_config = self.vllm_config
        if not self.compilation_config.cache_dir:
            # no provided cache dir, generate one based on the known factors
            # that affects the compilation. if none of the factors change,
            # the cache dir will be the same so that we can reuse the compiled
            # graph.

            factors = compilation_config_hash_factors(vllm_config)
            # 2. factors come from the code files that are traced by Dynamo (
            #    it mainly summarizes how the model is used in forward pass)
            code_hash = _compute_code_hash(self.compilation_config.traced_files)
            factors.append(code_hash)

            # 3. compiler hash
            compiler_hash = self.compiler_manager.compute_hash(vllm_config)
            factors.append(compiler_hash)

            # combine all factors to generate the cache dir
            hash_key = hashlib.md5(
                str(factors).encode(), usedforsecurity=False
            ).hexdigest()[:10]

            cache_dir = os.path.join(
                envs.VLLM_CACHE_ROOT,
                "torch_compile_cache",
                hash_key,
            )
            self.compilation_config.cache_dir = cache_dir

        cache_dir = self.compilation_config.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.compilation_config.cache_dir = cache_dir
        rank = vllm_config.parallel_config.rank
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        local_cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}", self.prefix)
        os.makedirs(local_cache_dir, exist_ok=True)
        self.compilation_config.local_cache_dir = local_cache_dir

        disable_cache = envs.VLLM_DISABLE_COMPILE_CACHE

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

        # Resolve splitting ops from strings to OpOverload objects
        from vllm.compilation.partition_rules import resolve_defined_ops

        resolved_splitting_ops = resolve_defined_ops(
            self.compilation_config.splitting_ops or []
        )

        self.split_gm, self.piecewise_graphs = split_graph(
            graph, resolved_splitting_ops
        )

        from torch._dynamo.utils import lazy_format_graph_code

        # depyf will hook lazy_format_graph_code and dump the graph
        # for debugging, no need to print the graph here
        lazy_format_graph_code("before split", self.graph)
        lazy_format_graph_code("after split", self.split_gm)

        compilation_counter.num_piecewise_graphs_seen += len(self.piecewise_graphs)
        submod_names_to_compile = [
            item.submod_name
            for item in self.piecewise_graphs
            if not item.is_splitting_graph
        ]

        # propagate the split graph to the piecewise backend,
        # compile submodules with symbolic shapes
        PiecewiseCompileInterpreter(
            self.split_gm, submod_names_to_compile, self.vllm_config, self
        ).run(*example_inputs)

        graph_path = os.path.join(local_cache_dir, "computation_graph.py")
        if not os.path.exists(graph_path):
            # code adapted from https://github.com/thuml/depyf/blob/dab831108a752d1facc00acdd6d4243891845c37/depyf/explain/patched_lazy_format_graph_code.py#L30 # noqa
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

        # Extract shape_env from the graph module's fake mode if available
        from torch._guards import detect_fake_mode

        fake_mode = detect_fake_mode()
        shape_env = fake_mode.shape_env if fake_mode else None

        # Extract submod_names from piecewise_graphs for serialization
        submod_names = [
            item.submod_name
            for item in self.piecewise_graphs
            if not item.is_splitting_graph
        ]

        # Collect inductor cache and sym_shape_indices from all
        # piecewise backends
        (
            inductor_compiled_artifacts,
            sym_shape_indices_map,
            returns_tuple_map,
        ) = self._collect_inductor_compiled_artifacts(submod_names)

        if (
            self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
            or not self.compilation_config.cudagraph_copy_inputs
        ):
            return VllmSerializableFunction(
                graph,
                example_inputs,
                self.prefix,
                self.split_gm,
                shape_env,
                submod_names,
                inductor_compiled_artifacts,
                sym_shape_indices_map,
                returns_tuple_map,
            )

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
            i
            for i, x in enumerate(fake_args)
            if isinstance(x, torch._subclasses.fake_tensor.FakeTensor)
            and any(is_symbolic(d) for d in x.size())
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

        return VllmSerializableFunction(
            graph,
            example_inputs,
            self.prefix,
            copy_and_call,
            shape_env,
            submod_names,
            inductor_compiled_artifacts,
            sym_shape_indices_map,
            returns_tuple_map,
        )
