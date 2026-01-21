# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import inspect
import os
import pickle
from collections.abc import Callable, Sequence
from typing import Any, Literal
from unittest.mock import patch

import torch
from torch.utils import _pytree as pytree

import vllm.envs as envs
from vllm.compilation.compiler_interface import get_inductor_factors
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.utils import hash_factors
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash

try:
    from torch._dynamo.aot_compile import SerializableCallable
except ImportError:
    SerializableCallable = object

assert isinstance(SerializableCallable, type)

logger = init_logger(__name__)


class StandaloneCompiledArtifacts:
    """Storage for standalone compiled artifacts with content-based deduplication.

    Deduplication works via a two-level indirection:
    1. `submodule_bytes` maps "{submod_name}_{shape}" -> SHA256 hash
    2. `submodule_bytes_store` maps SHA256 hash -> actual bytes

    When inserting, we compute the SHA256 hash of the bytes. If the hash
    already exists in `submodule_bytes_store`, we reuse the existing entry
    rather than storing duplicate bytes. This is common because submodules
    often compile to identical artifacts (e.g., identical transformer layers
    split on attn)
    """

    def __init__(self):
        # dict from submodule name to byte hash
        self.submodule_bytes = {}
        # dict from byte hash to bytes
        self.submodule_bytes_store = {}
        # dict from byte hash to loaded module
        self.loaded_submodule_store = {}

    def insert(self, submod_name: str, shape: str, entry: bytes):
        hasher = hashlib.sha256()
        hasher.update(entry)
        hex_digest = hasher.hexdigest()
        self.submodule_bytes[f"{submod_name}_{shape}"] = hex_digest
        if hex_digest not in self.submodule_bytes_store:
            self.submodule_bytes_store[hex_digest] = entry
            logger.debug(
                "inserting new artifact for submod %s with shape %s "
                "(%s bytes) at hash %s",
                submod_name,
                shape,
                len(entry),
                hex_digest,
            )
        else:
            logger.debug(
                "reusing existing cache artifact for submod %s "
                "with shape %s (%s bytes) at hash %s",
                submod_name,
                shape,
                len(entry),
                hex_digest,
            )

    def get(self, submod_name: str, shape: str) -> bytes:
        logger.debug(
            "getting artifact for submod %s with shape %s",
            submod_name,
            shape,
        )
        return self.submodule_bytes_store[
            self.submodule_bytes[f"{submod_name}_{shape}"]
        ]

    def get_loaded(self, submod_name: str, shape: str):
        logger.debug(
            "getting artifact for submod %s with shape %s",
            submod_name,
            shape,
        )
        return self.loaded_submodule_store[
            self.submodule_bytes[f"{submod_name}_{shape}"]
        ]

    def size_bytes(self) -> int:
        return sum(len(entry) for entry in self.submodule_bytes_store.values())

    def num_artifacts(self) -> int:
        return len(self.submodule_bytes_store)

    def num_entries(self) -> int:
        return len(self.submodule_bytes)

    def submodule_names(self) -> list[str]:
        # get unique "{submod_name}" from "{submod_name}_{shape}", preserving order
        names = [cache_key.rsplit("_", 1)[0] for cache_key in self.submodule_bytes]
        return list(dict.fromkeys(names))

    def load_all(self) -> None:
        import concurrent.futures

        # check already loaded
        if len(self.loaded_submodule_store) == len(self.submodule_bytes_store):
            return

        from torch._inductor.standalone_compile import AOTCompiledArtifact

        def _load_entry(entry_bytes) -> AOTCompiledArtifact:
            entry = pickle.loads(entry_bytes)
            return AOTCompiledArtifact.deserialize(entry)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            entries = list(self.submodule_bytes_store.values())
            loaded_entries = list(executor.map(_load_entry, entries))

        for i, k in enumerate(self.submodule_bytes_store.keys()):
            self.loaded_submodule_store[k] = loaded_entries[i]

        logger.debug("loaded all %s submodules", self.num_artifacts())

    def __getstate__(self):
        return {
            "submodule_bytes": self.submodule_bytes,
            "submodule_bytes_store": self.submodule_bytes_store,
        }

    def __setstate__(self, state):
        self.submodule_bytes = state["submodule_bytes"]
        self.submodule_bytes_store = state["submodule_bytes_store"]
        self.loaded_submodule_store = {}


class VllmSerializableFunction(SerializableCallable):  # type: ignore[misc]
    """
    A wrapper around a compiled function by vllm. It will forward the tensor
    inputs to the compiled function and return the result.
    It also implements a serialization interface to support PyTorch's precompile
    with custom backend, so that we can save and load the compiled function on
    disk. There's no need to wrap around the compiled function if we don't want
    to serialize them in particular cases.
    Right now serialization for the custom backend is done via
    serializing the Dynamo fx graph plus example inputs.
    """

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        example_inputs: Sequence[Any],
        prefix: str,
        optimized_call: Callable[..., Any],
        is_encoder: bool = False,
        vllm_backend: Any | None = None,
        sym_tensor_indices: list[int] | None = None,
    ) -> None:
        assert isinstance(graph_module, torch.fx.GraphModule)
        self.graph_module = graph_module
        self.example_inputs = example_inputs
        self.prefix = prefix
        self.optimized_call = optimized_call
        self.is_encoder = is_encoder
        self.shape_env = None
        self.vllm_backend = vllm_backend
        self.sym_tensor_indices = sym_tensor_indices
        sym_input = next(
            (i for i in self.example_inputs if isinstance(i, torch.SymInt)), None
        )
        if sym_input is not None:
            self.shape_env = sym_input.node.shape_env

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.optimized_call(*args, **kwargs)

    @classmethod
    def serialize_compile_artifacts(
        cls, compiled_fn: "VllmSerializableFunction"
    ) -> bytes:
        import sympy
        from torch._subclasses import FakeTensorMode
        from torch.fx._graph_pickler import GraphPickler, Options

        state = compiled_fn.__dict__.copy()
        state.pop("optimized_call")
        state.pop("shape_env")
        state.pop("vllm_backend", None)
        for node in state["graph_module"].graph.nodes:
            node.meta.pop("source_fn_stack", None)
            node.meta.pop("nn_module_stack", None)
        for name, submod in state["graph_module"].named_children():
            if hasattr(submod, "graph"):
                for node in submod.graph.nodes:
                    node.meta.pop("source_fn_stack", None)
                    node.meta.pop("nn_module_stack", None)

        graph_reducer_override = GraphPickler.reducer_override

        def _graph_reducer_override(
            self: GraphPickler, obj: Any
        ) -> tuple[Callable[..., Any], tuple[Any, ...]] | Any:
            if (
                inspect.isclass(obj)
                and issubclass(obj, sympy.Function)
                and hasattr(obj, "_torch_unpickler")
            ):
                return obj._torch_unpickler, (obj._torch_handler_name,)
            if isinstance(obj, FakeTensorMode):
                return type(None), ()
            return graph_reducer_override(self, obj)

        if state.get("sym_tensor_indices"):
            # put tensor inputs on meta device since their data
            # isn't needed, yet we need the meta for make_copy_and_call
            state["example_inputs"] = pytree.tree_map_only(
                torch.Tensor,
                lambda inp: torch.empty_like(inp, device="meta"),
                state["example_inputs"],
            )
        else:
            # mask off all tensor inputs since they are large and not needed.
            state["example_inputs"] = pytree.tree_map_only(
                torch.Tensor,
                lambda inp: torch.empty_like(inp, device="meta"),
                state["example_inputs"],
            )
        with patch.object(GraphPickler, "reducer_override", _graph_reducer_override):
            state["graph_module"] = GraphPickler.dumps(
                state["graph_module"], Options(ops_filter=None)
            )
            state["example_inputs"] = GraphPickler.dumps(state["example_inputs"])

        if compiled_fn.vllm_backend:
            (
                standalone_compile_artifacts,
                sym_shape_indices_map,
                returns_tuple_map,
            ) = compiled_fn.vllm_backend.collect_standalone_compile_artifacts()
            state["standalone_compile_artifacts"] = standalone_compile_artifacts
            state["sym_shape_indices_map"] = sym_shape_indices_map
            state["returns_tuple_map"] = returns_tuple_map
        return pickle.dumps(state)

    @classmethod
    def deserialize_compile_artifacts(cls, data: bytes) -> "VllmSerializableFunction":
        from torch._guards import TracingContext, tracing
        from torch._subclasses import FakeTensorMode
        from torch.fx._graph_pickler import GraphPickler
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        state = pickle.loads(data)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        state["graph_module"] = GraphPickler.loads(state["graph_module"], fake_mode)
        state["graph_module"].recompile()
        state["example_inputs"] = GraphPickler.loads(state["example_inputs"], fake_mode)

        standalone_compile_artifacts = state.pop("standalone_compile_artifacts", None)
        sym_shape_indices_map = state.pop("sym_shape_indices_map", {})
        returns_tuple_map = state.pop("returns_tuple_map", {})

        if envs.VLLM_USE_MEGA_AOT_ARTIFACT:
            assert standalone_compile_artifacts is not None
            submod_names = standalone_compile_artifacts.submodule_names()
            num_submods = len(submod_names)
            num_artifacts = standalone_compile_artifacts.num_artifacts()

            logger.info(
                "reconstructing serializable fn from standalone compile "
                "artifacts. num_artifacts=%d num_submods=%d",
                num_artifacts,
                num_submods,
            )

            fn = reconstruct_serializable_fn_from_mega_artifact(
                state=state,
                standalone_compile_artifacts=standalone_compile_artifacts,
                vllm_config=get_current_vllm_config(),
                sym_shape_indices_map=sym_shape_indices_map,
                returns_tuple_map=returns_tuple_map,
            )

            logger.info(
                "reconstructed serializable fn from standalone compile artifacts"
            )

            return fn

        # Fall back to standard VllmBackend
        from vllm.compilation.backends import VllmBackend

        is_encoder = state.get("is_encoder", False)
        vllm_backend: VllmBackend = VllmBackend(
            get_current_vllm_config(), state["prefix"], is_encoder
        )

        def optimized_call(*example_inputs: Any) -> Any:
            """
            On the first run of the optimized call, we rerun the compiler
            backend which should result in a cache hit. After the backend
            call returns, we just do a one-time replacement of the optimized
            call with the compiled function, so that subsequent calls are on
            the AOT compiled path.
            """
            compile_inputs = [
                inp if inp is not None else example_inputs[i]
                for i, inp in enumerate(fn.example_inputs)
            ]
            with tracing(TracingContext(fake_mode)):
                fn.optimized_call = vllm_backend(
                    state["graph_module"], compile_inputs
                ).optimized_call
            return fn.optimized_call(*example_inputs)

        fn = cls(**state, optimized_call=optimized_call)
        return fn

    @property
    def co_name(self) -> Literal["VllmSerializableFunction"]:
        """
        Used for depyf debugging.
        """
        return "VllmSerializableFunction"


def reconstruct_serializable_fn_from_mega_artifact(
    state: dict[str, Any],
    standalone_compile_artifacts: "StandaloneCompiledArtifacts",
    vllm_config: VllmConfig,
    sym_shape_indices_map: dict[str, list[int]],
    returns_tuple_map: dict[str, bool],
) -> "VllmSerializableFunction":
    """Construct a VllmSerializableFunction from cached inductor artifacts.

    This function reconstructs a callable model from pre-compiled inductor
    artifacts without re-running the compilation. It:
    1. Loads all cached artifacts
    2. Builds compiled callables for each submodule/shape
    3. Creates PiecewiseBackend instances that dispatch to cached artifacts
    4. Wraps with cudagraph if needed
    5. Returns the final VllmSerializableFunction

    Note: This function shares similar logic with PiecewiseCompileInterpreter
    in backends.py. Both create PiecewiseBackend instances and wrap them with
    cudagraph. The key difference is:
    - this function: PiecewiseBackend receives pre-compiled runnables
      (compiled_runnables is set, graph is None)
    - PiecewiseCompileInterpreter: PiecewiseBackend receives the FX graph
      to compile (graph is set, compiled_runnables is None)

    If modifying the backend creation/wrapping logic, consider updating both.

    Args:
        state: Deserialized state dict containing graph_module, example_inputs,
            prefix, sym_tensor_indices, is_encoder, etc.
        standalone_compile_artifacts: The StandaloneCompiledArtifacts containing
            pre-compiled artifacts for each submodule/shape combination.
        vllm_config: The vLLM configuration.
        sym_shape_indices_map: Mapping from submod_name to sym_shape_indices.
        returns_tuple_map: Mapping from submod_name to returns_tuple.

    Returns:
        A VllmSerializableFunction that can be called directly.
    """
    from vllm.compilation.backends import (
        VllmBackend,
        make_copy_and_call,
        wrap_with_cudagraph_if_needed,
    )
    from vllm.compilation.piecewise_backend import PiecewiseBackend

    prefix = state["prefix"]
    is_encoder = state.get("is_encoder", False)
    split_gm = state["graph_module"]
    compilation_config = vllm_config.compilation_config

    standalone_compile_artifacts.load_all()

    submod_names = standalone_compile_artifacts.submodule_names()
    compiled_callables: dict[str, dict[str, Callable]] = {}

    for cache_key in standalone_compile_artifacts.submodule_bytes:
        submod_name, shape_str = cache_key.rsplit("_", 1)
        compiled_callables.setdefault(submod_name, {})[shape_str] = (
            standalone_compile_artifacts.get_loaded(submod_name, shape_str)
        )

    vllm_backend = VllmBackend(vllm_config, prefix, is_encoder)
    dummy_cache_dir = os.path.join(envs.VLLM_CACHE_ROOT, "dummy_cache")
    os.makedirs(dummy_cache_dir, exist_ok=True)
    vllm_backend.compiler_manager.initialize_cache(
        cache_dir=dummy_cache_dir,
        disable_cache=True,
        prefix=prefix,
    )

    # spot check that cached submodules exist in the graph structure
    graph_children = {name for name, _ in split_gm.named_children()}
    missing = set(submod_names) - graph_children
    assert not missing, (
        f"artifacts reference submodules not in graph: {missing}. "
        f"graph has: {sorted(graph_children)}"
    )

    for i, submod_name in enumerate(submod_names):
        assert submod_name in sym_shape_indices_map and submod_name in returns_tuple_map

        sym_shape_indices = sym_shape_indices_map[submod_name]
        returns_tuple = returns_tuple_map[submod_name]
        runnables = compiled_callables[submod_name]

        piecewise_backend = PiecewiseBackend(
            graph=None,  # not needed for cached artifacts
            vllm_config=vllm_config,
            piecewise_compile_index=i,
            total_piecewise_compiles=len(submod_names),
            sym_shape_indices=sym_shape_indices,
            vllm_backend=vllm_backend,
            returns_tuple=returns_tuple,
            compiled_runnables=runnables,
        )

        is_first = i == 0
        is_last = i == len(submod_names) - 1
        wrapped_backend = wrap_with_cudagraph_if_needed(
            piecewise_backend,
            vllm_config,
            compilation_config,
            is_first,
            is_last,
        )

        split_gm.__dict__[submod_name] = wrapped_backend
        logger.debug(
            "Replaced submodule %s with piecewise backend from cache",
            submod_name,
        )

    if compilation_config.cudagraph_copy_inputs:
        sym_tensor_indices = state["sym_tensor_indices"]
        input_buffers = [
            torch.empty_like(
                state["example_inputs"][idx], device=vllm_config.device_config.device
            )
            for idx in sym_tensor_indices
        ]
        optimized_call = make_copy_and_call(sym_tensor_indices, input_buffers, split_gm)
    else:
        optimized_call = split_gm

    fn = VllmSerializableFunction(
        **state,
        optimized_call=optimized_call,
        vllm_backend=None,
    )
    return fn


def aot_compile_hash_factors(vllm_config: VllmConfig) -> list[str]:
    factors = []
    # 0. factors come from the env, for example, The values of
    # VLLM_PP_LAYER_PARTITION will affect the computation graph.
    env_hash = hash_factors(envs.compile_factors())
    factors.append(env_hash)

    # 1. factors come from the vllm_config (it mainly summarizes how the
    #    model is created)
    config_hash = vllm_config.compute_hash()
    factors.append(config_hash)

    # 2. inductor factors if applicable
    if envs.VLLM_USE_MEGA_AOT_ARTIFACT:
        factors.extend(get_inductor_factors())

    return factors


def _compute_code_hash_with_content(file_contents: dict[str, str]) -> str:
    items = list(sorted(file_contents.items(), key=lambda x: x[0]))
    hash_content = []
    for filepath, content in items:
        hash_content.append(filepath)
        if filepath == "<string>":
            # This means the function was dynamically generated, with
            # e.g. exec(). We can't actually check these.
            continue
        hash_content.append(content)
    return safe_hash(
        "\n".join(hash_content).encode(), usedforsecurity=False
    ).hexdigest()


def _compute_code_hash(files: set[str]) -> str:
    logger.debug(
        "Traced files (to be considered for compilation cache):\n%s", "\n".join(files)
    )
    file_contents = {}
    for filepath in files:
        # Skip files that don't exist (e.g., <string>, <frozen modules>, etc.)
        if not os.path.isfile(filepath):
            file_contents[filepath] = ""
        else:
            with open(filepath) as f:
                file_contents[filepath] = f.read()
    return _compute_code_hash_with_content(file_contents)
