# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import inspect
import os
import pickle
from unittest.mock import patch

import torch
from torch.utils import _pytree as pytree

import vllm.envs as envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger

try:
    from torch._dynamo.aot_compile import SerializableCallable
except ImportError:
    SerializableCallable = object

assert isinstance(SerializableCallable, type)

logger = init_logger(__name__)


class VllmSerializableFunction(SerializableCallable):
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

    class InductorCompiledArtifacts:
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
                    "Using stored inductor cache artifact for submod %s "
                    "with shape %s (%s bytes) at hash %s",
                    submod_name,
                    shape,
                    len(entry),
                    hex_digest,
                )
            else:
                logger.debug(
                    "Inserting inductor artifact for submod %s with shape %s "
                    "(%s bytes) at hash %s",
                    submod_name,
                    shape,
                    len(entry),
                    hex_digest,
                )

        def get(self, submod_name: str, shape: str) -> bytes:
            logger.debug(
                "Getting inductor artifact for submod %s with shape %s",
                submod_name,
                shape,
            )
            return self.submodule_bytes_store[
                self.submodule_bytes[f"{submod_name}_{shape}"]
            ]

        def get_loaded(self, submod_name: str, shape: str):
            logger.debug(
                "Getting inductor artifact for submod %s with shape %s",
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

        def load_all(self) -> None:
            import concurrent.futures

            # check if already loaded
            if len(self.loaded_submodule_store) == len(self.submodule_bytes_store):
                return

            from torch._inductor.standalone_compile import AOTCompiledArtifact

            def _load_entry(entry_bytes) -> AOTCompiledArtifact:
                # Unpickle the bundled cache entry first
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

    def __init__(
        self,
        graph_module,
        example_inputs,
        prefix,
        optimized_call,
        shape_env=None,
        submod_names=None,
        inductor_compiled_artifacts=None,
        sym_shape_indices_map=None,
        returns_tuple_map=None,
    ):
        assert isinstance(graph_module, torch.fx.GraphModule)
        self.graph_module = graph_module
        self.example_inputs = example_inputs
        self.prefix = prefix
        self.optimized_call = optimized_call
        self.shape_env = shape_env
        # Store submodule names for VllmBackendWithCache
        self.submod_names = submod_names or []
        # Store inductor cache for serialization/deserialization
        self.inductor_compiled_artifacts = inductor_compiled_artifacts
        # Store sym_shape_indices for each submodule
        self.sym_shape_indices_map = sym_shape_indices_map or {}
        # Store returns_tuple for each submodule
        self.returns_tuple_map = returns_tuple_map or {}
        if shape_env is None:
            sym_input = next(
                (i for i in self.example_inputs if isinstance(i, torch.SymInt)), None
            )
            if sym_input is not None:
                self.shape_env = sym_input.node.shape_env

    def __call__(self, *args, **kwargs):
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
        for node in state["graph_module"].graph.nodes:
            node.meta.pop("source_fn_stack", None)
            node.meta.pop("nn_module_stack", None)

        graph_reducer_override = GraphPickler.reducer_override

        def _graph_reducer_override(self, obj):
            if (
                inspect.isclass(obj)
                and issubclass(obj, sympy.Function)
                and hasattr(obj, "_torch_unpickler")
            ):
                return obj._torch_unpickler, (obj._torch_handler_name,)
            if isinstance(obj, FakeTensorMode):
                return type(None), ()
            return graph_reducer_override(self, obj)

        # Mask off tensor inputs since they are large and not needed.
        state["example_inputs"] = pytree.tree_map_only(
            torch.Tensor, lambda _: None, state["example_inputs"]
        )
        with patch.object(GraphPickler, "reducer_override", _graph_reducer_override):
            state["graph_module"] = GraphPickler.dumps(
                state["graph_module"], Options(ops_filter=None)
            )
            state["example_inputs"] = GraphPickler.dumps(state["example_inputs"])
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
        state["example_inputs"] = GraphPickler.loads(state["example_inputs"], fake_mode)

        # Check if we should use VllmBackendWithCache
        if envs.VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS:
            from vllm.compilation.backends import VllmBackendWithCache

            # Check if inductor cache has actual artifacts
            inductor_compiled_artifacts = state.get("inductor_compiled_artifacts")
            submod_names = state.get("submod_names")

            has_artifacts = (
                inductor_compiled_artifacts is not None
                and hasattr(inductor_compiled_artifacts, "num_artifacts")
                and inductor_compiled_artifacts.num_artifacts() > 0
            )

            num_artifacts = (
                inductor_compiled_artifacts.num_artifacts()
                if inductor_compiled_artifacts
                else 0
            )
            num_submods = len(submod_names) if submod_names else 0

            logger.info(
                "VllmBackendWithCache check: has_artifacts=%s, "
                "num_artifacts=%d, num_submods=%d",
                has_artifacts,
                num_artifacts,
                num_submods,
            )

            if not has_artifacts or not submod_names:
                # Cache doesn't exist yet or is incomplete
                # Fall back to standard compilation path which will populate it
                logger.warning(
                    "VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS is set "
                    "but inductor cache is empty (artifacts=%d) or submod_names "
                    "is missing (len=%d). Falling back to standard compilation "
                    "path to populate cache.",
                    num_artifacts,
                    num_submods,
                )
                # Continue to fallback path below instead of raising
            else:
                # Cache exists, use VllmBackendWithCache
                logger.info(
                    "Loading from VllmBackendWithCache with %d artifacts "
                    "and %d submodules",
                    num_artifacts,
                    num_submods,
                )

                sym_shape_indices_map = state.get("sym_shape_indices_map", {})
                returns_tuple_map = state.get("returns_tuple_map", {})

                vllm_backend_with_cache = VllmBackendWithCache(
                    inductor_compiled_artifacts=inductor_compiled_artifacts,
                    vllm_config=get_current_vllm_config(),
                    prefix=state["prefix"],
                    submod_names=submod_names,
                    sym_shape_indices_map=sym_shape_indices_map,
                    returns_tuple_map=returns_tuple_map,
                )

                # Get the split_gm from the deserialized state and populate it
                # with piecewise backends from cache
                split_gm = state["graph_module"]

                # Populate split_gm with piecewise backends from cache
                # This replaces the submodules with cached piecewise backends
                populated_split_gm = vllm_backend_with_cache.create_split_gm_from_cache(
                    split_gm
                )

                # Recompile the populated split_gm to generate its forward method.
                # This is necessary because:
                # 1. The graph may contain preprocessing operations beyond just
                #    submodule calls
                # 2. Executing operations eagerly (without compilation) can cause OOM
                # 3. The split_gm wrapper is small, so recompile overhead is minimal
                # 4. The heavy compute is in the cached piecewise backends, not
                #    the wrapper
                populated_split_gm.recompile()

                def optimized_call_with_cache(*args, **kwargs):
                    # Execute the compiled forward method
                    return populated_split_gm(*args, **kwargs)

                fn: VllmSerializableFunction = cls(
                    **state, optimized_call=optimized_call_with_cache
                )
                logger.info("Successfully created VllmBackendWithCache function")
                return fn

        # Fall back to standard VllmBackend
        from vllm.compilation.backends import VllmBackend

        vllm_backend: VllmBackend = VllmBackend(
            get_current_vllm_config(), state["prefix"]
        )

        def optimized_call(*example_inputs):
            """
            On the first run of the optimized call, we rerun the compiler
            backend which should result in a cache hit. After the backend
            call returns, we just do a one-time replacement of the optimized
            call with the compiled function, so that subsequent calls are on
            the AOT compiled path.
            """
            compile_inputs = [
                inp or example_inputs[i] for i, inp in enumerate(fn.example_inputs)
            ]
            with tracing(TracingContext(fake_mode)):
                fn.optimized_call = vllm_backend(
                    state["graph_module"], compile_inputs
                ).optimized_call
            return fn.optimized_call(*example_inputs)

        fn = cls(**state, optimized_call=optimized_call)
        return fn

    @property
    def co_name(self):
        """
        Used for depyf debugging.
        """
        return "VllmSerializableFunction"


def compilation_config_hash_factors(vllm_config: VllmConfig) -> list[str]:
    factors = []
    # 0. factors come from the env, for example, The values of
    # VLLM_PP_LAYER_PARTITION will affect the computation graph.
    env_hash = envs.compute_hash()
    factors.append(env_hash)

    # 1. factors come from the vllm_config (it mainly summarizes how the
    #    model is created)
    config_hash = vllm_config.compute_hash()
    factors.append(config_hash)
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
    return hashlib.md5(
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
