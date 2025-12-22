# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import os
import pickle
from unittest.mock import patch

import torch
from torch.utils import _pytree as pytree

import vllm.envs as envs
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

    def __init__(
        self, graph_module, example_inputs, prefix, optimized_call, is_encoder=False
    ):
        assert isinstance(graph_module, torch.fx.GraphModule)
        self.graph_module = graph_module
        self.example_inputs = example_inputs
        self.prefix = prefix
        self.optimized_call = optimized_call
        self.is_encoder = is_encoder
        self.shape_env = None
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

        from vllm.compilation.backends import VllmBackend

        state = pickle.loads(data)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        state["graph_module"] = GraphPickler.loads(state["graph_module"], fake_mode)
        state["graph_module"].recompile()
        state["example_inputs"] = GraphPickler.loads(state["example_inputs"], fake_mode)
        is_encoder = state.get("is_encoder", False)
        vllm_backend = VllmBackend(
            get_current_vllm_config(), state["prefix"], is_encoder
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
    def co_name(self):
        """
        Used for depyf debugging.
        """
        return "VllmSerializableFunction"


def compilation_config_hash_factors(vllm_config: VllmConfig) -> list[str]:
    factors = []
    # 0. factors come from the env, for example, The values of
    # VLLM_PP_LAYER_PARTITION will affect the computation graph.
    env_hash = hash_factors(envs.compile_factors())
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
