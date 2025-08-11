# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import inspect
import os
import pickle
import sys
import types
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from types import CodeType
from typing import Any, Callable, Optional

import torch

import vllm.envs as envs
from vllm.config import CompilationLevel, get_current_vllm_config
from vllm.logger import init_logger
from vllm.utils import is_torch_equal_or_newer

logger = init_logger(__name__)


def normalize_graph_module(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        node.meta.pop("source_fn_stack", None)
        node.meta.pop("nn_module_stack", None)
    return gm


def _unpickle_c_op(name):
    return getattr(torch.ops._C, name)


def _patch_torch_2_9_pickler():
    import sympy
    from torch._dynamo.guards import GuardsStatePickler
    from torch._subclasses import FakeTensorMode
    from torch.fx._graph_pickler import GraphPickler

    graph_reducer_override = GraphPickler.reducer_override

    def _graph_reducer_override(self, obj):
        if (inspect.isclass(obj) and issubclass(obj, sympy.Function)
                and hasattr(obj, "_torch_unpickler")):
            return obj._torch_unpickler, (obj._torch_handler_name, )
        if isinstance(obj, FakeTensorMode):
            return type(None), ()
        return graph_reducer_override(self, obj)

    GraphPickler.reducer_override = _graph_reducer_override

    if is_torch_equal_or_newer("2.9"):
        return

    reducer_override = GuardsStatePickler.reducer_override

    # Porting https://github.com/pytorch/pytorch/pull/158926 from torch 2.9
    def _reducer_override(self, obj):
        if isinstance(obj, torch._ops.OpOverloadPacket
                      ) and obj._qualified_op_name.startswith("_C::"):
            return _unpickle_c_op, (obj.__name__, )

        if (obj.__class__.__module__ == "builtins"
                and obj.__class__.__name__ == "PyCapsule"):
            return object, ()

        if isinstance(obj, types.CodeType):
            return object, ()

        if inspect.isfunction(obj) and obj.__qualname__ != obj.__name__:
            return object, ()

        return reducer_override(self, obj)

    GuardsStatePickler.reducer_override = _reducer_override


def _patch_torch_2_9_package_install(package):
    if is_torch_equal_or_newer("2.9"):
        return
    # Work around https://github.com/pytorch/pytorch/pull/157285 from torch 2.9
    from torch._C._dynamo.eval_frame import _load_precompile_entry
    from torch._dynamo.guards import GuardManagerWrapper, RootGuardManager
    from torch._dynamo.package import SerializedCode

    for code, entry in package._codes.items():
        assert len(entry.guarded_codes) == 1
        for guarded_code in entry.guarded_codes:
            _load_precompile_entry(
                code,
                GuardManagerWrapper(RootGuardManager()),
                SerializedCode.to_code_object(guarded_code.dynamo_code),
            )


@dataclass
class SymInput:
    expr: Any
    hint: Any
    pytype: type

    @classmethod
    def from_sym_node(cls, sym_node):
        return cls(
            expr=sym_node.expr,
            hint=sym_node.hint,
            pytype=sym_node.pytype,
        )

    @classmethod
    def to_sym_node(cls, sym_input, shape_env):
        from torch.fx.experimental.symbolic_shapes import SymNode

        return SymNode(
            sym_input.expr,
            shape_env,
            sym_input.pytype,
            sym_input.hint,
        )


@dataclass
class BackendInputs:
    inlined_sources: set[str]
    example_sym_inputs: dict[int, SymInput]

    @classmethod
    def save(cls, filename, **kwargs):
        with open(filename, "wb") as f:
            pickle.dump(cls(**kwargs), f)

    @classmethod
    def load(cls, filename) -> "BackendInputs":
        with open(filename, "rb") as f:
            ret = pickle.load(f)
        assert isinstance(ret, cls)
        return ret


class TorchCompileWrapperWithCustomDispatcher:
    """
    A wrapper class for torch.compile, with a custom dispatch logic.
    Subclasses should:
    1. Implement the forward method
    2. Implement the dispatch logic in the __call__ method
        It can use `self.compiled_codes` to access the compiled bytecode,
        and `with self.dispatch_to_code(index):` to dispatch to
        the compiled code.
    3. Implement the `__init__` method to determine how to call
        `torch.compile` over the forward method.
    """

    def __init__(
        self,
        compiled_callable: Optional[Callable] = None,
        compilation_level: int = 0,
    ):
        vllm_config = get_current_vllm_config()
        self.vllm_config = vllm_config
        if compiled_callable is None:
            # default compilation settings
            # compiling the forward method

            backend = vllm_config.compilation_config.init_backend(vllm_config)
            options = None
            if isinstance(backend, str) and backend == "inductor":
                options = get_current_vllm_config(
                ).compilation_config.inductor_compile_config

            torch_compile = torch.compile
            if envs.VLLM_USE_TORCH_DYNAMO_CACHING:
                torch_compile = self._torch_compile_with_dynamo_cache

            compiled_callable = torch_compile(
                self.forward,
                fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=backend,
                options=options,
            )

        self.compiled_callable = compiled_callable
        self.original_code_object = self.__class__.forward.__code__
        self.compiled_codes: list[CodeType] = []
        torch._dynamo.convert_frame.register_bytecode_hook(self.bytecode_hook)

        # read the env var to determine whether to use the custom dispatcher
        # subclasses can use this to switch between the custom dispatcher
        # and the default Dynamo guard mechanism.
        self.use_custom_dispatcher: bool = (compilation_level
                                            >= CompilationLevel.DYNAMO_ONCE)

    def _torch_compile_with_dynamo_cache(self, model, fullgraph, backend,
                                         options):
        from torch._dynamo.package import CompilePackage
        from torch._guards import TracingContext, tracing
        from torch._subclasses import FakeTensorMode
        from torch.fx._graph_pickler import GraphPickler, Options
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        try:
            # torch 2.9+
            from torch._dynamo.package import DiskDynamoStore
        except ImportError:
            # torch 2.8
            from torch._dynamo.package import DynamoStore as DiskDynamoStore

        _patch_torch_2_9_pickler()
        package = CompilePackage(model)
        hash_key = package.source_id

        cache_dir = os.path.join(
            envs.VLLM_CACHE_ROOT,
            "torch_dynamo_cache",
            hash_key,
        )

        rank = self.vllm_config.parallel_config.rank
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        local_cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}")

        loading = False
        store = DiskDynamoStore()
        eager_backends = {}
        try:
            package, eager_backends = store.load_package(
                model, local_cache_dir)
            logger.info("Loaded dynamo cache from %s", local_cache_dir)
            loading = True
        except Exception:
            pass

        inlined_sources = set()
        example_sym_inputs = {}

        if loading:

            loaded_backends = {}
            vllm_config = self.vllm_config
            vllm_extras = BackendInputs.load(
                os.path.join(local_cache_dir, "vllm_extras.pickle"))
            inlined_sources = vllm_extras.inlined_sources
            example_sym_inputs = vllm_extras.example_sym_inputs

            class CompiledGraph:

                def __init__(self, gm, fake_mode):
                    self._callback = None
                    self._gm = gm
                    self._fake_mode = fake_mode

                def __call__(self, *example_inputs):
                    if self._callback is None:
                        vllm_config.compilation_config.traced_files = (
                            inlined_sources)
                        compile_inputs = list(example_inputs)
                        for i, example_sym_input in example_sym_inputs.items():
                            compile_inputs[i] = torch.SymInt(
                                SymInput.to_sym_node(example_sym_input,
                                                     fake_mode.shape_env))
                        with tracing(TracingContext(fake_mode)):
                            self._callback = backend(gm, compile_inputs)
                    return self._callback(*example_inputs)

                def after_deserialization(self):
                    return self

            for k, v in eager_backends.items():
                fake_mode = FakeTensorMode(shape_env=ShapeEnv())
                backend_content = v
                if hasattr(backend_content, "after_deserialization"):
                    backend_content = backend_content.after_deserialization()
                assert isinstance(backend_content, bytes)
                gm = GraphPickler.loads(backend_content, fake_mode)

                loaded_backends[k] = CompiledGraph(gm, fake_mode)

            package.install(loaded_backends)
            _patch_torch_2_9_package_install(package)
            vllm_config.compilation_config.load_dynamo_cache = True

        vllm_to_eager_backends = {}

        def _custom_backend(gm, example_inputs):
            if not loading:
                inlined_sources.update(
                    self.vllm_config.compilation_config.traced_files)
                for i, example_input in enumerate(example_inputs):
                    if isinstance(example_input, torch.SymInt):
                        example_sym_inputs[i] = SymInput.from_sym_node(
                            example_input.node)
            ret = backend(gm, example_inputs)
            if not loading:
                vllm_to_eager_backends[ret] = gm
            return ret

        _compiled_callable = torch._dynamo.optimize(
            backend=_custom_backend,
            nopython=fullgraph,
            package=package,
            guard_filter_fn=lambda gs: [False for x in gs],
        )(model)

        @functools.wraps(_compiled_callable)
        def compiled_callable(*args, **kwargs):
            ret = _compiled_callable(*args, **kwargs)
            if not loading:
                for k, v in package.cached_backends.items():
                    store.record_eager_backend(
                        k,
                        GraphPickler.dumps(
                            normalize_graph_module(vllm_to_eager_backends[v]),
                            Options(ops_filter=None),
                        ),
                    )
                if not is_torch_equal_or_newer("2.9"):
                    os.makedirs(local_cache_dir, exist_ok=True)

                store.save_package(package, local_cache_dir)
                BackendInputs.save(
                    os.path.join(local_cache_dir, "vllm_extras.pickle"),
                    inlined_sources=inlined_sources,
                    example_sym_inputs=example_sym_inputs,
                )
            return ret

        return compiled_callable

    def __call__(self, *args, **kwargs):
        """Implement the dispatch logic here, beyond the torch.compile level.
        NOTE: this function can have additional arguments beyond the forward
         method, for directly dispatching to the compiled code.
        """
        return self.compiled_callable(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    def bytecode_hook(self, old_code: CodeType, new_code: CodeType):
        """Hook to save the compiled bytecode for direct execution."""
        if old_code is not self.original_code_object:
            return
        # code borrowed from https://github.com/thuml/depyf/blob/f4ad79fadee27ea113b4c75202db1eb1a11c0dbc/depyf/explain/enable_debugging.py#L25
        frame = sys._getframe()
        while frame and frame.f_back:
            frame = frame.f_back
            code_name = frame.f_code.co_name
            file_name = frame.f_code.co_filename.split(os.path.sep)[-1]
            if code_name == "_compile" and file_name == "convert_frame.py":
                break
        frame = frame.f_locals["frame"]
        assert frame.f_code == old_code

        if frame.f_locals["self"] is not self:
            return

        self.compiled_codes.append(new_code)
        debug_dump_dir = self.vllm_config.compilation_config.debug_dump_path
        if isinstance(debug_dump_dir, str) and debug_dump_dir != "":
            rank = self.vllm_config.parallel_config.rank
            decompiled_file = os.path.join(debug_dump_dir, f"rank_{rank}",
                                           "transformed_code.py")
            if not os.path.exists(decompiled_file):
                try:
                    # usually the decompilation will succeed for most models,
                    # as we guarantee a full-graph compilation in Dynamo.
                    # but there's no 100% guarantee, since decompliation is
                    # not a reversible process.
                    import depyf
                    src = depyf.decompile(new_code)

                    with open(decompiled_file, "w") as f:
                        f.write(src)

                    logger.debug("Dynamo transformed code saved to %s",
                                 decompiled_file)
                except Exception:
                    pass

        if self.vllm_config.compilation_config.use_cudagraph and \
            "update" in new_code.co_names:
            import depyf
            src = depyf.decompile(new_code)
            msg = "Assigning / modifying buffers of nn.Module during forward pass is not allowed when using cudagraph inside the compiler because it will cause silent errors. Please use eager mode or fix the code. The following code contains clues about which buffer is being modified (please search for the usage of the function `update`):\n" + src  # noqa
            raise RuntimeError(msg)

    @contextmanager
    def dispatch_to_code(self, index: int):
        """Context manager to dispatch to the compiled code.
        Why does this work? Because Dynamo guarantees that the compiled
        bytecode has exactly the same arguments, cell variables, and free
        variables as the original code. Therefore we can directly switch
        the code object in the function and call it.

        See https://dev-discuss.pytorch.org/t/what-is-the-relationship-requirement-among-original-bytecode-transformed-bytecode-and-bytecode-returned-by-hooks-in-dynamo/1693/7 for more details.
        """ # noqa
        self.__class__.forward.__code__ = self.compiled_codes[index]
        yield
        self.__class__.forward.__code__ = self.original_code_object
