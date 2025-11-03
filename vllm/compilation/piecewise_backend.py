# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import io
import pickle
from collections.abc import Callable
from pickle import Pickler
from typing import Any

import torch._functorch.config
import torch.fx as fx
from torch._inductor.runtime.triton_heuristics import CachingAutotuner

import vllm.envs as envs
from vllm.compilation.backends import VllmBackend
from vllm.compilation.monitor import end_monitoring_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class ConcreteSizeEntry:
    runtime_shape: int
    compiled: bool = False
    runnable: Callable = None  # type: ignore


class PiecewiseBackend:
    def __init__(
        self,
        graph: fx.GraphModule,
        vllm_config: VllmConfig,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        compiled_graph_for_general_shape: Callable,
        vllm_backend: VllmBackend,
        get_compiled_graph_for_size: Callable | None = None,
        returns_tuple: bool | None = None,
    ):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation of static shapes and
        dispatching based on runtime shape.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.
        """
        self.graph = graph
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.vllm_backend = vllm_backend
        self.get_compiled_graph_for_size = get_compiled_graph_for_size

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == total_piecewise_compiles - 1

        self.is_full_graph = total_piecewise_compiles == 1

        self.compile_sizes: set[int] = set(self.compilation_config.compile_sizes.copy())

        self.first_run_finished = False

        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  # noqa
        self.sym_shape_indices = sym_shape_indices
        self.returns_tuple = returns_tuple

        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # the entries for different shapes that we need to compile
        self.concrete_size_entries: dict[int, ConcreteSizeEntry] = {}

        # to_be_compiled_sizes tracks the remaining sizes to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_sizes: set[int] = self.compile_sizes.copy()

        # We only keep compilation management inside this class directly.
        for shape in self.compile_sizes:
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_shape=shape,
                runnable=self.compiled_graph_for_general_shape,
            )

        self.populate_precompiled_entries()

    def populate_precompiled_entries(self):
        if self.get_compiled_graph_for_size is None:
            return

        for shape, entry in self.concrete_size_entries.items():
            if entry.compiled:
                continue
            entry.runnable = self.get_compiled_graph_wrapper(
                self.get_compiled_graph_for_size(str(shape))
            )
            entry.compiled = True
            logger.debug(
                "setting runnable for shape %s to precompiled graph wrapper", shape
            )
            self.to_be_compiled_sizes.remove(shape)

        # finished compilations for all required shapes
        if self.is_last_graph and not self.to_be_compiled_sizes:
            self.check_for_ending_compilation()

    def get_compiled_graph_wrapper(self, compiled_graph):
        from torch._inductor.compile_fx import graph_returns_tuple

        # For deserialized functions from cache, the graph might be
        # empty. In that case, we can't check graph_returns_tuple
        # from the graph itself. Use the stored returns_tuple value.
        if not self.graph.graph.nodes:
            # Empty graph - use stored returns_tuple value
            if self.returns_tuple is None:
                # No stored value, assume it returns a tuple and just pass through
                def compiled_graph_wrapper_for_cache(*args):
                    return compiled_graph(*args)

                return compiled_graph_wrapper_for_cache
            else:
                # Use the stored returns_tuple value
                returns_tuple = self.returns_tuple

                def compiled_graph_wrapper_with_tuple(*args):
                    graph_output = compiled_graph(*args)
                    if returns_tuple:
                        return graph_output
                    else:
                        # Don't unpack - the AOTCompiledArtifact is returning a list
                        # but something else in the call chain expects to unpack it
                        return graph_output

                return compiled_graph_wrapper_with_tuple

        returns_tuple = graph_returns_tuple(self.graph)

        def compiled_graph_wrapper(*args):
            graph_output = compiled_graph(*args)
            # unpack the tuple if needed
            # TODO(rzou): the implication is that we're not
            # reading the python bytecode correctly in vLLM?
            if returns_tuple or not isinstance(graph_output, (tuple, list)):
                return graph_output
            else:
                return graph_output[0]

        return compiled_graph_wrapper

    def check_for_ending_compilation(self):
        if self.is_last_graph and not self.to_be_compiled_sizes:
            # no specific sizes to compile
            # save the hash of the inductor graph for the next run
            self.vllm_backend.compiler_manager.save_to_file()
            end_monitoring_torch_compile(self.vllm_config)

    def to_bytes(self) -> dict[str, bytes]:
        if not hasattr(self.compiled_graph_for_general_shape, "serialize"):
            return {}

        class InductorCompiledArtifactsPickler(Pickler):
            def reducer_override(self, obj):
                if isinstance(obj, CachingAutotuner):
                    obj.prepare_for_pickle()
                    return pickle.loads, (
                        pickle.dumps(
                            obj,
                        ),
                    )
                return NotImplemented

        def serialize(fn) -> bytes:
            assert hasattr(fn, "serialize"), "fn must have serialize method"
            with torch._functorch.config.patch("bundled_autograd_cache", True):
                entry = fn.serialize()
                # entry.pre_save()

                f = io.BytesIO()
                InductorCompiledArtifactsPickler(f).dump(entry)
                result = f.getvalue()
            return result

        out = {"None": serialize(self.compiled_graph_for_general_shape)}

        for entry in self.concrete_size_entries.values():
            if not entry.compiled:
                logger.debug(
                    "entry with shape %s not compiled, so cannot get its bytes",
                    entry.runtime_shape,
                )
                continue
            out[str(entry.runtime_shape)] = serialize(entry.runnable)

        return out

    def __call__(self, *args) -> Any:
        logger.debug(
            "calling piecewise backend on runtime_shape %s with "
            "remaining compile sizes %s",
            args[self.sym_shape_indices[0]],
            self.to_be_compiled_sizes,
        )

        if not self.first_run_finished:
            self.first_run_finished = True
            # Always wrap the general shape graph on first run if it has a
            # serialize method (meaning it's a deserialized function from cache)
            if hasattr(self.compiled_graph_for_general_shape, "serialize"):
                self.compiled_graph_for_general_shape = self.get_compiled_graph_wrapper(
                    self.compiled_graph_for_general_shape
                )
            self.check_for_ending_compilation()
            return self.compiled_graph_for_general_shape(*args)

        runtime_shape = args[self.sym_shape_indices[0]]

        if runtime_shape not in self.concrete_size_entries:
            # we don't need to do anything for this shape
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        if not entry.compiled:
            assert self.get_compiled_graph_for_size is None
            entry.compiled = True
            self.to_be_compiled_sizes.remove(runtime_shape)
            # args are real arguments

            shape_arg = args[self.sym_shape_indices[0]]
            logger.debug(
                "compiling runnable for piecewise backend on "
                "runtime_shape %s with remaining compile sizes %s",
                shape_arg,
                self.to_be_compiled_sizes,
            )

            with torch._functorch.config.patch("bundled_autograd_cache", True):
                entry.runnable = self.vllm_backend.compiler_manager.compile(
                    self.graph,
                    args,
                    self.compilation_config.inductor_compile_config,
                    self.compilation_config,
                    graph_index=self.piecewise_compile_index,
                    num_graphs=self.total_piecewise_compiles,
                    runtime_shape=runtime_shape,
                )

            # finished compilations for all required shapes
            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()

        return entry.runnable(*args)
