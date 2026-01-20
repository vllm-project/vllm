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

from vllm.compilation.backends import VllmBackend
from vllm.compilation.monitor import end_monitoring_torch_compile
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class RangeEntry:
    compile_range: Range
    compiled: bool = False
    runnable: Callable = None  # type: ignore


class PiecewiseBackend:
    def __init__(
        self,
        graph: fx.GraphModule | None,
        vllm_config: VllmConfig,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        vllm_backend: VllmBackend,
        returns_tuple: bool,
        compiled_runnables: dict[str, Callable] | None = None,
    ):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation of static shapes and
        dispatching based on runtime shape.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.

        This class supports two mutually exclusive modes:
        1. Compilation (graph is set, compiled_runnables is None):
           Used during initial compilation when we have the FX graph
           and need to compile it for each shape range.
        2. Precompilation (graph is None, compiled_runnables is set):
           Used when loading from cache/AOT artifacts where we already
           have pre-compiled callables and don't need the original graph.

        Exactly one of graph or compiled_runnables must be provided.
        """
        assert bool(graph is not None) ^ bool(compiled_runnables is not None), (
            "exactly one of graph and compiled_runnables should be set."
        )

        self.graph = graph
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.vllm_backend = vllm_backend
        self.compiled_runnables = compiled_runnables

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == total_piecewise_compiles - 1

        self.is_full_graph = total_piecewise_compiles == 1
        self.is_encoder_compilation = vllm_backend.is_encoder

        self.compile_ranges = self.compilation_config.get_compile_ranges()
        if self.is_encoder_compilation:
            # For encoder compilation we use the max int32 value
            # to set the upper bound of the compile ranges
            max_int32 = 2**31 - 1
            last_compile_range = self.compile_ranges[-1]
            assert (
                last_compile_range.end
                == vllm_config.scheduler_config.max_num_batched_tokens
            )
            self.compile_ranges[-1] = Range(
                start=last_compile_range.start, end=max_int32
            )

        log_string = f"PiecewiseBackend: compile_ranges: {self.compile_ranges}"
        logger.debug_once(log_string)

        self.compile_sizes = self.compilation_config.compile_sizes
        log_string = f"PiecewiseBackend: compile_sizes: {self.compile_sizes}"
        logger.debug_once(log_string)

        self.sym_shape_indices = sym_shape_indices
        self.returns_tuple = returns_tuple

        # the entries for ranges that we need to either
        self.range_entries: dict[Range, RangeEntry] = {}

        # to_be_compiled_ranges tracks the remaining ranges to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_ranges: set[Range] = set(self.compile_ranges)

        # We only keep compilation management inside this class directly.
        if self.compile_sizes is not None:
            for size in self.compile_sizes:
                if isinstance(size, str):
                    assert size == "cudagraph_capture_sizes"
                    raise NotImplementedError(
                        "cudagraph_capture_sizes not supported in compile_sizes."
                        "This should be handled in `post_init_cudagraph_sizes`."
                    )
                else:
                    assert isinstance(size, int)
                    range = Range(start=size, end=size)
                    if range not in self.compile_ranges:
                        self.range_entries[range] = RangeEntry(
                            compile_range=range,
                        )
                        self.to_be_compiled_ranges.add(range)

        for range in self.compile_ranges:
            self.range_entries[range] = RangeEntry(
                compile_range=range,
            )

        # get the on_compilation_complete callback from context...
        # PiecewiseBackend is created during the first call,
        # which is when the context is set (see compilation/decorators.py)
        from vllm.compilation.backends import _on_compilation_complete_callback

        self.on_compilation_complete = _on_compilation_complete_callback.get()

    def get_compiled_graph_wrapper(self, compiled_graph):
        def compiled_graph_wrapper(*args):
            graph_output = compiled_graph(*args)
            # unpack the tuple if needed
            # TODO(rzou): the implication is that we're not
            # reading the python bytecode correctly in vLLM?
            if self.returns_tuple or not isinstance(graph_output, (tuple, list)):
                return graph_output
            else:
                return graph_output[0]

        return compiled_graph_wrapper

    def check_for_ending_compilation(self) -> None:
        if self.is_last_graph and not self.to_be_compiled_ranges:
            # no specific sizes to compile
            # save the hash of the inductor graph for the next run
            self.vllm_backend.compiler_manager.save_to_file()
            end_monitoring_torch_compile(self.vllm_config)
            # Call the completion callback (e.g., to save AOT compiled function)
            if self.on_compilation_complete is not None:
                self.on_compilation_complete()

    def to_bytes(self) -> dict[str, bytes]:
        class StandaloneCompiledArtifactsPickler(Pickler):
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

                f = io.BytesIO()
                StandaloneCompiledArtifactsPickler(f).dump(entry)
                result = f.getvalue()
            return result

        out = {}

        for range_key, entry in self.range_entries.items():
            if not entry.compiled:
                logger.debug(
                    "entry with range %s not compiled, so cannot get its bytes",
                    range_key,
                )
                continue
            if hasattr(entry.runnable, "serialize"):
                out[str(range_key)] = serialize(entry.runnable)

        return out

    def _fakify_args(self, args: tuple[Any, ...]) -> list[Any]:
        # We need to pass fake example_inputs, otherwise torch.compile
        # will fakify the example_inputs potentially causing some non dynamic
        # dimension to be be duck shaped to other existing shapes that have hints
        # matching their values.
        # This is problem because it can lead to unintended specializations!
        # if the new wrongly dynamic dim is specialized
        # it will force specializing the whole shape
        # torch.compile probably should not accept
        # non fake tensors as example inputs!
        # See issue https://github.com/vllm-project/vllm/issues/27899
        fake_example_inputs = []
        assert self.graph is not None
        for node in self.graph.graph.nodes:
            # All place holders come first
            if node.op == "placeholder":
                fake_example_inputs.append(node.meta["example_value"])
            else:
                break
        assert len(fake_example_inputs) == len(args)
        return fake_example_inputs

    def _maybe_compile_for_range_entry(
        self, range_entry: RangeEntry, args: tuple[Any, ...]
    ) -> Any:
        if not range_entry.compiled:
            if self.compiled_runnables is not None:
                range_entry.runnable = self.get_compiled_graph_wrapper(
                    self.compiled_runnables[str(range_entry.compile_range)]
                )
            else:
                # args are real arguments
                # fakify for range, real args for concrete size.
                # For concrete size, we clear the shape env in
                # compiler_manager.compile() so no need to fakify.
                args_list = (
                    self._fakify_args(args)
                    if not range_entry.compile_range.is_single_size()
                    else list(args)
                )

                with (
                    torch._functorch.config.patch("bundled_autograd_cache", True),
                ):
                    range_entry.runnable = self.vllm_backend.compiler_manager.compile(
                        self.graph,
                        args_list,
                        self.vllm_backend.inductor_config,
                        self.compilation_config,
                        compile_range=range_entry.compile_range,
                        graph_index=self.piecewise_compile_index,
                        num_graphs=self.total_piecewise_compiles,
                    )

            range_entry.compiled = True
            self.to_be_compiled_ranges.remove(range_entry.compile_range)

            self.check_for_ending_compilation()

    def _find_range_for_shape(self, runtime_shape: int) -> RangeEntry | None:
        # First we try to find the range entry for the concrete compile size
        # If not found, we search for the range entry
        # that contains the runtime shape.
        if self.compile_sizes is None:
            return None

        if runtime_shape in self.compile_sizes:
            return self.range_entries[Range(start=runtime_shape, end=runtime_shape)]
        else:
            for range in self.compile_ranges:
                if runtime_shape in range:
                    return self.range_entries[range]
        return None

    def __call__(self, *args: Any) -> Any:
        runtime_shape = args[self.sym_shape_indices[0]]
        range_entry = self._find_range_for_shape(runtime_shape)

        assert range_entry is not None, (
            f"Shape: {runtime_shape} out of considered ranges: {self.compile_ranges}"
        )

        self._maybe_compile_for_range_entry(range_entry, args)
        return range_entry.runnable(*args)
