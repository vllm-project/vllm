# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import io
import json
import pickle
from collections.abc import Callable
from pickle import Pickler
from typing import Any

import torch._functorch.config
import torch.fx as fx
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._logging._internal import trace_structured

from vllm.compilation.backends import VllmBackend
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import init_logger

logger = init_logger(__name__)


def get_fake_args_from_graph(graph: fx.GraphModule) -> list[Any]:
    """Get fake args directly from graph placeholder nodes."""
    fake_args = []
    for node in graph.graph.nodes:
        if node.op == "placeholder":
            fake_args.append(node.meta["example_value"])
        else:
            break
    return fake_args


def create_concrete_args(graph: fx.GraphModule, size: int) -> list[Any]:
    """Create example inputs with symbolic dims replaced by a concrete size.

    Used for single-size eager compilation where we need concrete-shaped
    inputs but don't have real runtime tensors yet.
    """
    from torch.fx.experimental.symbolic_shapes import is_symbolic

    args: list[Any] = []
    for node in graph.graph.nodes:
        if node.op != "placeholder":
            break
        val = node.meta["example_value"]
        if isinstance(val, torch.SymInt):
            args.append(size)
        elif isinstance(val, torch.Tensor):
            new_shape = tuple(size if is_symbolic(d) else int(d) for d in val.shape)
            args.append(torch.zeros(new_shape, dtype=val.dtype, device=val.device))
        else:
            args.append(val)
    return args


@dataclasses.dataclass
class RangeEntry:
    compile_range: Range
    compiled: bool = False
    runnable: Callable[..., Any] = None  # type: ignore


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
        compiled_runnables: dict[str, Callable[..., Any]] | None = None,
        submod_name: str = "",
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
        self.submod_name = submod_name

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

        # Track whether we've logged the graph for this subgraph (only log once)
        self._graph_logged = False

    def get_compiled_graph_wrapper(
        self, compiled_graph: Callable[..., Any]
    ) -> Callable[..., Any]:
        def compiled_graph_wrapper(*args: Any) -> Any:
            graph_output = compiled_graph(*args)
            # unpack the tuple if needed
            # TODO(rzou): the implication is that we're not
            # reading the python bytecode correctly in vLLM?
            if self.returns_tuple or not isinstance(graph_output, (tuple, list)):
                return graph_output
            else:
                return graph_output[0]

        return compiled_graph_wrapper

    def to_bytes(self) -> dict[str, bytes]:
        class StandaloneCompiledArtifactsPickler(Pickler):
            def reducer_override(self, obj: object) -> Any:
                if isinstance(obj, CachingAutotuner):
                    obj.prepare_for_pickle()
                    return pickle.loads, (
                        pickle.dumps(
                            obj,
                        ),
                    )
                return NotImplemented

        def serialize(fn: Callable[..., Any]) -> bytes:
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

    def compile_all_ranges(self) -> None:
        """Compile all range entries for this piecewise subgraph.

        Called from VllmBackend.__call__ to compile all ranges up front
        before the callable is returned.
        """
        assert self.graph is not None, (
            "Cannot compile without a graph. "
            "When loading from cache/AOT artifacts, "
            "compile_all_ranges should not be called."
        )

        for range_entry in self.range_entries.values():
            if range_entry.compiled:
                continue

            self._log_compile_start(range_entry.compile_range)

            if range_entry.compile_range.is_single_size():
                args_list = create_concrete_args(
                    self.graph, range_entry.compile_range.start
                )
            else:
                args_list = get_fake_args_from_graph(self.graph)

            with torch._functorch.config.patch("bundled_autograd_cache", True):
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

    def _log_compile_start(self, compile_range: Range):
        """Log compilation event for TORCH_TRACE/tlparse."""
        is_cudagraph_size = (
            self.compile_sizes is not None and compile_range.start in self.compile_sizes
        )
        subgraph_index = self.piecewise_compile_index
        submod_name = self.submod_name
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "vllm_piecewise_compile_start",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(
                {
                    "piecewise_index": subgraph_index,
                    "submod_name": submod_name,
                    "total_piecewise_compiles": self.total_piecewise_compiles,
                    "compile_range_start": compile_range.start,
                    "compile_range_end": compile_range.end,
                    "is_single_size": compile_range.is_single_size(),
                    "is_cudagraph_capture_size": is_cudagraph_size,
                }
            ),
        )

        # Log the subgraph graph dump only once per subgraph (not per size)
        # to reduce log file size. The graph code is the same for all sizes.
        if not self._graph_logged:
            self._graph_logged = True
            assert self.graph is not None
            trace_structured(
                "graph_dump",
                metadata_fn=lambda: {
                    "name": f"vllm_{submod_name}",
                },
                payload_fn=lambda: self.graph.print_readable(print_output=False),
            )

    def _maybe_compile_for_range_entry(
        self, range_entry: RangeEntry, args: tuple[Any, ...]
    ) -> Any:
        if not range_entry.compiled:
            if self.compiled_runnables is not None:
                range_entry.runnable = self.get_compiled_graph_wrapper(
                    self.compiled_runnables[str(range_entry.compile_range)]
                )
                range_entry.compiled = True
            else:
                raise RuntimeError(
                    "All ranges should be compiled up front in "
                    "VllmBackend.__call__. This code path should not be "
                    f"reached. range_entry={range_entry.compile_range}"
                )

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
