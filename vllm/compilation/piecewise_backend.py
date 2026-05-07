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
from torch._dynamo.utils import dynamo_timed
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
    """Create Fake example inputs with symbolic dims replaced by a concrete size.

    Used for single-size compilation where we need concrete-shaped inputs.
    The Dynamo-captured graph gives us example inputs with SymInts in them.
    """
    from torch._prims_common import compute_required_storage_length
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, is_symbolic

    def concretize(sym_val: Any) -> int:
        """Replace all symbolic variables in a SymInt expression with size."""
        if not is_symbolic(sym_val):
            return int(sym_val)
        expr = sym_val.node.expr
        return int(expr.subs({s: size for s in expr.free_symbols}))

    fake_mode = FakeTensorMode(shape_env=ShapeEnv())

    args: list[Any] = []
    with fake_mode:
        for node in graph.graph.nodes:
            if node.op != "placeholder":
                break
            val = node.meta["example_value"]
            if isinstance(val, torch.SymInt):
                args.append(concretize(val))
            elif isinstance(val, torch.Tensor):
                new_shape = tuple(concretize(d) for d in val.shape)
                new_strides = tuple(concretize(s) for s in val.stride())
                new_storage_offset = concretize(val.storage_offset())
                needed_size = compute_required_storage_length(
                    new_shape, new_strides, new_storage_offset
                )
                t = torch.empty(needed_size, dtype=val.dtype, device=val.device)
                t = t.as_strided(new_shape, new_strides, new_storage_offset)
                args.append(t)
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

        for range in self.compile_ranges:
            self.range_entries[range] = RangeEntry(
                compile_range=range,
            )

        # Track whether we've logged the graph for this subgraph (only log once)
        self._graph_logged = False

        if self.graph is not None:
            self.compile_all_ranges()
        else:
            self.load_all_ranges()

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
        """Compile all range entries for this piecewise subgraph up front."""
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

            range_entry.runnable = self.vllm_backend.compiler_manager.compile(
                self.graph,
                args_list,
                self.vllm_backend.inductor_config,
                self.compilation_config,
                compile_range=range_entry.compile_range,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                is_encoder=self.vllm_backend.is_encoder,
            )

            range_entry.compiled = True

    @dynamo_timed("vllm_log_compile_start_torch_trace_only")
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

    def load_all_ranges(self) -> None:
        """Load all pre-compiled runnables for this piecewise subgraph.

        Called during warm start to wrap all cached compiled_runnables
        into range_entry.runnable up front, analogous to compile_all_ranges()
        for the cold start path.
        """
        assert self.compiled_runnables is not None, (
            "load_all_ranges should only be called when compiled_runnables "
            "is set (warm start / cache loading path)."
        )
        for range_entry in self.range_entries.values():
            if range_entry.compiled:
                continue
            key = str(range_entry.compile_range)
            assert key in self.compiled_runnables, (
                f"Missing compiled runnable for range {range_entry.compile_range}. "
                f"Available keys: {list(self.compiled_runnables.keys())}"
            )
            range_entry.runnable = self.get_compiled_graph_wrapper(
                self.compiled_runnables[key]
            )
            range_entry.compiled = True

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
        if self.sym_shape_indices:
            runtime_shape = args[self.sym_shape_indices[0]]
            range_entry = self._find_range_for_shape(runtime_shape)
            assert range_entry is not None, (
                f"Shape: {runtime_shape} out of considered ranges: "
                f"{self.compile_ranges}"
            )
        else:
            # All inputs have static shapes; use the only compiled range_entry
            compiled_entries = [re for re in self.range_entries.values() if re.compiled]
            assert len(compiled_entries) == 1, (
                f"Expected exactly one compiled range_entry for static shape "
                f"compilation, but found {len(compiled_entries)}"
            )
            range_entry = compiled_entries[0]

        assert range_entry.compiled, (
            "All ranges should be compiled or loaded up front in "
            "PiecewiseBackend.__init__. "
            f"range_entry={range_entry.compile_range}"
        )
        return range_entry.runnable(*args)
