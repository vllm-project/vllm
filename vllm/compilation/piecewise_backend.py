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
from vllm.config.compilation import Range
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
        graph: fx.GraphModule,
        vllm_config: VllmConfig,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
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
        for size in self.compile_sizes:
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

        self.populate_precompiled_entries()

    def populate_precompiled_entries(self):
        # Populate entries from cache if get_compiled_graph_for_size is provided
        if self.get_compiled_graph_for_size is not None:
            for range_key, entry in self.range_entries.items():
                if entry.compiled:
                    continue
                # Use str(range_key) to match the key format used in to_bytes()
                compiled_graph = self.get_compiled_graph_for_size(str(range_key))
                if compiled_graph is not None:
                    entry.runnable = self.get_compiled_graph_wrapper(compiled_graph)
                    entry.compiled = True
                    logger.debug(
                        "setting runnable for range %s to precompiled graph wrapper",
                        range_key,
                    )
                    self.to_be_compiled_ranges.discard(range_key)

        # finished compilations for all required shapes
        if self.is_last_graph and not self.to_be_compiled_ranges:
            self.check_for_ending_compilation()

    def get_compiled_graph_wrapper(self, compiled_graph):
        from torch._inductor.compile_fx import graph_returns_tuple

        # when deserializing functions from cache, the graph might be
        # empty. in this case, we can't check graph_returns_tuple
        # from the graph itself, and must use the stored returns_tuple value
        # (which is only set in create_piecewise_backend_from_cache)
        returns_tuple = (
            self.returns_tuple
            if self.returns_tuple is not None
            else graph_returns_tuple(self.graph)
        )

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
        if self.is_last_graph and not self.to_be_compiled_ranges:
            # no specific sizes to compile
            # save the hash of the inductor graph for the next run
            self.vllm_backend.compiler_manager.save_to_file()
            end_monitoring_torch_compile(self.vllm_config)
            # Call the completion callback (e.g., to save AOT compiled function)
            if self.vllm_backend.on_compilation_complete is not None:
                self.vllm_backend.on_compilation_complete()

    def to_bytes(self) -> dict[str, bytes]:
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

    def _fakify_args(self) -> list[Any]:
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
        for node in self.graph.graph.nodes:
            # All place holders come first
            if node.op == "placeholder":
                fake_example_inputs.append(node.meta["example_value"])
            else:
                break
        return fake_example_inputs

    def _maybe_compile_for_range_entry(
        self,
        range_entry: RangeEntry,
        args: list[Any] | None = None,
    ) -> Any:
        if args is not None:
            logger.debug(
                "calling piecewise backend on runtime_shape %s with "
                "remaining compile ranges %s",
                args[self.sym_shape_indices[0]],
                self.to_be_compiled_ranges,
            )

        if not range_entry.compiled:
            range_entry.compiled = True
            self.to_be_compiled_ranges.discard(range_entry.compile_range)

            # Determine compile_args based on range type and how we were called
            fake_mode = None
            if not range_entry.compile_range.is_single_size():
                # For general shape (non-single-size), use fakified args
                compile_args = self._fakify_args()
                # Get the fake mode from the fake tensors
                for arg in compile_args:
                    if isinstance(arg, torch.Tensor):
                        from torch._subclasses.fake_tensor import FakeTensor

                        if isinstance(arg, FakeTensor):
                            fake_mode = arg.fake_mode
                            break
            else:
                # For single-size, use the provided real args
                assert args is not None, "Single-size compilation requires real args"
                compile_args = args

            from contextlib import nullcontext

            with (
                fake_mode or nullcontext(),
                torch._functorch.config.patch("bundled_autograd_cache", True),
            ):
                range_entry.runnable = self.vllm_backend.compiler_manager.compile(
                    self.graph,
                    compile_args,
                    self.vllm_backend.inductor_config,
                    self.compilation_config,
                    compile_range=range_entry.compile_range,
                    graph_index=self.piecewise_compile_index,
                    num_graphs=self.total_piecewise_compiles,
                )

            self.check_for_ending_compilation()

    def _find_range_for_shape(self, runtime_shape: int) -> Range | None:
        # First we try to find the range entry for the concrete compile size
        # If not found, we search for the range entry
        # that contains the runtime shape.
        if runtime_shape in self.compile_sizes:
            return self.range_entries[Range(start=runtime_shape, end=runtime_shape)]
        else:
            for range in self.compile_ranges:
                if runtime_shape in range:
                    return self.range_entries[range]
        return None

    def __call__(self, *args) -> Any:
        runtime_shape = args[self.sym_shape_indices[0]]
        range_entry = self._find_range_for_shape(runtime_shape)

        assert range_entry is not None, (
            f"Shape: {runtime_shape} out of considered ranges: {self.compile_ranges}"
        )

        self._maybe_compile_for_range_entry(range_entry, list(args))
        return range_entry.runnable(*args)
