# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from typing import Any, Callable

import torch.fx as fx

import vllm.envs as envs
from vllm.compilation.backends import VllmBackend
from vllm.compilation.monitor import end_monitoring_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from typing import Optional

logger = init_logger(__name__)

@dataclasses.dataclass
class ConditionalEntry:
    runtime_shape: int
    compiled: bool = False
    runnable: Callable = None  # type: ignore
    runtime_range: Optional[tuple[int,
                                  int]] = None  # only used for range entries


class PiecewiseBackend:

    def __init__(self, graph: fx.GraphModule, vllm_config: VllmConfig,
                 piecewise_compile_index: int, total_piecewise_compiles: int,
                 sym_shape_indices: list[int],
                 compiled_graph_for_general_shape: Callable,
                 vllm_backend: VllmBackend):
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

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = (
            piecewise_compile_index == total_piecewise_compiles - 1)

        self.is_full_graph = total_piecewise_compiles == 1

        self.compile_sizes: set[int] = set(
            self.compilation_config.compile_sizes)
        self.compile_ranges: tuple[
            int, int] = self.compilation_config.compile_ranges
        self.is_in_range = lambda x, range: range[0] <= x <= range[1]

        self.first_run_finished = False

        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  # noqa

        self.sym_shape_indices = sym_shape_indices

        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # the entries for different shapes that we need to compile
        self.concrete_size_entries: dict[int, ConditionalEntry] = {}

        # the entries for ranges that we need to either
        # TODO: we should merge with concrete_size_entries
        self.range_entries: dict[tuple[int, int], ConditionalEntry] = {}

        # to_be_compiled_sizes tracks the remaining sizes to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_sizes: set[int] = self.compile_sizes.copy()
        self.to_be_compiled_ranges: set[tuple[int,
                                              int]] = set(self.compile_ranges)

        # We only keep compilation management inside this class directly.
        for shape in self.compile_sizes:
            self.concrete_size_entries[shape] = ConditionalEntry(
                runtime_shape=shape,
                runnable=self.compiled_graph_for_general_shape,
            )

    def check_for_ending_compilation(self):
        if (self.is_last_graph and not self.to_be_compiled_sizes
                and not self.to_be_compiled_ranges):
            # no specific sizes to compile
            # save the hash of the inductor graph for the next run
            self.vllm_backend.compiler_manager.save_to_file()
            end_monitoring_torch_compile(self.vllm_config)

    def __call__(self, *args) -> Any:
        if not self.first_run_finished:
            self.first_run_finished = True
            self.check_for_ending_compilation()
            return self.compiled_graph_for_general_shape(*args)

        runtime_shape = args[self.sym_shape_indices[0]]


        range_entry = None
        for range in self.compile_ranges:
            if self.is_in_range(runtime_shape, range):
                if range not in self.range_entries:
                    self.range_entries[range] = ConditionalEntry(
                        runtime_shape=runtime_shape,
                        runtime_range=range,
                    )
                range_entry = self.range_entries[range]
                break

        if (runtime_shape not in self.concrete_size_entries
                and range_entry is None):
            # we don't need to do anything for this shape
            return self.compiled_graph_for_general_shape(*args)

        if range_entry is not None:
            entry = range_entry
        else:
            entry = self.concrete_size_entries[runtime_shape]

        if not entry.compiled:
            entry.compiled = True
            if range_entry is not None:
                self.to_be_compiled_ranges.remove(range_entry.runtime_range)
            else:
                self.to_be_compiled_sizes.remove(runtime_shape)
            # args are real arguments
            entry.runnable = self.vllm_backend.compiler_manager.compile(
                self.graph,
                args,
                self.compilation_config.inductor_compile_config,
                self.compilation_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                runtime_shape=runtime_shape)

            # finished compilations for all required shapes
            if (self.is_last_graph and not self.to_be_compiled_sizes
                    and not self.to_be_compiled_ranges):
                self.check_for_ending_compilation()

        return entry.runnable(*args)
