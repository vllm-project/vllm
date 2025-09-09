# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from collections.abc import Callable
from typing import Any

import torch.fx as fx

from vllm.compilation.backends import VllmBackend
from vllm.compilation.monitor import end_monitoring_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class RangeEntry:
    compile_range: tuple[int, int]
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

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == total_piecewise_compiles - 1

        self.is_full_graph = total_piecewise_compiles == 1

        self.compile_ranges = self.compilation_config.get_compile_ranges()
        log_string = f"PiecewiseBackend: compile_ranges: {self.compile_ranges}"
        logger.debug_once(log_string)

        self.is_in_range = lambda x, range: range[0] <= x < range[1] if range[
            0] < range[1] else x == range[0]

        self.first_run_finished = False

        self.sym_shape_indices = sym_shape_indices

        # the entries for different shapes that we need to compile
        # self.concrete_size_entries: dict[int, RangeEntry] = {}

        # the entries for ranges that we need to either
        self.range_entries: dict[tuple[int, int], RangeEntry] = {}

        # to_be_compiled_ranges tracks the remaining ranges to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_ranges: set[tuple[int,
                                              int]] = set(self.compile_ranges)

        # We only keep compilation management inside this class directly.
        for range in self.compile_ranges:
            self.range_entries[range] = RangeEntry(compile_range=range, )

    def check_for_ending_compilation(self):
        if (self.is_last_graph and not self.to_be_compiled_ranges):
            # no specific sizes to compile
            # save the hash of the inductor graph for the next run
            self.vllm_backend.compiler_manager.save_to_file()
            end_monitoring_torch_compile(self.vllm_config)

    def _maybe_compile_for_range_entry(self, range_entry: RangeEntry,
                                       args) -> Any:
        if not range_entry.compiled:
            range_entry.compiled = True
            self.to_be_compiled_ranges.remove(range_entry.compile_range)

            # args are real arguments
            range_entry.runnable = self.vllm_backend.compiler_manager.compile(
                self.graph,
                args,
                self.compilation_config.inductor_compile_config,
                self.compilation_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                compile_range=range_entry.compile_range)

            # finished compilations for all required shapes
            self.check_for_ending_compilation()

    def __call__(self, *args) -> Any:
        if not self.first_run_finished:
            self.first_run_finished = True

            # Role of the general is taken by the last range
            range_entry = self.range_entries[self.compile_ranges[-1]]
            self._maybe_compile_for_range_entry(range_entry, args)
            return range_entry.runnable(*args)

        runtime_shape = args[self.sym_shape_indices[0]]

        range_found = False
        for range in self.compile_ranges:
            if self.is_in_range(runtime_shape, range):
                range_entry = self.range_entries[range]
                range_found = True
                break
        assert range_found, \
        f"Shape out of considered range: {runtime_shape} " \
        "[1, max_num_batched_tokens]"

        self._maybe_compile_for_range_entry(range_entry, args)

        return range_entry.runnable(*args)
