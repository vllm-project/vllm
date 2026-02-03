# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import dataclasses
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


@dataclasses.dataclass
class CompilationCounter:
    num_models_seen: int = 0
    num_graphs_seen: int = 0
    # including the splitting ops
    num_piecewise_graphs_seen: int = 0
    # not including the splitting ops
    num_piecewise_capturable_graphs_seen: int = 0
    num_backend_compilations: int = 0
    # Number of gpu_model_runner attempts to trigger CUDAGraphs capture
    num_gpu_runner_capture_triggers: int = 0
    # Number of CUDAGraphs captured
    num_cudagraph_captured: int = 0
    # InductorAdapter.compile calls
    num_inductor_compiles: int = 0
    # EagerAdapter.compile calls
    num_eager_compiles: int = 0
    # The number of time vLLM's compiler cache entry was updated
    num_cache_entries_updated: int = 0
    # The number of standalone_compile compiled artifacts saved
    num_compiled_artifacts_saved: int = 0
    # Number of times a model was loaded with CompilationMode.STOCK_TORCH_COMPILE
    stock_torch_compile_count: int = 0

    def clone(self) -> "CompilationCounter":
        return copy.deepcopy(self)

    @contextmanager
    def expect(self, **kwargs: Any) -> Generator[None, None, None]:
        old = self.clone()
        yield
        for k, v in kwargs.items():
            assert getattr(self, k) - getattr(old, k) == v, (
                f"{k} not as expected, before it is {getattr(old, k)}"
                f", after it is {getattr(self, k)}, "
                f"expected diff is {v}"
            )


compilation_counter = CompilationCounter()
