# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stock-torch.compile cudagraph wrapper.

Decoupled copy of the capture/replay logic in ``vllm/compilation/cuda_graph.py``
for the STOCK_TORCH_COMPILE path. It is intentionally NOT the shared
``CUDAGraphWrapper``: that class is also used by vLLM's non-torch.compile full
cudagraph path, and keeping the stock path on its own wrapper lets vLLM evolve
full cudagraphs without affecting torch.compile (and vice versa). The capture
semantics mirror ``CUDAGraphWrapper`` (batch-descriptor keyed capture/replay,
shared graph pool, weak-ref outputs); vLLM still owns padding, persistent input
buffers, and the forward-context mode/descriptor dispatch.
"""

import dataclasses
import weakref
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any, ClassVar
from unittest.mock import patch

import torch

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.utils.torch_utils import current_stream, weak_ref_tensors

logger = init_logger(__name__)


@dataclasses.dataclass
class StockCUDAGraphEntry:
    batch_descriptor: BatchDescriptor
    cudagraph: torch.cuda.CUDAGraph | None = None
    output: Any | None = None
    # For debugging: input addresses at capture, checked to be stable on replay.
    input_addresses: list[int] | None = None


@dataclasses.dataclass
class StockCUDAGraphOptions:
    debug_log_enable: bool = True
    gc_disable: bool = False
    weak_ref_output: bool = True


class StockTorchCompileCUDAGraphWrapper:
    """Capture/replay a runnable as a cudagraph, keyed on the forward-context
    batch descriptor. See the module docstring for why this is a separate class
    from the shared ``CUDAGraphWrapper``."""

    _all_instances: ClassVar[weakref.WeakSet["StockTorchCompileCUDAGraphWrapper"]] = (
        weakref.WeakSet()
    )

    @classmethod
    def clear_all_graphs(cls) -> None:
        for instance in list(cls._all_instances):
            instance.clear_graphs()

    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: StockCUDAGraphOptions | None = None,
    ) -> None:
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None

        assert self.runtime_mode != CUDAGraphMode.NONE
        self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = StockCUDAGraphOptions()
        self.cudagraph_options = cudagraph_options
        self.concrete_cudagraph_entries: dict[BatchDescriptor, StockCUDAGraphEntry] = {}

        StockTorchCompileCUDAGraphWrapper._all_instances.add(self)

    def __getattr__(self, key: str) -> Any:
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        if self.is_debugging_mode:
            raise AttributeError(
                f"Attribute {key} not exists in the runnable of "
                f"stock cudagraph wrapper: {self._runnable_str}"
            )
        raise AttributeError

    def unwrap(self) -> Callable[..., Any]:
        return self.runnable

    @property
    def cudagraph_wrapper(self) -> "StockTorchCompileCUDAGraphWrapper":
        return self

    def clear_graphs(self) -> None:
        self.concrete_cudagraph_entries.clear()

    def __call__(self, *args: Any, **kwargs: Any) -> Any | None:
        if not is_forward_context_available():
            # Outside the normal inference path (e.g. a vision encoder forward).
            return self.runnable(*args, **kwargs)

        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if (
            cudagraph_runtime_mode == CUDAGraphMode.NONE
            or cudagraph_runtime_mode != self.runtime_mode
        ):
            # Profile/warmup run, no cudagraph, or a mode meant for a different
            # (nested) wrapper: run directly.
            return self.runnable(*args, **kwargs)

        assert batch_descriptor is not None
        if batch_descriptor not in self.concrete_cudagraph_entries:
            self.concrete_cudagraph_entries[batch_descriptor] = StockCUDAGraphEntry(
                batch_descriptor=batch_descriptor
            )

        entry = self.concrete_cudagraph_entries[batch_descriptor]

        if entry.cudagraph is None:
            # The stock partition wrapper is installed as a process-global that is
            # never unset, so any inductor partition running under a PIECEWISE
            # forward context reaches this capture branch -- including one from an
            # unrelated compile or a different engine. Guard so a leaked or misowned
            # capture fails loudly here instead of silently corrupting memory.
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "StockTorchCompileCUDAGraphWrapper attempted a nested cudagraph capture: "
                    "the current stream is already capturing. A torch.compile "
                    "region likely ran under a PIECEWISE forward context and was "
                    "wrapped by the process-global stock partition wrapper; it must "
                    "go through maybe_disable_graph_partition."
                )
            if (
                forward_context.no_compile_layers
                is not self.compilation_config.static_forward_context
            ):
                raise RuntimeError(
                    "StockTorchCompileCUDAGraphWrapper is capturing a forward driven by a "
                    "different vLLM engine than the one it was built for. The "
                    "process-global stock partition wrapper carries a stale "
                    "compilation config (wrong capture options / gc / weak-ref "
                    "behavior)."
                )
            if self.cudagraph_options.debug_log_enable:
                logger.debug(
                    "Capturing a cudagraph on (%s,%s)",
                    self.runtime_mode.name,
                    entry.batch_descriptor,
                )
            validate_cudagraph_capturing_enabled()

            entry.input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if self.cudagraph_options.gc_disable:
                    # Piecewise mode captures one graph per partition; running gc
                    # per partition is very slow, so gc only for the first.
                    stack.enter_context(
                        patch("gc.collect", lambda *args, **kwargs: None)
                    )
                    stack.enter_context(
                        patch(
                            "torch.accelerator.empty_cache",
                            lambda *args, **kwargs: None,
                        )
                    )

                if self.graph_pool is not None:
                    set_graph_pool_id(self.graph_pool)
                else:
                    set_graph_pool_id(current_platform.graph_pool_handle())

                get_offloader().sync_prev_onload()

                with torch.cuda.graph(
                    cudagraph,
                    pool=self.graph_pool,
                    stream=current_stream(),
                ):
                    output = self.runnable(*args, **kwargs)
                    get_offloader().join_after_forward()
                    if self.cudagraph_options.weak_ref_output:
                        # Safe only for the last graph in piecewise mode: its
                        # output is not consumed by another captured graph.
                        output = weak_ref_tensors(output)

            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_captured += 1

            # Return the real output (not the weak ref) so pytorch manages the
            # cudagraph-pool memory correctly during capture.
            return output

        if self.is_debugging_mode:
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for cudagraphs are different during replay. "
                f"Expected {entry.input_addresses}, got {new_input_addresses}"
            )

        get_offloader().sync_prev_onload()
        entry.cudagraph.replay()
        return entry.output
