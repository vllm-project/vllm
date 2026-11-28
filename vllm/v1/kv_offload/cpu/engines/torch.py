# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch copy mechanism for the CPU offload worker.

The engine is one class configured by the platform's `TorchEngineConfig`, so
CUDA/ROCm/XPU do not need engine subclasses. Platform-specific copy functions,
including CUDA's Triton fast path, live in `platforms/`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.cpu.engines.base import (
    CopyBatch,
    CopyEngine,
    CopyStream,
)


@dataclass(frozen=True, slots=True)
class TorchEngineConfig:
    """Platform parameters for the torch copy mechanism."""

    device_module: Any
    copy_fn: Callable[..., None]
    ptr_dtype: torch.dtype = torch.int64
    clear_error: Callable[[], None] | None = None


class TorchCopyStream(CopyStream):
    """A torch stream plus the start/end events of one transfer."""

    def __init__(self, engine: "TorchCopyEngine"):
        self._engine = engine
        self._stream = engine.config.device_module.Stream()
        self._start = torch.Event(enable_timing=True)
        self._end = torch.Event(enable_timing=True)

    def wait_for_compute(self) -> None:
        self._stream.wait_stream(torch.accelerator.current_stream())

    def wait_for(self, previous: CopyStream) -> None:
        assert isinstance(previous, TorchCopyStream)
        self._stream.wait_event(previous._end)

    def submit(self, batch: CopyBatch, num_ops: int) -> None:
        config = self._engine.config
        with config.device_module.stream(self._stream):
            self._start.record(self._stream)
            if num_ops:
                src, dst, sizes = batch.payload
                # CPU->device may prefetch source bytes out of stream order:
                # the source is host memory no GPU stream writes. Device->CPU
                # reads the live KV cache, which the compute stream keeps
                # writing, so it must stay stream-ordered.
                config.copy_fn(
                    src[:num_ops],
                    dst[:num_ops],
                    sizes[:num_ops],
                    is_src_access_order_any=not self._engine.device_to_cpu,
                )
            self._end.record(self._stream)

    def abort(self) -> None:
        if self._engine.config.clear_error is not None:
            self._engine.config.clear_error()
        self._end.record(self._stream)

    def done(self) -> bool:
        return self._end.query()

    def synchronize(self) -> None:
        self._end.synchronize()

    def elapsed_seconds(self) -> float:
        # elapsed_time is in milliseconds.
        return self._start.elapsed_time(self._end) * 1e-3


class TorchCopyEngine(CopyEngine):
    """Torch implementation of `CopyEngine`."""

    def __init__(self, config: TorchEngineConfig, *, device_to_cpu: bool):
        self.config = config
        self.device_to_cpu = device_to_cpu
        self.ptr_dtype = config.ptr_dtype

    def new_stream(self) -> CopyStream:
        return TorchCopyStream(self)

    def new_batch(self, num_ops: int) -> CopyBatch:
        # All three descriptor tensors use the platform pointer dtype: the
        # XPU DMA op requires uint64 sizes as well as uint64 addresses.
        src = torch.empty(num_ops, dtype=self.ptr_dtype, pin_memory=PIN_MEMORY)
        dst = torch.empty(num_ops, dtype=self.ptr_dtype, pin_memory=PIN_MEMORY)
        sizes = torch.empty(num_ops, dtype=self.ptr_dtype, pin_memory=PIN_MEMORY)
        return CopyBatch(
            # uint64 and the tensors' int64 are bit-equivalent for addresses.
            src_ptrs=src.numpy().view(np.uint64),
            dst_ptrs=dst.numpy().view(np.uint64),
            sizes=sizes.numpy(),
            payload=(src, dst, sizes),
        )
