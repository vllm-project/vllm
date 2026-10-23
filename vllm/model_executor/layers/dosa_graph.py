# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private graph captures retained by one fixed-shape DoSA workspace."""

import weakref
from contextlib import contextmanager

import torch


def _destroy_stream(runtime, raw, device: torch.device):
    with torch.accelerator.device_index(device.index):
        runtime.cudaStreamDestroy(raw)


class _Graph(torch.cuda.CUDAGraph):
    capture_stream = None
    # Python reset rearms retained resources; finalization only resets native nodes.
    _native_reset = torch._C._CUDAGraph.reset

    def __del__(self):
        # Reset native nodes before releasing the unpooled capture stream.
        stream = self.capture_stream
        try:
            super().__del__()
        finally:
            self._native_reset()
            _ = stream


@contextmanager
def capture(device: torch.device, *, keep_graph=True):
    """Use an unpooled stream so graph reset cannot clear another BLAS cache."""
    from cuda.bindings import runtime

    with torch.accelerator.device_index(device.index):
        status, raw = runtime.cudaStreamCreateWithFlags(runtime.cudaStreamNonBlocking)
        if status != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA capture stream creation failed: {status}")
        stream = torch.cuda.ExternalStream(int(raw), device=device)
    cleanup = weakref.finalize(stream, _destroy_stream, runtime, raw, device)
    object.__setattr__(cleanup, "atexit", False)
    caller = torch.cuda.current_stream(device)
    stream.wait_stream(caller)
    graph = _Graph(keep_graph=keep_graph)
    graph.capture_stream = stream
    torch.accelerator.synchronize(device)
    with torch.cuda.stream(stream):
        graph.capture_begin(capture_error_mode="global")
        try:
            yield graph
        finally:
            graph.capture_end()
    caller.wait_stream(stream)
