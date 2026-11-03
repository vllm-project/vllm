# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-launch instantiation of the CUDA graphs vLLM captures
(``CompilationConfig.cudagraph_launch_mode="device"``)."""

from __future__ import annotations

import functools
import weakref
from typing import Any

import torch

from vllm.compilation.counter import compilation_counter
from vllm.config.compilation import CUDAGraphLaunchMode
from vllm.logger import init_logger

logger = init_logger(__name__)


@functools.cache
def _driver() -> Any:
    from cuda.bindings import driver

    return driver


def new_cuda_graph(launch_mode: CUDAGraphLaunchMode = "host") -> torch.cuda.CUDAGraph:
    """Create an empty CUDA graph that is instantiated as ``launch_mode`` says."""
    if launch_mode == "device":
        return DeviceLaunchCUDAGraph()
    return torch.cuda.CUDAGraph()


def device_launch_counts() -> tuple[int, int]:
    """Graphs instantiated for device launch and graphs that kept PyTorch's
    instantiation so far in this process."""
    return (
        compilation_counter.num_cudagraph_device_launched,
        compilation_counter.num_cudagraph_host_launch_fallbacks,
    )


def log_device_launch_summary(start: tuple[int, int]) -> None:
    """Log how the graphs captured since ``device_launch_counts()`` returned
    ``start`` were instantiated."""
    launched, fallbacks = (n - s for n, s in zip(device_launch_counts(), start))
    if fallbacks:
        logger.warning(
            "%d of %d CUDA graphs kept PyTorch's default instantiation instead of "
            "device launch; see the earlier warning for the reason.",
            fallbacks,
            launched + fallbacks,
        )
    else:
        logger.info("%d CUDA graphs were instantiated for device launch.", launched)


def _destroy_graph_exec(graph_exec: Any) -> None:
    # The driver frees an exec that is still in flight once it completes.
    drv = _driver()
    (err,) = drv.cuGraphExecDestroy(graph_exec)
    if err != drv.CUresult.CUDA_SUCCESS:
        logger.debug("cuGraphExecDestroy failed: %s", getattr(err, "name", err))


def _instantiate_for_device_launch(raw_graph: int, stream: int) -> tuple[Any, str]:
    """Instantiate ``raw_graph`` for device launch and upload it on ``stream``.

    Returns ``(graph_exec, "")``, or ``(None, reason)`` if the driver refuses;
    raises ``torch.OutOfMemoryError`` if it runs out of memory.
    """
    drv = _driver()
    flags = drv.CUgraphInstantiate_flags
    params = drv.CUDA_GRAPH_INSTANTIATE_PARAMS()
    # Same node-priority behavior as PyTorch's host-launch exec. PyTorch also
    # sets AUTO_FREE_ON_LAUNCH, which device launch does not allow; it only
    # matters for graphs with memory-allocation nodes, which device launch
    # rejects anyway.
    params.flags = (
        int(flags.CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH)
        | int(flags.CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY)
        | int(flags.CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD)
    )
    params.hUploadStream = drv.CUstream(stream)
    err, graph_exec = drv.cuGraphInstantiateWithParams(drv.CUgraph(raw_graph), params)
    if err == drv.CUresult.CUDA_SUCCESS:
        return graph_exec, ""
    if err == drv.CUresult.CUDA_ERROR_OUT_OF_MEMORY:
        # Same failure as running out of memory in any other capture.
        raise torch.OutOfMemoryError(
            "Out of memory instantiating a CUDA graph for device launch."
        )

    reason = str(getattr(err, "name", err))
    result = params.result_out
    if result != drv.CUgraphInstantiateResult.CUDA_GRAPH_INSTANTIATE_SUCCESS:
        reason += f", {getattr(result, 'name', result)}"
    if int(params.hErrNode_out):
        node_err, node_type = drv.cuGraphNodeGetType(params.hErrNode_out)
        if node_err == drv.CUresult.CUDA_SUCCESS:
            reason += f", node {getattr(node_type, 'name', node_type)}"
    return None, reason


class DeviceLaunchCUDAGraph(torch.cuda.CUDAGraph):
    """A ``torch.cuda.CUDAGraph`` whose executable graph is instantiated for
    device launch in ``capture_end()``, while the captured graph is live, so
    CUDA graph memory profiling counts it. A graph the driver refuses keeps
    PyTorch's instantiation.

    PyTorch's instantiate hooks do not run, and replays bypass
    ``torch.cuda.CUDAGraph.replay()``: its replay hooks, input-liveness checks
    and device guard do not apply (replay with the capture device current, as
    vLLM's workers do), and neither does its per-replay RNG setup. The seed
    and offset of random numbers drawn from PyTorch's CUDA generator inside
    the graph come from per-capture buffers that PyTorch fills only in
    ``replay()``, so they are undefined: they ignore the seed and repeat on
    every replay.
    """

    def __init__(self) -> None:
        super().__init__()
        self._graph_exec: Any = None
        self._graph_exec_finalizer: weakref.finalize | None = None
        self._device_index = 0
        self._instantiate_called = False

    @property
    def device_launched(self) -> bool:
        """Whether replays use the device-launch executable graph."""
        return self._graph_exec is not None

    def capture_begin(self, *args: Any, **kwargs: Any) -> None:
        if self._graph_exec is not None:
            # PyTorch cannot see this exec, so it would not refuse on its own.
            raise RuntimeError(
                "This CUDAGraph instance already owns a captured graph. "
                "Call reset() or create a new instance to capture again."
            )
        super().capture_begin(*args, **kwargs)

    def capture_end(self) -> None:
        self._instantiate_called = False
        super().capture_end()
        if not self._instantiate_called:
            raise RuntimeError(
                "cudagraph_launch_mode='device' needs torch.cuda.CUDAGraph."
                "capture_end() to call instantiate(), which PyTorch "
                f"{torch.__version__} does not; use cudagraph_launch_mode='host'."
            )

    def instantiate(self) -> None:
        self._instantiate_called = True
        self._device_index = torch.accelerator.current_device_index()
        stream = torch.cuda.current_stream()
        graph_exec, reason = _instantiate_for_device_launch(
            self.raw_cuda_graph(), stream.cuda_stream
        )
        if graph_exec is None:
            # Each rank decides this for its own graphs; log it on every rank.
            # log_device_launch_summary() reports how many graphs fell back.
            logger.warning_once(
                "CUDA graphs that could not be instantiated for device launch "
                "(%s) keep PyTorch's default instantiation.",
                reason,
                scope="process",
            )
            compilation_counter.num_cudagraph_host_launch_fallbacks += 1
            super().instantiate()
            return
        # Finish the upload before the first replay, which usually runs on
        # another stream; a launch is only ordered behind earlier work on its
        # own stream.
        stream.synchronize()
        compilation_counter.num_cudagraph_device_launched += 1
        self._graph_exec = graph_exec
        self._graph_exec_finalizer = weakref.finalize(
            self, _destroy_graph_exec, graph_exec
        )
        # Process teardown frees the exec; skip driver calls at exit.
        self._graph_exec_finalizer.atexit = False  # type: ignore[misc]

    def replay(self) -> None:
        if self._graph_exec is None:
            super().replay()
            return
        drv = _driver()
        stream = torch._C._cuda_getCurrentRawStream(self._device_index)
        (err,) = drv.cuGraphLaunch(self._graph_exec, stream)
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuGraphLaunch failed: {getattr(err, 'name', err)}")

    def reset(self) -> None:
        if self._graph_exec_finalizer is not None:
            self._graph_exec_finalizer()
        self._graph_exec = None
        self._graph_exec_finalizer = None
        super().reset()
