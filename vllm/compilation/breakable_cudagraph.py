# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Breakable CUDA graph capture/replay.

This is an alternative to :class:`CUDAGraphWrapper` that replaces vLLM's
torch.compile-based FX graph splitting with runtime stream-capture breaks.

The idea (mirroring sgl-project/sglang#19102): instead of splitting the
model FX graph at attention ops and wrapping each piece in its own
``torch.cuda.graph()``, we let torch.compile produce a single compiled
callable for the whole forward (``splitting_ops = []``) and intercept
attention / kv-cache custom ops at the dispatcher to end the current
stream capture, run the op eagerly, and resume capture.

The captured artifact is a list of ``("graph", CUDAGraph)`` /
``("eager", callable)`` segments, replayed in order at inference time.

Eager segments must operate on the same static buffers used during
capture so subsequent graph segments read the same memory addresses.
"""

from __future__ import annotations

import dataclasses
import functools
import threading
import weakref
from collections.abc import Callable
from typing import Any, ClassVar, TypeVar

import torch

import vllm.envs as envs
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import weak_ref_tensors

logger = init_logger(__name__)


def is_breakable_cudagraph_enabled() -> bool:
    return bool(envs.VLLM_USE_BREAKABLE_CUDAGRAPH)


F = TypeVar("F", bound=Callable[..., Any])


def eager_break_during_capture(fn: F) -> F:
    """Decorator that turns a custom-op Python kernel into a "break point"
    for the breakable cudagraph capture.

    When the decorated function is invoked outside of a
    :class:`BreakableCUDAGraphCapture` context, it executes normally.

    When invoked inside a capture context, it ends the current cudagraph
    segment, runs the function eagerly on the capture stream, records the
    callable for replay, and starts a fresh segment.

    This is the only seam ops should touch -- they don't need to import
    or know about :class:`BreakableCUDAGraphCapture` directly.

    Apply as the *outermost* decorator if there are other decorators
    (e.g. ``@maybe_transfer_kv_layer``) so the wrapped body, including
    those decorators' setup/teardown, runs inside the eager break.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        capture = BreakableCUDAGraphCapture.current()
        if capture is None:
            return fn(*args, **kwargs)
        return capture.add_eager(lambda: fn(*args, **kwargs))

    return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Capture context
# ---------------------------------------------------------------------------


class BreakableCUDAGraphCapture:
    """Stream-capture context that supports eager breaks via :meth:`add_eager`.

    Usage::

        cap = BreakableCUDAGraphCapture(pool=...)
        with cap:
            output = model(*static_inputs)
        # Later, after copying new inputs into the static buffers:
        cap.replay()
        # Output tensors live at the same addresses as during capture.

    Thread-local: only one capture may be active per thread.
    """

    _tls = threading.local()

    @classmethod
    def current(cls) -> BreakableCUDAGraphCapture | None:
        return getattr(cls._tls, "active", None)

    @classmethod
    def is_active(cls) -> bool:
        return cls.current() is not None

    def __init__(self, pool: Any | None = None) -> None:
        self.pool = pool
        # Each segment: ("graph", torch.cuda.CUDAGraph) | ("eager", callable).
        self.segments: list[tuple[str, Any]] = []
        self._current_graph: torch.cuda.CUDAGraph | None = None
        self._capturing: bool = False

    # --- context manager protocol ----------------------------------------

    def __enter__(self) -> BreakableCUDAGraphCapture:
        if getattr(BreakableCUDAGraphCapture._tls, "active", None) is not None:
            raise RuntimeError("Nested BreakableCUDAGraphCapture is not supported.")
        BreakableCUDAGraphCapture._tls.active = self
        self._begin_segment()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._end_segment()
        finally:
            BreakableCUDAGraphCapture._tls.active = None

    # --- segment management ----------------------------------------------

    def _begin_segment(self) -> None:
        assert not self._capturing
        g = torch.cuda.CUDAGraph()
        if self.pool is not None:
            g.capture_begin(pool=self.pool)
        else:
            g.capture_begin()
        self._current_graph = g
        self._capturing = True

    def _end_segment(self) -> None:
        if not self._capturing:
            return
        assert self._current_graph is not None
        # Some segments are legitimately empty (e.g. between consecutive
        # custom ops with no kernels in between, like kv_cache_update
        # immediately followed by attention_with_output). PyTorch warns on
        # those, but they are harmless -- empty replay is a no-op. Suppress
        # the warning so logs stay clean.
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*CUDA Graph is empty.*",
                category=UserWarning,
            )
            self._current_graph.capture_end()
        self.segments.append(("graph", self._current_graph))
        self._current_graph = None
        self._capturing = False

    def add_eager(self, fn: Callable[[], Any]) -> Any:
        """End the current capture segment, run ``fn`` eagerly on the
        capture stream, record ``fn`` for replay, and start a new segment.

        Returns whatever ``fn`` returned during this (capture-time) call.
        Replay does not return values; callers should propagate any
        downstream dependencies via static output buffers.
        """
        self._end_segment()
        result = fn()
        self.segments.append(("eager", fn))
        self._begin_segment()
        return result

    # --- replay ----------------------------------------------------------

    def replay(self) -> None:
        for kind, item in self.segments:
            if kind == "graph":
                item.replay()  # type: ignore[union-attr]
            else:
                item()

    # --- introspection ---------------------------------------------------

    @property
    def num_graphs(self) -> int:
        return sum(1 for kind, _ in self.segments if kind == "graph")

    @property
    def num_eager_breaks(self) -> int:
        return sum(1 for kind, _ in self.segments if kind == "eager")

    def __repr__(self) -> str:
        return (
            f"BreakableCUDAGraphCapture(graphs={self.num_graphs}, "
            f"eager_breaks={self.num_eager_breaks})"
        )


# ---------------------------------------------------------------------------
# Wrapper that mirrors CUDAGraphWrapper's interface
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _BreakableEntry:
    batch_descriptor: BatchDescriptor
    capture: BreakableCUDAGraphCapture | None = None
    output: Any = None
    input_addresses: list[int] | None = None


class BreakableCUDAGraphWrapper:
    """Drop-in replacement for :class:`CUDAGraphWrapper` that uses
    :class:`BreakableCUDAGraphCapture` instead of a single monolithic
    ``torch.cuda.graph()`` capture.

    Same dispatch contract as ``CUDAGraphWrapper``:
        * If no ``forward_context`` is available, run the underlying
          callable eagerly.
        * If runtime mode mismatch / NONE, run eagerly.
        * Otherwise, lazily capture per ``batch_descriptor`` and replay
          on subsequent invocations with the same descriptor.
    """

    _all_instances: ClassVar[weakref.WeakSet[BreakableCUDAGraphWrapper]] = (
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
    ) -> None:
        if runtime_mode == CUDAGraphMode.NONE:
            raise ValueError(
                "BreakableCUDAGraphWrapper requires a non-NONE runtime mode."
            )
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config
        self.graph_pool = current_platform.get_global_graph_pool()
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        self.entries: dict[BatchDescriptor, _BreakableEntry] = {}
        BreakableCUDAGraphWrapper._all_instances.add(self)

    # --- vllm-style attribute forwarding ---------------------------------

    def __getattr__(self, key: str) -> Any:
        runnable = self.__dict__.get("runnable")
        if runnable is not None and hasattr(runnable, key):
            return getattr(runnable, key)
        raise AttributeError(key)

    def unwrap(self) -> Callable[..., Any]:
        return self.runnable

    @property
    def cudagraph_wrapper(self) -> BreakableCUDAGraphWrapper:
        return self

    def clear_graphs(self) -> None:
        self.entries.clear()

    # --- dispatch --------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not is_forward_context_available():
            return self.runnable(*args, **kwargs)

        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if (
            cudagraph_runtime_mode == CUDAGraphMode.NONE
            or cudagraph_runtime_mode != self.runtime_mode
        ):
            return self.runnable(*args, **kwargs)

        assert batch_descriptor is not None
        entry = self.entries.get(batch_descriptor)
        if entry is None:
            entry = _BreakableEntry(batch_descriptor=batch_descriptor)
            self.entries[batch_descriptor] = entry

        if entry.capture is None:
            return self._capture(entry, args, kwargs)
        return self._replay(entry, args)

    # --- capture / replay paths -----------------------------------------

    def _capture(
        self,
        entry: _BreakableEntry,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        validate_cudagraph_capturing_enabled()

        entry.input_addresses = [
            x.data_ptr() for x in args if isinstance(x, torch.Tensor)
        ]

        if self.graph_pool is not None:
            set_graph_pool_id(self.graph_pool)
        else:
            set_graph_pool_id(current_platform.graph_pool_handle())

        capture = BreakableCUDAGraphCapture(pool=self.graph_pool)
        with capture:
            output = self.runnable(*args, **kwargs)

        entry.capture = capture
        # Hold a weak ref to outputs so the cudagraph pool can manage memory,
        # mirroring CUDAGraphWrapper.weak_ref_output behavior.
        entry.output = weak_ref_tensors(output)

        logger.debug(
            "Captured breakable cudagraph for %s: %r",
            entry.batch_descriptor,
            capture,
        )
        # Return the strong ref so the caller sees real tensors during the
        # initial run.
        return output

    def _replay(self, entry: _BreakableEntry, args: tuple[Any, ...]) -> Any:
        if self.is_debugging_mode and entry.input_addresses is not None:
            new_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            assert new_addresses == entry.input_addresses, (
                "Input tensor addresses changed between capture and replay "
                f"for {entry.batch_descriptor}. Expected "
                f"{entry.input_addresses}, got {new_addresses}."
            )
        assert entry.capture is not None
        entry.capture.replay()
        return entry.output
