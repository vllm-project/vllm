# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Breakable CUDA graph capture/replay.

This is an alternative to :class:`CUDAGraphWrapper` that replaces vLLM's
torch.compile-based FX graph splitting with runtime stream-capture
breaks.

The idea (inspired by sgl-project/sglang#19102): instead of pre-splitting
the model into many pieces at attention boundaries, a
single capture context drives the whole forward and intercepts
attention / kv-cache custom ops at the dispatcher to end the current
stream capture, run the op eagerly, and resume capture.

The captured artifact is a list of zero-arg callables -- the bound
``CUDAGraph.replay`` for graph segments, or the user fn for eager
segments -- replayed in order at inference time.

Eager segments must operate on the same static buffers used during
capture so subsequent graph segments read the same memory addresses.
"""

from __future__ import annotations

import dataclasses
import functools
import gc
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
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.utils.torch_utils import weak_ref_tensor, weak_ref_tensors

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

    **In-place output buffer required.** Decorated ops must write into a
    caller-provided output tensor; a fresh tensor returned by ``fn`` would
    change address each replay and break downstream graph segments.

    **Decorator order matters.** Apply as the *outermost* decorator if
    there are other decorators that introduce host-side side effects
    around the call -- the canonical example is
    ``@maybe_transfer_kv_layer`` for PD-disaggregation, whose
    ``wait_for_layer_load`` and ``save_kv_layer`` calls must run in the
    eager segment, not inside the captured cudagraph. Putting
    ``@eager_break_during_capture`` *inside* such a decorator would
    record those side effects into the graph and hang on replay.

    The correct order is::

        @eager_break_during_capture   # outermost
        @maybe_transfer_kv_layer
        def unified_attention_with_output(...):
            ...
    """
    if not is_breakable_cudagraph_enabled():
        return fn

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        capture = BreakableCUDAGraphCapture.current()
        if capture is None:
            return fn(*args, **kwargs)
        if not capture._capturing:
            return fn(*args, **kwargs)
        if is_forward_context_available():
            mode = get_forward_context().cudagraph_runtime_mode
            if mode == CUDAGraphMode.FULL:
                return fn(*args, **kwargs)

        # Weak-ref args: strong refs in the replay lambda pin cudagraph-pool
        # slots across batch descriptors. cudagraph owns the slot, so the
        # weak_ref is safe to deref on replay.
        weak_args = tuple(
            weak_ref_tensor(a) if isinstance(a, torch.Tensor) else a for a in args
        )
        weak_kwargs = {
            k: weak_ref_tensor(v) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        return capture.add_eager(lambda: fn(*weak_args, **weak_kwargs))

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
        self.segments: list[Callable[[], Any]] = []
        self._num_graphs: int = 0
        self._num_eager_breaks: int = 0
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
        self._current_graph.capture_end()
        self.segments.append(self._current_graph.replay)
        self._num_graphs += 1
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
        self.segments.append(fn)
        self._num_eager_breaks += 1
        self._begin_segment()
        return result

    # --- replay ----------------------------------------------------------

    def replay(self) -> None:
        for r in self.segments:
            r()

    # --- introspection ---------------------------------------------------

    @property
    def num_graphs(self) -> int:
        return self._num_graphs

    @property
    def num_eager_breaks(self) -> int:
        return self._num_eager_breaks

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
    ) -> None:
        # Unlike the original CUDAGraphWrapper which strictly matches a
        # single runtime_mode, this wrapper captures whatever the
        # dispatcher emits (any non-NONE runtime_mode) -- breakable's
        # capture is identical for prefill and decode, so there's nothing
        # to dispatch on at the runtime_mode level. Entries are keyed by
        # BatchDescriptor which already encodes batch shape / uniformity.
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.graph_pool = current_platform.get_global_graph_pool()
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        self.entries: dict[BatchDescriptor, _BreakableEntry] = {}
        BreakableCUDAGraphWrapper._all_instances.add(self)

        logger.info_once("Breakable CUDA graph enabled")

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

        # Capture whenever the dispatcher says "some cudagraph mode" --
        # breakable produces the same artifact regardless of PIECEWISE
        # vs FULL, so we match either. Entries are keyed by batch
        # descriptor, which already encodes prefill/decode distinctions.
        if cudagraph_runtime_mode == CUDAGraphMode.NONE:
            return self.runnable(*args, **kwargs)

        assert batch_descriptor is not None
        entry = self.entries.get(batch_descriptor)
        if entry is None:
            entry = _BreakableEntry(batch_descriptor=batch_descriptor)
            self.entries[batch_descriptor] = entry

        if entry.capture is None:
            return self._capture(entry, args, kwargs)
        return self._replay(entry, args, kwargs)

    # --- capture / replay paths -----------------------------------------

    @staticmethod
    def _collect_tensor_addresses(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> list[int]:
        """Flatten tensor data_ptrs from positional and keyword args in a
        stable order (positionals first, then kwargs in insertion order).

        Used for the DEBUG-mode address-stability check; covers both call
        styles since vLLM models are typically invoked with kwargs.
        """
        addrs = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
        addrs.extend(
            v.data_ptr() for v in kwargs.values() if isinstance(v, torch.Tensor)
        )
        return addrs

    def _capture(
        self,
        entry: _BreakableEntry,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        validate_cudagraph_capturing_enabled()

        entry.input_addresses = self._collect_tensor_addresses(args, kwargs)

        if self.graph_pool is not None:
            set_graph_pool_id(self.graph_pool)
        else:
            set_graph_pool_id(current_platform.graph_pool_handle())

        # Match torch.cuda.graph()'s pre-capture cleanup once per descriptor.
        # We drive capture_begin/end directly and bypass torch.cuda.graph(),
        # so its built-in gc + empty_cache never fire. Run them here once
        # per _capture call -- NOT inside _begin_segment, since this capture
        # session may issue many begin/end pairs (one per layer's break),
        # and repeated gc would tank capture time the way it did for the
        # pre-`gc_disable` piecewise path.
        gc.collect()
        torch.accelerator.empty_cache()
        # Sync the offloader's copy stream before capture so any in-flight
        # pre-capture prefetches are complete and don't leak into the graph.
        get_offloader().sync_prev_onload()

        capture = BreakableCUDAGraphCapture(pool=self.graph_pool)
        with capture:
            output = self.runnable(*args, **kwargs)
            # Join the offloader's copy stream while we still hold the last
            # segment open, so the join is captured into the graph (otherwise
            # we get an "unjoined stream" error on subsequent forwards).
            get_offloader().join_after_forward()
            # Convert output to a weak ref *inside* the capture context so the
            # strong ref is dropped before the last segment closes, letting
            # the cudagraph pool reclaim/reuse that memory immediately for
            # the next batch descriptor's capture.
            output = weak_ref_tensors(output)

        entry.capture = capture
        entry.output = weak_ref_tensors(output)

        logger.debug(
            "Captured breakable cudagraph for %s: %r",
            entry.batch_descriptor,
            capture,
        )
        # Return the (already-weak) output from the captured run so the
        # caller of model(...) gets a tensor pointing at the cudagraph pool's memory
        return output

    def _replay(
        self,
        entry: _BreakableEntry,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if self.is_debugging_mode and entry.input_addresses is not None:
            new_addresses = self._collect_tensor_addresses(args, kwargs)
            assert new_addresses == entry.input_addresses, (
                "Input tensor addresses changed between capture and replay "
                f"for {entry.batch_descriptor}. Expected "
                f"{entry.input_addresses}, got {new_addresses}."
            )
        # Sync the offloader's copy stream before replay so any external
        # dependencies from pre-capture prefetches are satisfied.
        get_offloader().sync_prev_onload()
        assert entry.capture is not None
        entry.capture.replay()
        return entry.output
