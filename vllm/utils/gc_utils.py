# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import time
from collections import Counter
from typing import Any

from vllm.envs import VLLM_GC_DEBUG, VLLM_GC_DEBUG_TOP_COLLECTED_OBJECTS
from vllm.logger import init_logger

logger = init_logger(__name__)


class GCDebugger:
    """
    Debugger for GC which logs helpful information for GC understanding.
    To enable, you should call maybe_attach_gc_debug_callback in the process.

    Options:
    - VLLM_GC_DEBUG=1: to enable basic GC elpased time logging
    - VLLM_GC_DEBUG_TOP_COLLECTED_OBJECTS=<k>: to enable top collected objects
      logging
    """

    def __init__(self) -> None:
        # Start time in micro second of this GC cycle
        self.start_time_ns: int = time.monotonic_ns()
        # If VLLM_GC_DEBUG_TOP_COLLECTED_OBJECTS is positive,
        # compute top collected objects by object types
        self.gc_top_collected_objects: str = ""

    def gc_callback(self, phase: str, info: dict[str, int]) -> None:
        """
        Callback entry point which could be attached to gc.callbacks
        """
        generation = info.get('generation')
        if generation is None:
            return
        if phase == "start":
            # Before GC started, record GC start time
            # and top collected objects
            self.start_time_ns = time.monotonic_ns()
            self.gc_top_collected_objects = _compute_top_gc_collected_objects(
                gc.get_objects(generation),
                VLLM_GC_DEBUG_TOP_COLLECTED_OBJECTS)
        elif phase == "stop":
            # After GC finished, Record GC elapsed time and
            # optionally top collected objects
            elpased_ms = (time.monotonic_ns() - self.start_time_ns) / 1e6
            logger.info(
                "GC took %.3fms to complete. "
                "Collected %s objects in GC generation %d.%s", elpased_ms,
                str(info.get('collected', '?')), generation,
                f" Top collected objects: \n{self.gc_top_collected_objects}"
                if self.gc_top_collected_objects else "")


def maybe_attach_gc_debug_callback() -> None:
    """
    Attached a callback for GC debug when VLLM_GC_DEBUG is enabled.
    """
    if VLLM_GC_DEBUG:
        gc.callbacks.append(_DEBUGGER.gc_callback)


_DEBUGGER: GCDebugger = GCDebugger()


def _compute_detailed_type(o: Any) -> str:
    """
    Detailed object type.

    TODO(Jialin): Further enhance the detailed type with element types for
    easier debugging. We tried but occasionally it would run into signals
    which kills the engine.
    """
    size_str: str = ""
    if hasattr(o, "__len__"):
        size_str = f"(size:{len(o)})"
    return f"{str(type(o))}{size_str}"


def _compute_top_gc_collected_objects(objects: list[Any], top: int) -> str:
    """
    Group collected objects by types.
    """
    if top <= 0:
        return ""
    object_types = [_compute_detailed_type(o) for o in objects]
    return "\n".join(
        f"{count:>5}:{object_type}"
        for object_type, count in Counter(object_types).most_common(top))
