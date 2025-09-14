# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import time
from collections import Counter
from typing import Any, Optional

from vllm.envs import VLLM_GC_DEBUG, VLLM_GC_DEBUG_TOP_COLLECTED_OBJECTS
from vllm.logger import init_logger

logger = init_logger(__name__)

GC_START_NS: int = time.monotonic_ns()
GC_TOP_COLLECTED_OBJECTS: str = ""


def _detailed_type(o: Any) -> str:
    """
    Detailed object type.

    TODO(Jialin): Further enhance the detailed type with element types for
    easier debugging. We tried but occasionally it would run into signals
    which kills the engine.
    """
    if type(o) is list:
        return f"list{len(o)}"
    if type(o) is dict:
        return f"dict{len(o)}"
    if type(o) is tuple:
        return f"tuple{len(o)}"
    return str(type(o))


def _top_gc_collected_objects(generation: Optional[int], top: int) -> str:
    """
    Group collected objects by types.
    """
    object_types = [_detailed_type(o) for o in gc.get_objects(generation or 0)]
    return "\n".join(
        f"{count:>5}:{object_type}" for object_type, count in Counter(
            object_types).most_common(VLLM_GC_DEBUG_TOP_COLLECTED_OBJECTS))


def _debug_gc_callback(phase: str, info: dict[str, int]) -> None:
    global GC_START_NS
    global GC_TOP_COLLECTED_OBJECTS
    if phase == "start":
        # Record GC start time and top collected objects
        GC_START_NS = time.monotonic_ns()
        if VLLM_GC_DEBUG_TOP_COLLECTED_OBJECTS > 0:
            GC_TOP_COLLECTED_OBJECTS = _top_gc_collected_objects(
                info.get('generation'), VLLM_GC_DEBUG_TOP_COLLECTED_OBJECTS)
    elif phase == "stop":
        # Record GC elapsed time and optionally top collected objects
        elpased_ms = (time.monotonic_ns() - GC_START_NS) / 1e6
        logger.info(
            "GC took %.3fms to complete. "
            "Collected %s objects in GC generation %s.%s", elpased_ms,
            str(info.get('collected', '?')), str(info.get('generation')),
            f" Top collected objects: \n{GC_TOP_COLLECTED_OBJECTS}"
            if GC_TOP_COLLECTED_OBJECTS else "")


def maybe_attach_gc_debug_callback() -> None:
    """
    Attached a callback for GC debug when VLLM_GC_DEBUG is enabled.
    """
    if VLLM_GC_DEBUG:
        gc.callbacks.append(_debug_gc_callback)
