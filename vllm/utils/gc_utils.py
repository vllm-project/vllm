# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import json
import time
from collections import Counter
from contextlib import suppress
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


class GCDebugConfig:
    """
    Config for GC Debugger.
    - 0: disable GC debugger
    - 1: enable GC debugger with gc.collect elapsed times
    - '{"top_objects":5}': enable GC debugger with top 5 collected objects
    """

    def __init__(self, gc_debug_conf: str | None = None) -> None:
        self.enabled: bool = False
        self.top_objects: int = -1

        if not gc_debug_conf or gc_debug_conf == "0":
            pass
        elif gc_debug_conf == "1":
            self.enabled = True
        else:
            try:
                json_conf = json.loads(gc_debug_conf)
                self.enabled = True
                self.top_objects = json_conf.get("top_objects", -1)
            except Exception:
                self.enabled = False
                logger.error("Failed to parse VLLM_GC_DEBUG(%s)", envs.VLLM_GC_DEBUG)
        logger.debug("GC Debug Config. %s", str(self))

    def __repr__(self) -> str:
        return f"enabled:{self.enabled},top_objects:{self.top_objects}"


class GCDebugger:
    """
    Debugger for GC which logs helpful information for GC understanding.
    To enable, you should call maybe_attach_gc_debug_callback in the process.
    """

    def __init__(self, config: GCDebugConfig) -> None:
        self.config = config
        # Start time in micro second of this GC cycle
        self.start_time_ns: int = time.monotonic_ns()
        self.num_objects: int = 0
        # If config.top_objects is positive,
        # compute top collected objects by object types
        self.gc_top_collected_objects: str = ""

    def handle(self, phase: str, info: dict[str, int]) -> None:
        """
        Handles a GC event (e.g. GC start or GC finish)
        """
        generation = info.get("generation")
        if generation is None:
            return
        if phase == "start":
            # Before GC started, record GC start time
            # and top collected objects
            self.start_time_ns = time.monotonic_ns()
            objects = gc.get_objects(generation)
            self.num_objects = len(objects)
            self.gc_top_collected_objects = _compute_top_gc_collected_objects(
                objects, self.config.top_objects
            )
        elif phase == "stop":
            # After GC finished, Record GC elapsed time and
            # optionally top collected objects
            elpased_ms = (time.monotonic_ns() - self.start_time_ns) / 1e6
            logger.info(
                "GC took %.3fms to complete. "
                "Collected %s objects (out of %d) in GC generation %d.%s",
                elpased_ms,
                str(info.get("collected", "?")),
                self.num_objects,
                generation,
                (
                    f" Top collected objects: \n{self.gc_top_collected_objects}"
                    if self.gc_top_collected_objects
                    else ""
                ),
            )


def freeze_gc_heap() -> None:
    """
    Freeze all objects tracked by the garbage collector. It should be invoked
    after server init / warmup, to reduce GC overhead from static objects
    during serving time.
    """
    # Ensure all static objects are pushed down to the oldest generation for
    # freeze
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    # Freeze all GC tracked objects
    gc.freeze()


def maybe_attach_gc_debug_callback() -> None:
    """
    Attached a callback for GC debug when VLLM_GC_DEBUG is enabled.
    """
    config = GCDebugConfig(envs.VLLM_GC_DEBUG)
    if config.enabled:
        debugger: GCDebugger = GCDebugger(config)

        def gc_callback(phase: str, info: dict[str, int]) -> None:
            debugger.handle(phase, info)

        gc.callbacks.append(gc_callback)


def _compute_detailed_type(o: Any) -> str:
    """
    Detailed object type.

    TODO(Jialin): Further enhance the detailed type with element types for
    easier debugging. We tried but occasionally it would run into signals
    which kills the engine.
    """
    size_str: str = ""
    # Object doesn't support len() - this can happen with type objects
    # or other objects that don't implement __len__ properly
    with suppress(Exception):
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
        for object_type, count in Counter(object_types).most_common(top)
    )
