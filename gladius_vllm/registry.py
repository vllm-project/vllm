"""Process-wide weakref registry bridging GladiusScheduler <-> GladiusStatLogger.

Exactly one Scheduler exists per engine-core process, so a simple
engine_id-keyed weakref map is sufficient -- no cross-process concerns within
a single process. See gladius_vllm.stat_logger for the (optional, secondary)
consumer and the cross-process caveat in the design plan.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gladius_vllm.scheduler import GladiusScheduler

_ACTIVE: dict[str, "weakref.ref[GladiusScheduler]"] = {}


def register_scheduler(scheduler: "GladiusScheduler") -> None:
    _ACTIVE[scheduler.engine_id] = weakref.ref(scheduler)


def get_scheduler(engine_id: str) -> "GladiusScheduler | None":
    ref = _ACTIVE.get(engine_id)
    return ref() if ref is not None else None
