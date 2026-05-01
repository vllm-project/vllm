# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DP routing policy — derives an "eligible engines" set from the fault bus.

Used by the load-balancing client (`DPLBAsyncMPClient`) to steer requests
away from engines that are PAUSED, UNHEALTHY, RECOVERING or DEAD. Today's
existing PRs only avoid DEAD engines (#38862's `_dead_engine_identities`);
this generalizes that to all non-HEALTHY states by subscribing to the bus.
"""

from __future__ import annotations

import threading
from contextlib import suppress
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.fault_tolerance.types import FaultInfo, FaultStatus

if TYPE_CHECKING:
    from vllm.v1.fault_tolerance.fault_state_bus import FaultStateBus

logger = init_logger(__name__)


_INELIGIBLE_STATES = frozenset(
    {
        FaultStatus.DEAD,
        FaultStatus.UNHEALTHY,
        FaultStatus.PAUSED,
        FaultStatus.RECOVERING,
    }
)


class DPRoutingPolicy:
    """Subscribe to the bus, expose ``eligible_engines()`` to the LB client.

    The LB client should call ``eligible_engines()`` when picking a target.
    The policy auto-updates whenever the bus publishes a status change.
    """

    def __init__(self, bus: FaultStateBus, total_engines: int):
        self._lock = threading.Lock()
        self._eligible: set[int] = set(range(total_engines))
        self._bus = bus
        bus.subscribe(self._on_event)

    def eligible_engines(self) -> frozenset[int]:
        with self._lock:
            return frozenset(self._eligible)

    def _on_event(self, info: FaultInfo) -> None:
        with self._lock:
            if info.status in _INELIGIBLE_STATES:
                if info.engine_index in self._eligible:
                    logger.info(
                        "DP rank %d → ineligible (status=%s)",
                        info.engine_index,
                        info.status.name,
                    )
                    self._eligible.discard(info.engine_index)
            elif (
                info.status is FaultStatus.HEALTHY
                and info.engine_index not in self._eligible
            ):
                logger.info(
                    "DP rank %d → eligible again (status=HEALTHY)",
                    info.engine_index,
                )
                self._eligible.add(info.engine_index)

    def close(self) -> None:
        with suppress(Exception):
            self._bus.unsubscribe(self._on_event)
