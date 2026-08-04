# SPDX-License-Identifier: Apache-2.0
"""Protocol and types for pluggable engine modules."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Protocol

# First Party
from lmcache.v1.multiprocess.protocol import RequestType

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext


class ThreadPoolType(Enum):
    """Declares which thread pool a handler should run in."""

    SYNC = auto()
    AFFINITY = auto()
    NORMAL = auto()


@dataclass
class HandlerSpec:
    """Specification for a single message queue handler.

    Args:
        request_type: The ZMQ request type this handler serves.
        handler: The callable that processes the request.
        pool: Which thread pool the handler runs in.
    """

    request_type: RequestType
    handler: Callable
    pool: ThreadPoolType


class EngineModule(Protocol):
    """Protocol for pluggable engine modules.

    Each module owns its internal state and exposes handlers
    that the compositor registers with the message queue server.
    """

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        ...

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves."""
        ...

    def report_status(self) -> dict:
        """Return module-specific status information."""
        ...

    def close(self) -> None:
        """Release resources owned by this module."""
        ...


class InstanceLivenessTarget(Protocol):
    """A module the periodic reaper drives, in either or both of two roles.

    * **Liveness owner** -- tracks per-worker registrations keyed by
      ``instance_id``, refreshed on PING and scanned for staleness
      (``touch_instance`` / ``reap_stale_instances`` /
      ``tracked_instance_count``). The transfer modules fill this role.
    * **State mirror** -- holds a second reference to a reaped instance's
      resources and releases it on demand (``drop_instance_state``).
      ``BlendV3Module`` fills this role for its per-instance CB state.

    Every method defaults to a no-op, so an implementer subclasses this
    protocol and overrides only the role it fills. The management module
    drives all targets from the PING handler and the reaper; no caller
    touches a module's private state directly.
    """

    def touch_instance(self, instance_id: int) -> None:
        """Refresh the worker's last-seen time and mark it ping-proven.

        A no-op if the instance is not tracked (already reaped or never
        registered), or for a target that owns no liveness state.

        Args:
            instance_id: The worker's opaque instance ID.
        """
        return

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Evict and clean up workers that have gone silent.

        An instance that has sent at least one PING is judged against
        ``reap_timeout_s``; one that has never pinged (warming up, or dead
        before its first request) is judged against ``registration_grace_s``.

        Args:
            reap_timeout_s: Silence budget for ping-proven instances.
            registration_grace_s: Silence budget for never-pinged instances;
                must be >= ``reap_timeout_s``.

        Returns:
            The instance IDs reaped during this scan; empty for a target
            that owns no liveness state.
        """
        return []

    def tracked_instance_count(self) -> int:
        """Return the number of currently tracked instances (0 if none)."""
        return 0

    def drop_instance_state(self, instance_id: int) -> None:
        """Release any state mirrored for a reaped instance.

        Called for every reaped ``instance_id``. A no-op unless the target
        keeps a second reference to that instance's resources (only mirrors
        such as ``BlendV3Module`` override this).

        Args:
            instance_id: The reaped worker's instance ID.
        """
        return
