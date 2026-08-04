# SPDX-License-Identifier: Apache-2.0
"""Base class for heartbeat commands"""

# Standard
from typing import TYPE_CHECKING, Any, Dict, Optional

# Third Party
import msgspec

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker


class HeartbeatCommand(msgspec.Struct, tag_field="command_type"):
    """Base class for heartbeat commands (polymorphic)

    This abstraction allows controller to send various commands to workers
    through the heartbeat mechanism. Each command type is a subclass with
    its own specific fields.

    Uses msgspec's tagged union for serialization/deserialization.
    """

    reason: Optional[str] = None  # Why this command is issued
    args: Optional[Dict[str, Any]] = None  # Extension arguments for command

    def describe(self) -> str:
        return f"{self.__class__.__name__}(reason={self.reason}, args={self.args})"

    def execute(self, worker: "LMCacheWorker") -> None:
        """Execute this command on the worker. Override in subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.execute() not implemented"
        )
