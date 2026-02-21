# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import time
from dataclasses import dataclass


class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""

    pass


class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        ENGINE_DEAD_MESSAGE = "EngineCore encountered an issue. See stack trace (above) for the root cause."  # noqa: E501

        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context


class EngineLoopPausedError(Exception):
    """
    Raised when the EngineCore loop is temporarily paused on purpose,
    e.g., to handle fault-tolerance.
    """

    pass


@dataclass
class FaultInfo:
    type: str
    message: str
    engine_id: str
    timestamp: str | None = None
    additional_info: dict | None = None

    def __post_init__(self):
        # If no exit time is specified, the current timestamp will be used by default.

        local_time = time.localtime(time.time())
        if self.timestamp is None:
            self.timestamp = time.strftime("%H:%M:%S", local_time)

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        engine_id: str | int,
        additional_info: dict | None = None,
    ) -> "FaultInfo":
        """Create FaultInfo from an exception."""
        return cls(
            type=type(exception).__name__,
            message=str(exception),
            engine_id=str(engine_id),
            additional_info=additional_info or {},
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type,
            "message": self.message,
            "timestamp": self.timestamp,
            "engine_id": self.engine_id,
            "additional_info": self.additional_info,
        }

    def serialize(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "FaultInfo":
        """Create FaultInfo from JSON string."""
        data = json.loads(json_str)
        return cls(
            type=data["type"],
            message=data["message"],
            timestamp=data["timestamp"],
            engine_id=data["engine_id"],
            additional_info=data["additional_info"],
        )
