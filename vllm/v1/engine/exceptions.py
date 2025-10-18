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


@dataclass
class FaultInfo:
    type: str
    message: str
    engine_index: int
    exit_time: str = None
    additional_info: str | None = None

    def __post_init__(self):
        # If no exit time is specified, the current timestamp will be used by default.
        local_time = time.localtime(time.time())
        if self.exit_time is None:
            self.exit_time = time.strftime("%H:%M:%S", local_time)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "message": self.message,
            "engine_index": self.engine_index,
            "exit_time": self.exit_time,
            "additional_info": self.additional_info,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "FaultInfo":
        data = json.loads(json_str)
        return cls(
            type=data["type"],
            message=data["message"],
            engine_index=data["engine_index"],
            exit_time=data.get("exit_time"),
            additional_info=data.get("additional_info"),
        )
