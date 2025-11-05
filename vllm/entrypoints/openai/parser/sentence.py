# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
)


class Channel(str, Enum):
    THINK = "think"
    FINAL = "final"  # this is the user facing


class Role(str, Enum):
    """The role of a message author (mirrors ``chat::Role``)."""

    USER = "user"
    ASSISTANT = "assistant"  # for minimaxM2, this is "ai"
    SYSTEM = "system"
    DEVELOPER = "developer"
    TOOL = "tool"

    @classmethod
    def _missing_(cls, value: object) -> "Role":  # type: ignore[override]
        raise ValueError(f"Unknown role: {value!r}")


class Author(BaseModel):
    role: Role
    name: str | None = None

    @classmethod
    def new(cls, role: Role, name: str) -> "Author":
        return cls(role=role, name=name)


class Content(BaseModel):  # noqa: D101 – simple wrapper
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


class TextContent(Content):
    text: str
    channel: str | None = None  # this could be reasoning, tool call, final

    def to_dict(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


class Sentence(BaseModel):
    author: Author
    content: list[Content]

# class
