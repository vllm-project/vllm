# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from typing import Literal

ToolStrictLevelName = Literal["auto", "function", "parameter"]


class ToolStrictLevel(enum.IntEnum):
    """Server-side floor for tool-call structural tags (``--tool-strict-level``).

    AUTO:      derive call constraints from ``tool_choice`` and argument-schema
               enforcement from each tool's explicit ``strict`` field.
    FUNCTION:  constrain the tool-call envelope for every request with tools.
    PARAMETER: additionally pin argument schemas, as if every tool were
               ``strict: true``.
    """

    AUTO = 0
    FUNCTION = 1
    PARAMETER = 2

    @classmethod
    def from_name(cls, name: str) -> "ToolStrictLevel":
        try:
            return cls[name.strip().upper()]
        except KeyError:
            expected = ", ".join(level.name.lower() for level in cls)
            raise ValueError(
                f"Unknown tool strict level {name!r}; expected one of {expected}."
            ) from None
