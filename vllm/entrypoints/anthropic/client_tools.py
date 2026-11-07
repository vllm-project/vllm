# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Built-in schemas for Anthropic-defined client tools.

Anthropic-defined client tools (``bash_*``, ``text_editor_*``, ``memory_*``)
are declared by a versioned ``type`` and a fixed ``name``, without an
``input_schema``; the schema is built into Claude. The client still runs the
tool, so vLLM only needs to hand the model an equivalent function schema and
return ``tool_use`` blocks under the expected name.
"""

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClientToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


_BASH = ClientToolSpec(
    name="bash",
    description=(
        "Run commands in a persistent bash shell session. State such as the "
        "working directory and environment variables is kept between calls. "
        "Set restart to true to restart the session."
    ),
    input_schema=_object(
        {
            "command": {
                "type": "string",
                "description": "The bash command to run.",
            },
            "restart": {
                "type": "boolean",
                "description": "Set to true to restart the bash session.",
            },
        },
        [],
    ),
)

_PATH = {
    "type": "string",
    "description": "Path to the file or directory.",
}
_VIEW_RANGE = {
    "type": "array",
    "items": {"type": "integer"},
    "minItems": 2,
    "maxItems": 2,
    "description": (
        "Optional [start, end] line range for view, 1-indexed; an end of -1 "
        "reads to the end of the file."
    ),
}
_FILE_TEXT = {"type": "string", "description": "Content of the file to create."}
_OLD_STR = {"type": "string", "description": "Exact text to replace."}
_NEW_STR = {"type": "string", "description": "Replacement text."}
_INSERT_LINE = {
    "type": "integer",
    "description": "Line number after which to insert text; 0 inserts at the top.",
}
_INSERT_TEXT = {"type": "string", "description": "Text to insert."}


def _text_editor(name: str, legacy: bool) -> ClientToolSpec:
    commands = ["view", "create", "str_replace", "insert"]
    properties: dict[str, Any] = {
        "command": {"type": "string", "enum": commands},
        "path": _PATH,
        "view_range": _VIEW_RANGE,
        "file_text": _FILE_TEXT,
        "old_str": _OLD_STR,
        "new_str": _NEW_STR,
        "insert_line": _INSERT_LINE,
    }
    if legacy:
        # The 2024-10-22 and 2025-01-24 editors take the inserted text in
        # new_str and also support undo_edit.
        commands.append("undo_edit")
    else:
        properties["insert_text"] = _INSERT_TEXT
    return ClientToolSpec(
        name=name,
        description=(
            "View, create and edit text files. view shows a file with line "
            "numbers or lists a directory; create writes a new file; "
            "str_replace replaces exactly one occurrence of old_str; insert "
            "adds text after insert_line"
            + ("; undo_edit reverts the last edit to path." if legacy else ".")
        ),
        input_schema=_object(properties, ["command", "path"]),
    )


_MEMORY = ClientToolSpec(
    name="memory",
    description=(
        "Read and write files in a persistent /memories directory to keep "
        "information across conversations."
    ),
    input_schema=_object(
        {
            "command": {
                "type": "string",
                "enum": [
                    "view",
                    "create",
                    "str_replace",
                    "insert",
                    "delete",
                    "rename",
                ],
            },
            "path": _PATH,
            "view_range": _VIEW_RANGE,
            "file_text": _FILE_TEXT,
            "old_str": _OLD_STR,
            "new_str": _NEW_STR,
            "insert_line": _INSERT_LINE,
            "insert_text": _INSERT_TEXT,
            "old_path": {"type": "string", "description": "Path to rename."},
            "new_path": {"type": "string", "description": "New path."},
        },
        ["command"],
    ),
)

CLIENT_TOOL_SPECS: dict[str, ClientToolSpec] = {
    "bash_20241022": _BASH,
    "bash_20250124": _BASH,
    "text_editor_20241022": _text_editor("str_replace_editor", legacy=True),
    "text_editor_20250124": _text_editor("str_replace_editor", legacy=True),
    "text_editor_20250429": _text_editor("str_replace_based_edit_tool", legacy=False),
    "text_editor_20250728": _text_editor("str_replace_based_edit_tool", legacy=False),
    "memory_20250818": _MEMORY,
}


def expand_client_tool(data: dict[str, Any]) -> dict[str, Any] | None:
    """Return ``data`` with the built-in schema filled in, or ``None`` if
    ``data`` is not a registered Anthropic-defined client tool."""
    tool_type = data.get("type")
    spec = CLIENT_TOOL_SPECS.get(tool_type) if isinstance(tool_type, str) else None
    if spec is None:
        return None
    name = data.get("name")
    if name != spec.name:
        raise ValueError(
            f"tool type {tool_type!r} must be named {spec.name!r}, got {name!r}"
        )
    return {
        **data,
        "description": data.get("description") or spec.description,
        "input_schema": copy.deepcopy(spec.input_schema),
    }
