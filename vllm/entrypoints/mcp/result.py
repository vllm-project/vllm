# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.types import CallToolResult


def _model_visible_result(result: "CallToolResult") -> str:
    payload = result.model_dump(mode="json", by_alias=True, exclude_none=True)
    payload.pop("_meta", None)
    for block in payload["content"]:
        block.pop("_meta", None)
        if block["type"] in ("image", "audio"):
            block["data"] = f"<omitted {len(block['data'])} base64 characters>"
        elif block["type"] == "resource":
            block["resource"].pop("_meta", None)
            if blob := block["resource"].get("blob"):
                block["resource"]["blob"] = f"<omitted {len(blob)} base64 characters>"
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def normalize_tool_result(result: "CallToolResult") -> "CallToolResult":
    """Convert rich MCP results into model-readable text."""
    if (
        len(result.content) == 1
        and result.content[0].type == "text"
        and result.structured_content is None
        and not result.is_error
        and getattr(result, "result_type", "complete") == "complete"
    ):
        return result

    from mcp.types import TextContent

    content = _model_visible_result(result)
    return result.model_copy(update={"content": [TextContent(text=content)]})


class MCPClientSessionAdapter:
    """Preserve MCP result semantics for text-only model input paths."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    async def call_tool(self, *args: Any, **kwargs: Any) -> "CallToolResult":
        result = await self._session.call_tool(*args, **kwargs)
        return normalize_tool_result(result)
