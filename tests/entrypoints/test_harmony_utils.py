# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from openai.types.responses.tool import (
    CodeInterpreterContainerCodeInterpreterToolAuto,
    LocalShell,
    Mcp,
    Tool,
)

from vllm.entrypoints.harmony_utils import extract_tool_types, has_custom_tools


def test_has_custom_tools() -> None:
    assert not has_custom_tools(set())
    assert not has_custom_tools({"web_search_preview", "code_interpreter", "container"})
    assert has_custom_tools({"others"})
    assert has_custom_tools(
        {"web_search_preview", "code_interpreter", "container", "others"}
    )


def test_extract_tool_types(monkeypatch: pytest.MonkeyPatch) -> None:
    tools: list[Tool] = []
    assert extract_tool_types(tools) == set()

    tools.append(LocalShell(type="local_shell"))
    assert extract_tool_types(tools) == {"local_shell"}

    tools.append(CodeInterpreterContainerCodeInterpreterToolAuto(type="auto"))
    assert extract_tool_types(tools) == {"local_shell", "auto"}

    tools.extend(
        [
            Mcp(type="mcp", server_label="random", server_url=""),
            Mcp(type="mcp", server_label="container", server_url=""),
            Mcp(type="mcp", server_label="code_interpreter", server_url=""),
            Mcp(type="mcp", server_label="web_search_preview", server_url=""),
        ]
    )
    # When envs.GPT_OSS_SYSTEM_TOOL_MCP_LABELS is not set,
    # mcp tool types are all ignored.
    assert extract_tool_types(tools) == {"local_shell", "auto"}

    # container is allowed, it would be extracted
    monkeypatch.setenv("GPT_OSS_SYSTEM_TOOL_MCP_LABELS", "container")
    assert extract_tool_types(tools) == {"local_shell", "auto", "container"}

    # code_interpreter and web_search_preview are allowed,
    # they would be extracted
    monkeypatch.setenv(
        "GPT_OSS_SYSTEM_TOOL_MCP_LABELS", "code_interpreter,web_search_preview"
    )
    assert extract_tool_types(tools) == {
        "local_shell",
        "auto",
        "code_interpreter",
        "web_search_preview",
    }
