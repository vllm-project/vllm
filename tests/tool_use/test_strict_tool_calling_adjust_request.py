# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for structured-output handling in the strict tool-calling
path of :meth:`ToolParser.adjust_request`.

When ``VLLM_ENFORCE_STRICT_TOOL_CALLING`` is set, ``adjust_request`` installs a
model structural tag for ``auto``/``required``/named tool choice. If the request
already carries a structured-output constraint (``json``/``regex``/...) or a
``response_format``, the original code set ``.structural_tag`` IN PLACE and
returned without nulling ``response_format`` -- leaving two mutually-exclusive
constraints on the same :class:`StructuredOutputsParams`. Because the in-place
assignment bypasses ``__post_init__``, the conflict is created silently and then
rejected on re-validation (e.g. in ``to_sampling_params``) with
``"You can only use one kind of structured outputs constraint but multiple are
specified"`` (HTTP 400).

``adjust_request`` must instead rebuild ``structured_outputs`` to hold only the
structural tag (preserving the whitespace / additional-properties knobs) and
null ``response_format`` -- mirroring what Step 2 of the same method already does
for the JSON-schema path.
"""

from __future__ import annotations

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.abstract_tool_parser import ToolParser

_STRICT_FLAG = "vllm.tool_parsers.abstract_tool_parser.VLLM_ENFORCE_STRICT_TOOL_CALLING"
_EXCLUSIVE = ("json", "regex", "choice", "grammar", "json_object", "structural_tag")


class _StubStructuralTag:
    """Stand-in for the structural tag returned by a model's
    ``get_structural_tag`` (only ``model_dump`` is used by ``adjust_request``)."""

    def model_dump(self) -> dict:
        return {"type": "structural_tag", "format": {"type": "any_text"}}


class _StructuralTagParser(ToolParser):
    """A parser that produces a structural tag, like deepseek_v4 / qwen3coder."""

    def get_structural_tag(self, request):  # type: ignore[override]
        return _StubStructuralTag()


def _parser() -> ToolParser:
    parser = _StructuralTagParser.__new__(_StructuralTagParser)
    parser.model_tokenizer = None
    return parser


def _tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def _request(*, tool_choice: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="strict-test",
        messages=[{"role": "user", "content": "What is the weather in Hanoi?"}],
        tools=[_tool()],
        tool_choice=tool_choice,
    )


def _active_constraints(so: StructuredOutputsParams) -> list[str]:
    return [f for f in _EXCLUSIVE if getattr(so, f) is not None]


@pytest.mark.parametrize("tool_choice", ["auto", "required"])
def test_strict_structural_tag_replaces_conflicting_constraint(
    monkeypatch, tool_choice
) -> None:
    """A pre-existing ``json`` constraint must be replaced by the structural
    tag (not left coexisting), and user whitespace knobs must be preserved."""
    monkeypatch.setattr(_STRICT_FLAG, True)

    request = _request(tool_choice=tool_choice)
    request.structured_outputs = StructuredOutputsParams(
        json={"type": "object", "properties": {"a": {"type": "string"}}},
        disable_any_whitespace=True,
    )

    result = _parser().adjust_request(request)
    so = result.structured_outputs

    assert so.structural_tag is not None
    assert so.json is None, "the conflicting json constraint must be dropped"
    assert _active_constraints(so) == ["structural_tag"], (
        "exactly one structured-output constraint must remain"
    )
    # The #40894 review's intent (preserve user-set sub-fields) is still honored.
    assert so.disable_any_whitespace is True


@pytest.mark.parametrize("tool_choice", ["auto", "required"])
def test_strict_structural_tag_nulls_response_format(monkeypatch, tool_choice) -> None:
    """``response_format`` must be nulled so a later ``to_sampling_params`` does
    not re-derive a JSON schema next to the structural tag (the 400)."""
    monkeypatch.setattr(_STRICT_FLAG, True)

    request = _request(tool_choice=tool_choice)
    request.response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    }

    result = _parser().adjust_request(request)
    so = result.structured_outputs

    assert result.response_format is None
    assert so.structural_tag is not None
    assert _active_constraints(so) == ["structural_tag"]


def test_strict_structural_tag_no_preexisting_constraint(monkeypatch) -> None:
    """Sanity: with no pre-existing constraint, the structural tag is installed
    and nothing else changes (the common path is unaffected)."""
    monkeypatch.setattr(_STRICT_FLAG, True)

    request = _request(tool_choice="auto")
    result = _parser().adjust_request(request)
    so = result.structured_outputs

    assert _active_constraints(so) == ["structural_tag"]
    assert result.response_format is None
