# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any, Literal

import pytest
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

import vllm.envs as envs
from tests.parser.engine.replay_harness import MockTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.parser_manager import ParserManager
from vllm.sampling_params import StructuredOutputsParams

_EOM = "<|eom|>"
_EOT = "<|eot|>"
_START = "<|start|>"
_MESSAGE = "<|message|>"
_FUNCTION_CALLS_OPEN = "<atem:function_calls>"
_FUNCTION_CALLS_CLOSE = "</atem:function_calls>"
_VOCAB = {_EOM: 200007, _EOT: 200008, _START: 200022, _MESSAGE: 200023}


def _tool_call(
    route: str,
    parameters: str = "",
    *,
    invoke_name: str | None = None,
    end_token: str = _EOT,
) -> str:
    invoke_name = route if invoke_name is None else invoke_name
    return (
        f"to={route}{_MESSAGE}{_FUNCTION_CALLS_OPEN}\n"
        f'<atem:invoke name="{invoke_name}">\n'
        f"{parameters}</atem:invoke>\n"
        f"{_FUNCTION_CALLS_CLOSE}{end_token}"
    )


def _answer(content: str) -> str:
    return f"to=user{_MESSAGE}{content}{_EOT}"


class TestAdjustRequest:
    """Check Muse request adjustment through grammar admission."""

    REQUEST_TEXT = "Hello"
    TOOL_1_NAME = "get_user_location"
    TOOL_2_NAME = "get_weather"
    TOOLS = [
        {
            "name": TOOL_1_NAME,
            "strict": True,
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": TOOL_2_NAME,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    ]
    OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    REASONING = (
        f"to=self{_MESSAGE}Thinking through the request.{_EOM}{_START}assistant "
    )
    COMMENTARY = f"to=commentary{_MESSAGE}Searching for an answer.{_EOM}"

    # These ATEM separators match the HF Muse Spark chat template. The second
    # parameter has a multiline value and is outside the tool's schema.
    TOOL_CALL_1 = _tool_call(TOOL_1_NAME)
    TOOL_CALL_1_CONTINUATION = _tool_call(TOOL_1_NAME, end_token=_EOM)
    TOOL_CALL_2 = _tool_call(
        TOOL_2_NAME,
        parameters=(
            '<atem:parameter name="city">Tokyo\nJapan</atem:parameter>\n'
            '<atem:parameter name="days">2</atem:parameter>\n'
        ),
    )
    REASONED_MULTIPLE_TOOL_CALLS = (
        REASONING
        + COMMENTARY
        + f"{_START}assistant "
        + TOOL_CALL_1_CONTINUATION
        + f"{_START}assistant "
        + TOOL_CALL_2
    )
    TOOL_CALL_MISMATCHED_INVOKE = _tool_call(TOOL_2_NAME, invoke_name=TOOL_1_NAME)
    TOOL_CALL_UNKNOWN = _tool_call("unknown_tool")

    ANSWER_TEXT = _answer("The answer is 42.")
    FINAL_JSON_SCHEMA = _answer('{"answer": "Tokyo"}')
    FINAL_JSON_OTHER_OBJECT = _answer('{"city": "Tokyo"}')
    REASONED_FINAL_JSON_SCHEMA = REASONING + FINAL_JSON_SCHEMA
    FINAL_REGEX = _answer("regex")

    ANY_ANSWER_SAMPLES = (
        "ANSWER_TEXT",
        "FINAL_JSON_SCHEMA",
        "FINAL_JSON_OTHER_OBJECT",
        "REASONED_FINAL_JSON_SCHEMA",
        "FINAL_REGEX",
    )
    ADMISSION_SAMPLES = (
        "COMMENTARY",
        "TOOL_CALL_1",
        "TOOL_CALL_2",
        "TOOL_CALL_MISMATCHED_INVOKE",
        "TOOL_CALL_UNKNOWN",
        "REASONED_MULTIPLE_TOOL_CALLS",
        *ANY_ANSWER_SAMPLES,
    )

    @classmethod
    def _build_request(
        cls,
        request_kind: Literal["chat", "responses"],
        tool_choice: str = "none",
        response_format_type: str | None = None,
        structured_outputs: StructuredOutputsParams | None = None,
    ) -> ChatCompletionRequest | ResponsesRequest:
        data: dict[str, Any] = {"model": "muse"}
        if request_kind == "chat":
            data["messages"] = [{"role": "user", "content": cls.REQUEST_TEXT}]
            data["tools"] = [
                {"type": "function", "function": tool} for tool in cls.TOOLS
            ]
            data["tool_choice"] = (
                {
                    "type": "function",
                    "function": {"name": cls.TOOL_2_NAME},
                }
                if tool_choice == "named"
                else tool_choice
            )
        else:
            data["input"] = cls.REQUEST_TEXT
            data["tools"] = [{"type": "function", **tool} for tool in cls.TOOLS]
            data["tool_choice"] = (
                {"type": "function", "name": cls.TOOL_2_NAME}
                if tool_choice == "named"
                else tool_choice
            )

        if response_format_type == "json_schema":
            schema_format = {
                "name": "answer_format",
                "schema": cls.OUTPUT_SCHEMA,
                "strict": True,
            }
            if request_kind == "chat":
                data["response_format"] = {
                    "type": "json_schema",
                    "json_schema": schema_format,
                }
            else:
                data["text"] = {"format": {"type": "json_schema", **schema_format}}
        elif response_format_type == "json_object":
            if request_kind == "chat":
                data["response_format"] = {"type": "json_object"}
            else:
                data["text"] = {"format": {"type": "json_object"}}
        if structured_outputs is not None:
            data["structured_outputs"] = structured_outputs

        if request_kind == "chat":
            return ChatCompletionRequest.model_validate(data)
        return ResponsesRequest.model_validate(data)

    @staticmethod
    def _assert_format_cleared(
        adjusted_request: ChatCompletionRequest | ResponsesRequest,
    ) -> None:
        if isinstance(adjusted_request, ResponsesRequest):
            assert adjusted_request.text is None or adjusted_request.text.format is None
        else:
            assert adjusted_request.response_format is None

        structured_outputs = adjusted_request.structured_outputs
        assert structured_outputs is not None
        assert structured_outputs.structural_tag is not None
        assert structured_outputs.all_non_structural_tag_constraints_none()

    @classmethod
    def _assert_structured_outputs_admission(
        cls,
        adjusted_request: ChatCompletionRequest | ResponsesRequest,
        expected_admission: Sequence[str],
    ) -> None:
        structured_outputs = adjusted_request.structured_outputs
        assert structured_outputs is not None
        assert structured_outputs.structural_tag is not None

        grammar = Grammar.from_structural_tag(structured_outputs.structural_tag)
        expected_admission_set = set(expected_admission)
        for sample_name in cls.ADMISSION_SAMPLES:
            should_admit = sample_name in expected_admission_set
            sample = " " + getattr(cls, sample_name)
            admitted = _is_grammar_accept_string(grammar, sample)
            assert admitted is should_admit, (
                f"Expected Muse structural-tag admission for {sample!r} "
                f"({sample_name}) to be {should_admit}, got {admitted}."
            )

    @pytest.mark.parametrize("request_kind", ["chat", "responses"])
    @pytest.mark.parametrize(
        ("request_kwargs", "expected_admission"),
        [
            pytest.param(
                {"tool_choice": "auto"},
                [
                    "TOOL_CALL_1",
                    "TOOL_CALL_2",
                    "REASONED_MULTIPLE_TOOL_CALLS",
                    *ANY_ANSWER_SAMPLES,
                ],
                id="tool_auto",
            ),
            pytest.param(
                {"tool_choice": "required"},
                [
                    "TOOL_CALL_1",
                    "TOOL_CALL_2",
                    "REASONED_MULTIPLE_TOOL_CALLS",
                ],
                id="tool_required",
            ),
            pytest.param(
                {"tool_choice": "named"},
                ["TOOL_CALL_2"],
                id="tool_named",
            ),
            pytest.param(
                {"tool_choice": "auto", "response_format_type": "json_schema"},
                [
                    "TOOL_CALL_1",
                    "TOOL_CALL_2",
                    "REASONED_MULTIPLE_TOOL_CALLS",
                    "FINAL_JSON_SCHEMA",
                    "REASONED_FINAL_JSON_SCHEMA",
                ],
                id="tool_auto_response_format_json_schema",
            ),
            pytest.param(
                {"response_format_type": "json_object"},
                [
                    "FINAL_JSON_SCHEMA",
                    "FINAL_JSON_OTHER_OBJECT",
                    "REASONED_FINAL_JSON_SCHEMA",
                ],
                id="response_format_json_object",
            ),
            pytest.param(
                {"structured_outputs": StructuredOutputsParams(regex=r"regex")},
                ["FINAL_REGEX"],
                id="structured_outputs_regex",
            ),
        ],
    )
    def test_adjust_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
        request_kind: Literal["chat", "responses"],
        request_kwargs: dict[str, Any],
        expected_admission: Sequence[str],
    ):
        """The adjusted grammar admits only the requested Muse outputs."""
        monkeypatch.setattr(envs, "VLLM_ENFORCE_STRICT_TOOL_CALLING", True)
        request = self._build_request(request_kind, **request_kwargs)
        parser_cls = ParserManager.get_parser(
            tool_parser_name="muse",
            reasoning_parser_name="muse",
            enable_auto_tools=True,
        )
        assert parser_cls is not None
        parser_tools = [
            ChatCompletionToolsParam.model_validate(
                {"type": "function", "function": tool}
            )
            for tool in self.TOOLS
        ]
        parser = parser_cls(MockTokenizer(dict(_VOCAB), []), parser_tools)

        adjusted_request = parser.adjust_request(request)
        self._assert_format_cleared(adjusted_request)
        self._assert_structured_outputs_admission(
            adjusted_request,
            expected_admission,
        )
