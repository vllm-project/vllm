# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses


class _RejectFlatNameFormatting(str):
    def __format__(self, format_spec: str) -> str:
        raise AssertionError("named tool validation flattened a namespace name")


def _namespace_tool(namespace: str, *, count: int = 1) -> dict:
    return {
        "type": "namespace",
        "name": namespace,
        "description": "namespace tool budget test",
        "tools": [
            {
                "type": "function",
                "name": f"lookup_{index}",
                "description": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            }
            for index in range(count)
        ],
    }


@pytest.mark.parametrize("tool_name", ["lookup_0", "ns__lookup_0"])
def test_named_namespace_tool_choice_does_not_flatten_every_tool_name(
    tool_name: str,
) -> None:
    request = ResponsesRequest(
        input="hello",
        tools=[_namespace_tool(_RejectFlatNameFormatting("ns"))],
        tool_choice={"type": "function", "name": tool_name},
    )

    assert getattr(request.tool_choice, "name", None) == tool_name


@pytest.fixture
def serving_responses() -> OpenAIServingResponses:
    engine_client = MagicMock()
    model_config = MagicMock()
    model_config.max_model_len = 16
    model_config.hf_config.model_type = "test"
    model_config.get_diff_sampling_param.return_value = {}
    engine_client.model_config = model_config
    engine_client.input_processor = MagicMock()
    engine_client.renderer = MagicMock()
    engine_client.renderer.get_tokenizer.return_value.max_chars_per_token = 1
    engine_client.errored = False

    instance = OpenAIServingResponses(
        engine_client=engine_client,
        models=MagicMock(),
        online_renderer=MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    instance._check_model = AsyncMock(return_value=None)
    return instance


def test_in_budget_namespace_tool_names_remain_allowed(
    serving_responses: OpenAIServingResponses,
) -> None:
    request = ResponsesRequest(
        input="hello",
        tools=[_namespace_tool("ns")],
    )

    assert serving_responses._validate_create_responses_input(request) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("truncation", ["disabled", "auto"])
async def test_oversized_namespace_tool_names_are_rejected_before_request_build(
    serving_responses: OpenAIServingResponses,
    truncation: str,
) -> None:
    request = ResponsesRequest(
        input="hello",
        tools=[_namespace_tool("namespace", count=2)],
        truncation=truncation,
    )
    serving_responses._make_request = AsyncMock(
        side_effect=AssertionError("request build reached namespace flattening")
    )

    result = await serving_responses._create_responses(request)

    assert isinstance(result, ErrorResponse)
    assert result.error.code == 400
    assert result.error.param == "tools"
    assert "namespace tool" in result.error.message.lower()
    serving_responses._make_request.assert_not_awaited()
