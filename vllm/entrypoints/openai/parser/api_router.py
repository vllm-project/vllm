# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native endpoint that runs a tool parser over already-generated text.

SGLang exposes the same endpoint. It lets an agent loop or an RL rollout
harness reuse the server's tool parser on text that was produced elsewhere
(for example by the generate tier of a disaggregated deployment) without
re-implementing the model-specific parsing on the client.
"""

from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vllm.entrypoints.generate.base.protocol import ToolCall
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import validate_json_request
from vllm.logger import init_logger
from vllm.tool_parsers import ToolParserManager

logger = init_logger(__name__)

router = APIRouter()


class ParseFunctionCallRequest(BaseModel):
    text: str
    """The model output to parse."""

    tools: list[ChatCompletionToolsParam] = Field(default_factory=list)
    """The tools that were offered to the model, in OpenAI format."""

    tool_call_parser: str | None = None
    """Tool parser to use. Defaults to the server's --tool-call-parser."""


class ParseFunctionCallResponse(BaseModel):
    normal_text: str
    """The part of ``text`` that is not a tool call."""

    calls: list[ToolCall]
    """The parsed tool calls."""


def _bad_request(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=HTTPStatus.BAD_REQUEST.value,
        content=ErrorResponse(
            error=ErrorInfo(
                message=message,
                type="BadRequestError",
                code=HTTPStatus.BAD_REQUEST.value,
            )
        ).model_dump(),
    )


@router.post(
    "/parse_function_call",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
    },
)
async def parse_function_call(
    request: ParseFunctionCallRequest, raw_request: Request
) -> ParseFunctionCallResponse | JSONResponse:
    parser_name = request.tool_call_parser or getattr(
        raw_request.app.state.args, "tool_call_parser", None
    )
    if not parser_name:
        return _bad_request(
            "No tool parser selected: pass `tool_call_parser` in the request "
            "body or start the server with --tool-call-parser."
        )

    try:
        parser_cls = ToolParserManager.get_tool_parser(parser_name)
    except KeyError:
        return _bad_request(f"Unknown tool parser {parser_name!r}.")

    tokenizer = raw_request.app.state.serving_tokenization.renderer.get_tokenizer()
    parser = parser_cls(tokenizer, request.tools)
    chat_request = ChatCompletionRequest(messages=[], tools=request.tools or None)
    info = parser.extract_tool_calls(request.text, chat_request)

    return ParseFunctionCallResponse(
        normal_text=info.content or "",
        calls=info.tool_calls,
    )


def register_parser_api_router(app: FastAPI):
    app.include_router(router)
