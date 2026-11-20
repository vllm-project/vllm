# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python implementation of the `vllm.Render` gRPC service.

Serves the same `rust/proto/render.proto` contract as `vllm-rs render`, backed
by the Python `ServingRender` that also handles the HTTP `/render` routes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import grpc
import pybase64
from google.protobuf import json_format, struct_pb2
from pydantic import BaseModel, ValidationError

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.generate.base.protocol import MMPayload
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.scale_out.render.grpc import (
    inference_pb2,
    render_pb2,
    render_pb2_grpc,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    MultiModalFeatures,
)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.logger import init_logger
from vllm.sampling_params import StructuredOutputsParams

if TYPE_CHECKING:
    from vllm.entrypoints.scale_out.render.serving import ServingRender

logger = init_logger(__name__)

_HTTP_TO_GRPC: dict[int, grpc.StatusCode] = {
    HTTPStatus.BAD_REQUEST: grpc.StatusCode.INVALID_ARGUMENT,
    HTTPStatus.NOT_FOUND: grpc.StatusCode.NOT_FOUND,
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE: grpc.StatusCode.RESOURCE_EXHAUSTED,
    HTTPStatus.UNPROCESSABLE_ENTITY: grpc.StatusCode.INVALID_ARGUMENT,
    HTTPStatus.TOO_MANY_REQUESTS: grpc.StatusCode.RESOURCE_EXHAUSTED,
    HTTPStatus.NOT_IMPLEMENTED: grpc.StatusCode.UNIMPLEMENTED,
    HTTPStatus.SERVICE_UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
}

_MODALITIES = {
    "image": inference_pb2.MODALITY_IMAGE,
    "video": inference_pb2.MODALITY_VIDEO,
    "audio": inference_pb2.MODALITY_AUDIO,
}

# Inline kwargs `Inference.Generate` accepts in one request; matches
# `MAX_INLINE_MM_BYTES` in `rust/src/server/src/grpc/media.rs`.
MAX_INLINE_KWARGS_BYTES = 16 * 1024 * 1024


class RenderRpcError(Exception):
    def __init__(self, code: grpc.StatusCode, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _mm_payload(value: int) -> MMPayload:
    if value in (render_pb2.MM_PAYLOAD_UNSPECIFIED, render_pb2.MM_PAYLOAD_TENSORS):
        return "tensors"
    if value == render_pb2.MM_PAYLOAD_NONE:
        return "none"
    raise RenderRpcError(
        grpc.StatusCode.INVALID_ARGUMENT, f"unknown mm_payload value {value}"
    )


def _structured_output(
    params: StructuredOutputsParams | None,
) -> inference_pb2.StructuredOutput | None:
    if params is None:
        return None
    if params.json is not None:
        schema = params.json
        return inference_pb2.StructuredOutput(
            json=schema if isinstance(schema, str) else json.dumps(schema)
        )
    if params.regex is not None:
        return inference_pb2.StructuredOutput(regex=params.regex)
    if params.choice is not None:
        return inference_pb2.StructuredOutput(
            choice=inference_pb2.StructuredOutput.StringChoices(choices=params.choice)
        )
    if params.grammar is not None:
        return inference_pb2.StructuredOutput(grammar=params.grammar)
    if params.json_object:
        return inference_pb2.StructuredOutput(json_object=True)
    if params.structural_tag is not None:
        return inference_pb2.StructuredOutput(structural_tag=params.structural_tag)
    return None


def _to_struct(value: dict[str, Any]) -> struct_pb2.Struct:
    struct = struct_pb2.Struct()
    json_format.ParseDict(value, struct)
    return struct


def _engine_reasoning_gate(
    rendered: GenerateRequest,
) -> inference_pb2.EngineReasoningGate | None:
    chat_template_kwargs = (
        rendered.reasoning_parser_kwargs.chat_template_kwargs
        if rendered.reasoning_parser_kwargs is not None
        else {}
    )
    if rendered.reasoning_ended is None and not chat_template_kwargs:
        return None
    gate = inference_pb2.EngineReasoningGate()
    if rendered.reasoning_ended is not None:
        gate.reasoning_ended = rendered.reasoning_ended
    if chat_template_kwargs:
        gate.chat_template_kwargs.CopyFrom(_to_struct(chat_template_kwargs))
    return gate


def _media(features: MultiModalFeatures | None) -> list[inference_pb2.MediaItem]:
    if features is None:
        return []

    items: list[tuple[int, inference_pb2.MediaItem]] = []
    kwargs_bytes = 0
    for modality, hashes in features.mm_hashes.items():
        proto_modality = _MODALITIES.get(modality)
        if proto_modality is None:
            raise RenderRpcError(
                grpc.StatusCode.UNIMPLEMENTED,
                f"modality {modality!r} has no vllm.Modality value",
            )
        encoded = (features.kwargs_data or {}).get(modality, [None] * len(hashes))
        placeholders = features.mm_placeholders[modality]
        for mm_hash, placeholder, kwargs_b64 in zip(
            hashes, placeholders, encoded, strict=True
        ):
            media_features = inference_pb2.PreprocessedMediaFeatures(
                # The processor cache key, as `vllm-rs` sets both fields.
                identifier=mm_hash,
                mm_hash=mm_hash,
                offset=placeholder.offset,
                length=placeholder.length,
                is_embed=placeholder.is_embed or [],
            )
            if kwargs_b64 is not None:
                kwargs = pybase64.b64decode(kwargs_b64)
                kwargs_bytes += len(kwargs)
                media_features.kwargs = kwargs
            items.append(
                (
                    placeholder.offset,
                    inference_pb2.MediaItem(
                        modality=proto_modality, features=media_features
                    ),
                )
            )

    if kwargs_bytes > MAX_INLINE_KWARGS_BYTES:
        raise RenderRpcError(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            f"rendered media kwargs total {kwargs_bytes} bytes, more than the "
            f"{MAX_INLINE_KWARGS_BYTES} bytes Inference.Generate accepts; "
            "render with mm_payload NONE",
        )
    items.sort(key=lambda item: item[0])
    return [item for _, item in items]


def to_render_response(rendered: GenerateRequest) -> render_pb2.RenderResponse:
    """Convert the HTTP render result into the `vllm.Render` response."""
    response = render_pb2.RenderResponse(
        prompt_token_ids=rendered.token_ids,
        media=_media(rendered.features),
    )
    structured_output = _structured_output(rendered.sampling_params.structured_outputs)
    if structured_output is not None:
        response.structured_output.CopyFrom(structured_output)
    gate = _engine_reasoning_gate(rendered)
    if gate is not None:
        response.engine_reasoning_gate.CopyFrom(gate)
    return response


def _parse_body(model: type[BaseModel], body: bytes, api_name: str) -> Any:
    try:
        return model.model_validate_json(body)
    except ValidationError as e:
        raise RenderRpcError(
            grpc.StatusCode.INVALID_ARGUMENT,
            f"failed to parse {api_name} request: {e}",
        ) from e


def _check_result(result: Any) -> Any:
    if isinstance(result, ErrorResponse):
        raise RenderRpcError(
            _HTTP_TO_GRPC.get(result.error.code, grpc.StatusCode.INTERNAL),
            result.error.message,
        )
    return result


class RenderServicer(render_pb2_grpc.RenderServicer):
    """`vllm.Render` backed by the Python `ServingRender`.

    Only the raw-JSON `Render` RPC is implemented. `RenderChat` takes the typed
    `ChatRequest`, which has no Python model, and answers `UNIMPLEMENTED`.
    """

    def __init__(self, serving_render: ServingRender):
        self.serving_render = serving_render
        self._handlers: dict[int, Callable[[bytes, MMPayload], Any]] = {
            render_pb2.API_CHAT: self._render_chat,
            render_pb2.API_COMPLETIONS: self._render_completion,
            render_pb2.API_ANTHROPIC_MESSAGES: self._render_messages,
            render_pb2.API_RESPONSES: self._render_responses,
        }

    async def Render(self, request, context):
        try:
            handler = self._handlers.get(request.api)
            if handler is None:
                if request.api == render_pb2.API_UNSPECIFIED:
                    raise RenderRpcError(
                        grpc.StatusCode.INVALID_ARGUMENT, "api is required"
                    )
                raise RenderRpcError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"unknown api value {request.api}",
                )
            rendered = await handler(request.body, _mm_payload(request.mm_payload))
            return to_render_response(rendered)
        except RenderRpcError as e:
            logger.warning("gRPC Render failed: %s", e.message)
            await context.abort(e.code, e.message)
        except (ValueError, TypeError) as e:
            logger.warning("gRPC Render failed: %s", e)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except NotImplementedError as e:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, str(e))

    async def _render_chat(self, body: bytes, mm_payload: MMPayload) -> GenerateRequest:
        request = _parse_body(ChatCompletionRequest, body, "chat completion")
        request.mm_payload = mm_payload
        return _check_result(await self.serving_render.render_chat_request(request))

    async def _render_messages(
        self, body: bytes, mm_payload: MMPayload
    ) -> GenerateRequest:
        request = _parse_body(AnthropicMessagesRequest, body, "messages")
        request.mm_payload = mm_payload
        return _check_result(await self.serving_render.render_messages_request(request))

    async def _render_completion(
        self, body: bytes, mm_payload: MMPayload
    ) -> GenerateRequest:
        request = _parse_body(CompletionRequest, body, "completion")
        request.mm_payload = mm_payload
        rendered = _check_result(
            await self.serving_render.render_completion_request(request)
        )
        if len(rendered) != 1:
            raise RenderRpcError(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Render takes one completion prompt, got {len(rendered)}",
            )
        return rendered[0]

    async def _render_responses(
        self, body: bytes, mm_payload: MMPayload
    ) -> GenerateRequest:
        request = _parse_body(ResponsesRequest, body, "responses")
        request.mm_payload = mm_payload
        return _check_result(
            await self.serving_render.render_responses_request(request)
        )
