# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from vllm.logger import init_logger

if TYPE_CHECKING:
    from fastapi import WebSocket

    from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
    from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

logger = init_logger(__name__)


def uuid7() -> str:
    """Generate a UUID v7 string (time-ordered, RFC 9562).

    Returns a 32-character lowercase hex string.
    Layout: 48-bit unix-ms timestamp | 4-bit version (7) | 12-bit random
            | 2-bit variant (10) | 62-bit random
    """
    timestamp_ms = int(time.time() * 1000)
    rand = os.urandom(10)
    hi = (timestamp_ms & 0xFFFF_FFFF_FFFF) << 80
    hi |= 0x7 << 76  # version 7
    hi |= (int.from_bytes(rand[:2], "big") & 0x0FFF) << 64
    lo = 0x80 << 56  # variant 10
    lo |= int.from_bytes(rand[2:], "big") & 0x3FFF_FFFF_FFFF_FFFF
    return f"{(hi | lo):032x}"


@dataclass
class ConnectionContext:
    """Per-connection state for WebSocket Responses mode."""

    connection_id: str
    last_response_id: str | None = None
    last_response: "ResponsesResponse | None" = None
    inflight: bool = False
    created_at: float = field(default_factory=time.monotonic)

    LIFETIME_SECONDS: ClassVar[int] = 3600  # 60 minutes
    WARNING_SECONDS: ClassVar[int] = 3300  # 55 minutes

    def is_expired(self) -> bool:
        return time.monotonic() - self.created_at >= self.LIFETIME_SECONDS

    def should_warn(self) -> bool:
        return time.monotonic() - self.created_at >= self.WARNING_SECONDS

    def evict_cache(self) -> None:
        self.last_response_id = None
        self.last_response = None


class WebSocketResponsesConnection:
    """Manages WebSocket lifecycle for Responses API WebSocket mode."""

    def __init__(self, websocket: "WebSocket", serving: "OpenAIServingResponses"):
        self.websocket = websocket
        self.serving = serving
        self.ctx = ConnectionContext(connection_id=f"ws-{uuid7()}")
        self._is_connected = False
        self._deadline_task: asyncio.Task | None = None

    async def send_event(self, event: BaseModel) -> None:
        """Send a Pydantic event as JSON text over WebSocket."""
        await self.websocket.send_text(event.model_dump_json())

    async def send_error(self, code: str, message: str, status: int = 400) -> None:
        """Send an error event in the OpenAI WebSocket error format."""
        payload = json.dumps(
            {
                "type": "error",
                "status": status,
                "error": {
                    "code": code,
                    "message": message,
                },
            }
        )
        await self.websocket.send_text(payload)

    async def handle_event(self, event: dict) -> None:
        """Parse and dispatch incoming WebSocket events."""
        event_type = event.get("type")

        if event_type != "response.create":
            await self.send_error(
                "unknown_event_type",
                f"Unknown event type: {event_type}",
            )
            return

        # Reject concurrent requests
        if self.ctx.inflight:
            await self.send_error(
                "concurrent_request",
                "A response is already being generated on this connection",
                409,
            )
            return

        # Resolve previous_response_id from connection-local cache
        prev_id = event.get("previous_response_id")
        if prev_id is not None and prev_id != self.ctx.last_response_id:
            await self.send_error(
                "previous_response_not_found",
                f"Response '{prev_id}' not found in connection cache",
                404,
            )
            return

        self.ctx.inflight = True
        try:
            await self._process_response_create(event)
        except Exception as e:
            logger.exception("Error processing response.create: %s", e)
            await self.send_error("processing_error", str(e), 500)
            self.ctx.evict_cache()
        finally:
            self.ctx.inflight = False

    async def _process_response_create(self, event: dict) -> None:
        """Process a response.create event."""
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.openai.responses.protocol import (
            ResponseCompletedEvent,
            ResponseCreatedEvent,
            ResponsesRequest,
            ResponsesResponse,
        )

        # Build request payload
        payload = {k: v for k, v in event.items() if k != "type"}
        generate = payload.pop("generate", True)
        payload["stream"] = True
        payload.pop("background", None)

        # Strip previous_response_id from the payload before constructing
        # ResponsesRequest. The WebSocket handler already validated it
        # against the connection-local cache in handle_event(). Passing
        # it through would cause the serving layer to look it up in the
        # shared response_store and fail.
        payload.pop("previous_response_id", None)

        request = ResponsesRequest(**payload)

        if not generate:
            # Warmup mode: emit created + completed, no inference.
            # Populate all required ResponsesResponse fields from request.
            response = ResponsesResponse(
                id=request.request_id,
                created_at=int(time.time()),
                model=request.model or "",
                output=[],
                status="completed",
                parallel_tool_calls=request.parallel_tool_calls or True,
                temperature=request.temperature or 1.0,
                tool_choice=request.tool_choice,
                tools=request.tools,
                top_p=request.top_p or 1.0,
                background=request.background or False,
                max_output_tokens=request.max_output_tokens or 0,
                service_tier=request.service_tier,
                truncation=request.truncation or "disabled",
            )
            await self.send_event(
                ResponseCreatedEvent(
                    type="response.created",
                    sequence_number=-1,
                    response=response,
                )
            )
            await self.send_event(
                ResponseCompletedEvent(
                    type="response.completed",
                    sequence_number=-1,
                    response=response,
                )
            )
            self.ctx.last_response_id = response.id
            self.ctx.last_response = response
            return

        # Call existing serving pipeline (always streaming)
        result = await self.serving.create_responses(request)

        if isinstance(result, ErrorResponse):
            await self.send_error(
                result.error.type, result.error.message, result.error.code
            )
            self.ctx.evict_cache()
            return

        if isinstance(result, ResponsesResponse):
            # Non-streaming response (shouldn't happen with stream=True)
            await self.send_event(ResponseCreatedEvent(response=result))
            await self.send_event(ResponseCompletedEvent(response=result))
            self.ctx.last_response_id = result.id
            self.ctx.last_response = result
            return

        # Streaming: iterate events, forward over WebSocket
        last_response = None
        async for event_obj in result:
            await self.send_event(event_obj)
            if hasattr(event_obj, "type") and event_obj.type == "response.completed":
                last_response = getattr(event_obj, "response", None)
            if not self._is_connected:
                break

        if last_response is not None:
            self.ctx.last_response_id = last_response.id
            self.ctx.last_response = last_response

    async def handle_connection(self) -> None:
        """Main WebSocket connection loop."""
        from starlette.websockets import WebSocketDisconnect

        self._is_connected = True
        logger.debug(
            "WebSocket responses connection accepted: %s", self.ctx.connection_id
        )

        # Start deadline enforcement task
        self._deadline_task = asyncio.create_task(self._enforce_deadline())

        try:
            while True:
                message = await self.websocket.receive_text()
                try:
                    event = json.loads(message)
                    await self.handle_event(event)
                except json.JSONDecodeError:
                    await self.send_error("invalid_json", "Invalid JSON")
                except Exception as e:
                    logger.exception("Error handling event: %s", e)
                    await self.send_error("processing_error", str(e), 500)
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected: %s", self.ctx.connection_id)
        except Exception as e:
            logger.exception(
                "Unexpected error in connection %s: %s", self.ctx.connection_id, e
            )
        finally:
            self._is_connected = False
            await self.cleanup()

    async def _enforce_deadline(self) -> None:
        """Background task: warn at WARNING_SECONDS, close at
        LIFETIME_SECONDS."""
        try:
            warn_delay = self.ctx.WARNING_SECONDS - (
                time.monotonic() - self.ctx.created_at
            )
            if warn_delay > 0:
                await asyncio.sleep(warn_delay)
            if self._is_connected:
                await self.send_error(
                    "connection_expiring",
                    "Connection will close in 5 minutes",
                )

            close_delay = self.ctx.LIFETIME_SECONDS - (
                time.monotonic() - self.ctx.created_at
            )
            if close_delay > 0:
                await asyncio.sleep(close_delay)
            if self._is_connected:
                self._is_connected = False
                await self.websocket.close(
                    code=1000,
                    reason="Connection lifetime exceeded",
                )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug(
                "Deadline task error (connection likely closed): %s",
                self.ctx.connection_id,
            )

    async def cleanup(self) -> None:
        """Release resources on disconnect."""
        if self._deadline_task and not self._deadline_task.done():
            self._deadline_task.cancel()
        logger.debug("Connection cleanup complete: %s", self.ctx.connection_id)
