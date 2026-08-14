# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pydantic
from fastapi import Request
from starlette.concurrency import iterate_in_threadpool

from vllm.logger import init_logger

logger = init_logger(__name__)


class SSEDecoder:
    """Robust Server-Sent Events decoder for streaming responses."""

    def __init__(self):
        self.buffer = ""
        self.content_buffer = []

    def decode_chunk(self, chunk: bytes) -> list[dict]:
        """Decode a chunk of SSE data and return parsed events."""
        import json

        try:
            chunk_str = chunk.decode("utf-8")
        except UnicodeDecodeError:
            # Skip malformed chunks
            return []

        self.buffer += chunk_str
        events = []

        # Process complete lines
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.rstrip("\r")  # Handle CRLF

            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    events.append({"type": "done"})
                elif data_str:
                    try:
                        event_data = json.loads(data_str)
                        events.append({"type": "data", "data": event_data})
                    except json.JSONDecodeError:
                        # Skip malformed JSON
                        continue

        return events

    def extract_content(self, event_data: dict) -> str:
        """Extract content from event data."""
        return _extract_content_from_chunk(event_data)

    def add_content(self, content: str) -> None:
        """Add content to the buffer."""
        if content:
            self.content_buffer.append(content)

    def get_complete_content(self) -> str:
        """Get the complete buffered content."""
        return "".join(self.content_buffer)


def _extract_content_from_chunk(chunk_data: dict) -> str:
    """Extract content from a streaming response chunk."""
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionStreamResponse,
        )
        from vllm.entrypoints.openai.completion.protocol import (
            CompletionStreamResponse,
        )

        # Try using Completion types for type-safe parsing
        if chunk_data.get("object") == "chat.completion.chunk":
            chat_response = ChatCompletionStreamResponse.model_validate(chunk_data)
            if chat_response.choices and chat_response.choices[0].delta.content:
                return chat_response.choices[0].delta.content
        elif chunk_data.get("object") == "text_completion":
            completion_response = CompletionStreamResponse.model_validate(chunk_data)
            if completion_response.choices and completion_response.choices[0].text:
                return completion_response.choices[0].text
    except pydantic.ValidationError:
        # Fallback to manual parsing
        if "choices" in chunk_data and chunk_data["choices"]:
            choice = chunk_data["choices"][0]
            if "delta" in choice and choice["delta"].get("content"):
                return choice["delta"]["content"]
            elif choice.get("text"):
                return choice["text"]
    return ""


def _log_streaming_response(response, response_body: list) -> None:
    """Log streaming response with robust SSE parsing."""
    from starlette.concurrency import iterate_in_threadpool

    sse_decoder = SSEDecoder()
    chunk_count = 0

    def buffered_iterator():
        nonlocal chunk_count

        for chunk in response_body:
            chunk_count += 1
            yield chunk

            # Parse SSE events from chunk
            events = sse_decoder.decode_chunk(chunk)

            for event in events:
                if event["type"] == "data":
                    content = sse_decoder.extract_content(event["data"])
                    sse_decoder.add_content(content)
                elif event["type"] == "done":
                    # Log complete content when done
                    full_content = sse_decoder.get_complete_content()
                    if full_content:
                        # Truncate if too long
                        if len(full_content) > 2048:
                            full_content = full_content[:2048] + ""
                            "...[truncated]"
                        logger.info(
                            "response_body={streaming_complete: content=%r, chunks=%d}",
                            full_content,
                            chunk_count,
                        )
                    else:
                        logger.info(
                            "response_body={streaming_complete: no_content, chunks=%d}",
                            chunk_count,
                        )
                    return

    response.body_iterator = iterate_in_threadpool(buffered_iterator())
    logger.info("response_body={streaming_started: chunks=%d}", len(response_body))


def _log_non_streaming_response(response_body: list) -> None:
    """Log non-streaming response."""
    try:
        decoded_body = response_body[0].decode()
        logger.info("response_body={%s}", decoded_body)
    except UnicodeDecodeError:
        logger.info("response_body={<binary_data>}")


async def log_response(request: Request, call_next):
    response = await call_next(request)
    response_body = [section async for section in response.body_iterator]
    response.body_iterator = iterate_in_threadpool(iter(response_body))
    # Check if this is a streaming response by looking at content-type
    content_type = response.headers.get("content-type", "")
    is_streaming = content_type == "text/event-stream; charset=utf-8"

    # Log response body based on type
    if not response_body:
        logger.info("response_body={<empty>}")
    elif is_streaming:
        _log_streaming_response(response, response_body)
    else:
        _log_non_streaming_response(response_body)
    return response
