# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OpenAI Realtime WebSocket API Server

This module implements the WebSocket handler for OpenAI's Realtime API.
Supports real audio model inference with Ultravox and similar models.

Reference: https://platform.openai.com/docs/guides/realtime-conversations
"""

import asyncio
import base64
import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.openai.protocol_realtime import (
    CLIENT_EVENT_TYPES,
    ContentPart,
    Conversation,
    ConversationCreatedEvent,
    ConversationItem,
    ConversationItemCreatedEvent,
    ConversationItemDeletedEvent,
    ConversationItemTruncatedEvent,
    ErrorEvent,
    InputAudioBufferClearedEvent,
    InputAudioBufferCommittedEvent,
    RateLimitsUpdatedEvent,
    RealtimeError,
    Response,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
    Session,
    SessionConfig,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    generate_event_id,
    generate_item_id,
    parse_client_event,
)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger

logger = init_logger(__name__)


class OpenAIServingRealtime:
    """Handler for OpenAI Realtime WebSocket API.

    This class manages WebSocket connections and handles the Realtime API
    protocol. Supports real audio model inference with Ultravox and similar models.
    """

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        model_config: ModelConfig,
        *,
        default_model: str = "gpt-4o-realtime-preview",
    ):
        """Initialize the Realtime API handler.

        Args:
            engine_client: The vLLM engine client for model inference
            models: The models handler for model information
            model_config: The model configuration
            default_model: Default model name for the session
        """
        self.engine_client = engine_client
        self.models = models
        self.model_config = model_config
        self.default_model = default_model

        # Get tokenizer from the processor
        self.processor = models.processor
        self.tokenizer = self.processor.tokenizer if self.processor else None

        # Audio configuration defaults
        self.default_sample_rate = 24000  # 24kHz for Ultravox/Whisper
        self.audio_placeholder = "<|audio|>"  # Ultravox audio placeholder token

    async def handle_session(self, websocket: WebSocket) -> None:
        """Handle a WebSocket session.

        This is the main entry point for a Realtime API connection.
        It manages the full lifecycle of the WebSocket session.

        Args:
            websocket: The FastAPI WebSocket connection
        """
        await websocket.accept()

        # Initialize session state
        session = Session(model=self.default_model)
        conversation = Conversation()
        conversation_items: dict[str, ConversationItem] = {}
        current_response: Response | None = None

        # Audio buffer for accumulating input audio chunks
        audio_buffer: bytes = b""

        try:
            # Send session.created event
            await self._send_event(
                websocket, SessionCreatedEvent(session=session)
            )

            # Send conversation.created event
            await self._send_event(
                websocket, ConversationCreatedEvent(conversation=conversation)
            )

            # Main message loop
            while True:
                try:
                    # Receive message from client
                    data = await websocket.receive_text()
                    event_data = json.loads(data)

                    logger.info("Received event: %s", event_data.get("type"))

                    # Process the event and update audio buffer
                    audio_buffer = await self._process_event(
                        websocket,
                        event_data,
                        session,
                        conversation,
                        conversation_items,
                        audio_buffer,
                    )

                except json.JSONDecodeError as e:
                    await self._send_error(
                        websocket,
                        code="invalid_json",
                        message=f"Invalid JSON: {e}",
                    )
                except ValueError as e:
                    await self._send_error(
                        websocket,
                        code="invalid_event",
                        message=str(e),
                    )

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected for session %s", session.id)
        except Exception as e:
            logger.exception("Error in WebSocket session: %s", e)
            try:
                await self._send_error(
                    websocket,
                    code="server_error",
                    message=f"Internal server error: {e}",
                )
            except Exception:
                pass

    async def _send_event(
        self, websocket: WebSocket, event: Any
    ) -> None:
        """Send an event to the client.

        Args:
            websocket: The WebSocket connection
            event: The event to send (Pydantic model)
        """
        await websocket.send_text(event.model_dump_json())

    async def _send_error(
        self,
        websocket: WebSocket,
        code: str,
        message: str,
        event_id: str | None = None,
    ) -> None:
        """Send an error event to the client.

        Args:
            websocket: The WebSocket connection
            code: Error code
            message: Error message
            event_id: Optional event ID that caused the error
        """
        error_event = ErrorEvent(
            error=RealtimeError(
                type="error",
                code=code,
                message=message,
                event_id=event_id,
            )
        )
        await self._send_event(websocket, error_event)

    async def _process_event(
        self,
        websocket: WebSocket,
        event_data: dict[str, Any],
        session: Session,
        conversation: Conversation,
        conversation_items: dict[str, ConversationItem],
        audio_buffer: bytes,
    ) -> bytes:
        """Process an incoming client event.

        Args:
            websocket: The WebSocket connection
            event_data: The parsed event data
            session: Current session state
            conversation: Current conversation state
            conversation_items: Dictionary of conversation items
            audio_buffer: Accumulated audio data (bytes)

        Returns:
            Updated audio buffer (may be cleared after commit)
        """
        event_type = event_data.get("type")
        event_id = event_data.get("event_id")

        if event_type not in CLIENT_EVENT_TYPES:
            await self._send_error(
                websocket,
                code="unknown_event",
                message=f"Unknown event type: {event_type}",
                event_id=event_id,
            )
            return audio_buffer

        # Parse and validate the event
        try:
            event = parse_client_event(event_data)
        except Exception as e:
            await self._send_error(
                websocket,
                code="invalid_event",
                message=f"Failed to parse event: {e}",
                event_id=event_id,
            )
            return audio_buffer

        # Route to appropriate handler
        if event_type == "session.update":
            await self._handle_session_update(websocket, event, session)

        elif event_type == "input_audio_buffer.append":
            # Accumulate audio data
            audio_buffer = self._handle_audio_append(event, audio_buffer)

        elif event_type == "input_audio_buffer.commit":
            audio_buffer = await self._handle_input_audio_commit(
                websocket, session, conversation_items, audio_buffer
            )

        elif event_type == "input_audio_buffer.clear":
            audio_buffer = await self._handle_input_audio_clear(websocket)

        elif event_type == "conversation.item.create":
            await self._handle_conversation_item_create(
                websocket, event, conversation_items
            )

        elif event_type == "conversation.item.truncate":
            await self._handle_conversation_item_truncate(websocket, event)

        elif event_type == "conversation.item.delete":
            await self._handle_conversation_item_delete(
                websocket, event, conversation_items
            )

        elif event_type == "response.create":
            await self._handle_response_create(
                websocket, event, session, conversation_items
            )

        elif event_type == "response.cancel":
            logger.debug("Received response cancel")

        return audio_buffer

    async def _handle_session_update(
        self,
        websocket: WebSocket,
        event: Any,
        session: Session,
    ) -> None:
        """Handle session.update event.

        Args:
            websocket: The WebSocket connection
            event: The SessionUpdateEvent
            session: Current session state
        """
        # Update session configuration
        update_data = event.session.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            if hasattr(session, key):
                setattr(session, key, value)

        # Send session.updated event
        await self._send_event(
            websocket, SessionUpdatedEvent(session=session)
        )

    def _handle_audio_append(
        self,
        event: Any,
        audio_buffer: bytes,
    ) -> bytes:
        """Handle input_audio_buffer.append event.

        Accumulates base64-encoded PCM16 audio chunks.

        Args:
            event: The InputAudioBufferAppendEvent
            audio_buffer: Current accumulated audio data

        Returns:
            Updated audio buffer with new data appended
        """
        try:
            # Decode base64 audio and append to buffer
            audio_chunk = base64.b64decode(event.audio)
            audio_buffer += audio_chunk
            logger.debug(
                "Appended %d bytes to audio buffer (total: %d bytes)",
                len(audio_chunk),
                len(audio_buffer),
            )
        except Exception as e:
            logger.warning("Failed to decode audio chunk: %s", e)
        return audio_buffer

    async def _handle_input_audio_commit(
        self,
        websocket: WebSocket,
        session: Session,
        conversation_items: dict[str, ConversationItem],
        audio_buffer: bytes,
    ) -> bytes:
        """Handle input_audio_buffer.commit event.

        Converts accumulated audio buffer to numpy array and creates
        a conversation item for the audio input.

        Args:
            websocket: The WebSocket connection
            session: Current session state
            conversation_items: Dictionary of conversation items
            audio_buffer: Accumulated audio data (PCM16 bytes)

        Returns:
            Empty bytes (buffer is cleared after commit)
        """
        if not audio_buffer:
            logger.warning("Committing empty audio buffer")
            await self._send_error(
                websocket,
                code="empty_audio_buffer",
                message="Cannot commit empty audio buffer",
            )
            return audio_buffer

        # Convert PCM16 bytes to numpy array
        # PCM16 is 16-bit signed integers, little-endian
        audio_array = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32)
        # Normalize to [-1.0, 1.0] range
        audio_array = audio_array / 32768.0

        # Get sample rate from session config or use default
        sample_rate = self.default_sample_rate
        if hasattr(session, "input_audio_format"):
            sample_rate = getattr(
                session.input_audio_format, "sample_rate", self.default_sample_rate
            )

        logger.debug(
            "Committed audio: %d samples at %d Hz (%.2f seconds)",
            len(audio_array),
            sample_rate,
            len(audio_array) / sample_rate,
        )

        # Create user message item with audio content
        item_id = generate_item_id()
        item = ConversationItem(
            id=item_id,
            type="message",
            role="user",
            content=[
                ContentPart(
                    type="input_audio",
                    transcript=None,  # Will be filled after transcription
                )
            ],
        )
        # Store audio data with the item for later processing
        item._audio_data = (audio_array, sample_rate)  # type: ignore
        conversation_items[item_id] = item

        # Send committed event
        previous_id = (
            list(conversation_items.keys())[-2]
            if len(conversation_items) > 1
            else None
        )
        await self._send_event(
            websocket,
            InputAudioBufferCommittedEvent(
                previous_item_id=previous_id,
                item_id=item_id,
            ),
        )

        # Send item created event
        await self._send_event(
            websocket,
            ConversationItemCreatedEvent(
                previous_item_id=previous_id,
                item=item,
            ),
        )

        # Clear buffer after commit
        return b""

    async def _handle_input_audio_clear(
        self,
        websocket: WebSocket,
    ) -> bytes:
        """Handle input_audio_buffer.clear event.

        Args:
            websocket: The WebSocket connection

        Returns:
            Empty bytes (buffer is cleared)
        """
        await self._send_event(websocket, InputAudioBufferClearedEvent())
        logger.debug("Cleared audio buffer")
        return b""

    async def _handle_conversation_item_create(
        self,
        websocket: WebSocket,
        event: Any,
        conversation_items: dict[str, ConversationItem],
    ) -> None:
        """Handle conversation.item.create event.

        Args:
            websocket: The WebSocket connection
            event: The ConversationItemCreateEvent
            conversation_items: Dictionary of conversation items
        """
        item = event.item
        conversation_items[item.id] = item

        await self._send_event(
            websocket,
            ConversationItemCreatedEvent(
                previous_item_id=event.previous_item_id,
                item=item,
            ),
        )

    async def _handle_conversation_item_truncate(
        self,
        websocket: WebSocket,
        event: Any,
    ) -> None:
        """Handle conversation.item.truncate event.

        Args:
            websocket: The WebSocket connection
            event: The ConversationItemTruncateEvent
        """
        await self._send_event(
            websocket,
            ConversationItemTruncatedEvent(
                item_id=event.item_id,
                content_index=event.content_index,
                audio_end_ms=event.audio_end_ms,
            ),
        )

    async def _handle_conversation_item_delete(
        self,
        websocket: WebSocket,
        event: Any,
        conversation_items: dict[str, ConversationItem],
    ) -> None:
        """Handle conversation.item.delete event.

        Args:
            websocket: The WebSocket connection
            event: The ConversationItemDeleteEvent
            conversation_items: Dictionary of conversation items
        """
        item_id = event.item_id
        if item_id in conversation_items:
            del conversation_items[item_id]

        await self._send_event(
            websocket,
            ConversationItemDeletedEvent(item_id=item_id),
        )

    async def _handle_response_create(
        self,
        websocket: WebSocket,
        event: Any,
        session: Session,
        conversation_items: dict[str, ConversationItem],
    ) -> None:
        """Handle response.create event with actual model inference.

        Builds a prompt from conversation history, including audio data,
        and generates a response using the vLLM engine.

        Args:
            websocket: The WebSocket connection
            event: The ResponseCreateEvent
            session: Current session state
            conversation_items: Dictionary of conversation items
        """
        # Create response object
        response = Response(status="in_progress")

        # Send response.created
        await self._send_event(
            websocket,
            ResponseCreatedEvent(response=response),
        )

        # Create assistant message item
        item_id = generate_item_id()
        item = ConversationItem(
            id=item_id,
            type="message",
            status="in_progress",
            role="assistant",
            content=[],
        )

        # Send response.output_item.added
        await self._send_event(
            websocket,
            ResponseOutputItemAddedEvent(
                response_id=response.id,
                output_index=0,
                item=item,
            ),
        )

        # Add text content part (Ultravox is audio-to-text only)
        text_content_index = 0
        text_part = ContentPart(type="text", text="")
        item.content.append(text_part)

        await self._send_event(
            websocket,
            ResponseContentPartAddedEvent(
                response_id=response.id,
                item_id=item_id,
                output_index=0,
                content_index=text_content_index,
                part=text_part,
            ),
        )

        # Build prompt and generate response
        full_text = ""
        input_tokens = 0
        output_tokens = 0

        try:
            # Build the prompt from conversation items
            prompt, multi_modal_data = self._build_prompt(conversation_items, session)

            if prompt is None:
                # No valid prompt could be built
                await self._send_error(
                    websocket,
                    code="invalid_request",
                    message="No valid conversation content to generate response from",
                )
                return

            print(f"[DEBUG] Built prompt: {prompt[:200] if len(prompt) > 200 else prompt}")
            if multi_modal_data:
                for key, val in multi_modal_data.items():
                    print(f"[DEBUG] Multi-modal data[{key}]: {len(val) if isinstance(val, list) else 1} items")
                    if key == "audio" and val:
                        for i, audio_item in enumerate(val):
                            if isinstance(audio_item, tuple) and len(audio_item) == 2:
                                arr, sr = audio_item
                                print(f"[DEBUG]   audio[{i}]: shape={arr.shape if hasattr(arr, 'shape') else 'unknown'}, dtype={arr.dtype if hasattr(arr, 'dtype') else 'unknown'}, sr={sr}")
            else:
                print("[DEBUG] Multi-modal data: None")

            # Generate response using the engine
            async for delta, tokens_info in self._generate_response(
                prompt, multi_modal_data, session
            ):
                full_text += delta
                if tokens_info:
                    input_tokens = tokens_info.get("input_tokens", input_tokens)
                    output_tokens = tokens_info.get("output_tokens", output_tokens)

                await self._send_event(
                    websocket,
                    ResponseTextDeltaEvent(
                        response_id=response.id,
                        item_id=item_id,
                        output_index=0,
                        content_index=text_content_index,
                        delta=delta,
                    ),
                )

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error("Error generating response: %s\n%s", e, error_trace)
            await self._send_error(
                websocket,
                code="generation_error",
                message=f"Failed to generate response: {type(e).__name__}: {e}",
            )
            # Update response status to failed
            response.status = "failed"
            await self._send_event(
                websocket,
                ResponseDoneEvent(response=response),
            )
            return

        # Send text done
        text_part.text = full_text
        await self._send_event(
            websocket,
            ResponseTextDoneEvent(
                response_id=response.id,
                item_id=item_id,
                output_index=0,
                content_index=text_content_index,
                text=full_text,
            ),
        )

        # Mark text content part done
        await self._send_event(
            websocket,
            ResponseContentPartDoneEvent(
                response_id=response.id,
                item_id=item_id,
                output_index=0,
                content_index=text_content_index,
                part=text_part,
            ),
        )

        # Mark item as completed
        item.status = "completed"
        conversation_items[item_id] = item

        # Send response.output_item.done
        await self._send_event(
            websocket,
            ResponseOutputItemDoneEvent(
                response_id=response.id,
                output_index=0,
                item=item,
            ),
        )

        # Update response and send response.done
        response.status = "completed"
        response.output = [item]
        response.usage = ResponseUsage(
            total_tokens=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        await self._send_event(
            websocket,
            ResponseDoneEvent(response=response),
        )

        # Send rate limits update
        await self._send_event(
            websocket,
            RateLimitsUpdatedEvent(
                rate_limits=[
                    {
                        "name": "requests",
                        "limit": 100,
                        "remaining": 99,
                        "reset_seconds": 60.0,
                    },
                    {
                        "name": "tokens",
                        "limit": 100000,
                        "remaining": 100000 - (input_tokens + output_tokens),
                        "reset_seconds": 60.0,
                    },
                ]
            ),
        )

    def _build_prompt(
        self,
        conversation_items: dict[str, ConversationItem],
        session: Session,
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Build a prompt from conversation items for model inference.

        Constructs a chat-formatted prompt with audio placeholders
        for audio content items.

        Args:
            conversation_items: Dictionary of conversation items
            session: Current session state

        Returns:
            Tuple of (prompt string, multi_modal_data dict) or (None, None)
        """
        if not conversation_items:
            return None, None

        messages = []
        audio_data_list = []

        # Process conversation items in order
        for item in conversation_items.values():
            if item.type != "message":
                continue

            role = item.role
            content_parts = []

            for part in item.content:
                # Handle both "text" (assistant output) and "input_text" (user input)
                if part.type in ("text", "input_text") and part.text:
                    content_parts.append(part.text)
                elif part.type == "input_audio":
                    # Add audio placeholder
                    content_parts.append(self.audio_placeholder)

                    # Get audio data if stored on the item
                    audio_tuple = getattr(item, "_audio_data", None)
                    if audio_tuple is not None:
                        audio_data_list.append(audio_tuple)

            if content_parts:
                content = "\n".join(content_parts)
                messages.append({"role": role, "content": content})

        if not messages:
            return None, None

        # Apply chat template to build the prompt
        if self.tokenizer is None:
            # Fallback: simple concatenation
            prompt = ""
            for msg in messages:
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += "assistant:"
        else:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.warning("Failed to apply chat template: %s, using fallback", e)
                prompt = ""
                for msg in messages:
                    prompt += f"{msg['role']}: {msg['content']}\n"
                prompt += "assistant:"

        # Build multi_modal_data
        multi_modal_data = None
        if audio_data_list:
            multi_modal_data = {"audio": audio_data_list}

        return prompt, multi_modal_data

    async def _generate_response(
        self,
        prompt: str,
        multi_modal_data: dict[str, Any] | None,
        session: Session,
    ) -> AsyncGenerator[tuple[str, dict[str, int] | None], None]:
        """Generate a response using the vLLM engine.

        Args:
            prompt: The text prompt
            multi_modal_data: Optional multi-modal data (audio)
            session: Current session state

        Yields:
            Tuples of (text_delta, tokens_info)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available for model inference")

        # Tokenize the prompt
        prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        # Build engine prompt
        engine_prompt: EngineTokensPrompt = {
            "prompt_token_ids": prompt_token_ids,
        }

        if multi_modal_data:
            engine_prompt["multi_modal_data"] = multi_modal_data

        # Create sampling params with safe type conversion
        temperature = getattr(session, "temperature", 0.8)
        if not isinstance(temperature, (int, float)):
            temperature = 0.8

        max_tokens = getattr(session, "max_response_output_tokens", 4096)
        if max_tokens == "inf" or not isinstance(max_tokens, int):
            max_tokens = 4096

        sampling_params = SamplingParams(
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

        # Generate request ID
        request_id = f"realtime-{uuid.uuid4().hex}"

        logger.debug(
            "Starting generation: request_id=%s, prompt_tokens=%d",
            request_id,
            len(prompt_token_ids),
        )

        # Stream generation from the engine
        prev_text_len = 0
        input_tokens = len(prompt_token_ids)
        output_tokens = 0

        try:
            generator = self.engine_client.generate(
                engine_prompt,
                sampling_params,
                request_id,
            )

            async for output in generator:
                # Get the generated text
                if output.outputs:
                    current_text = output.outputs[0].text
                    delta = current_text[prev_text_len:]
                    prev_text_len = len(current_text)

                    # Update token counts
                    output_tokens = len(output.outputs[0].token_ids)

                    if delta:
                        yield delta, {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                        }

        except Exception as e:
            logger.exception("Engine generation error: %s", e)
            raise

