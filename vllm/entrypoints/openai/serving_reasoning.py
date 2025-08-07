"""
Serving response utilities for reasoning responses.
Extends the base serving responses with reasoning-specific functionality.
"""
import json
import time
from typing import Any, Dict, List, Optional, Union

from vllm.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
    ErrorResponse,
    LogProbs,
    UsageInfo,
)
from vllm.logger import init_logger
from vllm.transformers_utils.openai_reasoning_parser import (
    OpenAIReasoningParser,
    ReasoningContent,
    FinalContent,
    extract_reasoning_and_final,
)

logger = init_logger(__name__)


class ReasoningDelta:
    """Delta for reasoning content in streaming responses."""
    
    def __init__(self, reasoning: Optional[str] = None):
        self.reasoning = reasoning


class ReasoningUsageInfo(UsageInfo):
    """Extended usage info with reasoning token counts."""
    
    def __init__(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        reasoning_tokens: Optional[int] = None,
    ):
        super().__init__(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        self.reasoning_tokens = reasoning_tokens


class ReasoningChatCompletionResponse(ChatCompletionResponse):
    """Chat completion response with reasoning support."""
    
    def __init__(
        self,
        id: str,
        choices: List[ChatCompletionResponseChoice],
        created: int,
        model: str,
        object: str = "chat.completion",
        system_fingerprint: Optional[str] = None,
        usage: Optional[ReasoningUsageInfo] = None,
        reasoning: Optional[str] = None,
    ):
        super().__init__(
            id=id,
            choices=choices,
            created=created,
            model=model,
            object=object,
            system_fingerprint=system_fingerprint,
            usage=usage,
        )
        self.reasoning = reasoning


class ReasoningChatCompletionStreamResponse(ChatCompletionStreamResponse):
    """Streaming chat completion response with reasoning support."""
    
    def __init__(
        self,
        id: str,
        choices: List[ChatCompletionResponseStreamChoice],
        created: int,
        model: str,
        object: str = "chat.completion.chunk",
        system_fingerprint: Optional[str] = None,
        usage: Optional[ReasoningUsageInfo] = None,
    ):
        super().__init__(
            id=id,
            choices=choices,
            created=created,
            model=model,
            object=object,
            system_fingerprint=system_fingerprint,
            usage=usage,
        )


def create_reasoning_response(
    request_id: str,
    model_name: str,
    output_text: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    include_reasoning: bool = True,
    finish_reason: str = "stop",
) -> ReasoningChatCompletionResponse:
    """Create a reasoning-aware chat completion response."""
    
    reasoning_content, final_content, is_tool_call = extract_reasoning_and_final(
        output_text
    )
    
    # Estimate reasoning tokens (simplified)
    reasoning_tokens = (
        len(reasoning_content.split()) if reasoning_content else 0
    )
    
    # Prepare the response content
    if include_reasoning and reasoning_content:
        response_content = final_content or ""
        reasoning_text = reasoning_content
    else:
        response_content = final_content or output_text
        reasoning_text = None
    
    # Create the choice
    choice = ChatCompletionResponseChoice(
        index=0,
        message={
            "role": "assistant",
            "content": response_content,
        },
        finish_reason=finish_reason,
        logprobs=None,
    )
    
    # Create usage info
    usage = ReasoningUsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        reasoning_tokens=reasoning_tokens,
    )
    
    return ReasoningChatCompletionResponse(
        id=request_id,
        choices=[choice],
        created=int(time.time()),
        model=model_name,
        usage=usage,
        reasoning=reasoning_text,
    )


def create_reasoning_stream_chunk(
    request_id: str,
    model_name: str,
    delta_content: str,
    chunk_index: int = 0,
    finish_reason: Optional[str] = None,
    is_reasoning: bool = False,
) -> ReasoningChatCompletionStreamResponse:
    """Create a streaming chunk for reasoning responses."""
    
    if is_reasoning:
        # For reasoning content, we might want special handling
        delta = DeltaMessage(
            role="assistant" if chunk_index == 0 else None,
            content=delta_content,
        )
    else:
        delta = DeltaMessage(
            role="assistant" if chunk_index == 0 else None,
            content=delta_content,
        )
    
    choice = ChatCompletionResponseStreamChoice(
        index=0,
        delta=delta,
        finish_reason=finish_reason,
        logprobs=None,
    )
    
    return ReasoningChatCompletionStreamResponse(
        id=request_id,
        choices=[choice],
        created=int(time.time()),
        model=model_name,
    )


def parse_reasoning_stream(
    output_text: str,
    previous_text: str = "",
) -> tuple[str, str, bool]:
    """
    Parse streaming output for reasoning content.
    
    Returns:
        tuple of (new_reasoning, new_final, is_complete)
    """
    
    # Simple incremental parsing
    # In practice, this would be more sophisticated
    
    if "<|reasoning|>" in output_text and "<|/reasoning|>" not in output_text:
        # Still building reasoning content
        reasoning_start = output_text.find("<|reasoning|>") + len("<|reasoning|>")
        new_reasoning = output_text[reasoning_start:].strip()
        return new_reasoning, "", False
    
    elif "<|final|>" in output_text and "<|/final|>" not in output_text:
        # Building final content
        final_start = output_text.find("<|final|>") + len("<|final|>")
        new_final = output_text[final_start:].strip()
        return "", new_final, False
    
    else:
        # Regular content or complete
        reasoning_content, final_content, _ = extract_reasoning_and_final(output_text)
        
        # Return only the new part
        new_content = output_text[len(previous_text):]
        return "", new_content, True


def format_error_response(
    error_message: str,
    error_type: str = "invalid_request_error",
    error_code: Optional[str] = None,
) -> ErrorResponse:
    """Format an error response for reasoning endpoints."""
    
    return ErrorResponse(
        message=error_message,
        type=error_type,
        code=error_code,
    )


def extract_usage_from_output(
    output_text: str,
    prompt_tokens: int = 0,
) -> ReasoningUsageInfo:
    """Extract token usage information from output."""
    
    reasoning_content, final_content, _ = extract_reasoning_and_final(output_text)
    
    # Simple token counting (word-based approximation)
    reasoning_tokens = (
        len(reasoning_content.split()) if reasoning_content else 0
    )
    final_tokens = len(final_content.split()) if final_content else 0
    completion_tokens = reasoning_tokens + final_tokens
    
    return ReasoningUsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        reasoning_tokens=reasoning_tokens,
    )
