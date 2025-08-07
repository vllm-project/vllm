"""
OpenAI reasoning parser for GPT-OSS model responses.
This module handles parsing of reasoning and final content from model outputs.
"""

import re
from typing import List, Optional, Tuple

from vllm.logger import init_logger

logger = init_logger(__name__)


class ReasoningContent:
    """Represents reasoning content from the model."""

    def __init__(self, text: str):
        self.text = text


class FinalContent:
    """Represents final content from the model."""

    def __init__(self, text: str):
        self.text = text


class OpenAIMessage:
    """Represents a parsed message from the model output."""

    def __init__(self, content: List, recipient: Optional[str] = None):
        self.content = content
        self.recipient = recipient


class OpenAIReasoningParser:
    """Parser for OpenAI-style reasoning responses."""

    def __init__(self):
        self.messages: List[OpenAIMessage] = []
        self.current_content: Optional[str] = None
        self._reasoning_pattern = re.compile(
            r"<\|reasoning\|>(.*?)<\|/reasoning\|>", re.DOTALL
        )
        self._final_pattern = re.compile(r"<\|final\|>(.*?)<\|/final\|>", re.DOTALL)

    def parse(self, text: str) -> "OpenAIReasoningParser":
        """Parse the text and extract reasoning and final content."""
        self.messages.clear()
        self.current_content = None

        # Look for reasoning content
        reasoning_matches = self._reasoning_pattern.findall(text)
        if reasoning_matches:
            reasoning_text = reasoning_matches[0].strip()
            reasoning_content = [ReasoningContent(reasoning_text)]
            self.messages.append(OpenAIMessage(reasoning_content))

        # Look for final content
        final_matches = self._final_pattern.findall(text)
        if final_matches:
            final_text = final_matches[0].strip()
            final_content = [FinalContent(final_text)]
            self.messages.append(OpenAIMessage(final_content))

        # If no structured content found, treat as current content
        if not self.messages:
            self.current_content = text.strip()

        return self

    def get_reasoning_content(self) -> Optional[str]:
        """Get the reasoning content if available."""
        for message in self.messages:
            for content in message.content:
                if isinstance(content, ReasoningContent):
                    return content.text
        return None

    def get_final_content(self) -> Optional[str]:
        """Get the final content if available."""
        for message in self.messages:
            for content in message.content:
                if isinstance(content, FinalContent):
                    return content.text
        return self.current_content


def parse_output_into_messages(token_ids: List[int]) -> OpenAIReasoningParser:
    """Parse token IDs into OpenAI message format."""
    # This is a simplified implementation
    # In practice, you'd need to decode the tokens first

    # For demonstration, assume we have a way to decode tokens to text
    # In real implementation, this would use the tokenizer
    text = f"Mock decoded text from {len(token_ids)} tokens"

    parser = OpenAIReasoningParser()
    return parser.parse(text)


def extract_reasoning_and_final(
    output_text: str,
) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Extract reasoning and final content from output text.

    Returns:
        Tuple of (reasoning_content, final_content, is_tool_call)
    """
    parser = OpenAIReasoningParser()
    parser.parse(output_text)

    reasoning_content = parser.get_reasoning_content()
    final_content = parser.get_final_content()

    # Simple heuristic to detect tool calls
    is_tool_call = final_content is not None and any(
        keyword in final_content.lower()
        for keyword in ["function_call", "tool_call", '"function":', '"name":']
    )

    return reasoning_content, final_content, is_tool_call


def format_reasoning_response(
    reasoning_content: Optional[str],
    final_content: Optional[str],
    include_reasoning: bool = True,
) -> str:
    """Format reasoning response for API output."""

    if not include_reasoning:
        return final_content or ""

    result = ""
    if reasoning_content:
        result += f"<|reasoning|>\n{reasoning_content}\n<|/reasoning|>\n\n"

    if final_content:
        result += f"<|final|>\n{final_content}\n<|/final|>"

    return result
