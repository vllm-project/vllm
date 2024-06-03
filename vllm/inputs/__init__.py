from vllm.multimodal import MULTIMODAL_REGISTRY

from .data import (LLMInputs, ParsedText, ParsedTokens, PromptInputs,
                   PromptStrictInputs, TextPrompt, TextTokensPrompt,
                   TokensPrompt, parse_and_batch_prompt)
from .registry import InputRegistry

INPUT_REGISTRY = InputRegistry(multimodal_registry=MULTIMODAL_REGISTRY)
"""The global :class:`~InputRegistry` which is used by model runners."""

del MULTIMODAL_REGISTRY

__all__ = [
    "ParsedText", "ParsedTokens", "parse_and_batch_prompt", "TextPrompt",
    "TokensPrompt", "TextTokensPrompt", "PromptStrictInputs", "PromptInputs",
    "LLMInputs", "INPUT_REGISTRY", "InputRegistry"
]
