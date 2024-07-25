from .data import (LLMInputs, ParsedText, ParsedTokens, PromptInputs,
                   TextPrompt, TokensPrompt, parse_and_batch_prompt)
from .registry import InputContext, InputRegistry

INPUT_REGISTRY = InputRegistry()
"""
The global :class:`~InputRegistry` which is used by :class:`~vllm.LLMEngine`
to dispatch data processing according to the target model.

See also:
    :ref:`input_processing_pipeline`
"""

__all__ = [
    "ParsedText", "ParsedTokens", "parse_and_batch_prompt", "TextPrompt",
    "TokensPrompt", "PromptInputs", "LLMInputs", "INPUT_REGISTRY",
    "InputContext", "InputRegistry"
]
