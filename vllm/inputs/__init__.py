from .data import (ExplicitEncoderDecoderPrompt,
                   ExplicitEncoderDecoderPromptStrict, LLMInputs, ParsedText,
                   ParsedTokens, PromptInputs, PromptStrictInputs, TextPrompt,
                   TextTokensPrompt, TokensPrompt, get_single_prompt_type,
                   is_valid_encoder_decoder_llm_inputs,
                   is_valid_encoder_decoder_prompt, parse_and_batch_prompt)
from .registry import InputContext, InputRegistry

INPUT_REGISTRY = InputRegistry()
"""
The global :class:`~InputRegistry` which is used by :class:`~vllm.LLMEngine`
to dispatch data processing according to the target model.

See also:
    :ref:`input_processing_pipeline`
"""

__all__ = [
    "ParsedText",
    "ParsedTokens",
    "parse_and_batch_prompt",
    "TextPrompt",
    "TokensPrompt",
    "TextTokensPrompt",
    "PromptStrictInputs",
    "PromptInputs",
    "LLMInputs",
    "INPUT_REGISTRY",
    "InputContext",
    "InputRegistry",
    "get_single_prompt_type",
    "is_valid_encoder_decoder_prompt",
    "is_valid_encoder_decoder_llm_inputs",
    "ExplicitEncoderDecoderPromptStrict",
    "ExplicitEncoderDecoderPrompt",
]
