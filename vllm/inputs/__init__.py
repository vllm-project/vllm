from .data import (EncoderDecoderLLMInputs, ExplicitEncoderDecoderPrompt,
                   LLMInputs, PromptType, SingletonPrompt, TextPrompt,
                   TokensPrompt, build_decoder_prompt, build_decoder_prompts,
                   build_explicit_enc_dec_prompt, to_enc_dec_tuple_list,
                   zip_enc_dec_prompts)
from .registry import InputContext, InputRegistry

INPUT_REGISTRY = InputRegistry()
"""
The global :class:`~InputRegistry` which is used by :class:`~vllm.LLMEngine`
to dispatch data processing according to the target model.

See also:
    :ref:`input_processing_pipeline`
"""

__all__ = [
    "TextPrompt",
    "TokensPrompt",
    "PromptType",
    "SingletonPrompt",
    "ExplicitEncoderDecoderPrompt",
    "LLMInputs",
    "EncoderDecoderLLMInputs",
    "build_decoder_prompt",
    "build_decoder_prompts",
    "build_explicit_enc_dec_prompt",
    "to_enc_dec_tuple_list",
    "zip_enc_dec_prompts",
    "INPUT_REGISTRY",
    "InputContext",
    "InputRegistry",
]
