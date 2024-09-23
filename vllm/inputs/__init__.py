from .data import (DecoderOnlyInputs, EncoderDecoderInputs,
                   ExplicitEncoderDecoderPrompt, PromptType, SingletonInputs,
                   SingletonPrompt, TextPrompt, TokenInputs, TokensPrompt,
                   build_explicit_enc_dec_prompt, to_enc_dec_tuple_list,
                   token_inputs, zip_enc_dec_prompts)
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
    "TokenInputs",
    "token_inputs",
    "SingletonInputs",
    "DecoderOnlyInputs",
    "EncoderDecoderInputs",
    "build_explicit_enc_dec_prompt",
    "to_enc_dec_tuple_list",
    "zip_enc_dec_prompts",
    "INPUT_REGISTRY",
    "InputContext",
    "InputRegistry",
]
