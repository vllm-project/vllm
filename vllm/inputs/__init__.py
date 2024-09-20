from .data import (DecoderOnlyInputs, EmbedInputs, EmbedsPrompt, EmptyInputs,
                   EncoderDecoderInputs, ExplicitEncoderDecoderPrompt,
                   PromptType, SingletonPrompt, TextPrompt, TokenInputs,
                   TokensPrompt, build_explicit_enc_dec_prompt, embed_inputs,
                   empty_inputs, to_enc_dec_tuple_list, token_inputs,
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
    "EmbedsPrompt",
    "PromptType",
    "SingletonPrompt",
    "ExplicitEncoderDecoderPrompt",
    "TokenInputs",
    "token_inputs",
    "EmbedInputs",
    "embed_inputs",
    "DecoderOnlyInputs",
    "EmptyInputs",
    "empty_inputs",
    "EncoderDecoderInputs",
    "build_explicit_enc_dec_prompt",
    "to_enc_dec_tuple_list",
    "zip_enc_dec_prompts",
    "INPUT_REGISTRY",
    "InputContext",
    "InputRegistry",
]
