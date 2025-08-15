from .data import (
    EncoderDecoderLLMInputs,
    ExplicitEncoderDecoderPrompt,
    LLMInputs,
    PromptType,
    SingletonPrompt,
    TextPrompt,
    TokensPrompt,
    build_explicit_enc_dec_prompt,
    to_enc_dec_tuple_list,
    zip_enc_dec_prompts,
)
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
    "build_explicit_enc_dec_prompt",
    "to_enc_dec_tuple_list",
    "zip_enc_dec_prompts",
    "INPUT_REGISTRY",
    "InputContext",
    "InputRegistry",
]


def __getattr__(name: str):
    if name == "PromptInput":
        import warnings

        msg = (
            "PromptInput has been renamed to PromptType. "
            "The original name will be removed in an upcoming version."
        )

        warnings.warn(DeprecationWarning(msg), stacklevel=2)

        return PromptType

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
