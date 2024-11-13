from .data import (DecoderOnlyInputs, EncoderDecoderInputs,
                   ExplicitEncoderDecoderPrompt, ProcessorInputs, PromptType,
                   SingletonInputs, SingletonInputsAdapter, SingletonPrompt,
                   TextPrompt, TokenInputs, TokensPrompt,
                   build_explicit_enc_dec_prompt, to_enc_dec_tuple_list,
                   token_inputs, zip_enc_dec_prompts)
from .registry import (DummyData, InputContext, InputProcessingContext,
                       InputRegistry)

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
    "DecoderOnlyInputs",
    "EncoderDecoderInputs",
    "ProcessorInputs",
    "SingletonInputs",
    "SingletonInputsAdapter",
    "build_explicit_enc_dec_prompt",
    "to_enc_dec_tuple_list",
    "zip_enc_dec_prompts",
    "INPUT_REGISTRY",
    "DummyData",
    "InputContext",
    "InputProcessingContext",
    "InputRegistry",
]


def __getattr__(name: str):
    import warnings

    if name == "PromptInput":
        msg = ("PromptInput has been renamed to PromptType. "
               "The original name will be removed in an upcoming version.")

        warnings.warn(DeprecationWarning(msg), stacklevel=2)

        return PromptType

    if name == "LLMInputs":
        msg = ("LLMInputs has been renamed to DecoderOnlyInputs. "
               "The original name will be removed in an upcoming version.")

        warnings.warn(DeprecationWarning(msg), stacklevel=2)

        return DecoderOnlyInputs

    if name == "EncoderDecoderLLMInputs":
        msg = (
            "EncoderDecoderLLMInputs has been renamed to EncoderDecoderInputs. "
            "The original name will be removed in an upcoming version.")

        warnings.warn(DeprecationWarning(msg), stacklevel=2)

        return EncoderDecoderInputs

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
