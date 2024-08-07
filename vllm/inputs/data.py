from typing import TYPE_CHECKING, List, Optional, Tuple, TypedDict, Union

from typing_extensions import NotRequired

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalDataDict


class TextPrompt(TypedDict):
    """Schema for a text prompt."""

    prompt: str
    """The input text to be tokenized before passing to the model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """


class TokensPrompt(TypedDict):
    """Schema for a tokenized prompt."""

    prompt_token_ids: List[int]
    """A list of token IDs to pass to the model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """


SingletonPromptInputs = Union[str, TextPrompt, TokensPrompt]
"""
Set of possible schemas for a single LLM input:

- A text prompt (:class:`str` or :class:`TextPrompt`)
- A tokenized prompt (:class:`TokensPrompt`)

Note that "singleton" is as opposed to a data structure
which encapsulates multiple prompts, i.e. of the sort
which may be utilized for encoder/decoder models when
the user desires to express both the encoder & decoder
prompts explicitly, i.e. ExplicitEncoderDecoderPrompt

A prompt of type SingletonPromptInputs may be employed
as (1) input to a decoder-only model, (2) input to
the encoder of an encoder/decoder model, in the scenario
where the decoder-prompt is not specified explicitly, or
(3) as a member of a larger data structure encapsulating
more than one prompt, i.e. ExplicitEncoderDecoderPrompt
"""


class ExplicitEncoderDecoderPrompt(TypedDict):
    """Represents an encoder/decoder model input prompt,
    comprising an explicit encoder prompt and a 
    decoder prompt.

    The encoder and decoder prompts, respectively,
    may formatted according to any of the
    SingletonPromptInputs schemas, and are not
    required to have the same schema.

    Only the encoder prompt may have multi-modal data.

    Note that an ExplicitEncoderDecoderPrompt may not
    be used as an input to a decoder-only model,
    and that the `encoder_prompt` and `decoder_prompt`
    fields of this data structure may not themselves
    must be SingletonPromptInputs instances.
    """

    encoder_prompt: SingletonPromptInputs

    decoder_prompt: SingletonPromptInputs


PromptInputs = Union[SingletonPromptInputs, ExplicitEncoderDecoderPrompt]
"""
Set of possible schemas for an LLM input, including
both decoder-only and encoder/decoder input types:

- A text prompt (:class:`str` or :class:`TextPrompt`)
- A tokenized prompt (:class:`TokensPrompt`)
- A single data structure containing both an encoder and a decoder prompt
  (:class:`ExplicitEncoderDecoderPrompt`)
"""


class LLMInputs(TypedDict):
    """
    The inputs in :class:`~vllm.LLMEngine` before they are
    passed to the model executor.

    This includes the data required for decoder-only models.
    """
    prompt_token_ids: List[int]
    """The token IDs of the prompt."""

    prompt: NotRequired[Optional[str]]
    """
    The original prompt text corresponding to the token IDs, if available.
    """

    multi_modal_data: NotRequired[Optional["MultiModalDataDict"]]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """


class EncoderDecoderLLMInputs(LLMInputs):
    """
    The inputs in :class:`~vllm.LLMEngine` before they are
    passed to the model executor.

    This includes the required data for encoder-decoder models.
    """
    encoder_prompt_token_ids: List[int]
    """The token IDs of the encoder prompt."""

    encoder_prompt: NotRequired[Optional[str]]
    """
    The original encoder prompt text corresponding to the token IDs, if
    available.
    """


def build_explicit_enc_dec_prompt(
    encoder_prompt: SingletonPromptInputs,
    decoder_prompt: SingletonPromptInputs,
) -> ExplicitEncoderDecoderPrompt:
    return ExplicitEncoderDecoderPrompt(encoder_prompt=encoder_prompt,
                                        decoder_prompt=decoder_prompt)


def zip_enc_dec_prompt_lists(
    enc_prompt_list: List[SingletonPromptInputs],
    dec_prompt_list: List[SingletonPromptInputs],
) -> List[ExplicitEncoderDecoderPrompt]:
    return [
        build_explicit_enc_dec_prompt(encoder_prompt, decoder_prompt)
        for (encoder_prompt,
             decoder_prompt) in zip(enc_prompt_list, dec_prompt_list)
    ]


def to_enc_dec_tuple_list(
    enc_dec_prompts: List[ExplicitEncoderDecoderPrompt],
) -> List[Tuple[PromptInputs, PromptInputs]]:
    return [(enc_dec_prompt['encoder_prompt'],
             enc_dec_prompt['decoder_prompt'])
            for enc_dec_prompt in enc_dec_prompts]
