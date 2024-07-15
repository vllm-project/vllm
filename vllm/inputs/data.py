from typing import (TYPE_CHECKING, List, Literal, Optional, Sequence,
                    TypedDict, Union, cast, overload)

from typing_extensions import NotRequired

from .utils import has_required_keys, is_dict, is_str

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalDataDict


class ParsedText(TypedDict):
    content: str
    is_tokens: Literal[False]


class ParsedTokens(TypedDict):
    content: List[int]
    is_tokens: Literal[True]


# https://github.com/vllm-project/vllm/pull/4028
@overload
def parse_and_batch_prompt(
        prompt: Union[str, List[str]]) -> Sequence[ParsedText]:
    ...


@overload
def parse_and_batch_prompt(
        prompt: Union[List[int], List[List[int]]]) -> Sequence[ParsedTokens]:
    ...


def parse_and_batch_prompt(
    prompt: Union[str, List[str], List[int], List[List[int]]],
) -> Union[Sequence[ParsedText], Sequence[ParsedTokens]]:
    if isinstance(prompt, str):
        # case 1: a string
        return [ParsedText(content=prompt, is_tokens=False)]

    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError("please provide at least one prompt")

        if isinstance(prompt[0], str):
            # case 2: array of strings
            return [
                ParsedText(content=elem, is_tokens=False)
                for elem in cast(List[str], prompt)
            ]
        if isinstance(prompt[0], int):
            # case 3: array of tokens
            elem = cast(List[int], prompt)
            return [ParsedTokens(content=elem, is_tokens=True)]
        if isinstance(prompt[0], list):
            if len(prompt[0]) == 0:
                raise ValueError("please provide at least one prompt")

            if isinstance(prompt[0][0], int):
                # case 4: array of token arrays
                return [
                    ParsedTokens(content=elem, is_tokens=True)
                    for elem in cast(List[List[int]], prompt)
                ]

    raise ValueError("prompt must be a string, array of strings, "
                     "array of tokens, or array of token arrays")


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


class TextTokensPrompt(TypedDict):
    """It is assumed that :attr:`prompt` is consistent with
    :attr:`prompt_token_ids`. This is currently used in
    :class:`AsyncLLMEngine` for logging both the text and token IDs."""

    prompt: str
    """The prompt text."""

    prompt_token_ids: List[int]
    """The token IDs of the prompt."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """


DecoderOnlyPromptInputs = Union[str, TextPrompt, TokensPrompt,
                                TextTokensPrompt]
StrictDecoderOnlyPromptInputs = Union[str, TextPrompt, TokensPrompt]


class ExplicitEncoderDecoderPrompt(TypedDict):
    """Represents an encoder/decoder model input prompt,
    comprising an encoder prompt and a decoder prompt.

    Only the encoder prompt may have multi-modal data.
    """

    encoder_prompt: DecoderOnlyPromptInputs

    decoder_prompt: DecoderOnlyPromptInputs


class ExplicitEncoderDecoderPromptStrict(TypedDict):
    """Represents an encoder/decoder model input prompt,
    comprising an encoder prompt and a decoder prompt.
    Strictly forbid a prompt containing both text and
    tokens.

    Only the encoder prompt may have multi-modal data.
    """

    encoder_prompt: StrictDecoderOnlyPromptInputs

    decoder_prompt: StrictDecoderOnlyPromptInputs


PromptStrictInputs = Union[StrictDecoderOnlyPromptInputs,
                           ExplicitEncoderDecoderPromptStrict]
"""
The inputs to the LLM, which can take one of the following forms:

- A text prompt (:class:`str` or :class:`TextPrompt`)
- A tokenized prompt (:class:`TokensPrompt`)
"""

PromptInputs = Union[DecoderOnlyPromptInputs, ExplicitEncoderDecoderPrompt]
"""Same as :const:`PromptStrictInputs` but additionally accepts
:class:`TextTokensPrompt`."""

AllPromptInputs = Union[PromptInputs, ExplicitEncoderDecoderPromptStrict]
"""All possible input prompt options, strict or non-strict"""


def get_single_prompt_type(prompt: AllPromptInputs, ) -> str:
    """
    Get the type-name of the prompt argument instance, given that
    isinstance() cannot apply to TypedDict subclasses directly.
    """

    required_keys_dict = {
        'TextPrompt': ['prompt'],
        'TokensPrompt': ['prompt_token_ids'],
        'TextTokensPrompt': ['prompt', 'prompt_token_ids'],
        'ExplicitEncoderDecoder': ['encoder_prompt', 'decoder_prompt'],
    }

    if is_dict(prompt):
        for ptype in required_keys_dict:
            # Ignore type checking in the conditional below because type
            # checker does not understand that is_dict(prompt) narrows
            # down the possible types
            if has_required_keys(
                    prompt,  # type: ignore
                    required_keys_dict[ptype]):
                return ptype

        raise ValueError(f"Invalid prompt {prompt}, valid types are "
                         "required_keys_dict={required_keys_dict}")
    elif is_str(prompt):
        return "str"

    raise ValueError(f"Invalid prompt {prompt}")


def is_valid_encoder_decoder_prompt(prompt: AllPromptInputs, ) -> bool:
    """
    Return True if prompt has the correct structure for an encoder/decoder
    prompt.
    """
    # Ignore type checking in the conditional below because type checker
    # does not understand that
    # get_single_prompt_type(prompt) == 'ExplicitEncoderDecoder' narrows
    # down the possible types
    if (get_single_prompt_type(prompt) == 'ExplicitEncoderDecoder' and
        (prompt['encoder_prompt'] is None  # type: ignore
         or prompt['decoder_prompt']['multi_modal_data']  # type: ignore
         is not None)):
        # For explicit encoder/decoder prompts, encoder prompt
        # must be non-None and decoder prompt must be free of
        # multi-modal data (which should instead be passed to
        # the encoder.)
        return False

    # Any valid prompt type other than an explicit encoder/decoder
    # prompt is a guaranteed-valid prompt
    return True


class LLMInputs(TypedDict):
    """
    The inputs in :class:`~vllm.LLMEngine` before they are
    passed to the model executor.
    """
    prompt_token_ids: List[int]
    """The token IDs of the prompt."""

    prompt: NotRequired[Optional[str]]
    """
    The original prompt text corresponding to the token IDs, if available.
    """

    encoder_prompt_token_ids: NotRequired[List[int]]
    """The token IDs of the encoder prompt."""

    encoder_prompt: NotRequired[Optional[str]]
    """
    The original encoder prompt text corresponding to the token IDs, if
    available.
    """

    multi_modal_data: NotRequired[Optional["MultiModalDataDict"]]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

def is_valid_encoder_decoder_llm_inputs(inputs: LLMInputs, ) -> bool:
    """
    Return True if the LLMInputs instance has the correct configuration
    for encoder/decoder.
    """

    if ('encoder_prompt_token_ids' in inputs and 
        inputs['encoder_prompt_token_ids'] is not None):
        # Encoder prompt token ids field exists &
        # is not None
        return True
    
    return False