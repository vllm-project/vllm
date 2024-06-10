from typing import (TYPE_CHECKING, List, Literal, Optional, Sequence,
                    TypedDict, Union, cast, overload)

from typing_extensions import NotRequired

if TYPE_CHECKING:
    from vllm.sequence import MultiModalData


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

    multi_modal_data: NotRequired["MultiModalData"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """


class TokensPrompt(TypedDict):
    """Schema for a tokenized prompt."""

    prompt_token_ids: List[int]
    """A list of token IDs to pass to the model."""

    multi_modal_data: NotRequired["MultiModalData"]
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
    """The token IDs of the prompt. If None, we use the
    tokenizer to convert the prompts to token IDs."""

    multi_modal_data: NotRequired["MultiModalData"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """


PromptStrictInputs = Union[str, TextPrompt, TokensPrompt]
"""
The inputs to the LLM, which can take one of the following forms:

- A text prompt (:class:`str` or :class:`TextPrompt`)
- A tokenized prompt (:class:`TokensPrompt`)
"""

PromptInputs = Union[str, TextPrompt, TokensPrompt, TextTokensPrompt]
"""Same as :const:`PromptStrictInputs` but additionally accepts
:class:`TextTokensPrompt`."""


class LLMInputs(TypedDict):
    prompt_token_ids: List[int]
    prompt: NotRequired[Optional[str]]
    multi_modal_data: NotRequired[Optional["MultiModalData"]]
