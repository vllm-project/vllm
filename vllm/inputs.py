from typing import (TYPE_CHECKING, List, Literal, Optional, Sequence,
                    TypedDict, Union, cast, overload)

if TYPE_CHECKING:
    from vllm.sequence import MultiModalData


class ParsedString(TypedDict):
    text: str
    is_tokens: Literal[False]


class ParsedTokens(TypedDict):
    text: List[int]
    is_tokens: Literal[True]


# https://github.com/vllm-project/vllm/pull/4028
@overload
def parse_and_batch_prompt(
        prompt: Union[str, List[str]]) -> Sequence[ParsedString]:
    ...


@overload
def parse_and_batch_prompt(
        prompt: Union[List[int], List[List[int]]]) -> Sequence[ParsedTokens]:
    ...


def parse_and_batch_prompt(
    prompt: Union[str, List[str], List[int], List[List[int]]],
) -> Union[Sequence[ParsedString], Sequence[ParsedTokens]]:
    if isinstance(prompt, str):
        # case 1: a string
        return [ParsedString(text=prompt, is_tokens=False)]

    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError("please provide at least one prompt")

        if isinstance(prompt[0], str):
            # case 2: array of strings
            return [
                ParsedString(text=elem, is_tokens=False)
                for elem in cast(List[str], prompt)
            ]
        if isinstance(prompt[0], int):
            # case 3: array of tokens
            elem = cast(List[int], prompt)
            return [ParsedTokens(text=elem, is_tokens=True)]
        if isinstance(prompt[0], list):
            if len(prompt[0]) == 0:
                raise ValueError("please provide at least one prompt")

            if isinstance(prompt[0][0], int):
                # case 4: array of token arrays
                return [
                    ParsedTokens(text=elem, is_tokens=True)
                    for elem in cast(List[List[int]], prompt)
                ]

    raise ValueError("prompt must be a string, array of strings, "
                     "array of tokens, or array of token arrays")


class MultiModalPrompt(TypedDict, total=False):
    multi_modal_data: Optional["MultiModalData"]
    """Multi modal data."""


class StringPrompt(MultiModalPrompt, TypedDict):
    prompt: str
    """The prompt string."""


class TokensPrompt(MultiModalPrompt, TypedDict):
    prompt_token_ids: List[int]
    """The token IDs of the prompt. If None, we use the
    tokenizer to convert the prompts to token IDs."""


class StringTokensPrompt(MultiModalPrompt, TypedDict):
    """It is assumed that :attr:`prompt` is consistent with
    :attr:`prompt_token_ids`. This is currently used in
    :class:`AsyncLLMEngine` for logging both the text and token IDs."""

    prompt: str
    """The prompt string."""

    prompt_token_ids: List[int]
    """The token IDs of the prompt. If None, we use the
    tokenizer to convert the prompts to token IDs."""


PromptStrictInputs = Union[str, StringPrompt, TokensPrompt]
"""The prompt string. More complex inputs should be represented by
:class:`StringPrompt` or :class:`TokensPrompt`."""

PromptInputs = Union[str, StringPrompt, TokensPrompt, StringTokensPrompt]
"""As :const:`PromptStrictInputs` but additionally accepts
:class:`StringTokensPrompt`."""


class LLMInputs(TypedDict):
    prompt_token_ids: List[int]
    prompt: Optional[str]
    multi_modal_data: Optional["MultiModalData"]
