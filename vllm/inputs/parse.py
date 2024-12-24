from typing import List, Literal, Sequence, TypedDict, Union, cast, overload

from typing_extensions import TypeIs

from vllm.utils import is_list_of

from .data import (EncoderDecoderInputs, ExplicitEncoderDecoderPrompt,
                   ProcessorInputs, PromptType, SingletonPrompt, TextPrompt,
                   TokensPrompt)


class ParsedText(TypedDict):
    content: str
    is_tokens: Literal[False]


class ParsedTokens(TypedDict):
    content: List[int]
    is_tokens: Literal[True]


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

        if is_list_of(prompt, str):
            # case 2: array of strings
            prompt = cast(List[str], prompt)
            return [
                ParsedText(content=elem, is_tokens=False) for elem in prompt
            ]
        if is_list_of(prompt, int):
            # case 3: array of tokens
            prompt = cast(List[int], prompt)
            return [ParsedTokens(content=prompt, is_tokens=True)]
        if is_list_of(prompt, list):
            prompt = cast(List[List[int]], prompt)
            if len(prompt[0]) == 0:
                raise ValueError("please provide at least one prompt")

            if is_list_of(prompt[0], int):
                # case 4: array of token arrays
                return [
                    ParsedTokens(content=elem, is_tokens=True)
                    for elem in prompt
                ]

    raise TypeError("prompt must be a string, array of strings, "
                    "array of tokens, or array of token arrays")


class ParsedStrPrompt(TypedDict):
    type: Literal["str"]
    content: str


class ParsedTextPrompt(TypedDict):
    type: Literal["text"]
    content: TextPrompt


class ParsedTokensPrompt(TypedDict):
    type: Literal["tokens"]
    content: TokensPrompt


def parse_singleton_prompt(
    prompt: SingletonPrompt,
) -> Union[ParsedStrPrompt, ParsedTextPrompt, ParsedTokensPrompt]:
    if isinstance(prompt, str):
        return ParsedStrPrompt(type="str", content=prompt)
    elif isinstance(prompt, dict):
        if "prompt_token_ids" in prompt:
            return ParsedTokensPrompt(type="tokens",
                                      content=prompt)  # type: ignore
        elif "prompt" in prompt:
            return ParsedTextPrompt(type="text", content=prompt)

    raise TypeError("inputs must be a string, TextPrompt, or TokensPrompt")


def is_token_prompt(prompt: PromptType) -> TypeIs[TokensPrompt]:
    return isinstance(prompt, dict) and "prompt_token_ids" in prompt


def is_explicit_encoder_decoder_prompt(
        prompt: PromptType) -> TypeIs[ExplicitEncoderDecoderPrompt]:
    return isinstance(prompt, dict) and "encoder_prompt" in prompt


def is_encoder_decoder_inputs(
        inputs: ProcessorInputs) -> TypeIs[EncoderDecoderInputs]:
    return "encoder" in inputs and "decoder" in inputs
