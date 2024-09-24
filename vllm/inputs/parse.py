from typing import List, Literal, Sequence, TypedDict, Union, overload

from typing_extensions import TypeIs

from vllm.utils import is_list_of

from .data import (EncoderDecoderLLMInputs, ExplicitEncoderDecoderPrompt,
                   LLMInputs, PromptInputs, SingletonPromptInputs, TextPrompt,
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
            return [
                ParsedText(content=elem, is_tokens=False) for elem in prompt
            ]
        if is_list_of(prompt, int):
            # case 3: array of tokens
            return [ParsedTokens(content=prompt, is_tokens=True)]
        if is_list_of(prompt, list):
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
    inputs: SingletonPromptInputs,
) -> Union[ParsedStrPrompt, ParsedTextPrompt, ParsedTokensPrompt]:
    if isinstance(inputs, str):
        return ParsedStrPrompt(type="str", content=inputs)
    elif isinstance(inputs, dict):
        if "prompt_token_ids" in inputs:
            return ParsedTokensPrompt(type="tokens",
                                      content=inputs)  # type: ignore
        elif "prompt" in inputs:
            return ParsedTextPrompt(type="text", content=inputs)

    raise TypeError("inputs must be a string, TextPrompt, or TokensPrompt")


def is_explicit_encoder_decoder_prompt(
        inputs: PromptInputs) -> TypeIs[ExplicitEncoderDecoderPrompt]:
    return isinstance(inputs, dict) and "encoder_prompt" in inputs


def is_valid_encoder_decoder_llm_inputs(
    inputs: Union[LLMInputs, EncoderDecoderLLMInputs],
) -> TypeIs[EncoderDecoderLLMInputs]:
    return "encoder_prompt_token_ids" in inputs
