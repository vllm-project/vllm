from typing import (List, Literal, Optional, Sequence, TypedDict, Union,
                    overload)

from vllm.utils import is_list_of

from .data import LLMInputs, PromptInputs


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

    raise ValueError("prompt must be a string, array of strings, "
                     "array of tokens, or array of token arrays")


def _has_required_keys(
    d: dict,
    required_keys: set,
) -> bool:
    return required_keys.issubset(d.keys())


def get_prompt_type(prompt: Optional[PromptInputs]) -> Optional[str]:
    """
    Get the type-name of the prompt argument instance, given that
    isinstance() cannot apply to TypedDict subclasses directly.
    If the prompt is None, return 'None' as the type name.

    Arguments:

    * prompt: LLM input prompt or None

    Returns:

    * String representation of prompt type
    """

    if prompt is None:
        return 'None'

    required_keys_dict = {
        'TextPrompt': {'prompt'},
        'TokensPrompt': {'prompt_token_ids'},
        'ExplicitEncoderDecoder': {'encoder_prompt', 'decoder_prompt'},
    }

    if isinstance(prompt, dict):
        for (ptype, required_keys) in required_keys_dict.items():
            # Ignore type checking in the conditional below because type
            # checker does not understand that is_dict(prompt) narrows
            # down the possible types
            if _has_required_keys(
                    prompt,  # type: ignore
                    required_keys):
                return ptype

        raise ValueError(f"Invalid prompt {prompt}, valid types are "
                         f"required_keys_dict={required_keys_dict}")

    if isinstance(prompt, str):
        return "str"

    raise ValueError(f"Invalid prompt {prompt}")


def is_valid_encoder_decoder_llm_inputs(inputs: LLMInputs) -> bool:
    """
    Return True if the LLMInputs instance has the correct configuration
    for encoder/decoder.
    """

    # True if encoder prompt token ids field exists &
    # is not None
    return ('encoder_prompt_token_ids' in inputs
            and inputs['encoder_prompt_token_ids'] is not None)
