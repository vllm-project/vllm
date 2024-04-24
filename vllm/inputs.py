from typing import TYPE_CHECKING, List, Optional, TypedDict, Union

if TYPE_CHECKING:
    from vllm.sequence import MultiModalData


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
