from dataclasses import dataclass
from functools import cached_property
from typing import (TYPE_CHECKING, Any, Dict, Generic, Iterable, List, Literal,
                    Optional, Tuple, Union, cast)

import torch
from typing_extensions import NotRequired, TypedDict, TypeVar, assert_never

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalDataDict, MultiModalPlaceholderDict
    from vllm.multimodal.inputs import MultiModalInputsV2


class TextPrompt(TypedDict):
    """Schema for a text prompt."""

    prompt: str
    """The input text to be tokenized before passing to the model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    mm_processor_kwargs: NotRequired[Dict[str, Any]]
    """
    Optional multi-modal processor kwargs to be forwarded to the
    multimodal input mapper & processor. Note that if multiple modalities
    have registered mappers etc for the model being considered, we attempt
    to pass the mm_processor_kwargs to each of them.
    """


class TokensPrompt(TypedDict):
    """Schema for a tokenized prompt."""

    prompt_token_ids: List[int]
    """A list of token IDs to pass to the model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    DEPRECATED: Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    mm_processor_kwargs: NotRequired[Dict[str, Any]]
    """
    DEPRECATED: Optional multi-modal processor kwargs to be forwarded to the
    multimodal input mapper & processor. Note that if multiple modalities
    have registered mappers etc for the model being considered, we attempt
    to pass the mm_processor_kwargs to each of them.
    """


SingletonPrompt = Union[str, TextPrompt, TokensPrompt]
"""
Set of possible schemas for a single prompt:

- A text prompt (:class:`str` or :class:`TextPrompt`)
- A tokenized prompt (:class:`TokensPrompt`)

Note that "singleton" is as opposed to a data structure
which encapsulates multiple prompts, i.e. of the sort
which may be utilized for encoder/decoder models when
the user desires to express both the encoder & decoder
prompts explicitly, i.e. :class:`ExplicitEncoderDecoderPrompt`

A prompt of type :class:`SingletonPrompt` may be employed
as (1) input to a decoder-only model, (2) input to
the encoder of an encoder/decoder model, in the scenario
where the decoder-prompt is not specified explicitly, or
(3) as a member of a larger data structure encapsulating
more than one prompt, i.e. :class:`ExplicitEncoderDecoderPrompt`
"""

_T1_co = TypeVar("_T1_co",
                 bound=SingletonPrompt,
                 default=SingletonPrompt,
                 covariant=True)
_T2_co = TypeVar("_T2_co",
                 bound=SingletonPrompt,
                 default=SingletonPrompt,
                 covariant=True)


# TODO: Make fields ReadOnly once mypy supports it
class ExplicitEncoderDecoderPrompt(TypedDict, Generic[_T1_co, _T2_co]):
    """
    Represents an encoder/decoder model input prompt,
    comprising an explicit encoder prompt and a decoder prompt.

    The encoder and decoder prompts, respectively, may be formatted
    according to any of the :class:`SingletonPrompt` schemas,
    and are not required to have the same schema.

    Only the encoder prompt may have multi-modal data. mm_processor_kwargs
    should be at the top-level, and should not be set in the encoder/decoder
    prompts, since they are agnostic to the encoder/decoder.

    Note that an :class:`ExplicitEncoderDecoderPrompt` may not
    be used as an input to a decoder-only model,
    and that the :code:`encoder_prompt` and :code:`decoder_prompt`
    fields of this data structure themselves must be
    :class:`SingletonPrompt` instances.
    """

    encoder_prompt: _T1_co

    decoder_prompt: Optional[_T2_co]

    mm_processor_kwargs: NotRequired[Dict[str, Any]]


PromptType = Union[SingletonPrompt, ExplicitEncoderDecoderPrompt]
"""
Set of possible schemas for an LLM input, including
both decoder-only and encoder/decoder input types:

- A text prompt (:class:`str` or :class:`TextPrompt`)
- A tokenized prompt (:class:`TokensPrompt`)
- A single data structure containing both an encoder and a decoder prompt
  (:class:`ExplicitEncoderDecoderPrompt`)
"""


class TokenInputs(TypedDict):
    """Represents token-based inputs."""

    type: Literal["token"]
    """The type of inputs."""

    prompt_token_ids: List[int]
    """The token IDs of the prompt."""

    prompt: NotRequired[str]
    """
    The original prompt text corresponding to the token IDs, if available.
    """

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    multi_modal_placeholders: NotRequired["MultiModalPlaceholderDict"]
    """
    Placeholder ranges for the multi-modal data.
    """

    mm_processor_kwargs: NotRequired[Dict[str, Any]]
    """
    Optional multi-modal processor kwargs to be forwarded to the
    multimodal input mapper & processor. Note that if multiple modalities
    have registered mappers etc for the model being considered, we attempt
    to pass the mm_processor_kwargs to each of them.
    """


def token_inputs(
    prompt_token_ids: List[int],
    prompt: Optional[str] = None,
    multi_modal_data: Optional["MultiModalDataDict"] = None,
    multi_modal_placeholders: Optional["MultiModalPlaceholderDict"] = None,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
) -> TokenInputs:
    """Construct :class:`TokenInputs` from optional values."""
    inputs = TokenInputs(type="token", prompt_token_ids=prompt_token_ids)

    if prompt is not None:
        inputs["prompt"] = prompt
    if multi_modal_data is not None:
        inputs["multi_modal_data"] = multi_modal_data
    if multi_modal_placeholders is not None:
        inputs["multi_modal_placeholders"] = multi_modal_placeholders
    if mm_processor_kwargs is not None:
        inputs["mm_processor_kwargs"] = mm_processor_kwargs

    return inputs


DecoderOnlyInputs = Union[TokenInputs, "MultiModalInputsV2"]
"""
The inputs in :class:`~vllm.LLMEngine` before they are
passed to the model executor.
This specifies the data required for decoder-only models.
"""


class EncoderDecoderInputs(TypedDict):
    """
    The inputs in :class:`~vllm.LLMEngine` before they are
    passed to the model executor.

    This specifies the required data for encoder-decoder models.
    """
    encoder: Union[TokenInputs, "MultiModalInputsV2"]
    """The inputs for the encoder portion."""

    decoder: Union[TokenInputs, "MultiModalInputsV2"]
    """The inputs for the decoder portion."""


SingletonInputs = Union[TokenInputs, "MultiModalInputsV2"]
"""
A processed :class:`SingletonPrompt` which can be passed to
:class:`vllm.sequence.Sequence`.
"""


@dataclass
class SingletonInputsAdapter:
    """
    Unified interface to access the components of :class:`SingletonInputs`.
    """
    inputs: SingletonInputs

    @cached_property
    def prompt(self) -> Optional[str]:
        inputs = self.inputs

        if inputs["type"] == "token" or inputs["type"] == "multimodal":
            return inputs.get("prompt")

        assert_never(inputs)

    @cached_property
    def prompt_token_ids(self) -> List[int]:
        inputs = self.inputs

        if inputs["type"] == "token" or inputs["type"] == "multimodal":
            return inputs.get("prompt_token_ids", [])

        assert_never(inputs)

    @cached_property
    def prompt_embeds(self) -> Optional[torch.Tensor]:
        inputs = self.inputs

        if inputs["type"] == "token" or inputs["type"] == "multimodal":
            return None

        assert_never(inputs)

    @cached_property
    def multi_modal_data(self) -> "MultiModalDataDict":
        inputs = self.inputs

        if inputs["type"] == "token":
            return inputs.get("multi_modal_data", {})

        if inputs["type"] == "multimodal":
            return inputs.get("mm_kwargs", {})

        assert_never(inputs)

    @cached_property
    def multi_modal_placeholders(self) -> "MultiModalPlaceholderDict":
        inputs = self.inputs

        if inputs["type"] == "token":
            return inputs.get("multi_modal_placeholders", {})

        if inputs["type"] == "multimodal":
            return inputs.get("mm_placeholders", {})

        assert_never(inputs)

    @cached_property
    def mm_processor_kwargs(self) -> Dict[str, Any]:
        inputs = self.inputs

        if inputs["type"] == "token":
            return inputs.get("mm_processor_kwargs", {})

        if inputs["type"] == "multimodal":
            return {}

        assert_never(inputs)


ProcessorInputs = Union[DecoderOnlyInputs, EncoderDecoderInputs]
"""
The inputs to :data:`vllm.inputs.InputProcessor`.
"""

_T1 = TypeVar("_T1", bound=SingletonPrompt, default=SingletonPrompt)
_T2 = TypeVar("_T2", bound=SingletonPrompt, default=SingletonPrompt)


def build_explicit_enc_dec_prompt(
    encoder_prompt: _T1,
    decoder_prompt: Optional[_T2],
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
) -> ExplicitEncoderDecoderPrompt[_T1, _T2]:
    if mm_processor_kwargs is None:
        mm_processor_kwargs = {}
    return ExplicitEncoderDecoderPrompt(
        encoder_prompt=encoder_prompt,
        decoder_prompt=decoder_prompt,
        mm_processor_kwargs=mm_processor_kwargs)


def zip_enc_dec_prompts(
    enc_prompts: Iterable[_T1],
    dec_prompts: Iterable[Optional[_T2]],
    mm_processor_kwargs: Optional[Union[Iterable[Dict[str, Any]],
                                        Dict[str, Any]]] = None,
) -> List[ExplicitEncoderDecoderPrompt[_T1, _T2]]:
    """
    Zip encoder and decoder prompts together into a list of
    :class:`ExplicitEncoderDecoderPrompt` instances.
    
    ``mm_processor_kwargs`` may also be provided; if a dict is passed, the same
    dictionary will be used for every encoder/decoder prompt. If an iterable is
    provided, it will be zipped with the encoder/decoder prompts.
    """
    if mm_processor_kwargs is None:
        mm_processor_kwargs = cast(Dict[str, Any], {})
    if isinstance(mm_processor_kwargs, dict):
        return [
            build_explicit_enc_dec_prompt(
                encoder_prompt, decoder_prompt,
                cast(Dict[str, Any], mm_processor_kwargs))
            for (encoder_prompt,
                 decoder_prompt) in zip(enc_prompts, dec_prompts)
        ]
    return [
        build_explicit_enc_dec_prompt(encoder_prompt, decoder_prompt,
                                      mm_proc_kwargs)
        for (encoder_prompt, decoder_prompt, mm_proc_kwargs
             ) in zip(enc_prompts, dec_prompts, mm_processor_kwargs)
    ]


def to_enc_dec_tuple_list(
    enc_dec_prompts: Iterable[ExplicitEncoderDecoderPrompt[_T1, _T2]],
) -> List[Tuple[_T1, Optional[_T2]]]:
    return [(enc_dec_prompt["encoder_prompt"],
             enc_dec_prompt["decoder_prompt"])
            for enc_dec_prompt in enc_dec_prompts]


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
