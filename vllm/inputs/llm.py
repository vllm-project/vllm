"""Schema and utilities for input prompts to the LLM API."""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, final

from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    import torch

    from vllm.multimodal.inputs import AudioItem, ImageItem, VideoItem, VisionChunk


_T = TypeVar("_T")

ModalityData: TypeAlias = _T | list[_T | None] | None
"""
Either a single data item, or a list of data items. Can only be None if UUID
is provided.

The number of data items allowed per modality is restricted by
`--limit-mm-per-prompt`.
"""


@final
class MultiModalDataBuiltins(TypedDict, total=False):
    """Type annotations for modality types predefined by vLLM."""

    image: ModalityData["ImageItem"]
    """The input image(s)."""

    video: ModalityData["VideoItem"]
    """The input video(s)."""

    audio: ModalityData["AudioItem"]
    """The input audio(s)."""

    vision_chunk: ModalityData["VisionChunk"]
    """The input visual atom(s) - unified modality for images and video chunks."""


MultiModalDataDict: TypeAlias = Mapping[str, ModalityData[Any]]
"""
A dictionary containing an entry for each modality type to input.

The built-in modalities are defined by
[`MultiModalDataBuiltins`][vllm.inputs.llm.MultiModalDataBuiltins].
"""

MultiModalUUIDDict: TypeAlias = Mapping[str, Sequence[str | None] | str]
"""
A dictionary containing user-provided UUIDs for items in each modality.
If a UUID for an item is not provided, its entry will be `None` and
MultiModalHasher will compute a hash for the item.

The UUID will be used to identify the item for all caching purposes
(input processing caching, embedding caching, prefix caching, etc).
"""


class _PromptOptions(TypedDict):
    """
    Additional options available to all
    [`SingletonPrompt`][vllm.inputs.llm.SingletonPrompt] types.
    """

    multi_modal_data: NotRequired[MultiModalDataDict | None]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    mm_processor_kwargs: NotRequired[dict[str, Any] | None]
    """
    Optional multi-modal processor kwargs to be forwarded to the
    multimodal input mapper & processor. Note that if multiple modalities
    have registered mappers etc for the model being considered, we attempt
    to pass the mm_processor_kwargs to each of them.
    """

    multi_modal_uuids: NotRequired[MultiModalUUIDDict]
    """
    Optional user-specified UUIDs for multimodal items, mapped by modality.
    Lists must match the number of items per modality and may contain `None`.
    For `None` entries, the hasher will compute IDs automatically; non-None
    entries override the default hashes for caching, and MUST be unique per
    multimodal item.
    """

    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
    """


class TextPrompt(_PromptOptions):
    """Schema for a text prompt."""

    prompt: str
    """The input text to be tokenized before passing to the model."""


class TokensPrompt(_PromptOptions):
    """Schema for a tokenized prompt."""

    prompt_token_ids: list[int]
    """A list of token IDs to pass to the model."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token IDs, if available."""

    token_type_ids: NotRequired[list[int]]
    """A list of token type IDs to pass to the cross encoder model."""


class EmbedsPrompt(_PromptOptions):
    """Schema for a prompt provided via token embeddings."""

    prompt_embeds: "torch.Tensor"
    """The embeddings of the prompt."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token embeddings, if available."""


DecoderOnlyPrompt: TypeAlias = (
    str | TextPrompt | list[int] | TokensPrompt | EmbedsPrompt
)
"""
Schema of a prompt for a decoder-only model:

- A text prompt (string or [`TextPrompt`][vllm.inputs.llm.TextPrompt])
- A tokenized prompt (list of token IDs, or
  [`TokensPrompt`][vllm.inputs.llm.TokensPrompt])
- An embeddings prompt ([`EmbedsPrompt`][vllm.inputs.llm.EmbedsPrompt])

For encoder-decoder models, passing a singleton prompt is shorthand for passing
`ExplicitEncoderDecoderPrompt(encoder_prompt=prompt, decoder_prompt=None)`.
"""


EncoderPrompt: TypeAlias = str | TextPrompt | list[int] | TokensPrompt
"""
Schema of a prompt for the encoder part of a encoder-decoder model:

- A text prompt (string or [`TextPrompt`][vllm.inputs.llm.TextPrompt])
- A tokenized prompt (list of token IDs, or
  [`TokensPrompt`][vllm.inputs.llm.TokensPrompt])
"""


DecoderPrompt: TypeAlias = str | TextPrompt | list[int] | TokensPrompt
"""
Schema of a prompt for the decoder part of an encoder-decoder model:

- A text prompt (string or [`TextPrompt`][vllm.inputs.llm.TextPrompt])
- A tokenized prompt (list of token IDs, or
  [`TokensPrompt`][vllm.inputs.llm.TokensPrompt])

Note:
    Multi-modal inputs are not supported for decoder prompts.
"""


class ExplicitEncoderDecoderPrompt(TypedDict):
    """
    Schema for a pair of encoder and decoder singleton prompts.

    Note:
        This schema is not valid for decoder-only models.
    """

    encoder_prompt: EncoderPrompt
    """The prompt for the encoder part of the model."""

    decoder_prompt: DecoderPrompt | None
    """
    The prompt for the decoder part of the model.

    Passing `None` will cause the prompt to be inferred automatically.
    """


EncoderDecoderPrompt: TypeAlias = EncoderPrompt | ExplicitEncoderDecoderPrompt
"""
Schema for a prompt for an encoder-decoder model.

You can pass a singleton encoder prompt, in which case the decoder prompt is
considered to be `None` (i.e., infer automatically).
"""


SingletonPrompt: TypeAlias = DecoderOnlyPrompt | EncoderPrompt | DecoderPrompt
"""
Schema for a single prompt. This is as opposed to a data structure
which encapsulates multiple prompts, such as
[`ExplicitEncoderDecoderPrompt`][vllm.inputs.llm.ExplicitEncoderDecoderPrompt].
"""


PromptType: TypeAlias = DecoderOnlyPrompt | EncoderDecoderPrompt
"""
Schema for any prompt, regardless of model type.

This is the input format accepted by most [`LLM`][vllm.entrypoints.llm.LLM] APIs.
"""


class DataPrompt(_PromptOptions):
    """
    Represents generic inputs that are converted to
    [`PromptType`][vllm.inputs.llm.PromptType] by IO processor plugins.
    """

    data: Any
    """The input data."""

    data_format: str
    """The input data format."""
