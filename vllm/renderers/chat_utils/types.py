# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal, Required, TypeAlias, TypedDict

from openai.types.chat import (
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam,
)
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam,
)
from openai_harmony import Message as OpenAIHarmonyMessage
from PIL import Image
from pydantic import BaseModel, ConfigDict


class AudioURL(TypedDict, total=False):
    url: Required[str]
    """
    Either a URL of the audio or a data URL with base64 encoded audio data.
    """


class ChatCompletionContentPartAudioParam(TypedDict, total=False):
    audio_url: Required[AudioURL]

    type: Required[Literal["audio_url"]]
    """The type of the content part."""


class VideoURL(TypedDict, total=False):
    url: Required[str]
    """
    Either a URL of the video or a data URL with base64 encoded video data.
    """


class ChatCompletionContentPartVideoParam(TypedDict, total=False):
    video_url: Required[VideoURL]

    type: Required[Literal["video_url"]]
    """The type of the content part."""


class PILImage(BaseModel):
    """A PIL.Image.Image object."""

    image_pil: Image.Image
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CustomChatCompletionContentPILImageParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a PIL image.

    Example:
    {
        "image_pil": ImageAsset('cherry_blossom').pil_image
    }

    """

    image_pil: PILImage | None
    uuid: str | None
    """
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    """


class CustomChatCompletionContentSimpleImageParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a plain image_url.
    This is supported by OpenAI API, although it is not documented.

    Example:
    {
        "image_url": "https://example.com/image.jpg"
    }

    """

    image_url: str | None
    uuid: str | None
    """
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    """


MultiModalEmbedsPayload: TypeAlias = str | dict[str, str | list[int | float]]


class ChatCompletionContentPartImageEmbedsParam(TypedDict, total=False):
    image_embeds: MultiModalEmbedsPayload | None
    """
    The image embeddings. It can be either:
    - A single base64 string.
    - A dictionary of base64 tensors or numeric JSON metadata arrays.
    """
    type: Required[Literal["image_embeds"]]
    """The type of the content part."""
    uuid: str | None
    """
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    """


class ChatCompletionContentPartAudioEmbedsParam(TypedDict, total=False):
    audio_embeds: MultiModalEmbedsPayload | None
    """
    The audio embeddings. It can be either:
    - A single base64 string representing a serialized torch tensor.
    - A dictionary of base64 tensors or numeric JSON metadata arrays.
    """
    type: Required[Literal["audio_embeds"]]
    """The type of the content part."""
    uuid: str | None
    """
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    """


class ChatCompletionContentPartVideoEmbedsParam(TypedDict, total=False):
    video_embeds: MultiModalEmbedsPayload | None
    """
    The video embeddings. It can be either:
    - A single base64 string representing a serialized torch tensor.
    - A dictionary of base64 tensors or numeric JSON metadata arrays.
    """
    type: Required[Literal["video_embeds"]]
    """The type of the content part."""
    uuid: str | None
    """
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    """


class ChatCompletionContentPartPromptEmbedsParam(TypedDict, total=False):
    data: Required[str]
    """
    Base64-encoded bytes of a serialized `torch.Tensor` of shape
    `(num_tokens, hidden_size)`. The tensor's `dtype` and `hidden_size` must
    match the model's input embedding layer.
    """
    type: Required[Literal["prompt_embeds"]]
    """The type of the content part."""


class CustomChatCompletionContentSimpleAudioParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a plain audio_url.

    Example:
    {
        "audio_url": "https://example.com/audio.mp3"
    }

    """

    audio_url: str | None


class CustomChatCompletionContentSimpleVideoParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a plain audio_url.

    Example:
    {
        "video_url": "https://example.com/video.mp4"
    }

    """

    video_url: str | None
    uuid: str | None
    """
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    """


class CustomChatCompletionContentToolReferenceParam(TypedDict, total=False):
    """A tool reference content param that only accepts a plain tool name.

    Example:
    {
        "name": "get_weather",
        "type": "tool_reference"
    }

    """

    name: str
    """The name of the tool being referenced."""

    type: Literal["tool_reference"]
    """The content type."""


class CustomThinkCompletionContentParam(TypedDict, total=False):
    """A Think Completion Content Param that accepts a plain text and a boolean.

    Example:
    {
        "thinking": "I am thinking about the answer",
        "closed": True,
        "type": "thinking"
    }

    """

    thinking: Required[str]
    """The thinking content."""

    closed: bool
    """Whether the thinking is closed."""

    type: Required[Literal["thinking"]]
    """The thinking type."""


ChatCompletionContentPartParam: TypeAlias = (
    OpenAIChatCompletionContentPartParam
    | ChatCompletionContentPartAudioParam
    | ChatCompletionContentPartInputAudioParam
    | ChatCompletionContentPartVideoParam
    | ChatCompletionContentPartRefusalParam
    | CustomChatCompletionContentPILImageParam
    | CustomChatCompletionContentSimpleImageParam
    | ChatCompletionContentPartImageEmbedsParam
    | ChatCompletionContentPartAudioEmbedsParam
    | ChatCompletionContentPartVideoEmbedsParam
    | ChatCompletionContentPartPromptEmbedsParam
    | CustomChatCompletionContentSimpleAudioParam
    | CustomChatCompletionContentSimpleVideoParam
    | CustomChatCompletionContentToolReferenceParam
    | CustomThinkCompletionContentParam
    | str
)


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""

    role: Required[str]
    """The role of the message's author."""

    content: str | list[ChatCompletionContentPartParam]
    """The contents of the message."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """

    tool_call_id: str | None
    """Tool call that this message is responding to."""

    tool_calls: list[ChatCompletionMessageToolCallParam] | None
    """The tool calls generated by the model, such as function calls."""

    reasoning: str | None
    """The reasoning content for interleaved thinking."""

    tools: list[ChatCompletionFunctionToolParam] | None
    """The tools for developer role."""

    task: str | None
    """Model-specific task marker. Currently passed through for DeepSeek V4."""


ChatCompletionMessageParam: TypeAlias = (
    OpenAIChatCompletionMessageParam
    | CustomChatCompletionMessageParam
    | OpenAIHarmonyMessage
)


# TODO: Make fields ReadOnly once mypy supports it
class ConversationMessage(TypedDict, total=False):
    role: Required[str]
    """The role of the message's author."""

    content: str | None | list[dict[str, str]]
    """The contents of the message"""

    tool_call_id: str | None
    """Tool call that this message is responding to."""

    name: str | None
    """The name of the function to call"""

    tool_calls: list[ChatCompletionMessageToolCallParam] | None
    """The tool calls generated by the model, such as function calls."""

    reasoning: str | None
    """The reasoning content for interleaved thinking."""

    reasoning_content: str | None
    """Deprecated: The reasoning content for interleaved thinking."""

    tools: list[ChatCompletionFunctionToolParam] | None
    """The tools for developer role."""

    task: str | None
    """Model-specific task marker. Currently passed through for DeepSeek V4."""


ModalityStr = Literal[
    "image",
    "audio",
    "video",
    "image_embeds",
    "audio_embeds",
    "video_embeds",
    "vision_chunk",
    "prompt_embeds",
]
