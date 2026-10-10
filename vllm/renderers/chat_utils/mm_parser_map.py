# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from ctypes import cast
from functools import partial
from typing import TypeAlias

from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
from openai.types.responses import ResponseInputImageParam
from pydantic import TypeAdapter

from vllm.renderers.chat_utils.types import (
    ChatCompletionContentPartAudioEmbedsParam,
    ChatCompletionContentPartAudioParam,
    ChatCompletionContentPartImageEmbedsParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartPromptEmbedsParam,
    ChatCompletionContentPartVideoEmbedsParam,
    ChatCompletionContentPartVideoParam,
    CustomChatCompletionContentPILImageParam,
    CustomChatCompletionContentToolReferenceParam,
    CustomThinkCompletionContentParam,
    MultiModalEmbedsPayload,
    PILImage,
)

# No need to validate using Pydantic again
_TextParser = partial(cast, ChatCompletionContentPartTextParam)
_ImageEmbedsParser = partial(cast, ChatCompletionContentPartImageEmbedsParam)
_AudioEmbedsParser = partial(cast, ChatCompletionContentPartAudioEmbedsParam)
_VideoEmbedsParser = partial(cast, ChatCompletionContentPartVideoEmbedsParam)
_PromptEmbedsParser = partial(cast, ChatCompletionContentPartPromptEmbedsParam)
_InputAudioParser = partial(cast, ChatCompletionContentPartInputAudioParam)
_RefusalParser = partial(cast, ChatCompletionContentPartRefusalParam)
_PILImageParser = partial(cast, CustomChatCompletionContentPILImageParam)
_ThinkParser = partial(cast, CustomThinkCompletionContentParam)
# Need to validate url objects
_ImageParser = TypeAdapter(ChatCompletionContentPartImageParam).validate_python
_AudioParser = TypeAdapter(ChatCompletionContentPartAudioParam).validate_python
_VideoParser = TypeAdapter(ChatCompletionContentPartVideoParam).validate_python

_ResponsesInputImageParser = TypeAdapter(ResponseInputImageParam).validate_python
_ContentPart: TypeAlias = (
    MultiModalEmbedsPayload | dict[str, str] | InputAudio | PILImage
)

# Define a mapping from part types to their corresponding parsing functions.
MM_PARSER_MAP: dict[
    str,
    Callable[[ChatCompletionContentPartParam], _ContentPart],
] = {
    "text": lambda part: _TextParser(part).get("text", None),
    "thinking": lambda part: _ThinkParser(part).get("thinking", None),
    "input_text": lambda part: _TextParser(part).get("text", None),
    "output_text": lambda part: _TextParser(part).get("text", None),
    "input_image": lambda part: _ResponsesInputImageParser(part).get("image_url", None),
    "image_url": lambda part: _ImageParser(part).get("image_url", {}).get("url", None),
    "image_embeds": lambda part: _ImageEmbedsParser(part).get("image_embeds", None),
    "audio_embeds": lambda part: _AudioEmbedsParser(part).get("audio_embeds", None),
    "video_embeds": lambda part: _VideoEmbedsParser(part).get("video_embeds", None),
    "prompt_embeds": lambda part: _PromptEmbedsParser(part).get("data", None),
    "image_pil": lambda part: _PILImageParser(part).get("image_pil", None),
    "audio_url": lambda part: _AudioParser(part).get("audio_url", {}).get("url", None),
    "input_audio": lambda part: _InputAudioParser(part).get("input_audio", None),
    "refusal": lambda part: _RefusalParser(part).get("refusal", None),
    "video_url": lambda part: _VideoParser(part).get("video_url", {}).get("url", None),
    "tool_reference": lambda part: cast(
        CustomChatCompletionContentToolReferenceParam, part
    ).get("name", None),
}
