# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import Counter
from collections.abc import Callable
from functools import partial
from typing import TypeAlias, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionToolMessageParam,
)
from openai.types.responses import ResponseInputImageParam
from pydantic import TypeAdapter

from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

from .types import (
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
    InputAudio,
    MultiModalEmbedsPayload,
    PILImage,
)

logger = init_logger(__name__)


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
_AssistantParser = partial(cast, ChatCompletionAssistantMessageParam)
_ToolParser = partial(cast, ChatCompletionToolMessageParam)

# Need to validate url objects
_ImageParser = TypeAdapter(ChatCompletionContentPartImageParam).validate_python
_AudioParser = TypeAdapter(ChatCompletionContentPartAudioParam).validate_python
_VideoParser = TypeAdapter(ChatCompletionContentPartVideoParam).validate_python

_ResponsesInputImageParser = TypeAdapter(ResponseInputImageParam).validate_python
ContentPart: TypeAlias = (
    MultiModalEmbedsPayload | dict[str, str] | InputAudio | PILImage
)

# Define a mapping from part types to their corresponding parsing functions.
MM_PARSER_MAP: dict[
    str,
    Callable[[ChatCompletionContentPartParam], ContentPart],
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


def _get_interleaved_text_prompt(
    placeholder_storage: dict[str, list], texts: list[str]
) -> str:
    for idx, elem in enumerate(texts):
        if elem in placeholder_storage:
            slots = placeholder_storage[elem]
            if not slots:
                raise VLLMValidationError(
                    f"Found more '{elem}' placeholders in input prompt than "
                    "actual multimodal data items."
                )
            texts[idx] = slots.pop(0)

    return "\n".join(texts)


# TODO: Let user specify how to insert multimodal tokens into prompt
# (similar to chat template)
def get_full_multimodal_text_prompt(
    placeholder_storage: dict[str, list],
    texts: list[str],
    interleave_strings: bool,
    multimodal_content_part_separator: str = "\n",
) -> str:
    """Combine multimodal prompts for a multimodal language model."""
    # flatten storage to make it looks like
    # {
    #   "<|image|>": 2,
    #   "<|audio|>": 1
    # }
    placeholder_counts = Counter(
        [v for elem in placeholder_storage.values() for v in elem]
    )

    if interleave_strings:
        text_prompt = _get_interleaved_text_prompt(placeholder_storage, texts)
    else:
        text_prompt = "\n".join(texts)

    # Pass interleaved text further in case the user used image placeholders
    # himself, but forgot to disable the 'interleave_strings' flag

    # Look through the text prompt to check for missing placeholders
    missing_placeholders: list[str] = []
    for placeholder in placeholder_counts:
        # For any existing placeholder in the text prompt, we leave it as is
        placeholder_counts[placeholder] -= text_prompt.count(placeholder)

        if placeholder_counts[placeholder] < 0:
            logger.error(
                "Placeholder count is negative! "
                "Ensure that the 'interleave_strings' flag is disabled "
                "(current value: %s) "
                "when manually placing image placeholders.",
                interleave_strings,
            )
            logger.debug("Input prompt: %s", text_prompt)
            raise VLLMValidationError(
                f"Found more '{placeholder}' placeholders in input prompt than "
                "actual multimodal data items.",
                parameter="messages",
            )

        missing_placeholders.extend([placeholder] * placeholder_counts[placeholder])

    # NOTE: Default behaviour: we always add missing placeholders
    # at the front of the prompt, if interleave_strings=False
    if text_prompt:
        return multimodal_content_part_separator.join(
            missing_placeholders + [text_prompt]
        )
    else:
        return multimodal_content_part_separator.join(missing_placeholders)
