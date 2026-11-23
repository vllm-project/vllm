# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render incremental image turns with the model's chat template."""

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from io import BytesIO
from typing import TYPE_CHECKING, cast

from PIL import Image

from vllm.engine.protocol import StreamingInput
from vllm.inputs.engine import MultiModalInput, TokensInput
from vllm.renderers.params import ChatParams
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.renderers import BaseRenderer
    from vllm.renderers.chat_utils import (
        ChatCompletionContentPartParam,
        ChatCompletionMessageParam,
    )


# Cap decoded frame size well below PIL's ~179 MP decompression-bomb error:
# a crafted ~80 MP JPEG otherwise decodes to ~240 MB of RGB per frame.
_MAX_FRAME_PIXELS = 4 * 1024 * 1024  # 4 MP; a 720p frame is ~0.9 MP


def decode_frame(data: bytes) -> Image.Image:
    """Decode encoded image bytes (PNG/JPEG/...) to a PIL RGB image."""
    with Image.open(BytesIO(data)) as img:
        w, h = img.size
        if w * h > _MAX_FRAME_PIXELS:
            raise ValueError(
                f"frame is {w}x{h} ({w * h} pixels); the per-frame limit is "
                f"{_MAX_FRAME_PIXELS} pixels"
            )
        return img.convert("RGB")


def build_messages(
    frame: Image.Image,
    system_prompt: str,
    question: str,
    is_first: bool,
) -> "list[ChatCompletionMessageParam]":
    """OpenAI-style messages for one frame's incremental turn.

    Frame 1 opens the conversation (system + image + question); later frames are
    image-only. The image is passed in-memory (`image_pil`) so no URL/base64
    round-trip is needed.
    """
    image_part = cast(
        "ChatCompletionContentPartParam", {"type": "image_pil", "image_pil": frame}
    )
    if not is_first:
        return [{"role": "user", "content": [image_part]}]
    user_content: list[ChatCompletionContentPartParam] = [image_part]
    if question:
        user_content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def trim_chunk_prefix(
    chunk: TokensInput | MultiModalInput,
    prefix: Sequence[int],
    replay_token: int | None,
    closing_token_ids: Sequence[int] = (),
) -> None:
    """Replace a per-render prefix with the prior assistant's closing suffix."""
    width = len(prefix)
    if chunk["prompt_token_ids"][:width] != list(prefix):
        raise ValueError("The chat template does not support incremental session turns")
    if chunk["type"] == "multimodal" and any(
        position.offset < width
        for positions in chunk["mm_placeholders"].values()
        for position in positions
    ):
        raise ValueError("A reference turn unexpectedly included image placeholders")
    leading = ([] if replay_token is None else [replay_token]) + list(closing_token_ids)
    chunk["prompt_token_ids"] = leading + chunk["prompt_token_ids"][width:]
    chunk.pop("prompt", None)
    if chunk["type"] == "multimodal":
        chunk["mm_placeholders"] = {
            modality: [
                replace(position, offset=position.offset - width + len(leading))
                for position in positions
            ]
            for modality, positions in chunk["mm_placeholders"].items()
        }
    else:
        chunk.pop("prompt_token_offsets", None)


async def build_chunk(
    renderer: "BaseRenderer",
    frame: Image.Image,
    is_first: bool,
    question: str,
    system_prompt: str,
    sp: SamplingParams,
    chat_params: ChatParams,
    replay_token: int | None = None,
) -> StreamingInput:
    """Render a frame into an engine input with replayable image data."""
    sp = sp.clone()
    sp.extra_args = {
        **(sp.extra_args or {}),
        "streaming_frame_replay_tokens": int(not is_first and replay_token is not None),
    }
    messages = build_messages(frame, system_prompt, question, is_first)
    # Re-prefill must be able to replay retained frames after GPU encoder eviction.
    # Processor-only caching keeps full tensors in every chunk's feature payload.
    if is_first:
        _, prompts = await renderer.render_chat_async(
            [messages], chat_params, skip_mm_cache=True
        )
        return StreamingInput(prompt=prompts[0], sampling_params=sp)

    chunk = await render_incremental_turn(renderer, messages, chat_params, replay_token)
    return StreamingInput(prompt=chunk, sampling_params=sp)


async def render_incremental_turn(
    renderer: "BaseRenderer",
    messages: "list[ChatCompletionMessageParam]",
    chat_params: ChatParams,
    replay_token: int | None = None,
) -> TokensInput | MultiModalInput:
    """Render a new turn, retaining the model's assistant-closing suffix."""
    previous, reply, current = "VLLM_PREVIOUS_TURN", "VLLM_REPLY", "VLLM_CURRENT_TURN"
    standalone: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": current}
    ]
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": previous},
        {"role": "assistant", "content": reply},
        *standalone,
    ]
    # Probe textual boundaries independently: some image templates expect user
    # content as a list but print assistant content directly as a string.
    text_params = replace(
        chat_params,
        chat_template_content_format="string",
        chat_template_kwargs={**chat_params.chat_template_kwargs, "tokenize": False},
    )
    (_, single_prompt), (_, full_prompt) = await asyncio.gather(
        renderer.render_messages_async(standalone, text_params),
        renderer.render_messages_async(conversation, text_params),
    )
    single_text = single_prompt.get("prompt")
    full_text = full_prompt.get("prompt")
    if (
        not isinstance(single_text, str)
        or not isinstance(full_text, str)
        or single_text.count(current) != 1
        or full_text.count(reply) != 1
        or full_text.count(current) != 1
    ):
        raise ValueError("The chat template does not support incremental session turns")
    tail = full_text.split(reply, 1)[1]
    common = 0
    for left, right in zip(reversed(single_text), reversed(tail)):
        if left != right:
            break
        common += 1
    if common == 0 or current not in single_text[-common:]:
        raise ValueError("The chat template rewrites earlier session turns")
    drop = single_text[: len(single_text) - common]
    closing = tail[: len(tail) - common]
    if previous in closing:
        raise ValueError("The chat template rewrites earlier session turns")
    tokenizer = renderer.get_tokenizer()
    _, (chunk,) = await renderer.render_chat_async(
        [messages], chat_params, skip_mm_cache=True
    )
    assert chunk["type"] in ("token", "multimodal")
    trim_chunk_prefix(
        chunk,
        tokenizer.encode(drop, add_special_tokens=False),
        replay_token,
        tokenizer.encode(closing, add_special_tokens=False),
    )
    return chunk
