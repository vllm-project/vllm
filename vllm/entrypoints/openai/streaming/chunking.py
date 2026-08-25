# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-frame prompt construction for the streaming server (model-independent).

A frame is rendered through the model's own chat template (via the engine's
renderer), so the per-model vision tokens come from the template — no hardcoded
markers. Each frame is one chat turn carrying an image: frame 1 opens the
conversation (system + question + image); later frames are image-only (the task
lives in the pinned system prompt). Per-chunk prompts are incremental — the
engine appends each to the running request.
"""

from io import BytesIO
from typing import TYPE_CHECKING, cast

from PIL import Image

from vllm.engine.protocol import StreamingInput
from vllm.renderers.params import ChatParams
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
    from vllm.renderers import BaseRenderer


# Cap decoded frame size well below PIL's ~179 MP decompression-bomb error:
# a crafted ~80 MP JPEG otherwise decodes to ~240 MB of RGB per frame.
_MAX_FRAME_PIXELS = 4 * 1024 * 1024  # 4 MP; a 720p frame is ~0.9 MP


def decode_frame(data: bytes) -> Image.Image:
    """Decode encoded image bytes (PNG/JPEG/...) to a PIL RGB image."""
    img = Image.open(BytesIO(data))  # lazy: parses the header only
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
) -> list[dict]:
    """OpenAI-style messages for one frame's incremental turn.

    Frame 1 opens the conversation (system + image + question); later frames are
    image-only. The image is passed in-memory (``image_pil``) so no URL/base64
    round-trip is needed.
    """
    image_part = {"type": "image_pil", "image_pil": frame}
    if not is_first:
        return [{"role": "user", "content": [image_part]}]
    user_content: list[dict] = [image_part]
    if question:
        user_content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


async def build_chunk(
    renderer: "BaseRenderer",
    frame: Image.Image,
    is_first: bool,
    question: str,
    system_prompt: str,
    sp: SamplingParams,
) -> StreamingInput:
    """Render one frame into a ``StreamingInput`` via the model's chat template.

    The full async chat pipeline is used so tokenization and the HF multimodal
    processing run off the event loop (async tokenizer + the renderer's mm
    executor) and the engine receives a typed ``EngineInput``, ready to append
    to the running request.
    """
    messages = cast(
        "list[ChatCompletionMessageParam]",
        build_messages(frame, system_prompt, question, is_first),
    )
    params = ChatParams(
        chat_template_content_format="auto",
        chat_template_kwargs={"add_generation_prompt": True},
    )
    _, eng_prompts = await renderer.render_chat_async([messages], params)
    return StreamingInput(prompt=eng_prompts[0], sampling_params=sp)
