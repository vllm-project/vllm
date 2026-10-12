# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stream obfuscation for SSE responses.

TLS hides the content of a streamed response but not the size of each record.
With one token per chunk, a passive network observer can recover every token's
length and use the sequence to infer the topic of a conversation (Whisper Leak,
https://arxiv.org/abs/2511.03675). ``with_sse_obfuscation`` wraps the final SSE
generator and adds an ``obfuscation`` field of random characters to each JSON
``data:`` payload, padding it to a multiple of ``OBFUSCATION_BLOCK_SIZE`` bytes.
This implements OpenAI's ``stream_options.include_obfuscation``.

Rounding sizes up to a block boundary, rather than padding by a random amount,
reduces what a chunk's size reveals to a single bit: whether its payload
crosses the next boundary. The block is large relative to typical token lengths
so that this is rare. The padding is random so that HTTP compression cannot
strip it.
"""

import secrets
from collections.abc import AsyncGenerator

OBFUSCATION_BLOCK_SIZE = 128
_FIELD_PREFIX = '"obfuscation":"'


def should_obfuscate(requested: bool | None, server_default: bool) -> bool:
    """Resolve a request's ``include_obfuscation`` against the server default."""
    return server_default if requested is None else requested


def with_sse_obfuscation(
    generator: AsyncGenerator[str, None],
    enabled: bool,
    event_suffix: str | None = None,
) -> AsyncGenerator[str, None]:
    """Pad the JSON payloads of ``generator`` when ``enabled``.

    A disabled call returns ``generator`` unchanged. If ``event_suffix`` is set,
    only SSE events whose ``event:`` name ends with it are padded.
    """
    if not enabled:
        return generator
    return _obfuscated_stream(generator, event_suffix)


async def _obfuscated_stream(
    generator: AsyncGenerator[str, None],
    event_suffix: str | None,
) -> AsyncGenerator[str, None]:
    try:
        async for chunk in generator:
            yield obfuscate_sse_chunk(chunk, event_suffix)
    finally:
        await generator.aclose()


def obfuscate_sse_chunk(chunk: str, event_suffix: str | None = None) -> str:
    """Pad every JSON ``data:`` payload in ``chunk``, which may hold several
    events. Comments, ``[DONE]`` and non-JSON payloads pass through."""
    if "data: {" not in chunk:
        return chunk
    return "\n\n".join(
        _obfuscate_event(event, event_suffix) for event in chunk.split("\n\n")
    )


def _obfuscate_event(event: str, event_suffix: str | None) -> str:
    lines = event.split("\n")
    if event_suffix is not None and not any(
        line.startswith("event: ") and line.endswith(event_suffix) for line in lines
    ):
        return event
    for i, line in enumerate(lines):
        if line.startswith("data: {") and line.endswith("}"):
            lines[i] = "data: " + _pad_json_object(line[len("data: ") :])
    return "\n".join(lines)


def _pad_json_object(payload: str) -> str:
    prefix = _FIELD_PREFIX if payload == "{}" else "," + _FIELD_PREFIX
    # The payload's closing brace is reused; the field adds its prefix and a
    # closing quote.
    size = len(payload.encode()) + len(prefix) + 1
    pad_len = -size % OBFUSCATION_BLOCK_SIZE
    # URL-safe base64 needs no JSON escaping, so each character is one byte.
    padding = secrets.token_urlsafe(pad_len)[:pad_len]
    return f'{payload[:-1]}{prefix}{padding}"}}'
