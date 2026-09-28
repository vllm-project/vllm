# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers shared by the streaming chat derender tests."""

from collections.abc import Callable

import httpx


def assemble_stream(choices: list[dict]) -> dict:
    """Fold streamed chat choices into one message.

    A tool call index must keep the first ID it was given. On the derender
    side that is what pinning the IDs in `stream_state` guarantees.
    """
    content = ""
    reasoning = ""
    tool_calls: list[dict] = []
    finish_reason = None
    for choice in choices:
        delta = choice["delta"]
        content += delta.get("content") or ""
        reasoning += delta.get("reasoning") or ""
        for tc in delta.get("tool_calls") or []:
            idx = tc["index"]
            while len(tool_calls) <= idx:
                tool_calls.append({"id": None, "name": None, "arguments": ""})
            if tc.get("id"):
                assert tool_calls[idx]["id"] in (None, tc["id"]), (
                    f"tool call {idx} changed ID mid stream"
                )
                tool_calls[idx]["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                tool_calls[idx]["name"] = fn["name"]
            tool_calls[idx]["arguments"] += fn.get("arguments") or ""
        finish_reason = choice.get("finish_reason") or finish_reason
    return {
        "content": content or None,
        "reasoning": reasoning or None,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
    }


async def stream_chat_derender(
    client: httpx.AsyncClient,
    output_ids: list[int],
    chunk_sizes: list[int],
    chat_request: dict,
    prompt_tokens: int,
    prompt_token_ids: list[int],
    finish_reason: str = "stop",
    on_chunk: Callable[[list[dict]], None] | None = None,
) -> dict:
    """Feed `output_ids` through the streaming chat derender endpoint in
    the given `chunk_sizes`, threading `stream_state` across calls and
    return the assembled message.

    `finish_reason` goes on the last chunk, so a trailing size of 0 sends it
    on a chunk with no tokens.

    If `on_chunk` is given, it is called after every chunk with the
    `tool_calls` assembled so far, letting callers assert properties of the
    intermediate deltas (e.g. monotonic argument growth) rather than only
    the final assembled result.
    """
    state = None
    choices: list[dict] = []
    pos = 0
    for i, size in enumerate(chunk_sizes):
        tids = output_ids[pos : pos + size]
        pos += size
        is_last = i == len(chunk_sizes) - 1
        resp = await client.post(
            "/v1/chat/completions/derender",
            json={
                "stream": True,
                "model": chat_request["model"],
                "generate_chunk": {
                    "request_id": "stream-test",
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": tids,
                            "finish_reason": finish_reason if is_last else None,
                        }
                    ],
                },
                "stream_state": state,
                "prompt_tokens": prompt_tokens,
                "prompt_token_ids": prompt_token_ids,
                "chat_request": chat_request,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        state = data["stream_state"]
        choices.extend(data["chunk"]["choices"])
        if on_chunk is not None:
            on_chunk(assemble_stream(choices)["tool_calls"])

    return assemble_stream(choices)
