# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import httpx
import pytest
import regex as re

from tests.entrypoints.scale_out.derender.utils import stream_chat_derender
from tests.utils import RemoteLaunchRenderServer, RemoteOpenAIServer
from vllm.tokenizers import get_tokenizer

MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"
START, END = "<|tool_call_start|>", "<|tool_call_end|>"
BODY = " ".join(["line"] * 128)
BODY_DELTAS = [f'{START}[f(text="line'] + [" line"] * 127


@pytest.fixture(scope="module")
def server():
    args = ["--enable-auto-tool-choice", "--tool-call-parser", "lfm2"]
    with RemoteLaunchRenderServer(MODEL, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish", "deltas", "arguments"),
    [
        ("stop", BODY_DELTAS + [f'")]{END}'], json.dumps({"text": BODY})),
        ("length", BODY_DELTAS, json.dumps({"text": BODY})[:-2]),
        (
            "stop",
            [f'{START}[f(text="a', "=bc:", f'def", text="ax")]{END}'],
            '{"text": "ax"}',
        ),
    ],
    ids=["complete", "cutoff", "withheld_before_overwrite"],
)
async def test_lfm2_string_stream(
    server, finish: str, deltas: list[str], arguments: str
):
    """Replay fixed tokens through a render server, including a cutoff.

    Derender uses the serving stream processor without relying on a model
    generating a particular argument or reproducing particular chunk boundaries.
    """
    tokenizer = get_tokenizer(MODEL)
    chunks = [tokenizer.encode(delta, add_special_tokens=False) for delta in deltas]
    ids = [token for chunk in chunks for token in chunk]
    assert tokenizer.decode(ids) == "".join(deltas)
    request = {
        "model": MODEL,
        "messages": [],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "skip_special_tokens": False,
    }
    async with httpx.AsyncClient(base_url=server.url_for(""), timeout=60) as client:
        result = await stream_chat_derender(
            client,
            ids,
            [len(chunk) for chunk in chunks],
            request,
            prompt_tokens=0,
            prompt_token_ids=[],
            finish_reason=finish,
        )
    assert result["content"] is None
    assert result["reasoning"] is None
    assert result["finish_reason"] == ("tool_calls" if finish == "stop" else "length")
    assert len(result["tool_calls"]) == 1
    call = result["tool_calls"][0]
    assert call["name"] == "f"
    assert call["arguments"] == arguments


def test_lfm2_chat_stream_preserves_null_fields():
    """Inspect raw SSE: derender and SDK defaults hide unset versus null."""
    output = f'{START}[f(text="{BODY}")]{END}'
    args = [
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "lfm2",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.1",
        "--stream-interval",
        "1",
    ]
    request = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Call f with the text."}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 1024,
        "stream": True,
        "structured_outputs": {"regex": re.escape(output, literal_spaces=True)},
    }
    arguments = ""
    continuations = 0
    with (
        RemoteOpenAIServer(MODEL, args, max_wait_seconds=480) as chat_server,
        httpx.stream(
            "POST",
            chat_server.url_for("v1/chat/completions"),
            json=request,
            timeout=60,
        ) as response,
    ):
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line.removeprefix("data: "))
            assert "error" not in chunk, chunk
            for choice in chunk["choices"]:
                delta = choice["delta"]
                for call in delta.get("tool_calls", []):
                    function = call["function"]
                    arguments += function.get("arguments") or ""
                    # The regex fixes the text, not how the model splits it.
                    if "name" not in function:
                        continuations += 1
                        assert delta["content"] is None
                        assert call["id"] is None
    assert continuations > 0
    assert arguments == json.dumps({"text": BODY})
