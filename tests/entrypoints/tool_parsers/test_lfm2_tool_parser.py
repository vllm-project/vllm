# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import httpx
import pytest

from tests.entrypoints.scale_out.derender.utils import stream_chat_derender
from tests.utils import RemoteLaunchRenderServer
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
