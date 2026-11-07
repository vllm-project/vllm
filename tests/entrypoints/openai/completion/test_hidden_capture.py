# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import pybase64 as base64
import pytest
import torch
from openai import BadRequestError

from tests.utils import RemoteOpenAIServer

MODEL_NAME = os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def capture_server():
    args = [
        "--max-model-len",
        "256",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.35",
        "--enforce-eager",
        "--no-async-scheduling",
        "--aux-output-config",
        '{"enable_hidden_states": true}',
    ]
    with RemoteOpenAIServer(
        MODEL_NAME, args, env_dict={"VLLM_USE_V2_MODEL_RUNNER": "1"}
    ) as remote_server:
        yield remote_server


@pytest.mark.asyncio
async def test_windowed_hidden_capture_http(capture_server):
    """Capture preserves generated tokens and aligns raw logits with hidden rows."""
    async with capture_server.get_async_client() as client:
        common: dict[str, Any] = dict(
            model=MODEL_NAME,
            prompt="Explain GPUs briefly.",
            max_tokens=8,
            temperature=0,
            extra_body={"ignore_eos": True, "return_token_ids": True},
        )
        plain = await client.completions.create(**common)
        common["extra_body"]["hidden_state_capture"] = {
            "start": 1,
            "end": 5,
            "min_rows": 4,
            "collection_id": "http-test",
            "logprob_topk": 3,
        }
        captured = await client.completions.create(**common)
        choice = captured.choices[0]
        assert choice.token_ids == plain.choices[0].token_ids
        result = choice.hidden_state_capture
        assert choice.hidden_capture_skip_reason is None
        assert result["collection_id"] == "http-test"
        assert result["hidden_positions"] == list(
            range(plain.usage.prompt_tokens, plain.usage.prompt_tokens + 4)
        )

        def decode(payload):
            return torch.frombuffer(
                bytearray(base64.b64decode(payload["data"])),
                dtype=getattr(torch, payload["dtype"]),
            ).reshape(payload["shape"])

        tensors = result["tensors"]
        hidden = decode(tensors["hidden_states"])
        logprobs = decode(tensors["logprobs"])
        token_ids = decode(tensors["logprob_token_ids"])
        assert result["segments"] == [
            {
                "kind": "output",
                "start": plain.usage.prompt_tokens,
                "end": plain.usage.prompt_tokens + 4,
            }
        ]
        assert result["logits_contract"]["vocab_size"] > 0
        assert hidden.shape[0] == 4
        assert logprobs.shape == token_ids.shape == (4, 3)
        assert torch.isfinite(hidden).all()
        assert torch.isfinite(logprobs).all()
        assert (logprobs <= 0).all()
        assert (logprobs[:, :-1] >= logprobs[:, 1:]).all()
        assert token_ids.dtype == torch.int64
        common["extra_body"]["hidden_state_capture"]["start"] = 20
        common["extra_body"]["hidden_state_capture"]["end"] = 24
        missed = await client.completions.create(**common)
        assert missed.choices[0].hidden_state_capture is None
        assert missed.choices[0].hidden_capture_skip_reason == "ended_before_window"
        common["extra_body"]["hidden_state_capture"]["end"] = 20
        with pytest.raises(BadRequestError):
            await client.completions.create(**common)
        common["extra_body"]["hidden_state_capture"]["end"] = 24
        with pytest.raises(BadRequestError):
            await client.completions.create(**common, stream=True)


@pytest.mark.asyncio
async def test_prompt_segment_hidden_capture_http(capture_server):
    """Teacher scoring returns every scored prompt row, even on a cache hit."""
    async with capture_server.get_async_client() as client:
        plain = await client.completions.create(
            model=MODEL_NAME,
            prompt="Explain GPUs briefly.",
            max_tokens=8,
            temperature=0,
            extra_body={"ignore_eos": True, "return_token_ids": True},
        )
        prompt_ids = list(plain.choices[0].prompt_token_ids)
        response_ids = list(plain.choices[0].token_ids)
        start = len(prompt_ids) - 1
        end = start + len(response_ids)
        scoring: dict[str, Any] = dict(
            model=MODEL_NAME,
            prompt=prompt_ids + response_ids,
            max_tokens=1,
            temperature=0,
            extra_body={
                "hidden_state_capture": {
                    "start": start,
                    "end": end,
                    "coordinate": "absolute",
                    "min_rows": end - start,
                    "collection_id": "teacher",
                }
            },
        )
        results = []
        for _ in range(2):
            choice = (await client.completions.create(**scoring)).choices[0]
            assert choice.hidden_capture_skip_reason is None
            results.append(choice.hidden_state_capture)
        for result in results:
            assert result["hidden_positions"] == list(range(start, end))
            assert result["segments"] == [
                {"kind": "prompt", "start": start, "end": end}
            ]
            assert set(result["tensors"]) == {"hidden_states"}
            assert result["tensors"]["hidden_states"]["shape"][0] == end - start
        scoring["extra_body"]["hidden_state_capture"]["logprob_topk"] = 2
        with pytest.raises(BadRequestError):
            await client.completions.create(**scoring)
