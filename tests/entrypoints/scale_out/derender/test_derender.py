# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the /derender endpoints (postprocessing counterpart to /render)."""

from unittest.mock import MagicMock

import httpx
import pytest
import pytest_asyncio

from tests.utils import RemoteLaunchRenderServer
from vllm.entrypoints.generate.base.protocol import FunctionCall
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateTokensResponse,
    GenerateTokensStreamResponse,
    ParseContext,
)
from vllm.outputs import CompletionOutput
from vllm.renderers.online_derenderer import OnlineDerenderer
from vllm.tokenizers import get_tokenizer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def server():
    with RemoteLaunchRenderServer(MODEL_NAME, []) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with httpx.AsyncClient(
        base_url=server.url_for(""), timeout=30.0
    ) as http_client:
        yield http_client


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL_NAME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _render_chat(client: httpx.AsyncClient) -> dict:
    """Render a minimal chat request and return the GenerateRequest dict."""
    resp = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert resp.status_code == 200
    return resp.json()


def _make_generate_response(
    token_ids: list[int] | None,
    request_id: str = "chatcmpl-test-id",
    finish_reason: str = "stop",
    logprobs: dict | None = None,
    prompt_logprobs: list | None = None,
    kv_transfer_params: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    choice: dict = {
        "index": 0,
        "token_ids": token_ids,
        "finish_reason": finish_reason,
        "logprobs": logprobs,
    }
    return {
        "request_id": request_id,
        "choices": [choice],
        "prompt_logprobs": prompt_logprobs,
        "kv_transfer_params": kv_transfer_params,
        "metrics": metrics,
    }


def _make_generate_logprobs(token_id: int = 1234) -> dict:
    """A `GenerateLogProbs` payload: integer token ids, no token/bytes."""
    entry = {
        "token_id": token_id,
        "logprob": -1.0,
        "rank": 1,
        "top_logprobs": [{"token_id": token_id + 1, "logprob": -2.0, "rank": 2}],
    }
    return {"content": [entry]}


# ---------------------------------------------------------------------------
# Chat derender tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derender_chat_roundtrip(client):
    """Render then derender: decoded content should be a non-empty string."""
    gen_req = await _render_chat(client)
    # Use the first 5 rendered token IDs as synthetic "generated" tokens.
    synthetic_ids = gen_req["token_ids"][:5]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(synthetic_ids),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"]
    assert data["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_derender_chat_passes_stop_reason_through(client):
    """The coupled server reports the stop string or token id that ended
    generation; derender must carry the generate value through unchanged."""
    gen_req = await _render_chat(client)
    generate_response = _make_generate_response(gen_req["token_ids"][:3])
    generate_response["choices"][0]["stop_reason"] = "\n\n"

    response = await client.post(
        "/v1/chat/completions/derender",
        json={"model": MODEL_NAME, "generate_response": generate_response},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["stop_reason"] == "\n\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("return_token_ids", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_derender_chat_returns_requested_prompt_token_ids(
    client, return_token_ids, stream
):
    """Like the coupled endpoint, prompt_token_ids is returned only on request."""
    gen_req = await _render_chat(client)
    prompt_ids = gen_req["token_ids"]
    generate_response = _make_generate_response(prompt_ids[:3])
    generate_response["prompt_token_ids"] = prompt_ids
    chat_request = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello"}],
        "return_token_ids": return_token_ids,
    }
    body: dict = {"model": MODEL_NAME, "chat_request": chat_request}
    if stream:
        body["stream"] = True
        body["generate_chunk"] = generate_response
    else:
        body["generate_response"] = generate_response

    response = await client.post("/v1/chat/completions/derender", json=body)

    assert response.status_code == 200
    data = response.json()["chunk"] if stream else response.json()
    assert data.get("prompt_token_ids") == (prompt_ids if return_token_ids else None)


@pytest.mark.asyncio
async def test_derender_completion_passes_stop_reason_through(client):
    gen_req = await _render_completion(client, "Hello world")
    generate_response = _make_completion_generate_response(
        gen_req["token_ids"][:3], "gen-stop"
    )
    generate_response["choices"][0]["stop_reason"] = 151645

    response = await client.post(
        "/v1/completions/derender",
        json={"model": MODEL_NAME, "generate_responses": [generate_response]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["stop_reason"] == 151645


@pytest.mark.asyncio
async def test_derender_chat_usage(client):
    """Supplied prompt_tokens flows through into usage correctly."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(synthetic_ids),
            "prompt_tokens": 10,
        },
    )
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == len(synthetic_ids)
    assert usage["total_tokens"] == 10 + len(synthetic_ids)


@pytest.mark.asyncio
async def test_derender_chat_usage_default(client):
    """Omitting prompt_tokens gives usage.prompt_tokens == 0."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(synthetic_ids),
        },
    )
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 0


@pytest.mark.asyncio
async def test_derender_chat_logprobs(client):
    """Integer token ids are decoded to real strings by derender."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]
    token_id = synthetic_ids[0]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(
                synthetic_ids,
                logprobs=_make_generate_logprobs(token_id),
            ),
        },
    )
    assert response.status_code == 200
    data = response.json()
    logprobs = data["choices"][0]["logprobs"]
    assert logprobs is not None
    content = logprobs["content"]
    assert content is not None and len(content) == 1
    token_str = content[0]["token"]
    assert token_str and not token_str.startswith("token_id:"), (
        f"token id was not decoded: {token_str!r}"
    )


@pytest.mark.asyncio
async def test_derender_chat_logprobs_bytes(client):
    """Resolved logprob entries have bytes populated as list[int]."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]
    token_id = synthetic_ids[0]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(
                synthetic_ids,
                logprobs=_make_generate_logprobs(token_id),
            ),
        },
    )
    assert response.status_code == 200
    content = response.json()["choices"][0]["logprobs"]["content"]
    bytes_field = content[0]["bytes"]
    assert isinstance(bytes_field, list)
    assert len(bytes_field) > 0
    assert all(isinstance(b, int) for b in bytes_field)


@pytest.mark.asyncio
async def test_derender_chat_top_logprobs(client):
    """top_logprobs entries are decoded too."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]
    token_id = synthetic_ids[0]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(
                synthetic_ids,
                logprobs=_make_generate_logprobs(token_id),
            ),
        },
    )
    assert response.status_code == 200
    content = response.json()["choices"][0]["logprobs"]["content"]
    top = content[0]["top_logprobs"]
    assert len(top) == 1
    assert not top[0]["token"].startswith("token_id:"), (
        f"top_logprobs token id not decoded: {top[0]['token']!r}"
    )


@pytest.mark.asyncio
async def test_derender_chat_prompt_logprobs_passthrough(client):
    """prompt_logprobs on GenerateResponse passes through unchanged."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]
    # prompt_logprobs is a list[dict[int, Logprob] | None]; use None entries.
    prompt_logprobs = [None, None]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(
                synthetic_ids, prompt_logprobs=prompt_logprobs
            ),
        },
    )
    assert response.status_code == 200
    assert response.json()["prompt_logprobs"] == prompt_logprobs


@pytest.mark.asyncio
async def test_derender_chat_kv_transfer_params_passthrough(client):
    """kv_transfer_params passes through to the ChatCompletionResponse."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]
    kv = {"key": "value"}

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(
                synthetic_ids, kv_transfer_params=kv
            ),
        },
    )
    assert response.status_code == 200
    assert response.json()["kv_transfer_params"] == kv


@pytest.mark.asyncio
async def test_derender_chat_metrics_passthrough(client):
    gen_req = await _render_chat(client)
    metrics = {
        "speculative_decoding": {
            "mean_acceptance_length": 2.0,
            "draft_acceptance_rate": 0.5,
            "acceptance_histogram": [0, 1],
            "num_spec_steps": 1,
            "num_accepted_draft_tokens": 1,
            "num_draft_tokens": 2,
            "num_spec_tokens": 1,
        }
    }
    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(
                gen_req["token_ids"][:3], metrics=metrics
            ),
        },
    )
    assert response.status_code == 200
    actual = response.json()["metrics"]
    assert set(actual) == {
        "time_to_first_token_ms",
        "generation_time_ms",
        "queue_time_ms",
        "mean_itl_ms",
        "tokens_per_second",
        "speculative_decoding",
    }
    assert all(
        actual[name] is None for name in actual if name != "speculative_decoding"
    )
    assert actual["speculative_decoding"] == {
        **metrics["speculative_decoding"],
        "per_step_accepted": None,
        "per_step_drafted": None,
    }


@pytest.mark.asyncio
async def test_derender_chat_empty_token_ids(client):
    """Empty token_ids list returns 400."""
    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response([]),
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_derender_chat_null_token_ids(client):
    """Null token_ids returns 400."""
    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(None),
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_derender_chat_unknown_model(client):
    """Unknown model returns 404."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": "does-not-exist",
            "generate_response": _make_generate_response(synthetic_ids),
        },
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_derender_chat_model_omitted_resolves_served_name(client):
    """Omitting `model` resolves the served name rather than rejecting.

    Mirrors test_serving_chat.py's "full name is returned when no model is
    specified" assertion for the derender path. Asserts the resolved value,
    not just the status, so the fallback is proven to have fired.
    """
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={"generate_response": _make_generate_response(synthetic_ids)},
    )
    assert response.status_code == 200
    assert response.json()["model"] == MODEL_NAME


@pytest.mark.asyncio
async def test_derender_chat_leading_space_seeded_from_prompt(client, tokenizer):
    """prompt_token_ids keeps the first token's leading space on this
    Metaspace tokenizer, matching the coupled path."""
    gen_req = await _render_chat(client)
    prompt_token_ids = gen_req["token_ids"]
    output_ids = tokenizer.encode("Hello there, output", add_special_tokens=False)

    unseeded = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(output_ids),
        },
    )
    seeded = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(output_ids),
            "prompt_token_ids": prompt_token_ids,
        },
    )
    assert unseeded.status_code == 200
    assert seeded.status_code == 200
    unseeded_content = unseeded.json()["choices"][0]["message"]["content"]
    seeded_content = seeded.json()["choices"][0]["message"]["content"]

    assert not unseeded_content.startswith(" ")
    assert seeded_content.startswith(" ")
    assert seeded_content.lstrip(" ") == unseeded_content


@pytest.mark.asyncio
async def test_derender_chat_oversized_prompt_token_ids_rejected(client):
    """prompt_token_ids longer than max_model_len returns 400."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]
    oversized_prompt_ids = [42] * 1_000_000

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(synthetic_ids),
            "prompt_token_ids": oversized_prompt_ids,
        },
    )
    assert response.status_code == 400
    assert "max_model_len" in response.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Completion derender tests
# ---------------------------------------------------------------------------


async def _render_completion(client: httpx.AsyncClient, prompt: str) -> dict:
    """Render a completion prompt and return the first GenerateRequest dict."""
    resp = await client.post(
        "/v1/completions/render",
        json={"model": MODEL_NAME, "prompt": prompt},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list) and len(data) >= 1
    return data[0]


def _make_completion_generate_response(
    token_ids: list[int],
    request_id: str,
    kv_transfer_params: dict | None = None,
    logprobs: dict | None = None,
) -> dict:
    return {
        "request_id": request_id,
        "choices": [
            {
                "index": 0,
                "token_ids": token_ids,
                "finish_reason": "stop",
                "logprobs": logprobs,
            }
        ],
        "prompt_logprobs": None,
        "kv_transfer_params": kv_transfer_params,
    }


@pytest.mark.asyncio
async def test_derender_completion_roundtrip(client):
    """Two prompts rendered, two GenerateResponses → two choices with indices 0, 1."""
    gr1 = await _render_completion(client, "Hello world")
    gr2 = await _render_completion(client, "Goodbye world")

    ids1 = gr1["token_ids"][:4]
    ids2 = gr2["token_ids"][:4]

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(ids1, gr1["request_id"]),
                _make_completion_generate_response(ids2, gr2["request_id"]),
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    choices = data["choices"]
    assert len(choices) == 2
    assert choices[0]["index"] == 0
    assert choices[1]["index"] == 1
    assert choices[0]["text"]
    assert choices[1]["text"]


@pytest.mark.asyncio
async def test_derender_completion_usage_aggregation(client):
    """prompt_tokens=[5, 10] is aggregated correctly into usage."""
    gr1 = await _render_completion(client, "Hello")
    gr2 = await _render_completion(client, "World")

    ids1 = gr1["token_ids"][:3]
    ids2 = gr2["token_ids"][:4]

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(ids1, gr1["request_id"]),
                _make_completion_generate_response(ids2, gr2["request_id"]),
            ],
            "prompt_tokens": [5, 10],
        },
    )
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 15
    assert usage["completion_tokens"] == len(ids1) + len(ids2)
    assert usage["total_tokens"] == 15 + len(ids1) + len(ids2)


@pytest.mark.asyncio
async def test_derender_completion_prompt_tokens_length_mismatch(client):
    """len(prompt_tokens) != len(generate_responses) returns 400."""
    gr1 = await _render_completion(client, "Hello")
    ids1 = gr1["token_ids"][:3]

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(ids1, gr1["request_id"]),
            ],
            "prompt_tokens": [5, 10],
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_derender_completion_prompt_token_ids_length_mismatch(client):
    """len(prompt_token_ids) != len(generate_responses) returns 400, the
    same as the existing prompt_tokens length check."""
    gr1 = await _render_completion(client, "Hello")
    ids1 = gr1["token_ids"][:3]

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(ids1, gr1["request_id"]),
            ],
            "prompt_token_ids": [[1, 2], [3, 4]],
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_derender_completion_leading_space_seeded_from_prompt(client, tokenizer):
    """Completions counterpart of
    test_derender_chat_leading_space_seeded_from_prompt."""
    gr1 = await _render_completion(client, "Hello world")
    prompt_token_ids = gr1["token_ids"]
    output_ids = tokenizer.encode("Hello there, output", add_special_tokens=False)

    unseeded = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(output_ids, gr1["request_id"]),
            ],
        },
    )
    seeded = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(output_ids, gr1["request_id"]),
            ],
            "prompt_token_ids": [prompt_token_ids],
        },
    )
    assert unseeded.status_code == 200
    assert seeded.status_code == 200
    unseeded_text = unseeded.json()["choices"][0]["text"]
    seeded_text = seeded.json()["choices"][0]["text"]

    assert not unseeded_text.startswith(" ")
    assert seeded_text.startswith(" ")
    assert seeded_text.lstrip(" ") == unseeded_text


@pytest.mark.asyncio
async def test_derender_completion_oversized_prompt_token_ids_rejected(client):
    """A prompt_token_ids entry longer than max_model_len returns 400."""
    gr1 = await _render_completion(client, "Hello")
    ids1 = gr1["token_ids"][:3]
    oversized_prompt_ids = [42] * 1_000_000

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(ids1, gr1["request_id"]),
            ],
            "prompt_token_ids": [oversized_prompt_ids],
        },
    )
    assert response.status_code == 400
    assert "max_model_len" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_derender_completion_empty_generate_responses(client):
    """Empty generate_responses list returns 400."""
    response = await client.post(
        "/v1/completions/derender",
        json={"model": MODEL_NAME, "generate_responses": []},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_derender_completion_logprobs(client):
    """Integer token ids are decoded; CompletionLogProbs flat-list structure is
    returned with non-empty tokens and text_offsets."""
    gr1 = await _render_completion(client, "Hello world")
    ids1 = gr1["token_ids"][:3]
    token_id = ids1[0]

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(
                    ids1,
                    gr1["request_id"],
                    logprobs=_make_generate_logprobs(token_id),
                ),
            ],
        },
    )
    assert response.status_code == 200
    logprobs = response.json()["choices"][0]["logprobs"]
    assert logprobs is not None
    tokens = logprobs["tokens"]
    assert len(tokens) == 1
    assert not tokens[0].startswith("token_id:"), (
        f"token id was not decoded: {tokens[0]!r}"
    )
    assert len(logprobs["token_logprobs"]) == 1
    assert isinstance(logprobs["token_logprobs"][0], float)
    assert len(logprobs["text_offset"]) == 1
    assert logprobs["text_offset"][0] == 0


@pytest.mark.asyncio
async def test_derender_completion_kv_transfer_params_passthrough(client):
    """kv_transfer_params passes through to CompletionResponse."""
    gr1 = await _render_completion(client, "Hello")
    ids1 = gr1["token_ids"][:3]
    kv = {"node": "abc"}

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(
                    ids1, gr1["request_id"], kv_transfer_params=kv
                ),
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["kv_transfer_params"] == kv


# ---------------------------------------------------------------------------
# Resource bounds regression tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derender_chat_bounded_payload_succeeds(client):
    """Normal bounded derender payload succeeds (positive control)."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:5]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(synthetic_ids),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_derender_chat_oversized_token_ids_rejected(client):
    """token_ids longer than max_model_len returns 400."""
    response = await client.get("/v1/models")
    assert response.status_code == 200

    # Use a token_ids list that exceeds any reasonable max_model_len.
    # The tiny-random model has max_model_len of 2048.
    oversized_ids = [42] * 1_000_000

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(oversized_ids),
        },
    )
    assert response.status_code == 400
    assert "max_model_len" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_derender_chat_too_many_choices_rejected(client):
    """Choices count exceeding VLLM_MAX_N_SEQUENCES returns 400."""
    # Default VLLM_MAX_N_SEQUENCES is 16384; use a larger count.
    oversized_choices = [
        {"index": i, "token_ids": [42], "finish_reason": "stop"} for i in range(20_000)
    ]
    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": {
                "request_id": "test-choices-bound",
                "choices": oversized_choices,
            },
        },
    )
    assert response.status_code == 400
    assert "choices count" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_derender_completion_too_many_generate_responses_rejected(client):
    """generate_responses count exceeding limit returns 400."""
    oversized_responses = [
        {
            "request_id": f"gen-{i}",
            "choices": [{"index": 0, "token_ids": [42], "finish_reason": "stop"}],
        }
        for i in range(20_000)
    ]
    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": oversized_responses,
        },
    )
    assert response.status_code == 400
    assert "generate_responses count" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_derender_chat_negative_token_ids_rejected(client):
    """Negative token_ids are rejected at the protocol validation level."""
    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response([-1, 42, 100]),
        },
    )
    # vLLM's validation_exception_handler converts Pydantic errors to 400
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_derender_chat_oversized_logprobs_rejected(client):
    """logprobs.content longer than max_model_len returns 400."""
    oversized_logprobs: dict = {
        "content": [
            {"token_id": 42, "logprob": -1.0, "rank": 1, "top_logprobs": []}
            for _ in range(1_000_000)
        ]
    }
    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": {
                "request_id": "test-logprobs-bound",
                "choices": [
                    {
                        "index": 0,
                        "token_ids": [42],
                        "finish_reason": "stop",
                        "logprobs": oversized_logprobs,
                    }
                ],
            },
        },
    )
    assert response.status_code == 400
    assert "logprobs.content length" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_derender_chat_oversized_top_logprobs_rejected(client):
    """top_logprobs count exceeding max_logprobs (default 20) returns 400."""
    oversized_top_logprobs = {
        "content": [
            {
                "token_id": 42,
                "logprob": -1.0,
                "rank": 1,
                "top_logprobs": [
                    {"token_id": i, "logprob": -float(i), "rank": i + 1}
                    for i in range(25)
                ],
            }
        ]
    }
    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": {
                "request_id": "test-top-logprobs-bound",
                "choices": [
                    {
                        "index": 0,
                        "token_ids": [42],
                        "finish_reason": "stop",
                        "logprobs": oversized_top_logprobs,
                    }
                ],
            },
        },
    )
    assert response.status_code == 400
    msg = response.json()["error"]["message"]
    assert "top_logprobs count" in msg
    assert "max_logprobs" in msg


@pytest.mark.asyncio
async def test_derender_completion_oversized_token_ids_rejected(client):
    """Completion endpoint also rejects oversized token_ids."""
    oversized_ids = [42] * 1_000_000
    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                {
                    "request_id": "gen-0",
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": oversized_ids,
                            "finish_reason": "stop",
                        }
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400
    assert "max_model_len" in response.json()["error"]["message"]


# ---------------------------------------------------------------------------
# E2E: render -> derender roundtrip with parser (reasoning + tool calls)
# ---------------------------------------------------------------------------

PARSER_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

_E2E_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]


@pytest.fixture(scope="module")
def parser_server():
    args = [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--reasoning-parser",
        "deepseek_r1",
    ]
    with RemoteLaunchRenderServer(PARSER_MODEL, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def parser_client(parser_server):
    async with httpx.AsyncClient(
        base_url=parser_server.url_for(""), timeout=60.0
    ) as http_client:
        yield http_client


@pytest.fixture(scope="module")
def parser_tokenizer():
    return get_tokenizer(PARSER_MODEL)


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _decoded(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _require_markers_survive(tokenizer, text: str, *markers: str) -> list[int]:
    """Encode text and skip the test if any marker is lost in roundtrip."""
    ids = _encode(tokenizer, text)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    for m in markers:
        if m not in decoded:
            pytest.skip(f"Marker {m!r} lost in encode->decode roundtrip")
    return ids


async def _e2e_render_chat(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict],
) -> dict:
    resp = await client.post(
        "/v1/chat/completions/render",
        json={"model": model, "messages": messages},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _e2e_generate_response(
    token_ids: list[int],
    request_id: str = "chatcmpl-e2e-test",
) -> dict:
    return {
        "request_id": request_id,
        "choices": [
            {
                "index": 0,
                "token_ids": token_ids,
                "finish_reason": "stop",
            }
        ],
    }


@pytest.mark.asyncio
async def test_e2e_plain_roundtrip(parser_client, parser_tokenizer):
    """Plain text without reasoning markers roundtrips correctly.

    Markerless output has no ``</think>``, which deepseek_r1 classifies
    wholly as reasoning, so the text lands there rather than in content.
    What this pins is detokenization fidelity through the parser path.
    """
    messages = [{"role": "user", "content": "What is 2+2?"}]
    gen_req = await _e2e_render_chat(parser_client, PARSER_MODEL, messages)

    answer = "The answer is four."
    output_ids = _encode(parser_tokenizer, answer)
    expected = _decoded(parser_tokenizer, output_ids)

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {"model": PARSER_MODEL, "messages": messages},
        },
    )
    assert resp.status_code == 200, resp.text
    message = resp.json()["choices"][0]["message"]
    assert message["reasoning"] == expected
    assert message["content"] is None


@pytest.mark.asyncio
async def test_e2e_token_identity(parser_client, parser_tokenizer):
    """encode(derender(token_ids)) == token_ids (RL invariant).

    Markerless output comes back as reasoning (see
    ``test_e2e_plain_roundtrip``), so re-encode that.
    """
    messages = [{"role": "user", "content": "Hi"}]
    gen_req = await _e2e_render_chat(parser_client, PARSER_MODEL, messages)

    answer = "Hello! How can I help?"
    output_ids = _encode(parser_tokenizer, answer)

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {"model": PARSER_MODEL, "messages": messages},
        },
    )
    assert resp.status_code == 200
    reasoning = resp.json()["choices"][0]["message"]["reasoning"]
    re_encoded = _encode(parser_tokenizer, reasoning)
    assert output_ids == re_encoded


@pytest.mark.asyncio
async def test_e2e_non_ascii_roundtrip(parser_client, parser_tokenizer):
    """CJK + emoji roundtrip without U+FFFD.

    Markerless output comes back as reasoning (see
    ``test_e2e_plain_roundtrip``).
    """
    messages = [{"role": "user", "content": "Reply in Chinese"}]
    gen_req = await _e2e_render_chat(parser_client, PARSER_MODEL, messages)

    answer = "你好世界 😀"
    output_ids = _encode(parser_tokenizer, answer)

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {"model": PARSER_MODEL, "messages": messages},
        },
    )
    assert resp.status_code == 200
    reasoning = resp.json()["choices"][0]["message"]["reasoning"]
    assert "�" not in reasoning


@pytest.mark.asyncio
async def test_e2e_parsed_reasoning(parser_client, parser_tokenizer):
    """<think>...</think> splits into reasoning + content."""
    messages = [{"role": "user", "content": "What is 2+3?"}]
    gen_req = await _e2e_render_chat(parser_client, PARSER_MODEL, messages)

    reasoning_text = "The user wants 2 plus 3. That is 5."
    answer_text = "The answer is 5."
    output_text = f"<think>{reasoning_text}</think>{answer_text}"
    output_ids = _require_markers_survive(parser_tokenizer, output_text, "</think>")

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {
                "model": PARSER_MODEL,
                "messages": messages,
                "include_reasoning": True,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    msg = resp.json()["choices"][0]["message"]
    assert msg["reasoning"] is not None
    assert reasoning_text in msg["reasoning"]
    assert answer_text in msg["content"]
    assert "<think>" not in msg["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("include_reasoning", [True, False])
async def test_e2e_parsed_logprobs_follow_include_reasoning(
    parser_client, parser_tokenizer, include_reasoning
):
    """With a parser configured, batch derender returns logprobs only when
    reasoning is shown. Hidden reasoning drops them, as the coupled chat
    endpoint does, because decoded logprob tokens would reveal it."""
    messages = [{"role": "user", "content": "What is 2+3?"}]
    gen_req = await _e2e_render_chat(parser_client, PARSER_MODEL, messages)
    output_ids = _require_markers_survive(
        parser_tokenizer, "<think>Two plus three.</think>Five.", "</think>"
    )
    generate_response = _e2e_generate_response(output_ids)
    generate_response["choices"][0]["logprobs"] = {
        "content": [
            {"token_id": tid, "logprob": -0.5, "rank": 1, "top_logprobs": []}
            for tid in output_ids
        ]
    }
    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": generate_response,
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {
                "model": PARSER_MODEL,
                "messages": messages,
                "include_reasoning": include_reasoning,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    assert choice["message"]["content"] == "Five."
    if include_reasoning:
        assert choice["message"]["reasoning"]
        assert len(choice["logprobs"]["content"]) == len(output_ids)
    else:
        assert choice["message"]["reasoning"] is None
        assert choice["logprobs"] is None


@pytest.mark.asyncio
async def test_e2e_parsed_tool_call(parser_client, parser_tokenizer):
    """<tool_call> extracted into tool_calls field."""
    messages = [{"role": "user", "content": "Weather in Paris?"}]
    gen_req = await _e2e_render_chat(parser_client, PARSER_MODEL, messages)

    output_text = (
        "<think>Let me check the weather.</think>"
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Paris"}}\n</tool_call>'
    )
    output_ids = _require_markers_survive(
        parser_tokenizer,
        output_text,
        "</think>",
        "<tool_call>",
        "</tool_call>",
    )

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {
                "model": PARSER_MODEL,
                "messages": messages,
                "tools": _E2E_TOOLS,
                "tool_choice": "auto",
            },
        },
    )
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    assert choice["message"]["tool_calls"]
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_e2e_parsed_reasoning_and_tool_call(parser_client, parser_tokenizer):
    """Reasoning + tool call in the same output."""
    messages = [{"role": "user", "content": "Weather in Paris?"}]
    gen_req = await _e2e_render_chat(parser_client, PARSER_MODEL, messages)

    reasoning_text = "I should look up the weather."
    tool_text = (
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Paris"}}\n</tool_call>'
    )
    output_text = f"<think>{reasoning_text}</think>{tool_text}"
    output_ids = _require_markers_survive(
        parser_tokenizer, output_text, "</think>", "<tool_call>"
    )

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {
                "model": PARSER_MODEL,
                "messages": messages,
                "tools": _E2E_TOOLS,
                "tool_choice": "auto",
                "include_reasoning": True,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    assert choice["message"]["reasoning"] is not None
    assert reasoning_text in choice["message"]["reasoning"]
    assert choice["message"]["tool_calls"]


@pytest.mark.asyncio
async def test_e2e_no_chat_request_rejected(parser_client, parser_tokenizer):
    """Without chat_request a parser configured model rejects with 400
    rather than silently falling back to plain detokenization. This is to
    prevent the leak of raw reasoning/tool markup into content."""
    messages = [{"role": "user", "content": "Hello"}]
    gen_req = await _e2e_render_chat(parser_client, PARSER_MODEL, messages)

    answer = "Hi there!"
    output_ids = _encode(parser_tokenizer, answer)

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
        },
    )
    assert resp.status_code == 400
    assert "chat_request" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# E2E: HarmonyParser + GPT-OSS
# ---------------------------------------------------------------------------

HARMONY_MODEL = "openai/gpt-oss-20b"


def _ensure_harmony_vocab():
    """Pre-cache the o200k_base BPE file needed by openai-harmony.

    The Rust tiktoken-rs backend downloads from Azure Blob Storage, which
    may be unreachable in some environments.  When the cache is cold we
    fetch the file ourselves and place it in ``/tmp/tiktoken-rs-cache/``
    using the SHA-1(URL) filename that tiktoken-rs expects.
    """
    import hashlib
    import urllib.request
    from pathlib import Path

    url = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
    cache_dir = Path("/tmp/tiktoken-rs-cache")
    cache_key = hashlib.sha1(url.encode()).hexdigest()
    cache_file = cache_dir / cache_key
    if not cache_file.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, cache_file)


@pytest.fixture(scope="module")
def harmony_server():
    _ensure_harmony_vocab()
    args = [
        "--trust-remote-code",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "openai",
        "--reasoning-parser",
        "openai_gptoss",
    ]
    with RemoteLaunchRenderServer(HARMONY_MODEL, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def harmony_client(harmony_server):
    async with httpx.AsyncClient(
        base_url=harmony_server.url_for(""), timeout=60.0
    ) as http_client:
        yield http_client


@pytest.fixture(scope="module")
def harmony_tokenizer():
    return get_tokenizer(HARMONY_MODEL, trust_remote_code=True)


def _harmony_extract_assistant_ids(
    tokenizer, assistant_msg: dict, user_content: str = "test"
) -> list[int]:
    """Extract assistant token IDs via apply_chat_template diff."""
    prompt = [{"role": "user", "content": user_content}]
    full = prompt + [assistant_msg]
    text_prompt = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=False
    )
    text_full = tokenizer.apply_chat_template(
        full, add_generation_prompt=False, tokenize=False
    )
    prompt_ids = tokenizer.encode(text_prompt)
    full_ids = tokenizer.encode(text_full)
    assistant_ids = list(full_ids[len(prompt_ids) :])
    if not assistant_ids:
        pytest.skip("Could not extract assistant tokens for Harmony")
    return assistant_ids


@pytest.mark.asyncio
async def test_e2e_harmony_plain_roundtrip(harmony_client, harmony_tokenizer):
    """GPT-OSS content-only roundtrip."""
    messages = [{"role": "user", "content": "What is 2+2?"}]
    gen_req = await _e2e_render_chat(harmony_client, HARMONY_MODEL, messages)

    assistant_msg = {"role": "assistant", "content": "Four."}
    output_ids = _harmony_extract_assistant_ids(harmony_tokenizer, assistant_msg)

    resp = await harmony_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": HARMONY_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {
                "model": HARMONY_MODEL,
                "messages": messages,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    content = resp.json()["choices"][0]["message"]["content"]
    assert content is not None and len(content) > 0
    assert "Four" in content


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_choice", ["auto", "none"])
async def test_e2e_harmony_tool_choice_none_drops_tool_calls(
    harmony_client, harmony_tokenizer, tool_choice, stream
):
    """HarmonyParser extracts tool calls whatever the tool_choice, so
    /derender must drop them for "none", like coupled chat, batch and
    streaming alike. "auto" checks the tokens really parse as a tool call."""
    messages = [{"role": "user", "content": "Weather in Paris?"}]
    gen_req = await _e2e_render_chat(harmony_client, HARMONY_MODEL, messages)

    assistant_msg = {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                },
            }
        ],
    }
    output_ids = _harmony_extract_assistant_ids(harmony_tokenizer, assistant_msg)
    if "get_weather" not in harmony_tokenizer.decode(output_ids):
        pytest.skip("Harmony template did not render the tool call")

    body: dict = {
        "model": HARMONY_MODEL,
        "prompt_tokens": len(gen_req["token_ids"]),
        "prompt_token_ids": gen_req["token_ids"],
        "chat_request": {
            "model": HARMONY_MODEL,
            "messages": messages,
            "tools": _E2E_TOOLS,
            "tool_choice": tool_choice,
        },
    }
    if stream:
        body["stream"] = True
        body["generate_chunk"] = _e2e_generate_response(output_ids)
    else:
        body["generate_response"] = _e2e_generate_response(output_ids)

    resp = await harmony_client.post("/v1/chat/completions/derender", json=body)
    assert resp.status_code == 200, resp.text
    if stream:
        choice = resp.json()["chunk"]["choices"][0]
        tool_calls = choice["delta"].get("tool_calls")
    else:
        choice = resp.json()["choices"][0]
        tool_calls = choice["message"].get("tool_calls")
    assert bool(tool_calls) == (tool_choice == "auto")
    assert choice["finish_reason"] == (
        "tool_calls" if tool_choice == "auto" else "stop"
    )


@pytest.fixture(scope="module")
def harmony_default_server():
    """gpt-oss render server with NO parser flags.

    Regression fixture: the model-default reasoning parser
    ("openai_gptoss", applied by verify_and_update_config) must be
    resolved by the render server without any --reasoning-parser flag,
    otherwise harmony markup leaks unparsed into derender output.
    """
    _ensure_harmony_vocab()
    with RemoteLaunchRenderServer(
        HARMONY_MODEL, ["--trust-remote-code"]
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def harmony_default_client(harmony_default_server):
    async with httpx.AsyncClient(
        base_url=harmony_default_server.url_for(""), timeout=60.0
    ) as http_client:
        yield http_client


@pytest.mark.asyncio
async def test_e2e_harmony_reasoning_default_parser(
    harmony_default_client, harmony_tokenizer
):
    """GPT-OSS reasoning parses on a bare launch (no parser flags)."""
    messages = [{"role": "user", "content": "Add 2 and 3."}]
    gen_req = await _e2e_render_chat(harmony_default_client, HARMONY_MODEL, messages)

    reasoning_text = "The user wants 2 plus 3."
    answer_text = "The answer is 5."
    assistant_msg = {
        "role": "assistant",
        "thinking": reasoning_text,
        "content": answer_text,
    }
    output_ids = _harmony_extract_assistant_ids(harmony_tokenizer, assistant_msg)

    decoded = harmony_tokenizer.decode(output_ids)
    if reasoning_text not in decoded:
        pytest.skip("Harmony template did not render thinking")

    resp = await harmony_default_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": HARMONY_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {
                "model": HARMONY_MODEL,
                "messages": messages,
                "include_reasoning": True,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    msg = resp.json()["choices"][0]["message"]
    assert msg["reasoning"] is not None
    assert reasoning_text in msg["reasoning"]
    assert answer_text in (msg["content"] or "")
    assert "<|channel|>" not in (msg["content"] or "")


@pytest.mark.asyncio
async def test_e2e_harmony_reasoning(harmony_client, harmony_tokenizer):
    """GPT-OSS reasoning: analysis channel extracted."""
    messages = [{"role": "user", "content": "Add 2 and 3."}]
    gen_req = await _e2e_render_chat(harmony_client, HARMONY_MODEL, messages)

    reasoning_text = "The user wants 2 plus 3."
    answer_text = "The answer is 5."
    assistant_msg = {
        "role": "assistant",
        "thinking": reasoning_text,
        "content": answer_text,
    }
    output_ids = _harmony_extract_assistant_ids(harmony_tokenizer, assistant_msg)

    decoded = harmony_tokenizer.decode(output_ids)
    if reasoning_text not in decoded:
        pytest.skip("Harmony template did not render thinking")

    resp = await harmony_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": HARMONY_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {
                "model": HARMONY_MODEL,
                "messages": messages,
                "include_reasoning": True,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    msg = resp.json()["choices"][0]["message"]
    assert msg["reasoning"] is not None
    assert reasoning_text in msg["reasoning"]
    assert answer_text in (msg["content"] or "")


_HERMES_TOOL_CALL_TEXT = (
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
)


def _derender_hermes_chat(
    text: str,
    tool_choice: str | None = "auto",
    engine_finish_reason: str = "stop",
    parser_cls: type | None = None,
):
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = MagicMock()
    renderer.get_tokenizer.return_value = tokenizer
    model_config = MagicMock()
    model_config.model = MODEL_NAME
    model_config.hf_config.model_type = "llama"
    derenderer = OnlineDerenderer(
        model_config,
        renderer,
        request_logger=None,
        chat_template="{{ messages }}",
        chat_template_content_format="auto",
        enable_auto_tools=True,
        tool_parser="hermes",
    )
    if parser_cls is not None:
        derenderer.parser = parser_cls
    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        tools=_E2E_TOOLS,
        tool_choice=tool_choice,
    )
    response = GenerateTokensResponse.model_validate(
        _make_generate_response(
            tokenizer.encode(text, add_special_tokens=False),
            finish_reason=engine_finish_reason,
        )
    )

    (choice,) = derenderer._derender_chat(response, request)
    return choice


@pytest.mark.parametrize(
    ("tool_choice", "text", "engine_finish_reason", "expected_finish_reason"),
    [
        ("auto", _HERMES_TOOL_CALL_TEXT, "stop", "tool_calls"),
        ("auto", _HERMES_TOOL_CALL_TEXT, "length", "length"),
        ("required", "Hello there", "stop", "stop"),
    ],
)
def test_derender_chat_finish_reason_matches_chat_completion(
    tool_choice, text, engine_finish_reason, expected_finish_reason
):
    """/derender applies the same finish_reason rule as the coupled chat
    path: ``tool_calls`` only for a real call at a natural stop."""
    choice = _derender_hermes_chat(text, tool_choice, engine_finish_reason)

    assert choice.finish_reason == expected_finish_reason
    assert bool(choice.message.tool_calls) == (text == _HERMES_TOOL_CALL_TEXT)


def test_derender_chat_tool_call_id_matches_chat_completion():
    """Batch /derender uses the coupled chat's default tool call ID format."""
    choice = _derender_hermes_chat(_HERMES_TOOL_CALL_TEXT)

    (tool_call,) = choice.message.tool_calls
    assert tool_call.id.startswith("chatcmpl-tool-")


def test_derender_chat_keeps_parser_tool_call_id():
    """Batch /derender keeps the parser's tool call ID (Kimi K2 style
    ``functions.<name>:<idx>``), since chat templates reference it in
    multi-turn history."""

    class _KimiIdParser:
        tool_parser_cls = object

        def __init__(self, *args, **kwargs):
            pass

        def parse(self, model_output, request, **kwargs):
            call = FunctionCall(
                id="functions.get_weather:1",
                name="get_weather",
                arguments='{"city": "Paris"}',
            )
            return None, None, [call]

    choice = _derender_hermes_chat(_HERMES_TOOL_CALL_TEXT, parser_cls=_KimiIdParser)

    (tool_call,) = choice.message.tool_calls
    assert tool_call.id == "functions.get_weather:1"


@pytest.mark.parametrize("use_stub_parser", [False, True])
def test_derender_chat_drops_tool_calls_without_tool_choice(use_stub_parser):
    """/derender drops parsed tool calls for ``tool_choice=None`` and for a
    parser that ignores ``tool_choice="none"`` (Harmony), like coupled chat."""

    class _IgnoresToolChoiceParser:
        tool_parser_cls = object

        def __init__(self, *args, **kwargs):
            pass

        def parse(self, model_output, request, **kwargs):
            call = FunctionCall(name="get_weather", arguments='{"city": "Paris"}')
            return None, None, [call]

    choice = _derender_hermes_chat(
        _HERMES_TOOL_CALL_TEXT,
        tool_choice="none" if use_stub_parser else None,
        parser_cls=_IgnoresToolChoiceParser if use_stub_parser else None,
    )

    assert not choice.message.tool_calls
    assert choice.finish_reason == "stop"


def _make_derenderer(tokenizer, generation_config: dict | None = None, **kwargs):
    renderer = MagicMock()
    renderer.get_tokenizer.return_value = tokenizer
    renderer.get_eos_token_id.return_value = tokenizer.eos_token_id
    renderer._executor = None
    model_config = MagicMock()
    model_config.model = MODEL_NAME
    model_config.hf_config.model_type = "llama"
    model_config.try_get_generation_config.return_value = generation_config or {}
    return OnlineDerenderer(
        model_config,
        renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="string",
        **kwargs,
    )


_STOP_TEXT = "one two three four"


@pytest.mark.parametrize("endpoint", ["chat", "completion"])
@pytest.mark.parametrize(
    ("last", "finish_reason", "stop_reason", "request_kwargs", "text", "reason"),
    [
        # Current generate hosts report the matched stop string.
        (None, "stop", "three", {}, "one two ", "three"),
        (None, "stop", "three", {"include_stop_str_in_output": True},
         "one two three", "three"),
        # Older hosts report nothing, so the request's stop list is scanned.
        (None, "stop", None, {"stop": ["three", "two"]}, "one ", "two"),
        (None, "stop", None, {"stop": "two", "min_tokens": 2}, _STOP_TEXT, None),
        (None, "length", None, {"stop": "two"}, _STOP_TEXT, None),
        # A reported stop token means no stop string matched.
        ("stop_id", "stop", "stop_id", {"stop": "two"}, _STOP_TEXT, "stop_id"),
        # Unreported stop tokens. Only the primary EOS keeps stop_reason None.
        ("eos", "stop", None, {}, _STOP_TEXT, None),
        ("stop_id", "stop", None, {"stop_token_ids": True}, _STOP_TEXT, "stop_id"),
        ("config_eos", "stop", None, {}, _STOP_TEXT, "config_eos"),
        ("config_eos", "stop", None, {"ignore_eos": True}, _STOP_TEXT + " ?", None),
        # The stop token is dropped first, then a stop string still wins.
        ("eos", "stop", None, {"stop": "four"}, "one two three ", "four"),
    ],
)  # fmt: skip
def test_derender_batch_trims_stop_like_engine(
    tokenizer, endpoint, last, finish_reason, stop_reason, request_kwargs, text, reason
):
    """Generate returns every token, so batch derender must cut the stop
    token and stop string from the text and report stop_reason the way the
    engine's detokenizer and check_stop do. "stop_id" is a client stop token
    ("!") and "config_eos" a generation_config EOS id ("?")."""
    ids = {
        "eos": tokenizer.eos_token_id,
        "stop_id": tokenizer.encode("!", add_special_tokens=False)[-1],
        "config_eos": tokenizer.encode("?", add_special_tokens=False)[-1],
    }
    stop_reason = ids.get(stop_reason, stop_reason)
    reason = ids.get(reason, reason)
    if request_kwargs.get("stop_token_ids"):
        request_kwargs = {**request_kwargs, "stop_token_ids": [ids["stop_id"]]}
    derenderer = _make_derenderer(
        tokenizer, {"eos_token_id": [tokenizer.eos_token_id, ids["config_eos"]]}
    )
    token_ids = tokenizer.encode(_STOP_TEXT, add_special_tokens=False)
    if last is not None:
        token_ids.append(ids[last])
    payload = _make_generate_response(token_ids, finish_reason=finish_reason)
    payload["choices"][0]["stop_reason"] = stop_reason
    response = GenerateTokensResponse.model_validate(payload)
    # Keep special tokens so a leaked EOS would show up in the text.
    request_kwargs = {"skip_special_tokens": False, **request_kwargs}

    if endpoint == "chat":
        request = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Count"}],
            **request_kwargs,
        )
        (choice,) = derenderer._derender_chat(response, request)
        output_text = choice.message.content
    else:
        request = CompletionRequest(model=MODEL_NAME, prompt="Count", **request_kwargs)
        (choice,), _, _ = derenderer._derender_completion([response], None, request)
        output_text = choice.text

    assert output_text == text
    assert choice.stop_reason == reason


@pytest.mark.parametrize("tool_parser", [None, "hermes"])
@pytest.mark.parametrize(
    ("add_generation_prompt", "echo", "role", "echoed"),
    [
        (True, True, "bot", ""),
        (False, False, "assistant", ""),
        (False, True, "assistant", "1, 2,"),
    ],
)
def test_derender_chat_role_and_echo_match_chat_completion(
    tokenizer, tool_parser, add_generation_prompt, echo, role, echoed
):
    """Batch /derender takes the role from `--response-role` or, when the
    last message is continued, from that message, and prepends it on
    `echo`, as `/v1/chat/completions` does."""
    derenderer = _make_derenderer(
        tokenizer,
        response_role="bot",
        tool_parser=tool_parser,
        enable_auto_tools=tool_parser is not None,
    )
    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "Count to five."},
            {"role": "assistant", "content": "1, 2,"},
        ],
        add_generation_prompt=add_generation_prompt,
        continue_final_message=not add_generation_prompt,
        echo=echo,
    )
    response = GenerateTokensResponse.model_validate(
        _make_generate_response(
            tokenizer.encode(" 3, 4, 5", add_special_tokens=False),
            finish_reason="length",
        )
    )

    (choice,) = derenderer._derender_chat(response, request)
    (plain,) = derenderer._derender_chat(
        response, request.model_copy(update={"echo": False})
    )

    assert choice.message.role == role
    assert choice.message.content == echoed + plain.message.content


def test_generate_derender_role_is_response_role(tokenizer):
    """Generate `output_mode: "derender"` returns `--response-role`."""
    derenderer = _make_derenderer(tokenizer, response_role="bot")
    output = CompletionOutput(
        index=0,
        text="Hello",
        token_ids=[1],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    )

    ((message, _),) = derenderer._derender_text_choices(
        [output], ParseContext().to_chat_request(), 0, None
    )

    assert message.role == "bot"


@pytest.mark.asyncio
async def test_derender_completion_stream_leaves_out_stop_token(tokenizer):
    """Streaming completions drops the final stop token like chat does and
    reports the stop_reason an older generate host leaves out."""
    derenderer = _make_derenderer(tokenizer)
    bang = tokenizer.encode("!", add_special_tokens=False)[-1]
    payload = _make_generate_response(
        tokenizer.encode(_STOP_TEXT, add_special_tokens=False) + [bang]
    )
    response, _ = await derenderer.derender_completion_stream(
        MODEL_NAME,
        GenerateTokensStreamResponse.model_validate(payload),
        completion_request=CompletionRequest(
            model=MODEL_NAME, prompt="Count", stop_token_ids=[bang]
        ),
    )
    assert response.choices[0].text == _STOP_TEXT
    assert response.choices[0].stop_reason == bang


@pytest.mark.asyncio
@pytest.mark.parametrize("return_token_ids", [None, False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_derender_completion_preserves_requested_token_ids(
    tokenizer, return_token_ids, stream
):
    derenderer = _make_derenderer(tokenizer)
    token_ids = tokenizer.encode("Hello world", add_special_tokens=False)
    request = CompletionRequest(
        model=MODEL_NAME, prompt="Hi", return_token_ids=return_token_ids
    )
    prompt_ids = tokenizer.encode("Hi")
    payload = _make_generate_response(token_ids, finish_reason="length")
    payload["prompt_token_ids"] = prompt_ids
    if stream:
        response, _ = await derenderer.derender_completion_stream(
            MODEL_NAME,
            GenerateTokensStreamResponse.model_validate(payload),
            completion_request=request,
        )
        choices = response.choices
    else:
        choices, _, _ = await derenderer.derender_completion(
            [GenerateTokensResponse.model_validate(payload)],
            completion_request=request,
        )
    assert choices[0].token_ids == (token_ids if return_token_ids else None)
    assert choices[0].prompt_token_ids == (prompt_ids if return_token_ids else None)
