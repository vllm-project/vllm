# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for streaming derender.

Tests are split into two layers:

1. Unit tests (no server): covers ``_detokenize_delta`` correctness
   (chunked == one-shot) and ``derender_completion_stream`` /
   ``derender_chat_stream`` logic via a real tokenizer on a tiny model.

2. Integration tests (require a running render server): covers the full
   HTTP round-trip through the streaming endpoint.  Marked with
   ``@pytest.mark.asyncio`` and gated by the ``server`` / ``client``
   fixtures from the sibling ``test_derender.py``.
"""

import pytest
import pytest_asyncio

from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    DerenderStreamState,
    GenerateResponseStreamChoice,
    GenerateStreamResponse,
)

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_stream_chunk(
    token_ids: list[int],
    index: int = 0,
    finish_reason: str | None = None,
    request_id: str = "test-req",
    usage: dict | None = None,
) -> GenerateStreamResponse:
    """Build a GenerateStreamResponse SSE chunk."""
    from vllm.entrypoints.openai.engine.protocol import UsageInfo

    return GenerateStreamResponse(
        request_id=request_id,
        choices=[
            GenerateResponseStreamChoice(
                index=index,
                token_ids=token_ids,
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(**usage) if usage else None,
    )


def _make_usage_chunk(
    completion_tokens: int,
    prompt_tokens: int = 0,
    request_id: str = "test-req",
) -> GenerateStreamResponse:
    """Build a usage only final SSE chunk (empty choices)."""
    from vllm.entrypoints.openai.engine.protocol import UsageInfo

    return GenerateStreamResponse(
        request_id=request_id,
        choices=[],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Unit tests — no running server
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    """Load the tiny tokenizer used across unit tests."""
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(MODEL_NAME)


@pytest.fixture(scope="module")
def derenderer(tokenizer):
    """Construct a minimal OnlineDerenderer backed by a stub renderer."""
    from unittest.mock import MagicMock

    from vllm.renderers.online_derenderer import OnlineDerenderer

    renderer = MagicMock()
    renderer.get_tokenizer.return_value = tokenizer

    model_config = MagicMock()
    model_config.hf_config.model_type = "llama"
    model_config.model = MODEL_NAME

    return OnlineDerenderer(
        model_config=model_config,
        renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="string",
        trust_request_chat_template=False,
        enable_auto_tools=False,
        tool_parser=None,
        reasoning_parser=None,
    )


class TestDetokenizeDelta:
    """_detokenize_delta: chunked decode must equal one shot decode."""

    def _one_shot(self, tokenizer, token_ids: list[int]) -> str:
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    def _chunked(self, derenderer, tokenizer, chunks: list[list[int]]) -> str:
        state = DerenderStreamState()
        parts: list[str] = []
        for delta in chunks:
            text, state = derenderer._detokenize_delta(
                tokenizer, delta, state, skip_special_tokens=True
            )
            parts.append(text)
        return "".join(parts)

    def test_single_chunk(self, derenderer, tokenizer):
        """All tokens in one chunk == one shot decode."""
        token_ids = tokenizer.encode("Hello world")[:8]
        assert self._chunked(derenderer, tokenizer, [token_ids]) == self._one_shot(
            tokenizer, token_ids
        )

    def test_two_equal_chunks(self, derenderer, tokenizer):
        """Split in half and reassemble == one shot."""
        token_ids = tokenizer.encode("Hello world from streaming derender")[:12]
        mid = len(token_ids) // 2
        chunks = [token_ids[:mid], token_ids[mid:]]
        assert self._chunked(derenderer, tokenizer, chunks) == self._one_shot(
            tokenizer, token_ids
        )

    def test_single_token_per_chunk(self, derenderer, tokenizer):
        """One token per chunk (most granular streaming) == one shot."""
        token_ids = tokenizer.encode("incremental detokenization test")[:10]
        chunks = [[t] for t in token_ids]
        assert self._chunked(derenderer, tokenizer, chunks) == self._one_shot(
            tokenizer, token_ids
        )

    def test_empty_delta_passthrough(self, derenderer, tokenizer):
        """An empty delta (usage only chunk) emits empty string and preserves state."""
        token_ids = tokenizer.encode("Hello")[:4]
        state = DerenderStreamState(prior_token_ids=token_ids)
        text, new_state = derenderer._detokenize_delta(
            tokenizer, [], state, skip_special_tokens=True
        )
        assert text == ""
        assert new_state.prior_token_ids == token_ids

    def test_multibyte_char_split_across_chunks(self, derenderer, tokenizer):
        """A CJK/emoji char straddling chunk boundaries == one shot.

        Regression test for held back trailing incomplete UTF-8 byte
        sequences being dropped when the rebuild window marks them as
        already read (see #46159).
        """
        token_ids = tokenizer.encode("Hello ✅ world 日本語")[:16]
        chunks = [[t] for t in token_ids]
        assert self._chunked(derenderer, tokenizer, chunks) == self._one_shot(
            tokenizer, token_ids
        )

    def test_state_accumulates_token_ids(self, derenderer, tokenizer):
        """prior_token_ids grows correctly across multiple calls."""
        t1 = tokenizer.encode("Hello")[:2]
        t2 = tokenizer.encode(" world")[:2]
        state = DerenderStreamState()
        _, state = derenderer._detokenize_delta(tokenizer, t1, state)
        _, state = derenderer._detokenize_delta(tokenizer, t2, state)
        assert state.prior_token_ids == t1 + t2

    def test_n_independent_streams_same_result(self, derenderer, tokenizer):
        """N parallel streams with the same token sequence give the same text."""
        token_ids = tokenizer.encode("parallel streams")[:8]
        mid = len(token_ids) // 2

        results = []
        for _ in range(3):
            state = DerenderStreamState()
            text, state = derenderer._detokenize_delta(
                tokenizer, token_ids[:mid], state
            )
            text2, _ = derenderer._detokenize_delta(tokenizer, token_ids[mid:], state)
            results.append(text + text2)

        assert len(set(results)) == 1, "All independent streams must produce same text"
        assert results[0] == self._one_shot(tokenizer, token_ids)


class TestDerenderCompletionStream:
    """derender_completion_stream: streaming output parity with one shot."""

    @pytest.mark.asyncio
    async def test_chunked_equals_oneshot(self, derenderer, tokenizer):
        """Sum of streaming text chunks == one shot tokenizer.decode."""
        token_ids = tokenizer.encode("streaming completion test")[:10]
        mid = len(token_ids) // 2

        state = DerenderStreamState()
        chunk1, state = await derenderer.derender_completion_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(token_ids[:mid]),
            state=state,
        )
        chunk2, _ = await derenderer.derender_completion_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(token_ids[mid:], finish_reason="stop"),
            state=state,
        )

        streamed_text = chunk1.choices[0].text + chunk2.choices[0].text
        one_shot = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert streamed_text == one_shot

    @pytest.mark.asyncio
    async def test_usage_chunk_passthrough(self, derenderer, tokenizer):
        """Usage only final chunk (empty choices) is passed through correctly."""
        usage_chunk = _make_usage_chunk(completion_tokens=10, prompt_tokens=5)
        chunk, state = await derenderer.derender_completion_stream(
            model=MODEL_NAME,
            generate_chunk=usage_chunk,
        )
        assert chunk.choices == []
        assert chunk.usage is not None
        assert chunk.usage.completion_tokens == 10
        assert chunk.usage.prompt_tokens == 5

    @pytest.mark.asyncio
    async def test_prompt_tokens_in_usage(self, derenderer, tokenizer):
        """prompt_tokens is correctly forwarded into usage on a usage chunk."""
        token_ids = tokenizer.encode("hello")[:3]
        usage_chunk = _make_usage_chunk(
            completion_tokens=len(token_ids), prompt_tokens=7
        )
        chunk, _ = await derenderer.derender_completion_stream(
            model=MODEL_NAME,
            generate_chunk=usage_chunk,
            prompt_tokens=7,
        )
        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens == 7

    @pytest.mark.asyncio
    async def test_none_state_initialises_correctly(self, derenderer, tokenizer):
        """Passing state=None (first call) initialises an empty DerenderStreamState."""
        token_ids = tokenizer.encode("hello")[:4]
        _, state = await derenderer.derender_completion_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(token_ids),
            state=None,
        )
        assert state.prior_token_ids == token_ids

    @pytest.mark.asyncio
    async def test_finish_reason_forwarded(self, derenderer, tokenizer):
        """finish_reason from the generate chunk reaches the derendered choice."""
        token_ids = tokenizer.encode("done")[:2]
        chunk, _ = await derenderer.derender_completion_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(token_ids, finish_reason="length"),
        )
        assert chunk.choices[0].finish_reason == "length"


class TestDerenderChatStream:
    """derender_chat_stream: plain detok branch (no parser)."""

    @pytest.mark.asyncio
    async def test_role_on_first_chunk_only(self, derenderer, tokenizer):
        """role='assistant' appears in the first chunk, not subsequent ones."""
        token_ids = tokenizer.encode("hello world")[:6]
        mid = len(token_ids) // 2

        state = DerenderStreamState()
        chunk1, state = await derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(token_ids[:mid]),
            state=state,
        )
        chunk2, _ = await derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(token_ids[mid:], finish_reason="stop"),
            state=state,
        )

        assert chunk1.choices[0].delta.role == "assistant"
        assert chunk2.choices[0].delta.role is None

    @pytest.mark.asyncio
    async def test_chunked_equals_oneshot(self, derenderer, tokenizer):
        """Sum of streaming content deltas == one shot decode."""
        token_ids = tokenizer.encode("streaming chat derender text")[:10]
        mid = len(token_ids) // 2

        state = DerenderStreamState()
        chunk1, state = await derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(token_ids[:mid]),
            state=state,
        )
        chunk2, _ = await derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(token_ids[mid:]),
            state=state,
        )

        streamed = (chunk1.choices[0].delta.content or "") + (
            chunk2.choices[0].delta.content or ""
        )
        one_shot = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert streamed == one_shot

    @pytest.mark.asyncio
    async def test_parser_raises_not_implemented(self, tokenizer):
        """Stream chat with a parser active raises NotImplementedError (TODO)."""
        from unittest.mock import MagicMock

        from vllm.renderers.online_derenderer import OnlineDerenderer

        renderer = MagicMock()
        renderer.get_tokenizer.return_value = tokenizer

        model_config = MagicMock()
        model_config.hf_config.model_type = "llama"
        model_config.model = MODEL_NAME

        # Construct a derenderer WITH a tool/reasoning parser active.
        # ParserManager.get_parser returns None when no parser is named, so
        # we inject a mock parser class directly.
        dr = OnlineDerenderer(
            model_config=model_config,
            renderer=renderer,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="string",
        )
        dr.parser = MagicMock()  # simulate a parser being active

        chat_request = MagicMock()
        token_ids = tokenizer.encode("hello")[:3]

        with pytest.raises(NotImplementedError):
            await dr.derender_chat_stream(
                model=MODEL_NAME,
                generate_chunk=_make_stream_chunk(token_ids),
                state=None,
                chat_request=chat_request,
            )


# ---------------------------------------------------------------------------
# Integration tests — require a live render server
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    from tests.utils import RemoteLaunchRenderServer

    with RemoteLaunchRenderServer(MODEL_NAME, []) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    import httpx

    async with httpx.AsyncClient(
        base_url=server.url_for(""), timeout=30.0
    ) as http_client:
        yield http_client


async def _render_chat(client) -> dict:
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


@pytest.mark.asyncio
async def test_streaming_completion_derender_roundtrip(client):
    """Streaming completions derender: chunked text == non streaming text."""
    gen_req = await _render_chat(client)
    token_ids: list[int] = gen_req["token_ids"][:8]
    mid = len(token_ids) // 2
    chunk1_ids, chunk2_ids = token_ids[:mid], token_ids[mid:]

    # Non streaming baseline.
    non_stream_resp = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                {
                    "request_id": "test-ns",
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": token_ids,
                            "finish_reason": "stop",
                        }
                    ],
                }
            ],
        },
    )
    assert non_stream_resp.status_code == 200
    expected_text = non_stream_resp.json()["choices"][0]["text"]

    # Streaming call 1.
    r1 = await client.post(
        "/v1/completions/derender",
        json={
            "stream": True,
            "model": MODEL_NAME,
            "generate_chunk": {
                "request_id": "test-s",
                "choices": [
                    {"index": 0, "token_ids": chunk1_ids, "finish_reason": None}
                ],
            },
            "stream_state": None,
        },
    )
    assert r1.status_code == 200
    d1 = r1.json()
    text1 = d1["chunk"]["choices"][0]["text"]
    state1 = d1["stream_state"]

    # Streaming call 2 (final chunk).
    r2 = await client.post(
        "/v1/completions/derender",
        json={
            "stream": True,
            "model": MODEL_NAME,
            "generate_chunk": {
                "request_id": "test-s",
                "choices": [
                    {"index": 0, "token_ids": chunk2_ids, "finish_reason": "stop"}
                ],
            },
            "stream_state": state1,
        },
    )
    assert r2.status_code == 200
    text2 = r2.json()["chunk"]["choices"][0]["text"]

    assert text1 + text2 == expected_text


@pytest.mark.asyncio
async def test_streaming_chat_derender_roundtrip(client):
    """Streaming chat derender (plain detok): chunked text == non streaming text."""
    gen_req = await _render_chat(client)
    token_ids: list[int] = gen_req["token_ids"][:8]
    mid = len(token_ids) // 2
    chunk1_ids, chunk2_ids = token_ids[:mid], token_ids[mid:]

    # Non streaming baseline.
    ns = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": {
                "request_id": "test-ns",
                "choices": [
                    {
                        "index": 0,
                        "token_ids": token_ids,
                        "finish_reason": "stop",
                    }
                ],
            },
        },
    )
    assert ns.status_code == 200
    expected_content = ns.json()["choices"][0]["message"]["content"]

    # Streaming call 1.
    r1 = await client.post(
        "/v1/chat/completions/derender",
        json={
            "stream": True,
            "model": MODEL_NAME,
            "generate_chunk": {
                "request_id": "test-s",
                "choices": [
                    {"index": 0, "token_ids": chunk1_ids, "finish_reason": None}
                ],
            },
            "stream_state": None,
        },
    )
    assert r1.status_code == 200
    d1 = r1.json()
    text1 = d1["chunk"]["choices"][0]["delta"].get("content") or ""
    state1 = d1["stream_state"]
    # role=assistant on the first chunk
    assert d1["chunk"]["choices"][0]["delta"].get("role") == "assistant"

    # Streaming call 2.
    r2 = await client.post(
        "/v1/chat/completions/derender",
        json={
            "stream": True,
            "model": MODEL_NAME,
            "generate_chunk": {
                "request_id": "test-s",
                "choices": [
                    {"index": 0, "token_ids": chunk2_ids, "finish_reason": "stop"}
                ],
            },
            "stream_state": state1,
        },
    )
    assert r2.status_code == 200
    d2 = r2.json()
    text2 = d2["chunk"]["choices"][0]["delta"].get("content") or ""
    # role must NOT be repeated on subsequent chunks
    assert d2["chunk"]["choices"][0]["delta"].get("role") is None

    assert text1 + text2 == expected_content


@pytest.mark.asyncio
async def test_streaming_derender_invalid_body_returns_422(client):
    """Missing required field in streaming request returns 422."""
    r = await client.post(
        "/v1/completions/derender",
        json={
            "stream": True,
            # missing required 'model' and 'generate_chunk'
        },
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_streaming_derender_non_object_body_returns_422(client):
    """A non object JSON body (e.g. a list) returns 422, not a 500."""
    r = await client.post(
        "/v1/completions/derender",
        json=[1, 2, 3],
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_streaming_usage_chunk(client):
    """Usage only final chunk is forwarded with correct token counts."""
    gen_req = await _render_chat(client)
    token_ids: list[int] = gen_req["token_ids"][:6]
    state: dict = {}

    # Send content chunk first.
    r1 = await client.post(
        "/v1/completions/derender",
        json={
            "stream": True,
            "model": MODEL_NAME,
            "generate_chunk": {
                "request_id": "usage-test",
                "choices": [
                    {"index": 0, "token_ids": token_ids, "finish_reason": "stop"}
                ],
            },
            "stream_state": None,
        },
    )
    assert r1.status_code == 200
    state = r1.json()["stream_state"]

    # Send usage only final chunk.
    r2 = await client.post(
        "/v1/completions/derender",
        json={
            "stream": True,
            "model": MODEL_NAME,
            "generate_chunk": {
                "request_id": "usage-test",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": len(token_ids),
                    "total_tokens": 10 + len(token_ids),
                },
            },
            "stream_state": state,
            "prompt_tokens": 10,
        },
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["chunk"]["choices"] == []
    assert d2["chunk"]["usage"]["prompt_tokens"] == 10
    assert d2["chunk"]["usage"]["completion_tokens"] == len(token_ids)
