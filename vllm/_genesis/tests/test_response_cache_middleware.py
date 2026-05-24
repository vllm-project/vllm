# SPDX-License-Identifier: Apache-2.0
"""TDD tests for P50 — ResponseCacheMiddleware.

Covers:
- Key extraction correctness for `/v1/completions` and
  `/v1/chat/completions`
- `stream=True` → not cache-eligible (returns None key)
- `temperature > 0` → not cache-eligible unless `allow_sampled=True`
- Missing model / missing prompt → None
- Malformed body → None
- Full middleware flow (mock ASGI):
  * cache miss → forward to app, store on 2xx response
  * cache hit → short-circuit return 200, NO app invocation
  * non-eligible path → pass-through
  * non-POST → pass-through
  * malformed JSON → pass-through without crash
  * downstream 500 → no cache store

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import asyncio
import json



# ── key extraction ─────────────────────────────────────────────────

class TestKeyExtraction:
    def test_completions_simple_string_prompt(self):
        from vllm._genesis.middleware import build_cache_key_from_request
        body = {"model": "m", "prompt": "hello"}
        out = build_cache_key_from_request(body)
        assert out is not None
        prompt, model, _sp = out
        assert prompt == "hello"
        assert model == "m"

    def test_completions_list_prompt_single(self):
        from vllm._genesis.middleware import build_cache_key_from_request
        out = build_cache_key_from_request(
            {"model": "m", "prompt": ["hi"]},
        )
        assert out is not None
        assert out[0] == "hi"

    def test_completions_list_prompt_multiple(self):
        from vllm._genesis.middleware import build_cache_key_from_request
        out = build_cache_key_from_request(
            {"model": "m", "prompt": ["a", "b"]},
        )
        assert out is not None
        # Serialised as JSON
        assert "[" in out[0]

    def test_chat_completions_messages_deterministic(self):
        from vllm._genesis.middleware import build_cache_key_from_request
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }
        out = build_cache_key_from_request(body)
        assert out is not None
        # Same body different key ordering → same serialised prompt
        body2 = {
            "messages": [{"content": "hi", "role": "user"}],
            "model": "m",
        }
        out2 = build_cache_key_from_request(body2)
        assert out[0] == out2[0]

    def test_stream_true_not_eligible(self):
        from vllm._genesis.middleware import build_cache_key_from_request
        out = build_cache_key_from_request(
            {"model": "m", "prompt": "p", "stream": True},
        )
        assert out is None

    def test_sampled_not_eligible_by_default(self):
        from vllm._genesis.middleware import build_cache_key_from_request
        out = build_cache_key_from_request(
            {"model": "m", "prompt": "p", "temperature": 0.7},
        )
        assert out is None

    def test_sampled_allowed_when_flag_true(self):
        from vllm._genesis.middleware import build_cache_key_from_request
        out = build_cache_key_from_request(
            {"model": "m", "prompt": "p", "temperature": 0.7},
            allow_sampled=True,
        )
        assert out is not None

    def test_top_p_less_than_1_not_eligible(self):
        from vllm._genesis.middleware import build_cache_key_from_request
        out = build_cache_key_from_request(
            {"model": "m", "prompt": "p", "top_p": 0.9},
        )
        assert out is None

    def test_top_k_greater_than_1_not_eligible(self):
        from vllm._genesis.middleware import build_cache_key_from_request
        out = build_cache_key_from_request(
            {"model": "m", "prompt": "p", "top_k": 40},
        )
        assert out is None

    def test_missing_model_not_eligible(self):
        from vllm._genesis.middleware import build_cache_key_from_request
        assert build_cache_key_from_request(
            {"prompt": "p"},
        ) is None

    def test_missing_prompt_and_messages_not_eligible(self):
        from vllm._genesis.middleware import build_cache_key_from_request
        assert build_cache_key_from_request({"model": "m"}) is None

    def test_non_dict_body_not_eligible(self):
        from vllm._genesis.middleware import build_cache_key_from_request
        assert build_cache_key_from_request("string") is None
        assert build_cache_key_from_request(None) is None

    def test_sampling_params_included(self):
        from vllm._genesis.middleware import build_cache_key_from_request
        out = build_cache_key_from_request(
            {
                "model": "m", "prompt": "p",
                "max_tokens": 100,
                "stop": ["\n\n"],
                "seed": 42,
            },
        )
        assert out is not None
        _, _, sp = out
        assert sp["max_tokens"] == 100
        assert sp["stop"] == ["\n\n"]
        assert sp["seed"] == 42


# ── ASGI middleware flow ───────────────────────────────────────────

class _FakeCache:
    """Minimal `ResponseCacheLRU`-compatible mock for tests."""
    def __init__(self):
        self.store_ = {}
        self.get_calls = 0
        self.store_calls = 0

    def get(self, prompt, model, params):
        self.get_calls += 1
        # Use a stable key — json because params is a dict
        k = (prompt, model, json.dumps(params, sort_keys=True))
        return self.store_.get(k)

    def store(self, prompt, model, params, response):
        self.store_calls += 1
        k = (prompt, model, json.dumps(params, sort_keys=True))
        self.store_[k] = response


class _CollectingSend:
    def __init__(self):
        self.messages = []

    async def __call__(self, message):
        self.messages.append(message)


def _make_downstream_app(status: int = 200, body: bytes = None):
    """Build a tiny fake ASGI app that returns a fixed status+body."""
    if body is None:
        body = b'{"choices":[{"text":"generated"}]}'

    async def app(scope, receive, send):
        # Drain request body (ASGI contract)
        more = True
        while more:
            msg = await receive()
            more = msg.get("more_body", False)
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({"type": "http.response.body", "body": body})
    return app


async def _make_receive(body: bytes):
    """Make an ASGI `receive` callable that yields exactly one body
    chunk then empty."""
    chunks = [
        {"type": "http.request", "body": body, "more_body": False},
    ]

    async def receive():
        if chunks:
            return chunks.pop(0)
        # After exhaustion, signal empty
        return {
            "type": "http.request", "body": b"", "more_body": False,
        }
    return receive


class TestMiddlewareFlow:
    def test_pass_through_non_post(self):
        from vllm._genesis.middleware import ResponseCacheMiddleware
        cache = _FakeCache()
        app = _make_downstream_app()
        mw = ResponseCacheMiddleware(app, cache=cache)
        scope = {
            "type": "http", "method": "GET", "path": "/v1/completions",
        }
        send = _CollectingSend()

        async def recv():
            return {"type": "http.request", "body": b"", "more_body": False}

        asyncio.get_event_loop().run_until_complete(
            mw(scope, recv, send),
        )
        # cache untouched
        assert cache.get_calls == 0
        assert cache.store_calls == 0

    def test_pass_through_non_intercepted_path(self):
        from vllm._genesis.middleware import ResponseCacheMiddleware
        cache = _FakeCache()
        app = _make_downstream_app()
        mw = ResponseCacheMiddleware(app, cache=cache)
        scope = {
            "type": "http", "method": "POST", "path": "/other/path",
        }
        send = _CollectingSend()

        async def recv():
            return {"type": "http.request", "body": b"{}", "more_body": False}

        asyncio.get_event_loop().run_until_complete(
            mw(scope, recv, send),
        )
        assert cache.get_calls == 0

    def test_cache_miss_forwards_and_stores(self):
        from vllm._genesis.middleware import ResponseCacheMiddleware
        cache = _FakeCache()
        app = _make_downstream_app(
            status=200,
            body=b'{"choices":[{"text":"hello"}]}',
        )
        mw = ResponseCacheMiddleware(app, cache=cache)
        body = json.dumps({"model": "m", "prompt": "hi"}).encode()
        scope = {
            "type": "http", "method": "POST", "path": "/v1/completions",
        }
        send = _CollectingSend()
        loop = asyncio.new_event_loop()
        recv = loop.run_until_complete(_make_receive(body))
        loop.run_until_complete(mw(scope, recv, send))
        loop.close()
        # Get attempted (miss) + store happened
        assert cache.get_calls == 1
        assert cache.store_calls == 1
        # Response header has MISS marker
        starts = [m for m in send.messages if m.get("type") == "http.response.start"]
        assert starts
        headers = starts[0]["headers"]
        assert any(
            name == b"x-genesis-cache" and val == b"MISS"
            for name, val in headers
        )

    def test_cache_hit_short_circuits(self):
        from vllm._genesis.middleware import ResponseCacheMiddleware
        cache = _FakeCache()
        # Pre-populate cache for the exact request
        cache.store(
            "hi", "m",
            {},
            {"choices": [{"text": "cached-response"}]},
        )
        # Downstream app — if invoked, test fails (setter 999 marker)
        downstream_invoked = {"flag": False}

        async def app(scope, receive, send):
            downstream_invoked["flag"] = True

        mw = ResponseCacheMiddleware(app, cache=cache)
        body = json.dumps({"model": "m", "prompt": "hi"}).encode()
        scope = {
            "type": "http", "method": "POST", "path": "/v1/completions",
        }
        send = _CollectingSend()
        loop = asyncio.new_event_loop()
        recv = loop.run_until_complete(_make_receive(body))
        loop.run_until_complete(mw(scope, recv, send))
        loop.close()

        assert downstream_invoked["flag"] is False, (
            "downstream was invoked on cache hit"
        )
        # Response sent from cache
        body_msgs = [
            m for m in send.messages if m.get("type") == "http.response.body"
        ]
        assert body_msgs
        payload = json.loads(body_msgs[0]["body"])
        assert payload["choices"][0]["text"] == "cached-response"
        # Header indicates HIT
        starts = [
            m for m in send.messages if m.get("type") == "http.response.start"
        ]
        assert starts
        headers = starts[0]["headers"]
        assert any(
            name == b"x-genesis-cache" and val == b"HIT"
            for name, val in headers
        )

    def test_malformed_body_passthrough(self):
        from vllm._genesis.middleware import ResponseCacheMiddleware
        cache = _FakeCache()
        app = _make_downstream_app()
        mw = ResponseCacheMiddleware(app, cache=cache)
        body = b"not-json-at-all{{{"
        scope = {
            "type": "http", "method": "POST", "path": "/v1/completions",
        }
        send = _CollectingSend()
        loop = asyncio.new_event_loop()
        recv = loop.run_until_complete(_make_receive(body))
        # Must not raise
        loop.run_until_complete(mw(scope, recv, send))
        loop.close()

    def test_non_2xx_response_not_stored(self):
        from vllm._genesis.middleware import ResponseCacheMiddleware
        cache = _FakeCache()
        app = _make_downstream_app(status=500, body=b'{"error":"x"}')
        mw = ResponseCacheMiddleware(app, cache=cache)
        body = json.dumps({"model": "m", "prompt": "hi"}).encode()
        scope = {
            "type": "http", "method": "POST", "path": "/v1/completions",
        }
        send = _CollectingSend()
        loop = asyncio.new_event_loop()
        recv = loop.run_until_complete(_make_receive(body))
        loop.run_until_complete(mw(scope, recv, send))
        loop.close()
        assert cache.store_calls == 0

    def test_stream_true_passthrough_not_cached(self):
        from vllm._genesis.middleware import ResponseCacheMiddleware
        cache = _FakeCache()
        app = _make_downstream_app()
        mw = ResponseCacheMiddleware(app, cache=cache)
        body = json.dumps(
            {"model": "m", "prompt": "p", "stream": True},
        ).encode()
        scope = {
            "type": "http", "method": "POST", "path": "/v1/completions",
        }
        send = _CollectingSend()
        loop = asyncio.new_event_loop()
        recv = loop.run_until_complete(_make_receive(body))
        loop.run_until_complete(mw(scope, recv, send))
        loop.close()
        # Cache-store must NOT fire — stream requests are not eligible
        assert cache.store_calls == 0


class TestCacheLookupErrorHandling:
    def test_cache_get_exception_is_treated_as_miss(self):
        """If cache.get() raises (Redis timeout etc.), forward to
        downstream rather than serving stale or erroring."""
        from vllm._genesis.middleware import ResponseCacheMiddleware

        class ErrorCache:
            def get(self, *a, **kw):
                raise RuntimeError("redis down")

            def store(self, *a, **kw):
                pass

        app = _make_downstream_app()
        mw = ResponseCacheMiddleware(app, cache=ErrorCache())
        body = json.dumps({"model": "m", "prompt": "p"}).encode()
        scope = {
            "type": "http", "method": "POST", "path": "/v1/completions",
        }
        send = _CollectingSend()
        loop = asyncio.new_event_loop()
        recv = loop.run_until_complete(_make_receive(body))
        # MUST NOT raise
        loop.run_until_complete(mw(scope, recv, send))
        loop.close()
        # Downstream DID get the request
        body_msgs = [
            m for m in send.messages if m.get("type") == "http.response.body"
        ]
        assert body_msgs
