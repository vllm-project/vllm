# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Error propagation in the XpYd disaggregated-prefill proxy demos.

Exercises the REAL modules loaded from their ``examples/`` paths, so a future
change to the demos is what these tests catch.

Two regressions are pinned here.

1. ``forward_request`` raised the upstream status through an ``HTTPException``
   from inside a ``try`` whose last clause was a bare ``except Exception``.
   ``HTTPException`` derives from ``Exception``, so the handler always caught
   the exception it had just raised and re-wrapped it as a 500: a 503/504 from
   a busy or dying prefill/decode instance never reached the client intact.
2. ``create_completion`` ended in a bare ``except Exception`` that only logged
   and fell off the end of the function, so FastAPI answered HTTP 200 with a
   ``null`` body for every failure; ``create_chat_completion`` returned the
   error text as an HTTP 200 ``text/event-stream`` body instead. The decode
   leg's ``remove_instance_endpoint`` was unreachable as well, because
   ``forward_request`` is an async generator: building it runs no user code, so
   the ``except HTTPException`` around the call could never fire and a broken
   decode node stayed in the round-robin cycle forever.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES = "examples/disaggregated/disaggregated_serving"
PULL_DEMO = f"{EXAMPLES}/disagg_proxy_demo.py"
PUSH_DEMO = f"{EXAMPLES}/disagg_proxy_pushconnector_demo.py"


def _load(name: str, rel_path: str):
    path = Path(__file__).parents[4] / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pull_demo():
    return _load("disagg_proxy_demo_error_handling", PULL_DEMO)


@pytest.fixture(scope="module")
def push_demo():
    return _load("disagg_proxy_pushconnector_demo_error_handling", PUSH_DEMO)


class _FakeRequest:
    def __init__(self, body: dict):
        self._body = body

    async def json(self):
        return self._body


def _drain_stream(response) -> None:
    """Consume a StreamingResponse body, discarding the chunks."""

    async def _run():
        async for _ in response.body_iterator:
            pass

    asyncio.run(_run())


def _make_proxy(demo):
    return demo.Proxy(
        prefill_instances=["prefill-0"],
        decode_instances=["decode-0", "decode-1"],
        model="test-model",
        scheduling_policy=demo.RoundRobinSchedulingPolicy(),
    )


# --------------------------------------------------------------------------- #
# 1. Upstream status codes must survive forward_request
# --------------------------------------------------------------------------- #


class _UpstreamResponse:
    """A 5xx upstream response: the branch that builds an HTTPException."""

    def __init__(self, status: int, body: str):
        self.status = status
        self._body = body

    async def text(self) -> str:
        return self._body


class _PostContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc_info):
        return False


class _FakeSession:
    """Stands in for ``aiohttp.ClientSession``; always returns one response."""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def post(self, **kwargs):
        return _PostContext(_UpstreamResponse(503, '{"error": "backend unavailable"}'))


@pytest.mark.parametrize(
    "demo_fixture, proxy_cls, extra_args",
    # forward_request(self, url, data, [headers,] use_chunked=True)
    [("pull_demo", "Proxy", ()), ("push_demo", "PushProxy", ({},))],
)
def test_forward_request_preserves_the_upstream_status(
    demo_fixture, proxy_cls, extra_args, request, monkeypatch
):
    demo = request.getfixturevalue(demo_fixture)
    monkeypatch.setattr(demo.aiohttp, "ClientSession", _FakeSession)

    async def _call():
        generator = getattr(demo, proxy_cls).forward_request(
            object(), "http://decode-0:8200/v1/completions", {}, *extra_args
        )
        async for _ in generator:
            pass

    with pytest.raises(demo.HTTPException) as excinfo:
        asyncio.run(_call())

    assert excinfo.value.status_code == 503
    assert "backend unavailable" in str(excinfo.value.detail)


# --------------------------------------------------------------------------- #
# 2. Failures must not be reported as a successful response
# --------------------------------------------------------------------------- #


def _raising_forward(error_factory):
    async def _forward(self, url, data, *args, **kwargs):
        if False:  # pragma: no cover - makes this an async generator
            yield b""
        raise error_factory()

    return _forward


@pytest.mark.parametrize("handler", ["create_completion", "create_chat_completion"])
def test_failure_is_not_reported_as_a_successful_response(
    pull_demo, monkeypatch, handler
):
    monkeypatch.setattr(
        pull_demo.Proxy,
        "forward_request",
        _raising_forward(lambda: RuntimeError("prefill exploded")),
    )
    proxy = _make_proxy(pull_demo)

    response = asyncio.run(getattr(proxy, handler)(_FakeRequest({"prompt": "hi"})))

    assert response is not None, "an error must not be serialised as a null 200 body"
    assert response.status_code == 500
    assert b"prefill exploded" in response.body


def test_upstream_http_error_is_not_reported_as_a_successful_response(
    pull_demo, monkeypatch
):
    monkeypatch.setattr(
        pull_demo.Proxy,
        "forward_request",
        _raising_forward(
            lambda: pull_demo.HTTPException(
                status_code=503, detail="prefill unavailable"
            )
        ),
    )
    proxy = _make_proxy(pull_demo)

    with pytest.raises(pull_demo.HTTPException) as excinfo:
        asyncio.run(proxy.create_completion(_FakeRequest({"prompt": "hi"})))

    assert excinfo.value.status_code == 503


# --------------------------------------------------------------------------- #
# 3. A decode node that cannot serve must leave the rotation
# --------------------------------------------------------------------------- #


def test_a_failing_decode_instance_is_removed_from_the_rotation(pull_demo, monkeypatch):
    async def _forward(self, url, data, *args, **kwargs):
        if "decode-0" in url:
            raise pull_demo.HTTPException(status_code=502, detail="decode-0 is down")
        yield b'{"choices": []}'

    monkeypatch.setattr(pull_demo.Proxy, "forward_request", _forward)
    proxy = _make_proxy(pull_demo)

    response = asyncio.run(proxy.create_completion(_FakeRequest({"prompt": "hi"})))

    with pytest.raises(pull_demo.HTTPException):
        _drain_stream(response)

    assert proxy.decode_instances == ["decode-1"]
