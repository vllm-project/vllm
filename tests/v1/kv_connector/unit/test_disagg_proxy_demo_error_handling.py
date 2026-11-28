# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for upstream errors and decoder eviction in the XpYd demos."""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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


def _make_proxy(demo):
    return demo.Proxy(
        prefill_instances=["prefill-0"],
        decode_instances=["decode-0", "decode-1"],
        model="test-model",
        scheduling_policy=demo.RoundRobinSchedulingPolicy(),
    )


class _UpstreamResponse:
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
    "demo_fixture, proxy_cls, extra_args, status",
    [("pull_demo", "Proxy", (), status) for status in (400, 429, 503, 504)]
    + [("push_demo", "PushProxy", ({},), status) for status in (503, 504)],
)
def test_forward_request_preserves_the_upstream_status(
    demo_fixture, proxy_cls, extra_args, request, monkeypatch, status
):
    demo = request.getfixturevalue(demo_fixture)
    monkeypatch.setattr(demo.aiohttp, "ClientSession", _FakeSession)
    monkeypatch.setattr(
        _FakeSession,
        "post",
        lambda self, **kwargs: _PostContext(
            _UpstreamResponse(status, '{"error": "backend unavailable"}')
        ),
    )

    async def _call():
        generator = getattr(demo, proxy_cls).forward_request(
            object(), "http://decode-0:8200/v1/completions", {}, *extra_args
        )
        async for _ in generator:
            pass

    with pytest.raises(demo.HTTPException) as excinfo:
        asyncio.run(_call())

    assert excinfo.value.status_code == status
    assert "backend unavailable" in str(excinfo.value.detail)


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


def test_a_failing_decode_instance_is_removed_from_the_rotation(pull_demo, monkeypatch):
    async def _forward(self, url, data, *args, **kwargs):
        if "decode-0" in url:
            yield b'{"choices": []}'
            raise pull_demo.HTTPException(status_code=502, detail="decode-0 is down")
        yield b'{"choices": []}'

    monkeypatch.setattr(pull_demo.Proxy, "forward_request", _forward)
    proxy = _make_proxy(pull_demo)

    async def run():
        response = await proxy.create_completion(_FakeRequest({"prompt": "hi"}))
        async for _ in response.body_iterator:
            pass

    with pytest.raises(pull_demo.HTTPException):
        asyncio.run(run())

    assert proxy.decode_instances == ["decode-1"]


@pytest.mark.parametrize("path", ["/v1/completions", "/v1/chat/completions"])
@pytest.mark.parametrize("status", [400, 429, 503])
def test_initial_decode_error_reaches_client_before_response_starts(
    pull_demo, monkeypatch, path, status
):
    async def forward(self, url, data, *args, **kwargs):
        if "decode-0" in url:
            raise pull_demo.HTTPException(
                status_code=status, detail="decode unavailable"
            )
        yield b'{"choices": []}'

    monkeypatch.setattr(pull_demo.Proxy, "forward_request", forward)
    proxy = _make_proxy(pull_demo)
    app = pull_demo.FastAPI()
    app.include_router(proxy.router)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(path, json={"prompt": "hi"})

    assert response.status_code == status
    assert response.json()["detail"] == "decode unavailable"
    assert proxy.decode_instances == (
        ["decode-1"] if status >= 500 else ["decode-0", "decode-1"]
    )


@pytest.mark.parametrize("path", ["/v1/completions", "/v1/chat/completions"])
def test_client_prefill_error_does_not_evict_a_healthy_node(
    pull_demo, monkeypatch, path
):
    monkeypatch.setattr(
        pull_demo.Proxy,
        "forward_request",
        _raising_forward(
            lambda: pull_demo.HTTPException(status_code=400, detail="invalid request")
        ),
    )
    proxy = _make_proxy(pull_demo)
    app = pull_demo.FastAPI()
    app.include_router(proxy.router)
    with TestClient(app) as client:
        response = client.post(path, json={"prompt": "hi"})
    assert response.status_code == 400
    assert proxy.prefill_instances == ["prefill-0"]


@pytest.mark.parametrize("path", ["/v1/completions", "/v1/chat/completions"])
def test_preopening_decode_replays_all_chunks_and_closes_upstream(
    pull_demo, monkeypatch, path
):
    closed = []

    async def forward(self, url, data, *args, **kwargs):
        try:
            yield b"first"
            yield b"second"
        finally:
            closed.append(url)

    monkeypatch.setattr(pull_demo.Proxy, "forward_request", forward)
    proxy = _make_proxy(pull_demo)
    app = pull_demo.FastAPI()
    app.include_router(proxy.router)
    with TestClient(app) as client:
        response = client.post(path, json={"prompt": "hi"})

    assert response.status_code == 200
    assert response.content == b"firstsecond"
    assert closed == [f"http://prefill-0{path}", f"http://decode-0{path}"]
    assert proxy.decode_instances == ["decode-0", "decode-1"]
