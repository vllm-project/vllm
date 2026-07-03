# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import builtins
import sys
from types import ModuleType

import pytest

from vllm import envs
from vllm.entrypoints.openai import llm_sign
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponsesResponse,
)


@pytest.fixture(autouse=True)
def reset_llm_sign(monkeypatch: pytest.MonkeyPatch):
    llm_sign.clear_cached_signer()
    envs.disable_envs_cache()
    monkeypatch.delenv("VLLM_LLM_SIGN_ENABLED", raising=False)
    monkeypatch.delenv("VLLM_LLM_SIGN_CERTFILE", raising=False)
    monkeypatch.delenv("VLLM_LLM_SIGN_KEYFILE", raising=False)
    yield
    llm_sign.clear_cached_signer()
    envs.disable_envs_cache()


def test_llm_sign_skips_non_chat_completion_response():
    """``maybe_sign_chat_completion`` is a no-op for non-chat responses.

    The caller (``OpenAIServingChat.create_chat_completion``) is
    responsible for gating on ``envs.VLLM_LLM_SIGN_ENABLED`` *before*
    invoking this helper. The hook's only remaining input check is a
    type guard against streaming / unexpected payload types, so that
    ``return maybe_sign_chat_completion(request, response)`` is safe
    even when the caller passes through a non-``ChatCompletionResponse``
    value (defensive — should not happen on the gated non-streaming
    code path, but guarantees the hook never crashes the response).

    The byte-for-byte equivalence with upstream when
    ``VLLM_LLM_SIGN_ENABLED`` is unset is now a property of the
    *caller* (the hook is not invoked at all in that case); it is
    covered by the integration tests that exercise the full
    ``serving_chat`` path.
    """
    request = _request()
    response = "not-a-chat-completion-response"
    assert llm_sign.maybe_sign_chat_completion(request, response) is response


def test_llm_sign_skips_non_responses_response():
    """``maybe_sign_responses_response`` is a no-op for non-responses payloads.

    Same contract as ``test_llm_sign_skips_non_chat_completion_response``:
    the caller (``OpenAIServingResponses.responses_full_generator``)
    pre-checks ``envs.VLLM_LLM_SIGN_ENABLED``; the hook only retains a
    defensive type guard.
    """
    request = _responses_request()
    response = "not-a-responses-response"
    assert llm_sign.maybe_sign_responses_response(request, response) is response


def test_llm_sign_filters_vllm_only_request_fields(monkeypatch: pytest.MonkeyPatch):
    """vLLM-only request knobs (e.g. ``min_tokens``) must not leak into the
    signed payload, otherwise the strict ``openai.chat-completions.input.v1``
    profile rejects the artifact with ``unknown fields``.
    """
    monkeypatch.setenv("VLLM_LLM_SIGN_ENABLED", "1")
    monkeypatch.setenv("VLLM_LLM_SIGN_CERTFILE", "cert.pem")
    monkeypatch.setenv("VLLM_LLM_SIGN_KEYFILE", "key.pem")

    captured: list[dict] = []
    _install_fake_llm_sign(monkeypatch, captured)

    # A request that includes a vLLM-only knob plus a standard one
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "ping"}],
        min_tokens=3,  # vLLM extension
        temperature=0.0,  # OpenAI standard
    )
    llm_sign.maybe_sign_chat_completion(request, _response())

    signed_request = captured[1]["request"]
    assert "model" in signed_request
    assert "messages" in signed_request
    assert "temperature" in signed_request
    # vLLM-only knobs filtered out
    assert "min_tokens" not in signed_request


def test_llm_sign_filters_vllm_only_response_fields(monkeypatch: pytest.MonkeyPatch):
    """vLLM-only response fields (``prompt_logprobs``, ``prompt_token_ids``,
    ``kv_transfer_params``) must not leak into the signed payload, otherwise
    a verifier re-canonicalizing the HTTP body fails with ``unknown fields``.
    """
    monkeypatch.setenv("VLLM_LLM_SIGN_ENABLED", "1")
    monkeypatch.setenv("VLLM_LLM_SIGN_CERTFILE", "cert.pem")
    monkeypatch.setenv("VLLM_LLM_SIGN_KEYFILE", "key.pem")

    captured: list[dict] = []
    _install_fake_llm_sign(monkeypatch, captured)

    response = _response()
    response.prompt_logprobs = [None]
    response.prompt_token_ids = [1, 2, 3]
    response.kv_transfer_params = {"foo": "bar"}

    llm_sign.maybe_sign_chat_completion(_request(), response)

    signed_response = captured[1]["response"]
    assert "choices" in signed_response
    assert "model" in signed_response
    # vLLM-only response fields filtered out
    assert "prompt_logprobs" not in signed_response
    assert "prompt_token_ids" not in signed_response
    assert "kv_transfer_params" not in signed_response
    assert "llm_sign" not in signed_response


def test_llm_sign_filters_vllm_only_responses_request_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_LLM_SIGN_ENABLED", "1")
    monkeypatch.setenv("VLLM_LLM_SIGN_CERTFILE", "cert.pem")
    monkeypatch.setenv("VLLM_LLM_SIGN_KEYFILE", "key.pem")

    captured: list[dict] = []
    _install_fake_llm_sign(monkeypatch, captured)

    request = ResponsesRequest(
        model="test-model",
        input="ping",
        temperature=0.0,
        kv_transfer_params={"foo": "bar"},
        prompt_cache_key="cache-key",
        cache_salt="salt",
        vllm_xargs={"internal": "value"},
    )
    llm_sign.maybe_sign_responses_response(request, _responses_response())

    signed_request = captured[1]["request"]
    assert signed_request["model"] == "test-model"
    assert signed_request["input"] == "ping"
    assert signed_request["temperature"] == 0.0
    assert "kv_transfer_params" not in signed_request
    assert "prompt_cache_key" not in signed_request
    assert "cache_salt" not in signed_request
    assert "vllm_xargs" not in signed_request
    assert "request_id" not in signed_request


def test_llm_sign_strips_client_supplied_previous_response_hash(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_LLM_SIGN_ENABLED", "1")
    monkeypatch.setenv("VLLM_LLM_SIGN_CERTFILE", "cert.pem")
    monkeypatch.setenv("VLLM_LLM_SIGN_KEYFILE", "key.pem")

    captured: list[dict] = []
    _install_fake_llm_sign(monkeypatch, captured)

    request = ResponsesRequest(
        model="test-model",
        input="ping",
        previous_response_id="resp-parent",
        previous_response_hash="client-forged-hash",
    )
    llm_sign.maybe_sign_responses_response(
        request,
        _responses_response(previous_response_id="resp-parent"),
        parent_hash="server-observed-parent-hash",
    )

    signed_request = captured[1]["request"]
    assert signed_request["previous_response_id"] == "resp-parent"
    assert "previous_response_hash" not in signed_request
    assert captured[1]["parent_hash"] == "server-observed-parent-hash"


def test_llm_sign_reads_parent_hash_from_saved_response_only():
    parent = _responses_response()
    parent.llm_sign = {
        "artifact_hash": "stored-parent-hash",
        "artifact": {"chain": [{"block": {"seq": 7}}]},
    }

    assert llm_sign.responses_parent_signing_context(parent) == (
        "stored-parent-hash",
        8,
    )


def test_llm_sign_parent_context_ignores_missing_signature():
    assert llm_sign.responses_parent_signing_context(_responses_response()) == (None, 0)


def test_llm_sign_filters_vllm_only_responses_response_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_LLM_SIGN_ENABLED", "1")
    monkeypatch.setenv("VLLM_LLM_SIGN_CERTFILE", "cert.pem")
    monkeypatch.setenv("VLLM_LLM_SIGN_KEYFILE", "key.pem")

    captured: list[dict] = []
    _install_fake_llm_sign(monkeypatch, captured)

    response = _responses_response()
    response.kv_transfer_params = {"foo": "bar"}
    response.llm_sign = {"stale": "metadata"}

    llm_sign.maybe_sign_responses_response(_responses_request(), response)

    signed_response = captured[1]["response"]
    assert signed_response["model"] == "test-model"
    assert signed_response["status"] == "completed"
    assert "output" in signed_response
    assert "kv_transfer_params" not in signed_response
    assert "llm_sign" not in signed_response


def _install_fake_llm_sign(
    monkeypatch: pytest.MonkeyPatch,
    captured: list[dict],
) -> None:
    """Install a fake ``llm_sign`` package so tests don't need cryptography."""

    fake_pkg = ModuleType("llm_sign")
    fake_server = ModuleType("llm_sign.server")

    _CHAT_REQUEST_ALLOWED = {
        "messages",
        "model",
        "temperature",
        "top_p",
        "stop",
        "max_tokens",
        "tools",
        "tool_choice",
        "n",
        "stream",
        "user",
        "metadata",
    }
    _CHAT_RESPONSE_ALLOWED = {
        "choices",
        "model",
        "response_format",
        "created",
        "id",
        "object",
        "usage",
        "system_fingerprint",
    }
    _RESPONSES_REQUEST_ALLOWED = {
        "background",
        "frequency_penalty",
        "include",
        "input",
        "instructions",
        "logit_bias",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "model",
        "parallel_tool_calls",
        "presence_penalty",
        "previous_response_hash",
        "previous_response_id",
        "prompt",
        "reasoning",
        "service_tier",
        "store",
        "stream",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
    }
    _RESPONSES_RESPONSE_ALLOWED = {
        "background",
        "created_at",
        "frequency_penalty",
        "id",
        "incomplete_details",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "model",
        "object",
        "output",
        "parallel_tool_calls",
        "presence_penalty",
        "previous_response_id",
        "prompt",
        "reasoning",
        "service_tier",
        "status",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "truncation",
        "usage",
        "user",
    }

    def fake_project_openai_chat_request(payload):
        return {k: v for k, v in payload.items() if k in _CHAT_REQUEST_ALLOWED}

    def fake_project_openai_chat_response(payload):
        return {k: v for k, v in payload.items() if k in _CHAT_RESPONSE_ALLOWED}

    def fake_project_openai_responses_request(payload):
        return {k: v for k, v in payload.items() if k in _RESPONSES_REQUEST_ALLOWED}

    def fake_project_openai_responses_response(payload):
        return {k: v for k, v in payload.items() if k in _RESPONSES_RESPONSE_ALLOWED}

    fake_pkg.project_openai_chat_request = fake_project_openai_chat_request  # type: ignore[attr-defined]
    fake_pkg.project_openai_chat_response = fake_project_openai_chat_response  # type: ignore[attr-defined]
    fake_pkg.project_openai_responses_request = fake_project_openai_responses_request  # type: ignore[attr-defined]
    fake_pkg.project_openai_responses_response = fake_project_openai_responses_response  # type: ignore[attr-defined]

    class FakeCredential:
        @classmethod
        def from_files(cls, **kwargs):
            captured.append(kwargs)
            return cls()

        def signer(self):
            return "signer"

        def certificate_chain_pem(self):
            return ["cert-chain"]

    def fake_sign_openai_chat_turn(**kwargs):
        captured.append(kwargs)
        return {
            "schema": "llm-sign.artifact.v1",
            "kind": "chat",
            "chain": [{"block": {"seq": 0}}, {"block": {"seq": 1}}],
        }

    def fake_sign_openai_responses_turn(**kwargs):
        captured.append(kwargs)
        start_seq = kwargs.get("start_seq", 0)
        return {
            "schema": "llm-sign.artifact.v1",
            "kind": "responses",
            "chain": [
                {"block": {"seq": start_seq}},
                {"block": {"seq": start_seq + 1}},
            ],
        }

    def fake_attach(envelope, *, artifact, credential=None, certificate_chain_pem=None):
        envelope["llm_sign"] = {
            "artifact": dict(artifact),
            "artifact_hash": f"fake-{artifact.get('kind', 'artifact')}-hash",
        }
        chain = certificate_chain_pem
        if chain is None and credential is not None:
            chain = credential.certificate_chain_pem()
        if chain is not None:
            envelope["llm_sign"]["certificate_chain"] = list(chain)
        captured.append({"attached": envelope["llm_sign"]})
        return envelope

    fake_server.TLSCertificateCredential = FakeCredential  # type: ignore[attr-defined]
    fake_server.sign_openai_chat_turn = fake_sign_openai_chat_turn  # type: ignore[attr-defined]
    fake_server.sign_openai_responses_turn = fake_sign_openai_responses_turn  # type: ignore[attr-defined]
    fake_server.attach_signed_artifact_to_openai_response = fake_attach  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "llm_sign", fake_pkg)
    monkeypatch.setitem(sys.modules, "llm_sign.server", fake_server)


def test_llm_sign_attaches_artifact_when_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_LLM_SIGN_ENABLED", "1")
    monkeypatch.setenv("VLLM_LLM_SIGN_CERTFILE", "cert.pem")
    monkeypatch.setenv("VLLM_LLM_SIGN_KEYFILE", "key.pem")

    calls: list[dict] = []
    _install_fake_llm_sign(monkeypatch, calls)

    response = llm_sign.maybe_sign_chat_completion(_request(), _response())

    assert response.llm_sign["artifact"]["schema"] == "llm-sign.artifact.v1"
    assert response.llm_sign["artifact"]["kind"] == "chat"
    assert response.llm_sign["artifact_hash"] == "fake-chat-hash"
    assert response.llm_sign["certificate_chain"] == ["cert-chain"]
    # The issuer (host name) is derived from the certificate by llm_sign;
    # vLLM forwards only the cert and key paths.
    assert calls[0] == {
        "ssl_certfile": "cert.pem",
        "ssl_keyfile": "key.pem",
    }
    assert calls[1]["request"]["model"] == "test-model"
    assert calls[1]["response"]["choices"][0]["message"]["content"] == "pong"
    assert calls[1]["signer"] == "signer"
    # The envelope was produced by llm_sign's official helper, not by
    # vLLM inlining the dict layout.
    assert calls[-1] == {"attached": response.llm_sign}


def test_llm_sign_responses_attaches_artifact_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_LLM_SIGN_ENABLED", "1")
    monkeypatch.setenv("VLLM_LLM_SIGN_CERTFILE", "cert.pem")
    monkeypatch.setenv("VLLM_LLM_SIGN_KEYFILE", "key.pem")

    calls: list[dict] = []
    _install_fake_llm_sign(monkeypatch, calls)

    response = llm_sign.maybe_sign_responses_response(
        _responses_request(previous_response_id="resp-parent"),
        _responses_response(previous_response_id="resp-parent"),
        parent_hash="parent-artifact-hash",
        start_seq=4,
    )

    assert response.llm_sign["artifact"]["schema"] == "llm-sign.artifact.v1"
    assert response.llm_sign["artifact"]["kind"] == "responses"
    assert response.llm_sign["artifact"]["chain"][0]["block"]["seq"] == 4
    assert response.llm_sign["artifact_hash"] == "fake-responses-hash"
    assert response.llm_sign["certificate_chain"] == ["cert-chain"]
    assert calls[0] == {
        "ssl_certfile": "cert.pem",
        "ssl_keyfile": "key.pem",
    }
    assert calls[1]["request"]["previous_response_id"] == "resp-parent"
    assert calls[1]["response"]["previous_response_id"] == "resp-parent"
    assert calls[1]["signer"] == "signer"
    assert calls[1]["parent_hash"] == "parent-artifact-hash"
    assert calls[1]["start_seq"] == 4
    assert calls[-1] == {"attached": response.llm_sign}


def test_llm_sign_requires_cert_and_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_LLM_SIGN_ENABLED", "1")

    with pytest.raises(ValueError, match="VLLM_LLM_SIGN_CERTFILE"):
        llm_sign.maybe_sign_chat_completion(_request(), _response())


def test_llm_sign_import_is_optional_until_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_LLM_SIGN_ENABLED", "1")
    monkeypatch.setenv("VLLM_LLM_SIGN_CERTFILE", "cert.pem")
    monkeypatch.setenv("VLLM_LLM_SIGN_KEYFILE", "key.pem")
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "llm_sign.server":
            raise ImportError("missing llm_sign")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="llm_sign must be installed"):
        llm_sign.maybe_sign_chat_completion(_request(), _response())


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "ping"}],
    )


def _response() -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="pong"),
            )
        ],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _responses_request(previous_response_id: str | None = None) -> ResponsesRequest:
    return ResponsesRequest(
        model="test-model",
        input="ping",
        previous_response_id=previous_response_id,
        store=False,
    )


def _responses_response(previous_response_id: str | None = None) -> ResponsesResponse:
    return ResponsesResponse(
        model="test-model",
        output=[],
        parallel_tool_calls=True,
        temperature=0.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        background=False,
        max_output_tokens=16,
        previous_response_id=previous_response_id,
        service_tier="auto",
        status="completed",
        truncation="disabled",
    )
