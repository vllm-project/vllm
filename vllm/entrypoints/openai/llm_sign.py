# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse
from vllm.entrypoints.openai.responses.protocol import ResponsesResponse


def responses_parent_signing_context(parent_response: Any) -> tuple[str | None, int]:
    parent_hash: str | None = None
    start_seq = 0

    parent_llm_sign = getattr(parent_response, "llm_sign", None)
    if not isinstance(parent_llm_sign, Mapping):
        return parent_hash, start_seq

    candidate = parent_llm_sign.get("artifact_hash")
    if isinstance(candidate, str):
        parent_hash = candidate

    parent_artifact = parent_llm_sign.get("artifact")
    if not isinstance(parent_artifact, Mapping):
        return parent_hash, start_seq

    parent_chain = parent_artifact.get("chain")
    if not isinstance(parent_chain, list) or not parent_chain:
        return parent_hash, start_seq

    last_signed = parent_chain[-1]
    if not isinstance(last_signed, Mapping):
        return parent_hash, start_seq

    last_block = last_signed.get("block")
    if not isinstance(last_block, Mapping):
        return parent_hash, start_seq

    last_seq = last_block.get("seq")
    if isinstance(last_seq, int):
        start_seq = last_seq + 1

    return parent_hash, start_seq


def maybe_sign_chat_completion(
    request: Any,
    response: Any,
) -> Any:
    """Attach llm_sign metadata to a Chat Completions response.

    This helper assumes the caller has already gated on
    ``envs.VLLM_LLM_SIGN_ENABLED``; it does not re-check the env var.
    When the response is a regular :class:`ChatCompletionResponse`
    (not a streaming generator), this helper:

    1. Projects the request and response to the OpenAI Chat Completions
       v1 canonical schemas (vLLM-only knobs such as ``min_tokens``,
       ``prompt_logprobs`` are dropped; they are not covered by the
       signing profile).
    2. Signs the projected turn with the TLS private key loaded from
       ``VLLM_LLM_SIGN_CERTFILE`` / ``VLLM_LLM_SIGN_KEYFILE``.
    3. Attaches ``{"artifact": ..., "certificate_chain": [...pem...]}``
       under the ``llm_sign`` field of the response envelope, using the
       official :func:`llm_sign.server.attach_signed_artifact_to_openai_response`
       helper so the wire format stays in lockstep with ``llm_sign``'s
       spec.

    Downstream clients verify the response with
    :func:`llm_sign.client.verify_openai_response`, which authenticates
    the embedded certificate chain under the standard TLS / X.509
    server-certificate validation rules (using the system TLS trust
    store by default) and then verifies the transcript under the
    validated leaf public key.
    """

    if not isinstance(response, ChatCompletionResponse):
        return response

    signer = _get_signer()
    # The request payload uses ``exclude_unset=True``: only the fields the
    # client *explicitly set* on the wire enter the signed digest. Pydantic
    # otherwise materializes every schema default (``frequency_penalty=0.0``,
    # ``n=1``, ``logprobs=False``, ...), and a verifier that re-canonicalizes
    # the bytes the client actually sent would compute a different digest and
    # reject a legitimate signature. The response payload uses
    # ``exclude_none=False`` because that mirrors what FastAPI's default JSON
    # encoder produces on the wire (null fields kept), and the client reads
    # exactly those bytes.
    from llm_sign import project_openai_chat_request, project_openai_chat_response

    request_payload = project_openai_chat_request(
        request.model_dump(mode="json", exclude_unset=True)
    )
    response_payload = project_openai_chat_response(
        response.model_dump(mode="json", exclude_none=False)
    )
    envelope: dict[str, Any] = {}
    signer.sign_chat_and_attach(envelope, request_payload, response_payload)
    # ``ChatCompletionResponse`` (via ``OpenAIBaseModel``) sets
    # ``model_config = ConfigDict(extra="allow")``, so the ``llm_sign`` field
    # is a valid dynamic attribute at runtime. mypy cannot see that through
    # pydantic's config, hence the targeted ignore.
    response.llm_sign = envelope["llm_sign"]  # type: ignore[attr-defined]
    return response


def maybe_sign_responses_response(
    request: Any,
    response: Any,
    parent_hash: str | None = None,
    start_seq: int = 0,
) -> Any:
    """Attach llm_sign metadata to a ``/v1/responses`` response.

    Mirror of :func:`maybe_sign_chat_completion` for the OpenAI
    Responses API. This helper assumes the caller has already gated on
    ``envs.VLLM_LLM_SIGN_ENABLED``; it does not re-check the env var.

    Projects the request and response to the Responses canonical
    schemas (vLLM-only knobs such as ``kv_transfer_params``,
    ``vllm_xargs``, ``prompt_cache_key``, ``cache_salt`` are dropped;
    they are not part of the signing whitelist), signs the turn with
    the shared TLS credential, and attaches the artifact envelope.

    The Responses API is stateful (``previous_response_id`` points at
    a prior turn held in the server's store), and the
    ``previous_response_id`` pointer is part of the signed input
    payload. When ``parent_hash`` is supplied, it is also injected
    into the signed input payload under ``previous_response_hash`` —
    this is the server's own record of the parent turn's artifact
    hash (looked up from the session store, not read from client
    input), which binds the current turn's signature to that specific
    prior artifact rather than to whatever content is sitting under
    the id string at verification time.

    ``start_seq`` lets a multi-turn Responses session continue ``seq``
    monotonically across turns: turn 1 signs blocks at seq=0,1; turn 2
    at seq=2,3; etc. The chain-level structure is still independent
    per turn (``prev_block_digest`` on each turn's first block is
    ``null``); the seq numbering is the only artifact-level signal
    of a session's turn ordering. Cross-turn integrity is by
    ``previous_response_hash`` in the signed input.

    Forking — the same ``previous_response_id`` used on multiple
    create calls — produces independent artifacts. Each fork starts
    its first block at the same ``start_seq``; that's intentional and
    fine, llm_sign does not attempt to detect or prevent forks.
    """

    if not isinstance(response, ResponsesResponse):
        return response

    signer = _get_signer()
    from llm_sign import (
        project_openai_responses_request,
        project_openai_responses_response,
    )

    # See ``maybe_sign_chat_completion`` for the rationale: the request
    # uses ``exclude_unset=True`` to capture only what the client actually
    # sent on the wire; the response uses ``exclude_none=False`` because
    # FastAPI's encoder emits null fields and the client reads exactly
    # those bytes.
    raw_request_payload = request.model_dump(mode="json", exclude_unset=True)
    # ``previous_response_hash`` is a server-injected binding derived from
    # ``response_store``. Never let a client-supplied extra field choose it.
    raw_request_payload.pop("previous_response_hash", None)
    request_payload = project_openai_responses_request(raw_request_payload)
    response_payload = project_openai_responses_response(
        response.model_dump(mode="json", exclude_none=False)
    )
    envelope: dict[str, Any] = {}
    signer.sign_responses_and_attach(
        envelope,
        request_payload,
        response_payload,
        parent_hash=parent_hash,
        start_seq=start_seq,
    )
    # ``ResponsesResponse`` (via ``OpenAIBaseModel``) sets
    # ``model_config = ConfigDict(extra="allow")``, so the ``llm_sign``
    # field is a valid dynamic attribute at runtime. mypy cannot see
    # that through pydantic's config, hence the targeted ignore.
    response.llm_sign = envelope["llm_sign"]  # type: ignore[attr-defined]
    return response


class _OpenAILLMSigner:
    def __init__(self) -> None:
        certfile = envs.VLLM_LLM_SIGN_CERTFILE
        keyfile = envs.VLLM_LLM_SIGN_KEYFILE
        if not certfile or not keyfile:
            raise ValueError(
                "VLLM_LLM_SIGN_CERTFILE and VLLM_LLM_SIGN_KEYFILE must be set "
                "when VLLM_LLM_SIGN_ENABLED=1"
            )

        try:
            from llm_sign.server import (
                TLSCertificateCredential,
                attach_signed_artifact_to_openai_response,
                sign_openai_chat_turn,
                sign_openai_responses_turn,
            )
        except ImportError as exc:
            raise ImportError(
                "llm_sign must be installed when VLLM_LLM_SIGN_ENABLED=1"
            ) from exc

        self._credential = TLSCertificateCredential.from_files(
            ssl_certfile=certfile,
            ssl_keyfile=keyfile,
        )
        self._signer = self._credential.signer()
        self._sign_openai_chat_turn = sign_openai_chat_turn
        self._sign_openai_responses_turn = sign_openai_responses_turn
        self._attach = attach_signed_artifact_to_openai_response

    def sign_chat_and_attach(
        self,
        envelope: dict[str, Any],
        request: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> None:
        artifact = self._sign_openai_chat_turn(
            request=request,
            response=response,
            signer=self._signer,
        )
        self._attach(envelope, artifact=artifact, credential=self._credential)

    def sign_responses_and_attach(
        self,
        envelope: dict[str, Any],
        request: Mapping[str, Any],
        response: Mapping[str, Any],
        *,
        parent_hash: str | None = None,
        start_seq: int = 0,
    ) -> None:
        artifact = self._sign_openai_responses_turn(
            request=request,
            response=response,
            signer=self._signer,
            parent_hash=parent_hash,
            start_seq=start_seq,
        )
        self._attach(envelope, artifact=artifact, credential=self._credential)


_signer: _OpenAILLMSigner | None = None
_signer_lock = threading.Lock()


def _get_signer() -> _OpenAILLMSigner:
    global _signer
    if _signer is None:
        with _signer_lock:
            # Double-checked locking: another thread may have
            # initialized the singleton while we were waiting on
            # the lock.
            if _signer is None:
                _signer = _OpenAILLMSigner()
    return _signer


def clear_cached_signer() -> None:
    global _signer
    with _signer_lock:
        _signer = None
