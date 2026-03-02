# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Stateless conversation state carrier for the Responses API.

Serializes conversation message history into the ``encrypted_content`` field of
a synthetic ResponseReasoningItem, enabling multi-turn conversations without
server-side storage.  Implements the approach proposed by @grs in
vllm-project/vllm#26934.

Wire format:  ``vllm:1:<base64(json(messages))>:<hmac-sha256-hex>``

Security note:
    The payload is **signed**, not encrypted.  The HMAC-SHA256 signature
    prevents client-side tampering with the serialized history, but the
    contents (serialized message dicts) are readable by anyone who holds
    the response object.  This matches the spirit of the ``encrypted_content``
    field in the OpenAI protocol: an opaque blob the client stores and
    returns verbatim.

Multi-node deployments:
    Set ``VLLM_RESPONSES_STATE_SIGNING_KEY`` to the same 64-character hex
    string on all nodes.  Without it each process generates a random key at
    startup, making state carriers incompatible across restarts/replicas.
"""

import base64
import hashlib
import hmac
import json
import os
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_FORMAT_VERSION = "vllm:1"
_SIGNING_KEY: bytes | None = None


def _get_signing_key() -> bytes:
    global _SIGNING_KEY
    if _SIGNING_KEY is None:
        key_hex = os.environ.get("VLLM_RESPONSES_STATE_SIGNING_KEY", "")
        if key_hex:
            try:
                _SIGNING_KEY = bytes.fromhex(key_hex)
            except ValueError as exc:
                raise ValueError(
                    "VLLM_RESPONSES_STATE_SIGNING_KEY must be a valid hex "
                    "string (e.g. a 64-char / 32-byte hex key)."
                ) from exc
        else:
            _SIGNING_KEY = os.urandom(32)
            logger.warning(
                "VLLM_RESPONSES_STATE_SIGNING_KEY is not set. "
                "Stateless multi-turn state carriers are valid only for this "
                "server instance and will break across restarts or on "
                "multi-node deployments. "
                "Set VLLM_RESPONSES_STATE_SIGNING_KEY to a shared 64-char "
                "hex key to enable multi-node / restart-safe operation."
            )
    return _SIGNING_KEY


def _harmony_serializer(obj: Any) -> Any:
    """JSON serializer for OpenAIHarmonyMessage and similar pydantic models."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def serialize_state(messages: list[Any]) -> str:
    """Serialize *messages* into a signed, base64-encoded state carrier string.

    The returned string is suitable for embedding in
    ``ResponseReasoningItem.encrypted_content``.

    Args:
        messages: A list of message objects (OpenAIHarmonyMessage instances or
            plain ``ChatCompletionMessageParam`` dicts).

    Returns:
        A wire-format state carrier string: ``vllm:1:<b64>:<hmac>``.
    """
    payload_b64 = base64.b64encode(
        json.dumps(messages, default=_harmony_serializer).encode()
    ).decode()
    sig = hmac.new(
        _get_signing_key(), payload_b64.encode(), hashlib.sha256
    ).hexdigest()
    return f"{_FORMAT_VERSION}:{payload_b64}:{sig}"


def deserialize_state(encrypted_content: str) -> list[Any] | None:
    """Deserialize a state carrier string back into a message list.

    Args:
        encrypted_content: The value of ``ResponseReasoningItem.encrypted_content``.

    Returns:
        The deserialized message list (as plain dicts), or ``None`` if the
        string is not a vLLM state carrier (e.g. a real encrypted_content
        from an external model).

    Raises:
        ValueError: If the string looks like a vLLM state carrier but the
            HMAC signature does not match (indicates tampering).
    """
    if not encrypted_content.startswith(f"{_FORMAT_VERSION}:"):
        return None
    # Expected: "vllm:1:<payload_b64>:<sig>"
    # Split into exactly 4 parts on the first 3 colons.
    parts = encrypted_content.split(":", 3)
    if len(parts) != 4:
        raise ValueError(
            "Malformed vLLM state carrier: expected "
            f"'{_FORMAT_VERSION}:<payload>:<sig>', got {encrypted_content!r}"
        )
    _, _, payload_b64, sig = parts
    expected = hmac.new(
        _get_signing_key(), payload_b64.encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig, expected):
        raise ValueError(
            "vLLM state carrier HMAC verification failed. "
            "The state may have been tampered with, or was produced by a "
            "different server instance (check VLLM_RESPONSES_STATE_SIGNING_KEY)."
        )
    return json.loads(base64.b64decode(payload_b64))


def is_state_carrier(item: Any) -> bool:
    """Return True if *item* is a vLLM state-carrier ReasoningItem.

    A state carrier is a ``ResponseReasoningItem`` whose ``encrypted_content``
    field starts with the vLLM format prefix.  These items are synthetic:
    they carry serialized conversation history and should not be sent to the
    LLM as part of the chat message list.

    Args:
        item: Any object, typically a ``ResponseOutputItem``.

    Returns:
        True if the item is a vLLM-generated state carrier.
    """
    ec = getattr(item, "encrypted_content", None)
    return bool(ec and isinstance(ec, str) and ec.startswith(f"{_FORMAT_VERSION}:"))
