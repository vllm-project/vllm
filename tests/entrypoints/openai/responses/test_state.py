# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the stateless Responses API state carrier (state.py)."""

import importlib
import os

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_signing_key():
    """Force state.py to re-derive the signing key on next call."""
    import vllm.entrypoints.openai.responses.state as state_mod

    state_mod._SIGNING_KEY = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_key(monkeypatch):
    """Each test gets a fresh, deterministic signing key."""
    monkeypatch.setenv(
        "VLLM_RESPONSES_STATE_SIGNING_KEY", "ab" * 32  # 64 hex chars = 32 bytes
    )
    _reset_signing_key()
    yield
    _reset_signing_key()


# ---------------------------------------------------------------------------
# Import after env is patched (in case module was already imported)
# ---------------------------------------------------------------------------


@pytest.fixture()
def state():
    import vllm.entrypoints.openai.responses.state as m

    return m


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


def test_roundtrip_plain_dicts(state):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    blob = state.serialize_state(messages)
    recovered = state.deserialize_state(blob)
    assert recovered == messages


def test_roundtrip_empty_list(state):
    blob = state.serialize_state([])
    recovered = state.deserialize_state(blob)
    assert recovered == []


def test_roundtrip_nested_structure(state):
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is 2+2?"}],
        },
        {"role": "assistant", "content": "4", "extra": {"key": [1, 2, 3]}},
    ]
    blob = state.serialize_state(messages)
    recovered = state.deserialize_state(blob)
    assert recovered == messages


def test_roundtrip_pydantic_model(state):
    """Objects with model_dump() should serialize transparently."""

    class FakeModel:
        def model_dump(self):
            return {"author": {"role": "user"}, "content": "hi"}

    messages = [FakeModel()]
    blob = state.serialize_state(messages)
    recovered = state.deserialize_state(blob)
    # After JSON round-trip, FakeModel becomes a plain dict
    assert recovered == [{"author": {"role": "user"}, "content": "hi"}]


# ---------------------------------------------------------------------------
# is_state_carrier
# ---------------------------------------------------------------------------


def test_is_state_carrier_true(state):
    blob = state.serialize_state([{"role": "user", "content": "hi"}])

    class FakeItem:
        encrypted_content = blob

    assert state.is_state_carrier(FakeItem())


def test_is_state_carrier_false_external(state):
    """Real encrypted_content from external models should not be detected."""

    class FakeItem:
        encrypted_content = "some-opaque-blob-from-openai"

    assert not state.is_state_carrier(FakeItem())


def test_is_state_carrier_false_no_field(state):
    class FakeItem:
        pass

    assert not state.is_state_carrier(FakeItem())


def test_is_state_carrier_false_none(state):
    class FakeItem:
        encrypted_content = None

    assert not state.is_state_carrier(FakeItem())


# ---------------------------------------------------------------------------
# deserialize_state — non-carrier inputs
# ---------------------------------------------------------------------------


def test_deserialize_returns_none_for_non_carrier(state):
    assert state.deserialize_state("some-random-opaque-string") is None


def test_deserialize_returns_none_for_empty_string(state):
    assert state.deserialize_state("") is None


# ---------------------------------------------------------------------------
# HMAC tamper detection
# ---------------------------------------------------------------------------


def test_tampered_payload_raises(state):
    blob = state.serialize_state([{"role": "user", "content": "original"}])
    # Corrupt the payload part (index 2 when split on ':')
    parts = blob.split(":", 3)
    assert len(parts) == 4
    parts[2] = parts[2][:-4] + "XXXX"  # corrupt the base64 payload
    tampered = ":".join(parts)
    with pytest.raises(ValueError, match="HMAC verification failed"):
        state.deserialize_state(tampered)


def test_tampered_sig_raises(state):
    blob = state.serialize_state([{"role": "user", "content": "hello"}])
    parts = blob.split(":", 3)
    parts[3] = "0" * 64  # replace HMAC with zeros
    tampered = ":".join(parts)
    with pytest.raises(ValueError, match="HMAC verification failed"):
        state.deserialize_state(tampered)


def test_malformed_carrier_raises(state):
    malformed = "vllm:1:onlythreeparts"
    with pytest.raises(ValueError, match="Malformed vLLM state carrier"):
        state.deserialize_state(malformed)


# ---------------------------------------------------------------------------
# Cross-key incompatibility
# ---------------------------------------------------------------------------


def test_different_keys_are_incompatible(monkeypatch):
    """A blob signed with key A must not validate with key B."""
    import vllm.entrypoints.openai.responses.state as state_mod

    monkeypatch.setenv("VLLM_RESPONSES_STATE_SIGNING_KEY", "aa" * 32)
    state_mod._SIGNING_KEY = None
    blob = state_mod.serialize_state([{"role": "user", "content": "secret"}])

    # Switch to a different key
    monkeypatch.setenv("VLLM_RESPONSES_STATE_SIGNING_KEY", "bb" * 32)
    state_mod._SIGNING_KEY = None

    with pytest.raises(ValueError, match="HMAC verification failed"):
        state_mod.deserialize_state(blob)


# ---------------------------------------------------------------------------
# Random-key warning (no env var)
# ---------------------------------------------------------------------------


def test_no_env_var_generates_random_key(monkeypatch):
    """Without the env var, a random 32-byte key is generated.

    The warning is emitted via vLLM's logger (visible in test output) but is
    not capturable via capsys/caplog since vLLM writes to sys.__stdout__ directly.
    """
    import vllm.entrypoints.openai.responses.state as state_mod

    monkeypatch.delenv("VLLM_RESPONSES_STATE_SIGNING_KEY", raising=False)
    state_mod._SIGNING_KEY = None

    key = state_mod._get_signing_key()

    assert key is not None
    assert len(key) == 32
    # A second call returns the same cached key (warning only fires once)
    key2 = state_mod._get_signing_key()
    assert key == key2


# ---------------------------------------------------------------------------
# Invalid hex key
# ---------------------------------------------------------------------------


def test_invalid_hex_key_raises(monkeypatch):
    import vllm.entrypoints.openai.responses.state as state_mod

    monkeypatch.setenv("VLLM_RESPONSES_STATE_SIGNING_KEY", "not-valid-hex!!")
    state_mod._SIGNING_KEY = None

    with pytest.raises(ValueError, match="valid hex string"):
        state_mod._get_signing_key()
