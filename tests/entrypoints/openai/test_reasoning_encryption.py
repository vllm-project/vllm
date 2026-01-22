# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for reasoning content encryption module."""

import json
import os
from unittest import mock

import pytest


def _has_cryptography() -> bool:
    """Check if cryptography package is available."""
    try:
        import cryptography  # noqa: F401

        return True
    except ImportError:
        return False


class TestEncryptionDisabled:
    """Tests when encryption is disabled (default)."""

    def setup_method(self):
        """Reset encryption state before each test."""
        from vllm.entrypoints.openai.reasoning_encryption import (
            _reset_encryption_state,
        )

        _reset_encryption_state()

    def teardown_method(self):
        """Reset encryption state after each test."""
        from vllm.entrypoints.openai.reasoning_encryption import (
            _reset_encryption_state,
        )

        _reset_encryption_state()

    def test_is_encryption_disabled_by_default(self):
        """Test that encryption is disabled by default."""
        with mock.patch.dict(os.environ, {}, clear=False):
            # Ensure the env var is not set
            os.environ.pop("VLLM_ENCRYPT_REASONING_CONTENT", None)

            from vllm.entrypoints.openai.reasoning_encryption import (
                _reset_encryption_state,
                is_encryption_enabled,
            )

            _reset_encryption_state()
            assert is_encryption_enabled() is False

    def test_encrypt_returns_plain_json_when_disabled(self):
        """Test that encrypt returns plain JSON when encryption is disabled."""
        with mock.patch.dict(
            os.environ, {"VLLM_ENCRYPT_REASONING_CONTENT": "0"}, clear=False
        ):
            from vllm.entrypoints.openai.reasoning_encryption import (
                _reset_encryption_state,
                encrypt_reasoning_content,
            )

            _reset_encryption_state()

            content = [{"type": "reasoning_text", "text": "test reasoning"}]
            result = encrypt_reasoning_content("rs_123", content)

            # Should be valid JSON
            parsed = json.loads(result)
            assert parsed["id"] == "rs_123"
            assert parsed["content"] == content

    def test_decrypt_parses_plain_json_when_disabled(self):
        """Test that decrypt parses plain JSON when encryption is disabled."""
        with mock.patch.dict(
            os.environ, {"VLLM_ENCRYPT_REASONING_CONTENT": "0"}, clear=False
        ):
            from vllm.entrypoints.openai.reasoning_encryption import (
                _reset_encryption_state,
                decrypt_reasoning_content,
            )

            _reset_encryption_state()

            content = [{"type": "reasoning_text", "text": "test reasoning"}]
            plain_json = json.dumps({"id": "rs_456", "content": content})
            result = decrypt_reasoning_content(plain_json)

            assert result is not None
            assert result["id"] == "rs_456"
            assert result["content"] == content


class TestEncryptionEnabled:
    """Tests when encryption is enabled."""

    def setup_method(self):
        """Reset encryption state before each test."""
        from vllm.entrypoints.openai.reasoning_encryption import (
            _reset_encryption_state,
        )

        _reset_encryption_state()

    def teardown_method(self):
        """Reset encryption state after each test."""
        from vllm.entrypoints.openai.reasoning_encryption import (
            _reset_encryption_state,
        )

        _reset_encryption_state()

    @pytest.mark.skipif(
        not _has_cryptography(), reason="cryptography package not installed"
    )
    def test_is_encryption_enabled_when_env_set(self):
        """Test that encryption is enabled when env var is set."""
        with mock.patch.dict(
            os.environ, {"VLLM_ENCRYPT_REASONING_CONTENT": "1"}, clear=False
        ):
            from vllm.entrypoints.openai.reasoning_encryption import (
                _reset_encryption_state,
                is_encryption_enabled,
            )

            _reset_encryption_state()
            assert is_encryption_enabled() is True

    @pytest.mark.skipif(
        not _has_cryptography(), reason="cryptography package not installed"
    )
    def test_encrypt_produces_base64_ciphertext(self):
        """Test that encrypt produces base64-encoded ciphertext when enabled."""
        with mock.patch.dict(
            os.environ, {"VLLM_ENCRYPT_REASONING_CONTENT": "1"}, clear=False
        ):
            from vllm.entrypoints.openai.reasoning_encryption import (
                _reset_encryption_state,
                encrypt_reasoning_content,
            )

            _reset_encryption_state()

            content = [{"type": "reasoning_text", "text": "test reasoning"}]
            result = encrypt_reasoning_content("rs_123", content)

            # AES-256-GCM output is base64-encoded (nonce + ciphertext + tag)
            import base64

            decoded = base64.b64decode(result)
            # Should be at least nonce(12) + tag(16) + some ciphertext
            assert len(decoded) >= 28
            # Should not be valid JSON (it's encrypted)
            with pytest.raises(json.JSONDecodeError):
                json.loads(result)

    @pytest.mark.skipif(
        not _has_cryptography(), reason="cryptography package not installed"
    )
    def test_encrypt_decrypt_roundtrip(self):
        """Test that content can be encrypted and decrypted."""
        with mock.patch.dict(
            os.environ, {"VLLM_ENCRYPT_REASONING_CONTENT": "1"}, clear=False
        ):
            from vllm.entrypoints.openai.reasoning_encryption import (
                _reset_encryption_state,
                decrypt_reasoning_content,
                encrypt_reasoning_content,
            )

            _reset_encryption_state()

            content = [{"type": "reasoning_text", "text": "secret reasoning"}]
            encrypted = encrypt_reasoning_content("rs_secret", content)
            decrypted = decrypt_reasoning_content(encrypted)

            assert decrypted is not None
            assert decrypted["id"] == "rs_secret"
            assert decrypted["content"] == content

    @pytest.mark.skipif(
        not _has_cryptography(), reason="cryptography package not installed"
    )
    def test_decrypt_fallback_to_plain_json(self):
        """Test that decrypt falls back to plain JSON if decryption fails."""
        with mock.patch.dict(
            os.environ, {"VLLM_ENCRYPT_REASONING_CONTENT": "1"}, clear=False
        ):
            from vllm.entrypoints.openai.reasoning_encryption import (
                _reset_encryption_state,
                decrypt_reasoning_content,
            )

            _reset_encryption_state()

            # Plain JSON from older server or when encryption was disabled
            content = [{"type": "reasoning_text", "text": "plain content"}]
            plain_json = json.dumps({"id": "rs_plain", "content": content})
            result = decrypt_reasoning_content(plain_json)

            assert result is not None
            assert result["id"] == "rs_plain"
            assert result["content"] == content


class TestCustomEncryptionKey:
    """Tests for custom encryption key configuration."""

    def setup_method(self):
        """Reset encryption state before each test."""
        from vllm.entrypoints.openai.reasoning_encryption import (
            _reset_encryption_state,
        )

        _reset_encryption_state()

    def teardown_method(self):
        """Reset encryption state after each test."""
        from vllm.entrypoints.openai.reasoning_encryption import (
            _reset_encryption_state,
        )

        _reset_encryption_state()

    @pytest.mark.skipif(
        not _has_cryptography(), reason="cryptography package not installed"
    )
    def test_uses_custom_key(self):
        """Test that a custom encryption key can be used."""
        import base64
        import secrets

        # Generate a 32-byte key for AES-256
        custom_key = base64.b64encode(secrets.token_bytes(32)).decode()

        with mock.patch.dict(
            os.environ,
            {
                "VLLM_ENCRYPT_REASONING_CONTENT": "1",
                "VLLM_REASONING_ENCRYPTION_KEY": custom_key,
            },
            clear=False,
        ):
            from vllm.entrypoints.openai.reasoning_encryption import (
                _reset_encryption_state,
                decrypt_reasoning_content,
                encrypt_reasoning_content,
            )

            _reset_encryption_state()

            content = [{"type": "reasoning_text", "text": "with custom key"}]
            encrypted = encrypt_reasoning_content("rs_custom", content)
            decrypted = decrypt_reasoning_content(encrypted)

            assert decrypted is not None
            assert decrypted["id"] == "rs_custom"

    @pytest.mark.skipif(
        not _has_cryptography(), reason="cryptography package not installed"
    )
    def test_invalid_custom_key_disables_encryption(self):
        """Test that invalid custom key disables encryption."""
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_ENCRYPT_REASONING_CONTENT": "1",
                "VLLM_REASONING_ENCRYPTION_KEY": "invalid_key",
            },
            clear=False,
        ):
            from vllm.entrypoints.openai.reasoning_encryption import (
                _reset_encryption_state,
                is_encryption_enabled,
            )

            _reset_encryption_state()
            # Should be disabled due to invalid key
            assert is_encryption_enabled() is False


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def setup_method(self):
        """Reset encryption state before each test."""
        from vllm.entrypoints.openai.reasoning_encryption import (
            _reset_encryption_state,
        )

        _reset_encryption_state()

    def teardown_method(self):
        """Reset encryption state after each test."""
        from vllm.entrypoints.openai.reasoning_encryption import (
            _reset_encryption_state,
        )

        _reset_encryption_state()

    def test_decrypt_empty_string_returns_none(self):
        """Test that decrypting empty string returns None."""
        from vllm.entrypoints.openai.reasoning_encryption import (
            decrypt_reasoning_content,
        )

        assert decrypt_reasoning_content("") is None

    def test_decrypt_none_returns_none(self):
        """Test that decrypting None returns None."""
        from vllm.entrypoints.openai.reasoning_encryption import (
            decrypt_reasoning_content,
        )

        assert decrypt_reasoning_content(None) is None

    def test_decrypt_invalid_content_returns_none(self):
        """Test that decrypting invalid content returns None."""
        from vllm.entrypoints.openai.reasoning_encryption import (
            decrypt_reasoning_content,
        )

        assert decrypt_reasoning_content("not_valid_json_or_encrypted") is None

    def test_encrypt_empty_content(self):
        """Test encrypting with empty content list."""
        with mock.patch.dict(
            os.environ, {"VLLM_ENCRYPT_REASONING_CONTENT": "0"}, clear=False
        ):
            from vllm.entrypoints.openai.reasoning_encryption import (
                _reset_encryption_state,
                decrypt_reasoning_content,
                encrypt_reasoning_content,
            )

            _reset_encryption_state()

            result = encrypt_reasoning_content("rs_empty", [])
            decrypted = decrypt_reasoning_content(result)

            assert decrypted is not None
            assert decrypted["id"] == "rs_empty"
            assert decrypted["content"] == []
