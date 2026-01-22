# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encryption utilities for reasoning content in the Responses API.

This module provides AES-256-GCM encryption/decryption for reasoning items,
enabling secure round-tripping in multi-turn conversations. When enabled via
VLLM_ENCRYPT_REASONING_CONTENT=1, reasoning content is encrypted before being
sent to clients and decrypted when received back.

Key management:
- A 256-bit key is generated at startup (per-process)
- The key can be overridden via VLLM_REASONING_ENCRYPTION_KEY env var
- Keys are 32 bytes, base64-encoded (44 characters)

Wire format:
- 12-byte nonce || ciphertext || 16-byte auth tag, all base64-encoded

Security notes:
- Keys generated at startup will be lost on restart, invalidating any
  encrypted content from previous sessions
- For production deployments with multiple instances, configure a shared
  key via VLLM_REASONING_ENCRYPTION_KEY
"""

import base64
import json
import logging
import os
import secrets

from vllm import envs

logger = logging.getLogger(__name__)

# Constants for AES-256-GCM
_NONCE_SIZE = 12  # 96 bits, recommended for GCM
_KEY_SIZE = 32  # 256 bits
_TAG_SIZE = 16  # 128 bits

# Lazy-loaded encryption components
_cipher_key: bytes | None = None
_encryption_enabled: bool | None = None


def _get_cipher_key() -> bytes | None:
    """Get or create the AES-256 key for encryption/decryption.

    Returns None if encryption is disabled or cryptography is unavailable.
    """
    global _cipher_key, _encryption_enabled

    if _encryption_enabled is not None:
        return _cipher_key

    # Check if encryption is enabled
    if not envs.VLLM_ENCRYPT_REASONING_CONTENT:
        _encryption_enabled = False
        _cipher_key = None
        return None

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
    except ImportError:
        logger.warning(
            "VLLM_ENCRYPT_REASONING_CONTENT=1 but cryptography package "
            "is not installed. Falling back to unencrypted content. "
            "Install with: pip install cryptography"
        )
        _encryption_enabled = False
        _cipher_key = None
        return None

    # Get or generate encryption key
    key_b64 = os.environ.get("VLLM_REASONING_ENCRYPTION_KEY")
    if key_b64:
        try:
            key = base64.b64decode(key_b64)
            if len(key) != _KEY_SIZE:
                raise ValueError(f"Key must be {_KEY_SIZE} bytes, got {len(key)}")
            _cipher_key = key
            logger.info("Using configured VLLM_REASONING_ENCRYPTION_KEY")
        except Exception as e:
            logger.error(
                "Invalid VLLM_REASONING_ENCRYPTION_KEY: %s. "
                "Key must be %d bytes, base64-encoded. "
                "Generate with: python -c 'import secrets, base64; "
                "print(base64.b64encode(secrets.token_bytes(32)).decode())'",
                e,
                _KEY_SIZE,
            )
            _encryption_enabled = False
            _cipher_key = None
            return None
    else:
        # Generate a new key for this session
        _cipher_key = secrets.token_bytes(_KEY_SIZE)
        logger.info(
            "Generated new reasoning encryption key for this session. "
            "Set VLLM_REASONING_ENCRYPTION_KEY to persist across restarts."
        )

    _encryption_enabled = True
    return _cipher_key


def encrypt_reasoning_content(reasoning_id: str, content: list[dict]) -> str:
    """Encrypt reasoning content for round-tripping.

    Args:
        reasoning_id: The reasoning item's ID
        content: List of content dicts with 'type' and 'text' keys

    Returns:
        Base64-encoded encrypted string if encryption is enabled,
        otherwise plain JSON string
    """
    payload = json.dumps({"id": reasoning_id, "content": content})

    key = _get_cipher_key()
    if key is None:
        # Encryption disabled, return plain JSON
        return payload

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        # Generate random nonce
        nonce = secrets.token_bytes(_NONCE_SIZE)

        # Encrypt with AES-256-GCM
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, payload.encode(), None)

        # Combine nonce + ciphertext (tag is appended by encrypt())
        encrypted = nonce + ciphertext

        # Base64 encode for transport
        return base64.b64encode(encrypted).decode()
    except Exception as e:
        logger.warning("Failed to encrypt reasoning content: %s", e)
        return payload


def decrypt_reasoning_content(encrypted_content: str) -> dict | None:
    """Decrypt reasoning content from encrypted_content field.

    Args:
        encrypted_content: The encrypted string from the client

    Returns:
        Dict with 'id' and 'content' keys, or None if decryption fails
    """
    if not encrypted_content:
        return None

    key = _get_cipher_key()
    if key is None:
        # Encryption disabled, try to parse as plain JSON
        try:
            return json.loads(encrypted_content)
        except (json.JSONDecodeError, TypeError):
            return None

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        # Base64 decode
        encrypted = base64.b64decode(encrypted_content)

        # Extract nonce and ciphertext
        if len(encrypted) < _NONCE_SIZE + _TAG_SIZE:
            raise ValueError("Encrypted content too short")

        nonce = encrypted[:_NONCE_SIZE]
        ciphertext = encrypted[_NONCE_SIZE:]

        # Decrypt with AES-256-GCM
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)

        return json.loads(plaintext)
    except Exception as e:
        # Could be:
        # - Content from a different server/session (different key)
        # - Content from when encryption was disabled (plain JSON)
        # - Corrupted/tampered content
        logger.debug("Failed to decrypt reasoning content: %s", e)

        # Fall back to trying plain JSON parse
        try:
            return json.loads(encrypted_content)
        except (json.JSONDecodeError, TypeError):
            return None


def is_encryption_enabled() -> bool:
    """Check if reasoning content encryption is enabled and working."""
    _get_cipher_key()  # Initialize if needed
    return _encryption_enabled or False


def _reset_encryption_state() -> None:
    """Reset encryption state for testing purposes.

    This function is intended for use in unit tests to ensure clean state
    between test cases. It should not be used in production code.
    """
    global _cipher_key, _encryption_enabled
    _cipher_key = None
    _encryption_enabled = None
