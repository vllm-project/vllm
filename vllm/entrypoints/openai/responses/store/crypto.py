from __future__ import annotations

import os
import struct

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .key_provider import KeyProvider


class DataDecryptionError(ValueError):
    """Raised when encrypted disk data is malformed or fails authentication."""


class FramedAESGCMCipher:
    """Encrypts independently appendable frames with AES-GCM."""

    _MAGIC = b"VRS1"
    _NONCE_SIZE = 12
    _TAG_SIZE = 16
    _HEADER = struct.Struct(">4sI12s")

    def __init__(self, key_provider: KeyProvider) -> None:
        key = key_provider.get_key()
        if len(key) not in (16, 24, 32):
            raise ValueError("AES-GCM key must be 16, 24, or 32 bytes")
        self._cipher = AESGCM(key)

    def encrypt(self, plaintext: bytes, associated_data: bytes) -> bytes:
        nonce = os.urandom(self._NONCE_SIZE)
        ciphertext = self._cipher.encrypt(nonce, plaintext, associated_data)
        return self._HEADER.pack(self._MAGIC, len(ciphertext), nonce) + ciphertext

    def decrypt(self, framed_data: bytes, associated_data: bytes) -> bytes:
        data = memoryview(framed_data)
        plaintext = bytearray()
        offset = 0

        while offset < len(data):
            if len(data) - offset < self._HEADER.size:
                raise DataDecryptionError("truncated encrypted frame header")

            magic, ciphertext_size, nonce = self._HEADER.unpack_from(data, offset)
            if magic != self._MAGIC:
                raise DataDecryptionError("invalid encrypted frame marker")
            if ciphertext_size < self._TAG_SIZE:
                raise DataDecryptionError("invalid encrypted frame size")

            offset += self._HEADER.size
            frame_end = offset + ciphertext_size
            if frame_end > len(data):
                raise DataDecryptionError("truncated encrypted frame payload")

            try:
                plaintext.extend(
                    self._cipher.decrypt(
                        nonce,
                        data[offset:frame_end],
                        associated_data,
                    )
                )
            except InvalidTag as exc:
                raise DataDecryptionError(
                    "encrypted frame authentication failed"
                ) from exc
            offset = frame_end

        return bytes(plaintext)
