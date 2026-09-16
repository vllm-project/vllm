from __future__ import annotations

import base64
import binascii
import os
import secrets
from pathlib import Path
from typing import Protocol


class KeyProvider(Protocol):
    """Supplies symmetric encryption keys to the disk store."""

    def get_key(self) -> bytes: ...


class EphemeralKeyProvider:
    """Keeps a randomly generated AES-256 key in process memory."""

    _KEY_SIZE_BYTES = 32

    def __init__(self) -> None:
        self._key = secrets.token_bytes(self._KEY_SIZE_BYTES)

    def get_key(self) -> bytes:
        return self._key


class StaticKeyProvider:
    """Supplies a caller-provided key, primarily for dependency injection."""

    def __init__(self, key: bytes) -> None:
        self._key = bytes(key)

    def get_key(self) -> bytes:
        return self._key


class FileKeyProvider:
    """Loads and atomically rotates a Base64-encoded AES-256 key file."""

    _KEY_SIZE_BYTES = 32

    def __init__(self, key_path: str) -> None:
        if not key_path:
            raise ValueError("key_path must not be empty")

        self._path = Path(key_path)
        self._pending_path = self._path.with_name(f"{self._path.name}.pending")
        self._key = self._read_key(self._path)

    def get_key(self) -> bytes:
        return self._key

    def get_pending_key(self) -> bytes | None:
        """Return a staged key left by an interrupted rotation, if present."""
        if not self._pending_path.exists():
            return None
        return self._read_key(self._pending_path)

    def stage_new_key(self) -> bytes:
        """Generate and durably stage a new key beside the active key file."""
        key = secrets.token_bytes(self._KEY_SIZE_BYTES)
        self._atomic_write(self._pending_path, base64.b64encode(key) + b"\n")
        return key

    def promote_pending_key(self, key: bytes) -> None:
        """Atomically replace the active key file with its staged successor."""
        pending_key = self._read_key(self._pending_path)
        if not secrets.compare_digest(pending_key, key):
            raise ValueError("Responses store pending key changed during rotation")

        try:
            os.replace(self._pending_path, self._path)
            self._sync_parent_directory()
        except OSError as exc:
            raise RuntimeError(
                f"Unable to promote Responses store key file: {self._path}"
            ) from exc
        self._key = bytes(key)

    def discard_pending_key(self) -> None:
        """Remove a staged key that is not referenced by the database."""
        try:
            self._pending_path.unlink(missing_ok=True)
            self._sync_parent_directory()
        except OSError as exc:
            raise RuntimeError(
                f"Unable to remove stale Responses store key: {self._pending_path}"
            ) from exc

    @classmethod
    def _read_key(cls, path: Path) -> bytes:
        try:
            encoded_key = path.read_bytes().strip()
        except OSError as exc:
            raise ValueError(
                f"Unable to read Responses store key file: {path}"
            ) from exc

        try:
            key = base64.b64decode(encoded_key, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                "Responses store key file must contain valid Base64"
            ) from exc

        if len(key) != cls._KEY_SIZE_BYTES:
            raise ValueError(
                "Responses store key file must contain a Base64-encoded "
                "32-byte AES-256 key"
            )
        return key

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        temporary_path = path.with_name(
            f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
        )
        file_descriptor: int | None = None
        try:
            file_descriptor = os.open(
                temporary_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(file_descriptor, "wb") as file:
                file_descriptor = None
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
            FileKeyProvider._sync_directory(path.parent)
        except OSError as exc:
            raise RuntimeError(
                f"Unable to stage Responses store key file: {path}"
            ) from exc
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)
            temporary_path.unlink(missing_ok=True)

    def _sync_parent_directory(self) -> None:
        self._sync_directory(self._path.parent)

    @staticmethod
    def _sync_directory(path: Path) -> None:
        if os.name == "nt":
            return
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
