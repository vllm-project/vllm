# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Valkey client surface the valkey tier depends on.

Kept free of glide imports so the tier and its tests load without
valkey-glide-sync installed; the glide implementation lives in glide_client.
"""

from typing import Protocol


class ValkeyClient(Protocol):
    def set(self, key: str, value: memoryview, ttl_seconds: int | None) -> None:
        """Store ``value`` at ``key``, expiring after ``ttl_seconds`` if set."""
        ...

    def get_into(self, key: str, buffer: memoryview) -> int | None:
        """Copy the value at ``key`` into ``buffer``.

        Returns:
            Bytes written, or None if the key does not exist.

        Raises:
            Exception: If the value is larger than ``buffer``.

        """
        ...

    def exists_each(self, keys: list[str]) -> list[bool]:
        """Whether each key exists, in one pipelined round trip."""
        ...

    def close(self) -> None: ...
