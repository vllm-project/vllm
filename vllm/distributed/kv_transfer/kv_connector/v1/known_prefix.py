# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Advisory prefix lengths obtained by normal connector lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.request import Request


@dataclass(frozen=True, slots=True)
class _KnownPrefix:
    request: Request
    num_tokens: int
    hashes: tuple[bytes, ...]
    matched_tokens: int


class KnownPrefixHits:
    """Retain reduced lookup hints while requests wait for allocation."""

    def __init__(self) -> None:
        self._hits: dict[str, _KnownPrefix] = {}

    def record(self, request: Request, matched_tokens: int) -> None:
        self._hits[request.request_id] = _KnownPrefix(
            request, request.num_tokens, tuple(request.block_hashes), matched_tokens
        )

    def get(self, request: Request) -> int | None:
        hit = self._hits.get(request.request_id)
        if (
            hit is None
            or request.skip_reading_external_kv
            or hit.request is not request
            or hit.num_tokens != request.num_tokens
            or hit.hashes != tuple(request.block_hashes)
        ):
            return None
        return hit.matched_tokens

    def discard(self, request_id: str) -> None:
        self._hits.pop(request_id, None)

    def clear(self) -> None:
        self._hits.clear()
