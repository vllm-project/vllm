# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request limits on KV offloading, shared by store-backed connectors."""

from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def _parse_token_limit(params: dict[str, Any] | None, key: str) -> int | None:
    """Parse a non-negative int limit, preserving zero and rejecting booleans."""
    raw = params.get(key) if params else None
    if type(raw) is int and raw >= 0:
        return raw
    if raw is not None:
        logger.warning("%s must be a non-negative int, got %r; ignoring", key, raw)
    return None


@dataclass(frozen=True, slots=True)
class RequestOffloadLimits:
    """Token limits a request sets through its `kv_transfer_params`.

    NOTE: These fields are experimental and subject to change in the future.
    """

    # Upper bound on the prefix stored externally; 0 disables stores.
    max_offload_tokens: int | None = None
    # Upper bound on tokens loaded beyond the locally computed prefix.
    max_load_tokens: int | None = None

    @classmethod
    def from_kv_transfer_params(
        cls, params: dict[str, Any] | None
    ) -> "RequestOffloadLimits":
        return cls(
            max_offload_tokens=_parse_token_limit(params, "max_offload_tokens"),
            max_load_tokens=_parse_token_limit(params, "max_load_tokens"),
        )

    @property
    def store_disabled(self) -> bool:
        return self.max_offload_tokens == 0

    def cap_store(self, num_tokens: int) -> int:
        """Clamp a stored-prefix length to `max_offload_tokens`."""
        if self.max_offload_tokens is None:
            return num_tokens
        return min(num_tokens, self.max_offload_tokens)

    def can_store(self, num_tokens: int) -> bool:
        """Whether a prefix of `num_tokens` may be stored."""
        return self.max_offload_tokens is None or num_tokens <= self.max_offload_tokens

    def cap_load(self, end_tokens: int, num_computed_tokens: int) -> int:
        """Clamp a load ending at `end_tokens` to `max_load_tokens` past
        `num_computed_tokens`."""
        if self.max_load_tokens is None:
            return end_tokens
        return min(end_tokens, num_computed_tokens + self.max_load_tokens)
