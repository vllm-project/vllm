# SPDX-License-Identifier: Apache-2.0
"""Genesis cache subpackage — v7.6+.

Houses the exact-match response cache (P41) and — in v7.7+ — a
semantic-similarity cache shim. Import entries only when enabled via
the corresponding env gate; module import itself is cheap and
side-effect-free."""

from vllm._genesis.cache.response_cache import (
    ResponseCacheLRU,
    is_p41_enabled,
    get_default_cache,
)

__all__ = [
    "ResponseCacheLRU",
    "is_p41_enabled",
    "get_default_cache",
]
