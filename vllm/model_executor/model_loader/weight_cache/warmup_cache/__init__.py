# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Daemon warmup caches, seeded daemon-to-daemon over the shared transport."""

from typing import TYPE_CHECKING, Any, Protocol

from .common import (
    BUNDLE_LIMIT,
    CacheManifest,
    CacheRoot,
    _bundle,
    _cache_roots,
    _install,
    _manifest,
    _safe_path,
)
from .compiled import CompileArtifactWarmupCache
from .flashinfer_autotune import FlashInferWarmupCache

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.weight_cache.daemon import WeightCacheDaemon
    from vllm.model_executor.model_loader.weight_cache.seed import SeedTransport


class WarmupCache(Protocol):
    """One daemon warmup cache and its cache-specific D2D semantics."""

    kind: str

    def warmup(
        self, daemon: "WeightCacheDaemon", transport: "SeedTransport"
    ) -> None: ...

    def request_params(self, ctx: Any) -> dict[str, Any]: ...
    def resolve(self, request: dict[str, Any]) -> Any | None: ...
    def validate_payload(self, payload: Any) -> Any: ...
    def install(self, ctx: Any, payload: Any) -> None: ...


__all__ = [
    "BUNDLE_LIMIT",
    "CacheManifest",
    "CacheRoot",
    "CompileArtifactWarmupCache",
    "FlashInferWarmupCache",
    "WarmupCache",
    "_bundle",
    "_cache_roots",
    "_install",
    "_manifest",
    "_safe_path",
]
