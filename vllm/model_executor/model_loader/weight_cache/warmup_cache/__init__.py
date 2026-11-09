# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Daemon warmup caches, seeded daemon-to-daemon over the shared transport."""

from typing import TYPE_CHECKING, Protocol

from vllm.model_executor.model_loader.weight_cache.seed import Seedable

from .compiled import CompileArtifactWarmupCache
from .flashinfer_autotune import FlashInferWarmupCache

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.weight_cache.daemon import WeightCacheDaemon
    from vllm.model_executor.model_loader.weight_cache.seed import SeedTransport


class WarmupCache(Seedable, Protocol):
    """One daemon warmup cache and its cache-specific D2D semantics."""

    def warmup(
        self, daemon: "WeightCacheDaemon", transport: "SeedTransport"
    ) -> None: ...


__all__ = [
    "CompileArtifactWarmupCache",
    "FlashInferWarmupCache",
    "WarmupCache",
]
