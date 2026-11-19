# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup cache lifecycle interface."""

from typing import TYPE_CHECKING

from vllm.model_executor.model_loader.weight_cache.seed import Seedable

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.weight_cache.daemon import WeightCacheDaemon
    from vllm.model_executor.model_loader.weight_cache.seed import SeedTransport


class WarmupCache(Seedable):
    """One daemon warmup cache and its cache-specific D2D semantics."""

    def adopt_before_load(
        self, daemon: "WeightCacheDaemon", transport: "SeedTransport"
    ) -> None:
        pass

    def warmup(
        self, daemon: "WeightCacheDaemon", transport: "SeedTransport"
    ) -> None: ...
