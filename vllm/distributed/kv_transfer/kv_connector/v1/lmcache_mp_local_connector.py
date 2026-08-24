# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy bridge to LMCache's process-local MP connector implementation."""

try:
    from lmcache.integration.vllm.lmcache_mp_local_connector import (
        LMCacheMPLocalConnector,
    )
except ImportError as error:
    raise ImportError(
        "The configured LMCache in-process MP deployment requires an LMCache "
        "build that provides LMCacheMPLocalConnector. Install the matching "
        "LMCache version or select lmcache.mp.deployment='external'."
    ) from error

__all__ = ["LMCacheMPLocalConnector"]
