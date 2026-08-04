# SPDX-License-Identifier: Apache-2.0
"""
SDK for retrieving Q tensors.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.sdk.context import LMCacheSDKCacheKind, LMCacheSDKContext

logger = init_logger(__name__)


def connect(
    url: str, http_url: str, model_name: str, timeout: float = 60.0
) -> LMCacheSDKContext:
    """Connect to the LMCache server and return a context for Q cache.
    The technique to get the query tensors are the same as KV cache,
    however, specific model name prefix needs to be used.

    Args:
        url: The MQ URL of the LMCache server.
        http_url: The HTTP URL of the LMCache server.
        model_name: The original model name.
        timeout: The timeout for the connection.

    Returns:
        An LMCacheSDKContext instance for Q cache.
    """
    ctx = LMCacheSDKContext(
        url=url,
        http_url=http_url,
        model_name=model_name,
        kind=LMCacheSDKCacheKind.QUERY,
        timeout=timeout,
    )
    ctx.register_caches()
    return ctx
