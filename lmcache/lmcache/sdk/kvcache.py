# SPDX-License-Identifier: Apache-2.0
"""
SDK for retrieving and storing KV cache tensors.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.sdk.context import LMCacheSDKCacheKind, LMCacheSDKContext

logger = init_logger(__name__)


def connect(
    url: str,
    http_url: str,
    model_name: str,
    timeout: float = 60.0,
) -> "LMCacheSDKContext":
    """Create and initialize the LMCache SDK context.

    Args:
        url: ZMQ endpoint URL for the LMCache message queue.
        http_url: HTTP endpoint URL for fetching server configuration.
        model_name: Model name used by the running LMCache server instance.
        timeout: Timeout in seconds for blocking MQ calls. Defaults to 60.

    Returns:
        An initialized LMCacheSDKContext instance.
        Ready to be passed to close(), retrieve(), and store() functions.
    """
    ctx = LMCacheSDKContext(
        url=url,
        http_url=http_url,
        model_name=model_name,
        kind=LMCacheSDKCacheKind.KV,
        timeout=timeout,
    )
    ctx.register_caches()
    return ctx
