# SPDX-License-Identifier: Apache-2.0
"""Public LMCache SDK helpers."""

# First Party
from lmcache.sdk import batch, context, kvcache, qcache, request
from lmcache.sdk.cache_kind import LMCacheSDKCacheKind

__all__ = ["batch", "context", "kvcache", "qcache", "request", "LMCacheSDKCacheKind"]


def connect(
    kind: LMCacheSDKCacheKind,
    url: str,
    http_url: str,
    model_name: str,
    timeout: float = 60.0,
) -> context.LMCacheSDKContext:
    """Connect to the LMCache server and return a context for the given kind.
    Calls ``kvcache.connect`` or ``qcache.connect`` based on ``kind``.

    Args:
        kind: Which cache to connect to (``LMCacheSDKCacheKind.KV`` or
            ``LMCacheSDKCacheKind.QUERY``).
        url: ZMQ endpoint URL for the LMCache message queue.
        http_url: HTTP endpoint URL for fetching server configuration.
        model_name: Model name used by the running LMCache server instance.
        timeout: Timeout in seconds for blocking MQ calls. Defaults to 60.

    Returns:
        An initialized LMCacheSDKContext for the requested kind.

    Raises:
        ValueError: If ``kind`` is not a supported cache kind.
    """
    if kind == LMCacheSDKCacheKind.KV:
        return kvcache.connect(
            url=url, http_url=http_url, model_name=model_name, timeout=timeout
        )
    if kind == LMCacheSDKCacheKind.QUERY:
        return qcache.connect(
            url=url, http_url=http_url, model_name=model_name, timeout=timeout
        )
    raise ValueError(f"unsupported cache kind: {kind}")
