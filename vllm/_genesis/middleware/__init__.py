# SPDX-License-Identifier: Apache-2.0
"""Genesis middleware subpackage — v7.8+.

Drop-in middleware modules for the deployment layer (cliproxyapi /
FastAPI sidecar). They bridge the Genesis internal caches / metrics
to external HTTP-facing surfaces without touching the vLLM engine.

Current members:
  - `response_cache_middleware` — P50 cliproxyapi integration for
    P41 `ResponseCacheLRU` / `RedisResponseCache`. Short-circuits
    `POST /v1/chat/completions` + `POST /v1/completions` on cache
    hit without forwarding to vLLM.
"""

from vllm._genesis.middleware.response_cache_middleware import (
    ResponseCacheMiddleware,
    build_cache_key_from_request,
)

__all__ = [
    "ResponseCacheMiddleware",
    "build_cache_key_from_request",
]
