# SPDX-License-Identifier: Apache-2.0
"""REST resource for the global CacheBlend fingerprint directory.

Blend mp-servers publish chunk fingerprints on STORE (``POST
/blend/fingerprints``) and query them on LOOKUP (``POST /blend/match``).
Endpoints operate on the shared :class:`GlobalBlendMatcher` reached via
``app.state.blend_directory``.
"""

# Third Party
from fastapi import APIRouter, Request

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.blend_directory import (
    GlobalBlendMatcher,
    StoreRange,
)
from lmcache.v1.mp_coordinator.schemas import (
    BlendEvictRequest,
    BlendEvictResponse,
    BlendFingerprintRequest,
    BlendFingerprintResponse,
    BlendMatchRequest,
    BlendMatchResponse,
    GlobalMatchModel,
    decode_tokens,
)

logger = init_logger(__name__)

router = APIRouter()


def _directory(request: Request) -> GlobalBlendMatcher:
    """Return the shared blend directory from app state.

    Args:
        request: The incoming request.

    Returns:
        The shared :class:`GlobalBlendMatcher`.

    Raises:
        RuntimeError: If the directory is not initialized (wired by
            ``create_app``, so this should not happen in practice).
    """
    directory = getattr(request.app.state, "blend_directory", None)
    if directory is None:
        raise RuntimeError("blend directory not initialized")
    return directory


@router.post("/blend/fingerprints")
def publish_fingerprints(
    body: BlendFingerprintRequest, request: Request
) -> BlendFingerprintResponse:
    """Register published chunk fingerprints (idempotent).

    Returns:
        Number of fingerprints newly inserted.
    """
    directory = _directory(request)
    ranges = [
        StoreRange(
            model_scope=r.model_scope,
            tokens=r.tokens,
            object_keys=r.object_keys,
            old_st_base=r.old_st_base,
        )
        for r in body.ranges
    ]
    inserted = directory.register(ranges) if ranges else 0
    return BlendFingerprintResponse(inserted=inserted)


@router.delete("/blend/fingerprints")
def evict_fingerprints(body: BlendEvictRequest, request: Request) -> BlendEvictResponse:
    """Evict fingerprints by storage key (idempotent).

    Returns:
        Number of fingerprint entries removed.
    """
    directory = _directory(request)
    removed = directory.remove(body.object_keys) if body.object_keys else 0
    return BlendEvictResponse(removed=removed)


@router.post("/blend/match")
def match_fingerprints(body: BlendMatchRequest, request: Request) -> BlendMatchResponse:
    """Match a request's rolling-hash array against the directory.

    The request tokens arrive base64-packed (``tokens_b64``); they are decoded
    to a ``uint64`` array and handed straight to the matcher.

    Returns:
        Matched chunks (``object_key`` / ``old_st`` / ``cur_st``), ascending by
        ``cur_st``.
    """
    directory = _directory(request)
    matches = directory.match(body.model_scope, decode_tokens(body.tokens_b64))
    return BlendMatchResponse(
        matches=[
            GlobalMatchModel(object_key=m.object_key, old_st=m.old_st, cur_st=m.cur_st)
            for m in matches
        ]
    )
