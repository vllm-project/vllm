# SPDX-License-Identifier: Apache-2.0
"""Key-directory endpoints on the coordinator (fleet-level).

The ``/directory`` surface, thin over the :class:`KeyDirectory` carried on
the typed :class:`CoordinatorContext`: cache-event ingestion from MP
servers, placement lookup, and stats. See
``docs/design/v1/mp_coordinator/key_directory.md``.
"""

# Third Party
from fastapi import APIRouter, HTTPException, Request

# First Party
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.key_directory import ApplyResult, DirectoryStats
from lmcache.v1.mp_coordinator.schemas import (
    DirectoryEventsRequest,
    DirectoryEventsResponse,
    DirectoryKeyPlacements,
    DirectoryLookupRequest,
    DirectoryLookupResponse,
    TokenPlacementLookupRequest,
    TokenPlacementLookupResponse,
)
from lmcache.v1.multiprocess.cache_control.key_resolver import resolve_object_keys

router = APIRouter()


@router.post("/directory/events")
async def report_cache_events(
    body: DirectoryEventsRequest, request: Request
) -> DirectoryEventsResponse:
    """Apply a batch of cache-event batches to the key directory.

    Batches are applied in list order; per instance they must be sent in
    emission order. Duplicate and stale-incarnation batches are dropped
    and counted, not errors.

    Args:
        body: The event batches to apply.

    Returns:
        Counts of applied and dropped batches.
    """
    directory = get_context(request).key_directory
    response = DirectoryEventsResponse()
    for batch in body.batches:
        result = directory.apply_batch(batch)
        if result == ApplyResult.APPLIED:
            response.applied += 1
        elif result == ApplyResult.DUPLICATE:
            response.duplicates += 1
        else:
            response.stale += 1
    return response


@router.post("/directory/lookup")
async def lookup_placements(
    body: DirectoryLookupRequest, request: Request
) -> DirectoryLookupResponse:
    """Resolve keys to their known placements across the fleet.

    Args:
        body: The keys to resolve.

    Returns:
        One result per requested key, in request order; placements are
        empty for unknown keys.
    """
    directory = get_context(request).key_directory
    keys = [encoded.to_object_key() for encoded in body.keys]
    return DirectoryLookupResponse(
        results=[
            DirectoryKeyPlacements(key=encoded, placements=placements)
            for encoded, placements in zip(
                body.keys, directory.lookup(keys), strict=True
            )
        ]
    )


@router.post("/directory/lookup_tokens")
async def lookup_placements_by_tokens(
    body: TokenPlacementLookupRequest, request: Request
) -> TokenPlacementLookupResponse:
    """Resolve a token sequence to keys and return their placements.

    Hashes ``token_ids`` with the fleet's token hasher, expands each
    complete chunk into its per-rank object keys, and looks each key up
    in the directory.

    Args:
        body: The token sequence and the key-resolution parameters.

    Returns:
        Chunk count plus one result per resolved key; empty when the
        sequence is shorter than one chunk.

    Raises:
        HTTPException: 400 when the token sequence exceeds the
            per-request cap or a key field is invalid.
    """
    ctx = get_context(request)
    try:
        obj_keys, chunks = resolve_object_keys(
            token_hasher=ctx.token_hasher,
            model_name=body.model_name,
            world_size=body.world_size,
            token_ids=body.token_ids,
            cache_salt=body.cache_salt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TokenPlacementLookupResponse(
        chunks=chunks,
        results=[
            DirectoryKeyPlacements(
                key=key.to_encoded_object_key(), placements=placements
            )
            for key, placements in zip(
                obj_keys, ctx.key_directory.lookup(obj_keys), strict=True
            )
        ],
    )


@router.get("/directory/stats")
async def directory_stats(request: Request) -> DirectoryStats:
    """Return a point-in-time summary of directory contents.

    Returns:
        Key/placement counts plus per-instance stream state (incarnation,
        last applied seq, gap flag), keyed by ``instance_id``.
    """
    return get_context(request).key_directory.stats()
