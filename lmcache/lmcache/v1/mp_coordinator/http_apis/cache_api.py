# SPDX-License-Identifier: Apache-2.0
"""Cache-control endpoints on the coordinator (fleet-level).

Warm-prefetch dispatch to a named MP server, thin over the
:class:`PrefetchManager` on the typed :class:`CoordinatorContext` (resolved via
:func:`get_context`). Handlers map fleet-routing failures to HTTP directly --
``404`` for an unknown ``instance_id`` and ``502`` when an MP server is
unreachable or rejects a proxied call.

Quota writes, combined quota+usage status, and usage-event ingestion are
accounting concerns and live in the ``/quota`` group (:mod:`quota_api`).
"""

# Standard
from dataclasses import asdict

# Third Party
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx

# First Party
from lmcache.v1.distributed.api import Tier
from lmcache.v1.mp_coordinator.http_apis.dependencies import (
    get_context,
    get_outbound_client,
)
from lmcache.v1.mp_coordinator.schemas import (
    DeleteRequest,
    DeleteResponse,
    PinRequest,
    PinResponse,
    PrefetchRequest,
    PrefetchResponse,
)
from lmcache.v1.multiprocess.cache_control.key_resolver import resolve_object_keys

router = APIRouter()


# -- Prefetch dispatch -------------------------------------------------------


@router.post("/cache/prefetches")
async def request_prefetch(body: PrefetchRequest, request: Request) -> PrefetchResponse:
    """Submit a warm prefetch of a token sequence on one MP server.

    Resolves ``body.instance_id`` in the registry and proxies to that server's
    ``POST /cache/prefetches``, which submits the load and returns a
    ``request_id``. Poll ``GET /cache/prefetches/{instance_id}/{request_id}``.

    Args:
        body: Target instance, model/world_size, the token_ids to warm, and the
            per-tenant cache_salt.

    Returns:
        ``PrefetchResponse`` carrying the server's ``request_id`` (empty when the
        sequence was shorter than one chunk -- ``status`` ``"noop"``).

    Raises:
        HTTPException: 404 if ``instance_id`` is not registered; 502 if the
            target server is unreachable or rejects the submit.
    """
    ctx = get_context(request)
    target = ctx.registry.get(body.instance_id)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"no MP server registered with instance_id={body.instance_id!r}",
        )

    try:
        result = await ctx.prefetch_manager.submit_prefetch(
            target=target,
            http_client=get_outbound_client(request),
            model_name=body.model_name,
            world_size=body.world_size,
            token_ids=body.token_ids,
            cache_salt=body.cache_salt,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"prefetch submit to {body.instance_id!r} failed: {exc}",
        ) from None

    return PrefetchResponse(
        instance_id=body.instance_id,
        request_id=result.get("request_id", ""),
        chunks=result.get("chunks", 0),
        status=result.get("status", "submitted"),
    )


@router.get("/cache/prefetches/{instance_id}/{request_id}")
async def get_prefetch_status(
    instance_id: str, request_id: str, request: Request
) -> JSONResponse:
    """Proxy a warm-prefetch status poll to the owning MP server.

    The warm holds no lock, so this poll only reports progress; the first poll
    that observes completion drops the job on the server (exactly-once). Poll
    until ``"completed"``.

    Args:
        instance_id: The MP server the prefetch was submitted to.
        request_id: The id returned by ``POST /cache/prefetches``.

    Returns:
        The server's status body relayed verbatim with its status code (200
        ``pending`` / ``completed``, or 404 for an unknown id).

    Raises:
        HTTPException: 404 if ``instance_id`` is not registered; 502 if the
            target server is unreachable.
    """
    ctx = get_context(request)
    target = ctx.registry.get(instance_id)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"no MP server registered with instance_id={instance_id!r}",
        )

    try:
        code, payload = await ctx.prefetch_manager.get_status(
            target=target,
            http_client=get_outbound_client(request),
            request_id=request_id,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"prefetch status from {instance_id!r} failed: {exc}",
        ) from None

    return JSONResponse(status_code=code, content=payload)


# -- Pin / unpin (L2 eviction protection) ------------------------------------


@router.post("/cache/pins")
async def request_pin(body: PinRequest, request: Request) -> PinResponse:
    """Pin a token sequence's keys in the L2 eviction plan.

    The coordinator resolves the token sequence to its object keys locally and
    records them so they are excluded from its L2 eviction plan (fleet-wide,
    per ``cache_salt``). No MP-server round-trip.

    Args:
        body: model/world_size, token_ids, cache_salt.

    Returns:
        ``PinResponse`` with ``requested`` chunks and ``affected`` L2 keys pinned.

    Raises:
        HTTPException: 400 if the token cap is exceeded or a key field is invalid.
    """
    ctx = get_context(request)
    try:
        resolved, chunks = resolve_object_keys(
            ctx.token_hasher,
            body.model_name,
            body.world_size,
            body.token_ids,
            body.cache_salt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    ctx.eviction_manager.pin(resolved)
    return PinResponse(
        requested=chunks,
        affected=len(resolved),
        status="pinned",
    )


@router.delete("/cache/pins")
async def request_unpin(body: PinRequest, request: Request) -> PinResponse:
    """Unpin a token sequence's keys from the L2 eviction plan.

    Symmetric with pin: resolves the keys locally and releases them, making them
    eligible for L2 eviction again.

    Args:
        body: model/world_size, token_ids, cache_salt.

    Returns:
        ``PinResponse`` with ``requested`` chunks and ``affected`` L2 keys unpinned.

    Raises:
        HTTPException: 400 if the token cap is exceeded or a key field is invalid.
    """
    ctx = get_context(request)
    try:
        resolved, chunks = resolve_object_keys(
            ctx.token_hasher,
            body.model_name,
            body.world_size,
            body.token_ids,
            body.cache_salt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    ctx.eviction_manager.unpin(resolved)
    return PinResponse(
        requested=chunks,
        affected=len(resolved),
        status="unpinned",
    )


# -- Delete dispatch ---------------------------------------------------------


@router.post("/cache/delete")
async def request_delete(body: DeleteRequest, request: Request) -> DeleteResponse:
    """Delete a token sequence on one MP server, per ``body.tier`` (l1 / l2 / all).

    Args:
        body: Target instance, model/world_size, token_ids, cache_salt, tier,
            force.

    Returns:
        ``DeleteResponse`` with ``requested`` chunks, ``affected`` keys removed,
        and ``skipped`` keys refused (L1 locks reported by the node, plus L2 pins
        held back by the coordinator).

    Raises:
        HTTPException: 400 if the token cap is exceeded or a key field is invalid;
            404 if ``instance_id`` is not registered; 502 if the target server is
            unreachable or rejects the delete.
    """
    ctx = get_context(request)
    target = ctx.registry.get(body.instance_id)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"no MP server registered with instance_id={body.instance_id!r}",
        )
    try:
        resolved, chunks = resolve_object_keys(
            ctx.token_hasher,
            body.model_name,
            body.world_size,
            body.token_ids,
            body.cache_salt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    if not chunks:
        return DeleteResponse(
            instance_id=body.instance_id,
            requested=0,
            affected=0,
            skipped=0,
            status="noop",
        )

    # When the delete touches L2, hold back L2-pinned keys (non-force). They are
    # dropped from the delete set entirely, so a pinned key is retained in both
    # tiers; ``force`` deletes them and clears the pins.
    touches_l2 = body.tier in (Tier.L2, Tier.ALL)
    if touches_l2 and not body.force:
        delete_keys = ctx.eviction_manager.filter_unpinned(resolved)
        pin_skipped = len(resolved) - len(delete_keys)
    else:
        delete_keys = resolved
        pin_skipped = 0

    node_deleted = 0
    node_skipped = 0
    if delete_keys:
        client = get_outbound_client(request)
        url = f"http://{target.ip}:{target.http_port}/cache/objects"
        payload = {
            "keys": [asdict(k.to_encoded_object_key()) for k in delete_keys],
            "tier": body.tier.value,
            "force": body.force,
        }
        try:
            # httpx ``.delete`` can't take ``json=``; use ``request(...)``.
            resp = await client.request("DELETE", url, json=payload)
            resp.raise_for_status()
            result = resp.json()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"delete to {body.instance_id!r} failed: {exc}",
            ) from None
        node_deleted = result.get("deleted", 0)
        node_skipped = result.get("skipped", 0)

    if touches_l2 and body.force:
        ctx.eviction_manager.drop_pins(resolved)

    # ``skipped`` = L1 keys the node refused (locks) + L2 keys the coordinator
    # held back for a pin.
    return DeleteResponse(
        instance_id=body.instance_id,
        requested=chunks,
        affected=node_deleted,
        skipped=node_skipped + pin_skipped,
        status="deleted",
    )
