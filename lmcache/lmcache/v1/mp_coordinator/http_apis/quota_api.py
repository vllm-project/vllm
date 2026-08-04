# SPDX-License-Identifier: Apache-2.0
"""Quota and usage-accounting endpoints on the coordinator (fleet-level).

The ``/quota`` surface, thin over the collaborators on the typed
:class:`CoordinatorContext` (resolved via :func:`get_context`): quota writes,
combined quota+usage status reads, and usage-event ingestion from MP servers.
This mirrors the MP server's node-local ``/quota`` group; warm-prefetch dispatch
is genuine cache control and lives in :mod:`cache_api` instead.
"""

# Third Party
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.v1.distributed.api import Tier
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.schemas import (
    EventType,
    QuotaConfigRequest,
    QuotaConfigResponse,
    QuotaResponse,
    ReportUsageRequest,
    ReportUsageResponse,
    SetQuotaRequest,
    StatusListResponse,
    StatusResponse,
)

router = APIRouter()

_GB = 1024**3
_DEFAULT_SALT_SENTINEL = "_default"
# Quota / usage / status are L2-tier accounting today; other tiers are rejected.
_SUPPORTED_TIER = Tier.L2


def _resolve_salt_from_api_path(cache_salt: str) -> str:
    """Map the ``_default`` sentinel to the empty string."""
    return "" if cache_salt == _DEFAULT_SALT_SENTINEL else cache_salt


def _require_supported_tier(tier: Tier) -> None:
    """Raise ``HTTPException(400)`` unless ``tier`` is the supported one (``l2``)."""
    if tier != _SUPPORTED_TIER:
        raise HTTPException(
            status_code=400,
            detail=f"tier {tier.value!r} not supported; only {_SUPPORTED_TIER.value!r}",
        )


def _gb(n_bytes: int) -> float:
    """Convert bytes to GiB."""
    return n_bytes / _GB


# -- Quota config  ---------------------------------------------


@router.put("/quota/config")
async def set_quota_config(
    body: QuotaConfigRequest, request: Request
) -> QuotaConfigResponse:
    """Set the default quota applied to salts with no explicit entry.

    Args:
        body: The default limit to apply (and the ``tier`` it applies to).

    Returns:
        The applied config.
    """
    _require_supported_tier(body.tier)
    default_limit_bytes = (
        None if body.default_limit_gb is None else int(body.default_limit_gb * _GB)
    )
    get_context(request).quota_manager.set_default_limit_bytes(default_limit_bytes)
    return QuotaConfigResponse(default_limit_gb=body.default_limit_gb)


@router.get("/quota/config")
async def get_quota_config(
    request: Request, tier: Tier = Tier.L2
) -> QuotaConfigResponse:
    """Read the default quota applied to salts with no explicit entry.

    Args:
        tier: Cache tier (only ``l2`` is supported today).

    Returns:
        The current config; ``default_limit_gb`` is ``null`` while
        unquota'd salts are exempt from eviction (the boot default).
    """
    _require_supported_tier(tier)
    default_limit = get_context(request).quota_manager.get_default_limit_bytes()
    return QuotaConfigResponse(
        default_limit_gb=None if default_limit is None else _gb(default_limit)
    )


# -- Quota writes ------------------------------------------------------------


@router.put("/quota/{cache_salt}", response_model=None)
async def set_quota(
    cache_salt: str, body: SetQuotaRequest, request: Request
) -> QuotaResponse | JSONResponse:
    """Create or update a quota.

    Args:
        cache_salt: Tenant identifier; use ``_default`` for the empty salt.
        body: Quota limit to apply (and the ``tier`` it applies to).

    Returns:
        The applied quota, or a 400 JSON response if the limit is invalid.
    """
    _require_supported_tier(body.tier)
    cache_salt = _resolve_salt_from_api_path(cache_salt)
    limit_bytes = int(body.limit_gb * _GB)
    try:
        get_context(request).quota_manager.set_quota(cache_salt, limit_bytes)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "invalid quota limit"})
    return QuotaResponse(cache_salt=cache_salt, limit_gb=body.limit_gb, status="ok")


@router.delete("/quota/{cache_salt}")
async def delete_quota(
    cache_salt: str, request: Request, tier: Tier = Tier.L2
) -> QuotaResponse:
    """Remove a salt's quota entry.

    Args:
        cache_salt: Tenant identifier; use ``_default`` for the empty salt.
        tier: Cache tier (only ``l2`` is supported today).

    Returns:
        ``QuotaResponse`` with ``status`` ``"removed"`` or ``"not_found"``.
    """
    _require_supported_tier(tier)
    cache_salt = _resolve_salt_from_api_path(cache_salt)
    removed = get_context(request).quota_manager.delete_quota(cache_salt)
    return QuotaResponse(
        cache_salt=cache_salt,
        limit_gb=0.0,
        status="removed" if removed else "not_found",
    )


# -- Combined status queries -------------------------------------------------


@router.get("/quota/{cache_salt}")
async def get_status(
    cache_salt: str, request: Request, tier: Tier = Tier.L2
) -> StatusResponse:
    """Read quota and usage for a single salt.

    Args:
        cache_salt: Tenant identifier; use ``_default`` for the empty salt.
        tier: Cache tier (only ``l2`` is supported today).

    Returns:
        Combined quota and live usage detail.
    """
    _require_supported_tier(tier)
    cache_salt = _resolve_salt_from_api_path(cache_salt)
    ctx = get_context(request)
    usage = ctx.usage_manager.get(cache_salt)
    exists = ctx.quota_manager.has_quota(cache_salt)
    limit = ctx.quota_manager.get_limit_bytes(cache_salt)
    return StatusResponse(
        cache_salt=cache_salt,
        quota_limit_gb=_gb(limit) if exists else 0.0,
        quota_exists=exists,
        usage_gb=_gb(usage),
    )


@router.get("/quota")
async def list_status(request: Request, tier: Tier = Tier.L2) -> StatusListResponse:
    """List quota and usage across all cache salts.

    Args:
        tier: Cache tier (only ``l2`` is supported today).

    Returns:
        Total usage plus per-salt breakdown with quota info.
    """
    _require_supported_tier(tier)
    ctx = get_context(request)
    tracker = ctx.usage_manager
    store = ctx.quota_manager
    by_salt = tracker.get_all()
    total = tracker.get_total()
    quota_entries = {e.cache_salt: e.limit_bytes for e in store.list_quotas()}
    all_salts = sorted(set(by_salt) | set(quota_entries))
    return StatusListResponse(
        total_gb=_gb(total),
        by_cache_salt=[
            StatusResponse(
                cache_salt=salt,
                quota_limit_gb=_gb(quota_entries[salt])
                if salt in quota_entries
                else 0.0,
                quota_exists=salt in quota_entries,
                usage_gb=_gb(by_salt.get(salt, 0)),
            )
            for salt in all_salts
        ],
    )


# -- Usage-event ingestion ---------------------------------------------------


@router.post("/quota/events")
async def report_events(
    body: ReportUsageRequest, request: Request
) -> ReportUsageResponse:
    """Record a batch of ``store`` / ``lookup`` / ``delete`` events.

    Usage events feed the same per-``cache_salt`` ledger the quota status reads
    expose, so they live in the ``/quota`` group alongside the quota writes.

    Args:
        body: Event batch tagged with the reporter's ``instance_id``, ``seq``,
            and the ``tier`` the events apply to.

    Returns:
        Number of events processed.
    """
    _require_supported_tier(body.tier)
    ctx = get_context(request)
    tracker = ctx.usage_manager
    ctrl = ctx.eviction_manager
    for event in body.events:
        ok = event.key.to_object_key()
        if event.type == EventType.STORE:
            tracker.record_stored(ok, event.bytes)
            ctrl.on_store(ok)
        elif event.type == EventType.LOOKUP:
            ctrl.on_lookup(ok)
        elif event.type == EventType.DELETE:
            tracker.record_evicted(ok)
            ctrl.on_remove(ok)
    return ReportUsageResponse(recorded=len(body.events))
