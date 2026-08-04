# SPDX-License-Identifier: Apache-2.0
"""Cache-management HTTP endpoints on the MP server (node-local).

A single ``/cache/*`` surface, thin over the typed services in
``lmcache/v1/multiprocess/cache_control/`` (resolved via :func:`get_context`):

- Objects   -- ``GET /cache/objects`` and ``DELETE /cache/objects``
  (:class:`ObjectService`); the delete is key-addressed and spans L1, L2, or
  both via its ``tier`` field. Adapter listing lives in the config group
  (``GET /config/adapters``).
- Prefetch  -- ``POST /cache/prefetches``, ``GET /cache/prefetches/{id}``
  (:class:`PrefetchService`).
- Diagnostics -- ``POST /cache/clear``, ``POST /cache/checksums`` (engine-local).

The object / prefetch routes are pass-throughs: the services validate and raise
domain errors, which the central handler in :mod:`error_handlers` maps to status
codes -- so there is no per-route ``try/except``. The diagnostics routes reach
engine internals directly (no service) and validate inline.
"""

# Standard
from http import HTTPStatus
from typing import Any, Optional
import asyncio
import hashlib

# Third Party
from fastapi import APIRouter, HTTPException, Query, Request
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import compress_slot_mapping
from lmcache.v1.distributed.api import Tier
from lmcache.v1.multiprocess.http_apis.dependencies import get_context
from lmcache.v1.multiprocess.http_apis.schemas import (
    ChecksumRequest,
    ClearRequest,
    DeleteObjectsRequest,
    PrefetchRequest,
)
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)

router = APIRouter()

_MAX_PAGE_SIZE = 5000
_DEFAULT_PAGE_SIZE = 500
_CLEAR_TIER = Tier.L1


# ---------------------------------------------------------------------------
# Objects -- pass-throughs to ObjectService
# ---------------------------------------------------------------------------


@router.get("/cache/objects", response_model=None)
async def list_cache_objects(
    request: Request,
    tier: Tier = Tier.L2,
    adapter: str | None = Query(default=None),
    model_name: str | None = Query(default=None),
    page_size: int = Query(default=_DEFAULT_PAGE_SIZE, ge=1, le=_MAX_PAGE_SIZE),
    page_token: str | None = Query(default=None),
) -> dict[str, object]:
    """List cache objects resident in one tier/adapter, paginated.

    Responses:
        200: ``{"adapter", "entries", "next_page_token"}``.
        400: ``tier`` unsupported or malformed ``page_token``. 404: adapter
            matches none.
        503: server not initialized, no adapters configured, or the adapter does
            not support listing.
    """
    return await get_context(request).object_service.list_objects(
        tier, adapter, model_name, page_size, page_token
    )


@router.delete("/cache/objects", response_model=None)
async def delete_cache_objects(
    body: DeleteObjectsRequest, request: Request
) -> dict[str, object]:
    """Delete a caller-supplied list of object keys from L1, L2, or both.

    ``tier`` selects the tier(s) (``l1`` / ``l2`` / ``all``); ``force`` deletes
    L1 keys even if locked.

    Responses:
        200: ``{"deleted", "skipped", "ok"[, "error"]}``.
        400: batch too large or an ``ObjectKey`` invariant violation.
            404: adapter matches none. 422: body validation.
        503: server not initialized, or no L2 adapters configured (tier
            includes L2).
    """
    return await get_context(request).object_service.delete_objects(
        body.tier, body.adapter, body.keys, body.force
    )


# ---------------------------------------------------------------------------
# Prefetch -- pass-throughs to PrefetchService
# ---------------------------------------------------------------------------


@router.post("/cache/prefetches", response_model=None, status_code=202)
async def submit_prefetch(body: PrefetchRequest, request: Request) -> dict[str, object]:
    """Submit a warm prefetch of a token sequence from L2 into L1.

    Responses:
        202: ``{"request_id", "chunks", "status": "submitted"}``, or
            ``{"chunks": 0, "status": "noop"}`` for a sub-chunk sequence.
        400: token cap exceeded, invalid ``cache_salt``, or unsupported tiers.
        422: body validation.
        503: not initialized, or no layout registered for the model.
    """
    return get_context(request).prefetch_service.submit(
        body.model_name,
        body.world_size,
        body.token_ids,
        body.cache_salt,
        body.source_tier,
        body.target_tier,
    )


@router.get("/cache/prefetches/{request_id}", response_model=None)
async def get_prefetch(request_id: str, request: Request) -> dict[str, object]:
    """Poll a submitted warm prefetch.

    Responses:
        200: ``{"request_id", "status": "pending"}`` or ``{"request_id",
            "status": "completed", "found_keys", "total_keys"}``.
        404: unknown id. 503: server not initialized.
    """
    return get_context(request).prefetch_service.status(request_id)


# ---------------------------------------------------------------------------
# Diagnostics (clear, checksums) -- engine-local; validated inline
# ---------------------------------------------------------------------------

# Per-format axis of the ``num_blocks`` dimension inside a per-layer KV tensor.
# The checksum endpoint gathers KV data by block IDs along this axis, which
# preserves the block_size dimension verbatim so chunking is a clean slice on a
# known axis. Formats that fuse num_blocks and block_size into a single
# page-buffer dimension are intentionally not listed: the block-level semantics
# don't map cleanly, and the endpoint declines them with 501.
_BLOCK_AXIS_BY_FORMAT: dict[Any, int] = {
    lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS: 1,  # [2, NB, BS, NH, HS]
    lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS: 0,  # [NB, 2, BS, NH, HS]
    lmc_ops.EngineKVFormat.NL_X_NB_BS_HS: 0,  # MLA: [NB, BS, HS]
    lmc_ops.EngineKVFormat.NL_X_NB_BSV_BSS: 0,  # DSA indexer: [NB, BS, 132]
    lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS: 1,  # [2, NB, NH, BS, HS]
    lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS: 0,  # [NB, 2, NH, BS, HS]
}


@router.post("/cache/clear", response_model=None)
async def clear_cache(
    request: Request, body: ClearRequest | None = None
) -> dict[str, object]:
    """Force-clear a tier's resident cache.

    Clears all objects in the tier, including those with active read/write
    locks; in-flight store/prefetch operations may be corrupted.

    The body is optional: an absent (or empty) body defaults to
    ``{"tier": "l1", "force": true}``.

    Responses:
        200: ``{"status": "ok", "cleared": {"tier": "l1"}}``.
        400: unsupported tier. 503: server not initialized.
    """
    if body is None:
        body = ClearRequest()
    if body.tier != _CLEAR_TIER:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=(
                f"tier {body.tier.value!r} not supported; only {_CLEAR_TIER.value!r}"
            ),
        )
    # TODO(cache-control): ``body.force`` is accepted for API forward-compat but
    # not honored -- the engine's CLEAR path always force-clears. Wiring it
    # through would require extending the ZMQ ``RequestType.CLEAR`` payload
    # (which currently carries no fields) so the cross-process op can pass force.
    get_context(request).engine.clear()
    logger.info("Cache cleared via HTTP API")
    return {"status": "ok", "cleared": {"tier": _CLEAR_TIER.value}}


def _resolve_per_layer_block_axes(
    formats_per_layer: list[Optional["lmc_ops.EngineKVFormat"]],
) -> tuple[Optional[list[int]], Optional[str]]:
    """Map each layer to its ``num_blocks`` axis from its Engine KV format.

    Each layer uses its own axis, so mixed-format models are checksummed rather
    than rejected. A ``None`` layer (cross-layer KV sharing) inherits a
    single-format model's axis and is rejected in a mixed-format one.

    Returns ``(block_axes, None)``, or ``(None, error)`` if a format is
    unsupported or a ``None`` layer can't be resolved.
    """
    axis_by_format: dict[int, int] = {}
    for fmt in formats_per_layer:
        if fmt is None or int(fmt) in axis_by_format:
            continue
        axis = _BLOCK_AXIS_BY_FORMAT.get(fmt)
        if axis is None:
            return None, "checksum not supported for GPU KV format %s" % fmt.name
        axis_by_format[int(fmt)] = axis

    shared_axis = (
        next(iter(axis_by_format.values())) if len(axis_by_format) == 1 else None
    )
    block_axes: list[int] = []
    for fmt in formats_per_layer:
        if fmt is not None:
            block_axes.append(axis_by_format[int(fmt)])
        elif shared_axis is not None:
            block_axes.append(shared_axis)
        else:
            return None, (
                "checksum not supported: a cross-layer KV-sharing layer in a "
                "mixed-format model has no resolvable block axis"
            )
    return block_axes, None


def _compute_block_checksums(
    kv_tensors: list[torch.Tensor],
    block_ids: list[int],
    block_axes: list[int],
    chunk_size: int,
    layerwise: bool,
) -> dict[str, Any]:
    """Compute MD5 checksums over KV cache blocks, grouped ``chunk_size`` blocks
    per chunk.

    The gather is performed in block space on each layer's own block axis via
    :func:`torch.index_select`, so mixed-format models gather each layer
    correctly. Each layer's gathered tensor is moved to CPU once, then chunk
    checksums slice it along its block axis in steps of ``chunk_size`` blocks.
    ``bfloat16`` is upcast to ``float32`` only if present.
    """
    num_blocks = len(block_ids)
    num_chunks = (num_blocks + chunk_size - 1) // chunk_size

    block_idx_cpu = torch.tensor(block_ids, dtype=torch.long)
    block_idx_by_device: dict[torch.device, torch.Tensor] = {
        block_idx_cpu.device: block_idx_cpu,
    }

    layer_blocks: list[torch.Tensor] = []
    for li, kv in enumerate(kv_tensors):
        idx = block_idx_by_device.get(kv.device)
        if idx is None:
            idx = block_idx_cpu.to(kv.device)
            block_idx_by_device[kv.device] = idx
        gathered = kv.index_select(block_axes[li], idx).cpu()
        if gathered.dtype == torch.bfloat16:
            gathered = gathered.to(torch.float32)
        layer_blocks.append(gathered)

    def _chunk_slice(t: torch.Tensor, axis: int, start: int, end: int) -> torch.Tensor:
        return t.narrow(axis, start, end - start).contiguous()

    if layerwise:
        per_layer: dict[str, list[str]] = {}
        for li, blocks in enumerate(layer_blocks):
            axis = block_axes[li]
            digests: list[str] = []
            for ci in range(num_chunks):
                s = ci * chunk_size
                e = min(s + chunk_size, num_blocks)
                chunk = _chunk_slice(blocks, axis, s, e)
                digests.append(hashlib.md5(chunk.numpy().tobytes()).hexdigest())
            per_layer["layer_%d" % li] = digests
        return {
            "status": "success",
            "chunk_size": chunk_size,
            "num_chunks": num_chunks,
            "chunk_checksums": per_layer,
            "layerwise": True,
        }

    aggregated: list[str] = []
    for ci in range(num_chunks):
        s = ci * chunk_size
        e = min(s + chunk_size, num_blocks)
        md5 = hashlib.md5()
        for li, blocks in enumerate(layer_blocks):
            chunk = _chunk_slice(blocks, block_axes[li], s, e)
            md5.update(chunk.numpy().tobytes())
        aggregated.append(md5.hexdigest())
    return {
        "status": "success",
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "chunk_checksums": aggregated,
        "layerwise": False,
    }


@router.post("/cache/checksums", response_model=None)
async def cache_checksums(body: ChecksumRequest, request: Request) -> dict[str, Any]:
    """Compute MD5 checksums over KV cache blocks, ``chunk_size`` blocks/chunk.

    Responses:
        200: ``{"status": "success", "chunk_size", "num_chunks",
            "chunk_checksums", "layerwise", "block_id_ranges"}``.
        400: no ``block_ids`` or non-positive ``chunk_size``.
        404: unknown ``instance_id`` or empty KV. 501: unsupported engine/format.
        503: server not initialized.
    """
    engine = get_context(request).engine
    cache_ctxs = getattr(engine, "cache_contexts", None)
    if cache_ctxs is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            detail="checksum not supported for this engine type",
        )
    ctx = cache_ctxs.get(body.instance_id)
    if ctx is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="instance_id %d not registered" % body.instance_id,
        )
    if not body.block_ids:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="block_ids is required"
        )
    if body.chunk_size <= 0:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="chunk_size must be positive"
        )
    kv_tensors = ctx.kv_tensors
    if not kv_tensors:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="kv_caches empty")

    block_axes, axis_err = _resolve_per_layer_block_axes(
        ctx.engine_kv_format_per_layer()
    )
    if block_axes is None:
        raise HTTPException(status_code=HTTPStatus.NOT_IMPLEMENTED, detail=axis_err)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _compute_block_checksums(
            kv_tensors, body.block_ids, block_axes, body.chunk_size, body.layerwise
        ),
    )
    result["block_id_ranges"] = compress_slot_mapping(body.block_ids)
    return result
