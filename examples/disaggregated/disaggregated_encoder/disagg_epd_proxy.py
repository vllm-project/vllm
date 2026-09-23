#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible proxy for E+PD and E+P+D disaggregation.

Use static server URLs, or --dynamic-registration for launcher-managed HTTP
registration of E, P/PD and D instances. Both modes route OpenAI-compatible
``/v1/chat/completions`` requests through encoder and inference clusters and
share encoder fan-out, metadata-only rewriting, prefill, and retry/streaming
forwarding.

For MM input we:
    1. Extract *every* image/audio/video item.
    2. Send concurrent encoder requests with all text removed,
       grouping images assigned to the same encoder.
    3. Wait for all of them to succeed.
    4. Forward the *original* request to a decode server.
"""

from __future__ import annotations

import argparse
import asyncio
import enum
import hashlib
import itertools
import json
import logging
import os
import random
import secrets
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import msgspec
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import AnyHttpUrl, BaseModel, Field, model_validator

###############################################################################
# FastAPI app & global state
###############################################################################

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("proxy")

DEFAULT_PROBE_INTERVAL = 5.0
DEFAULT_PROBE_TIMEOUT = 2.0
DEFAULT_FAIL_THRESHOLD = 3
# Stop probing an instance that has been down this long. 0 probes forever,
# which is what a cluster that restarts instances in place wants.
DEFAULT_EVICTED_TTL = 900.0


class InstanceRole(str, enum.Enum):
    ENCODE = "encode"
    PREFILL = "prefill"
    DECODE = "decode"
    PREFILL_DECODE = "prefill_decode"


@dataclass
class InstanceRecord:
    """One registered instance.

    Attributes:
        role: Which stage this instance serves.
        url: Base OpenAI-compatible URL, e.g. ``http://host:8000``.
        ec_zmq_addrs: Mooncake TP-rank-0 control addresses, one per DP replica,
            supplied by the launcher from the consumer's fixed port config.
        dp_size: Data-parallel replicas behind `url`, so the proxy can pick
            a replica and name the same one to both halves of a request.

    """

    role: InstanceRole
    url: str
    ec_zmq_addrs: list[str] = field(default_factory=list)
    dp_size: int = 1


class InstanceRegistry:
    def __init__(
        self,
        probe_interval: float = DEFAULT_PROBE_INTERVAL,
        probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
        fail_threshold: int = DEFAULT_FAIL_THRESHOLD,
        evicted_ttl: float = DEFAULT_EVICTED_TTL,
    ):
        self._probe_interval = probe_interval
        self._probe_timeout = probe_timeout
        self._fail_threshold = fail_threshold
        self._evicted_ttl = evicted_ttl

        self._live: dict[str, InstanceRecord] = {}
        self._evicted: dict[str, InstanceRecord] = {}
        self._evicted_since: dict[str, float] = {}
        self._fail_counts: dict[str, int] = {}
        # One cursor per role, only ever incremented. Rebuilding it whenever
        # the roster changes -- what an `itertools.cycle` over a mutable list
        # forces -- restarts every fan-out at the first instance and hot-spots
        # it after each registration.
        self._cursors: dict[InstanceRole, int] = {role: 0 for role in InstanceRole}
        self._replica_cursors: dict[str, int] = {}
        self._probe_task: asyncio.Task | None = None

    def register(self, record: InstanceRecord) -> bool:
        """Add or refresh an instance without overriding its health status."""
        roles = {
            other.role
            for other in itertools.chain(self._live.values(), self._evicted.values())
        }
        split_roles = {InstanceRole.PREFILL, InstanceRole.DECODE}
        if (record.role is InstanceRole.PREFILL_DECODE and roles & split_roles) or (
            record.role in split_roles and InstanceRole.PREFILL_DECODE in roles
        ):
            raise ValueError("Cannot mix prefill_decode with standalone prefill/decode")
        key = record.url
        previous = self._live.get(key) or self._evicted.get(key)
        if previous is not None:
            if previous == record:
                return False
            if previous.role is not record.role:
                raise ValueError("Unregister the instance before changing its role")
            target = self._live if key in self._live else self._evicted
            target[key] = record
            return False
        self._live[key] = record
        logger.info("Registered instance %s: %s", record.role.value, record.url)
        return True

    def unregister(self, url: str) -> bool:
        """Drop an instance for good, so a probe cannot bring it back."""
        key = url
        if key not in self._live and key not in self._evicted:
            return False
        self._live.pop(key, None)
        self._evicted.pop(key, None)
        self._evicted_since.pop(key, None)
        self._fail_counts.pop(key, None)
        self._replica_cursors.pop(key, None)
        logger.info("Unregistered instance: %s", url)
        return True

    def instances(self, role: InstanceRole) -> list[InstanceRecord]:
        return [record for record in self._live.values() if record.role is role]

    def urls(self, role: InstanceRole) -> list[str]:
        return [record.url for record in self.instances(role)]

    def pick(self, role: InstanceRole) -> InstanceRecord | None:
        """Take the next instance of `role` in round-robin order."""
        picked = self.pick_many(role, 1)
        return picked[0] if picked else None

    def pick_many(self, role: InstanceRole, count: int) -> list[InstanceRecord]:
        """Take `count` instances, continuing the rotation across calls.

        A multimodal request fans out one encoder request per item, so the
        assignment has to be contiguous with the previous request's rather
        than restart at the first instance every time.
        """
        alive = self.instances(role)
        if not alive or count <= 0:
            return []
        start = self._cursors[role]
        self._cursors[role] = start + count
        return [alive[(start + offset) % len(alive)] for offset in range(count)]

    def next_replica(self, record: InstanceRecord) -> int:
        """Take the next data-parallel replica of `record`, round-robin.

        The encoder pushes to one replica's receive channel, so the request
        has to run on that same replica; the caller names the rank to both
        halves.
        """
        if record.dp_size <= 1:
            return 0
        cursor = self._replica_cursors.get(record.url, 0)
        self._replica_cursors[record.url] = cursor + 1
        return cursor % record.dp_size

    def status(self) -> dict[str, Any]:
        return {
            role.value: {
                "live": [record.url for record in self.instances(role)],
                "evicted": [
                    record.url
                    for record in self._evicted.values()
                    if record.role is role
                ],
            }
            for role in InstanceRole
        }

    def start_probing(self) -> None:
        if self._probe_task is None and self._probe_interval > 0:
            self._probe_task = asyncio.create_task(self._probe_loop())

    async def stop_probing(self) -> None:
        if self._probe_task is None:
            return
        self._probe_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._probe_task
        self._probe_task = None

    async def _probe_loop(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self._probe_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                await asyncio.sleep(self._probe_interval)
                try:
                    await self._probe_once(session)
                except Exception:
                    logger.exception("EPD registry probe round failed")

    async def _probe_once(self, session: aiohttp.ClientSession) -> None:
        targets = list(self._live.values()) + list(self._evicted.values())
        if not targets:
            return
        results = await asyncio.gather(
            *(self._probe(session, record.url) for record in targets),
            return_exceptions=True,
        )
        now = time.monotonic()
        for record, healthy in zip(targets, results):
            current = self._live.get(record.url) or self._evicted.get(record.url)
            if current is not record:
                continue
            if healthy is True:
                self._on_probe_success(record)
            else:
                self._on_probe_failure(record, now)
        self._drop_expired(now)

    async def _probe(self, session: aiohttp.ClientSession, url: str) -> bool:
        async with session.get(f"{url}/health") as resp:
            return resp.status == 200

    def _on_probe_success(self, record: InstanceRecord) -> None:
        key = record.url
        self._fail_counts.pop(key, None)
        if key in self._evicted:
            self._evicted.pop(key, None)
            self._evicted_since.pop(key, None)
            self._live[key] = record
            logger.info(
                "Instance %s (%s) is healthy again; routing resumed",
                record.url,
                record.role.value,
            )

    def _on_probe_failure(self, record: InstanceRecord, now: float) -> None:
        key = record.url
        if key not in self._live:
            # Either already evicted, or unregistered while this probe was in
            # flight. A round snapshots its targets and then awaits, so a
            # removal inside that window would otherwise be undone by a result
            # describing a registry that no longer exists.
            return
        failures = self._fail_counts.get(key, 0) + 1
        self._fail_counts[key] = failures
        if failures < self._fail_threshold:
            return
        self._live.pop(key, None)
        self._evicted[key] = record
        self._evicted_since[key] = now
        logger.warning(
            "Instance %s (%s) failed %d consecutive probes; stopped routing "
            "to it. It rejoins on its own once it responds again.",
            record.url,
            record.role.value,
            failures,
        )

    def _drop_expired(self, now: float) -> None:
        if self._evicted_ttl <= 0:
            return
        for key, since in list(self._evicted_since.items()):
            if now - since < self._evicted_ttl:
                continue
            record = self._evicted.pop(key, None)
            self._evicted_since.pop(key, None)
            self._fail_counts.pop(key, None)
            self._replica_cursors.pop(key, None)
            if record is not None:
                logger.warning("Instance %s stayed down; forgetting it", record.url)


class InstanceRegistration(BaseModel):
    role: InstanceRole
    url: AnyHttpUrl
    ec_zmq_addrs: list[str] = Field(default_factory=list)
    dp_size: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_consumer_addresses(self):
        if self.ec_zmq_addrs and len(self.ec_zmq_addrs) != self.dp_size:
            raise ValueError("Provide one Mooncake control address per DP replica")
        return self


def require_admin_key(x_api_key: str = Header(default="")) -> None:
    expected = os.getenv("ADMIN_API_KEY", "")
    if not expected or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(403, "Invalid admin API key")


@dataclass
class EPDProxyConfig:
    probe_interval: float = 5.0
    probe_timeout: float = 2.0
    fail_threshold: int = 3
    evicted_ttl: float = 900.0


@dataclass
class _Route:
    """The instances one request was assigned to.

    Attributes:
        consumer: The stage that receives the embedding, if any stage does.
        consumer_zmq: The receive address named to the encoders.
        dp_rank: Which replica of `consumer` was named, so the request can be
            pinned to it.

    """

    encoder_urls: list[str]
    prefill: InstanceRecord | None
    decode: InstanceRecord
    consumer: InstanceRecord | None = None
    consumer_zmq: str | None = None
    dp_rank: int | None = None


class EPDProxy:
    def __init__(self, registry: InstanceRegistry):
        self.registry = registry

    # ---------------------------------------------------------------- #
    # Routing                                                          #
    # ---------------------------------------------------------------- #
    def route(self, num_items: int) -> _Route:
        decode = self.registry.pick(InstanceRole.PREFILL_DECODE) or self.registry.pick(
            InstanceRole.DECODE
        )
        if decode is None:
            raise HTTPException(
                status_code=503, detail="No decode instance is registered"
            )
        prefill = (
            self.registry.pick(InstanceRole.PREFILL)
            if decode.role is InstanceRole.DECODE
            else None
        )
        if decode.role is InstanceRole.DECODE and prefill is None:
            raise HTTPException(
                status_code=503, detail="No prefill instance is registered"
            )
        encoders = self.registry.urls(InstanceRole.ENCODE)
        if num_items and not encoders:
            raise HTTPException(
                status_code=503, detail="No encode instance is registered"
            )
        route = _Route(encoder_urls=encoders, prefill=prefill, decode=decode)
        self._name_consumer(route)
        return route

    def _name_consumer(self, route: _Route) -> None:
        """Pin the EC consumer replica and, for push connectors, its endpoint."""
        candidate = route.prefill or route.decode
        route.consumer = candidate
        rank = self.registry.next_replica(candidate)
        route.dp_rank = rank if candidate.dp_size > 1 else None
        if candidate.ec_zmq_addrs:
            route.consumer_zmq = candidate.ec_zmq_addrs[rank]


app = FastAPI()
encode_session: aiohttp.ClientSession | None = None
prefill_session: aiohttp.ClientSession | None = None
decode_session: aiohttp.ClientSession | None = None

# Cursor for round-robin encoder assignment, shared across requests so the
# fan-out doesn't restart from e_urls[0] every time.
encoder_rr_idx = 0
encoder_rr_lock = asyncio.Lock()

###############################################################################
# Utils
###############################################################################


MM_TYPES = {"image_url", "audio_url", "input_audio", "video_url"}

# The embeds content type each MM item is rewritten to once the encoder has
# published its embedding out of band.
EMBEDS_TYPES = {
    "image_url": "image_embeds",
    "audio_url": "audio_embeds",
    "input_audio": "audio_embeds",
    "video_url": "video_embeds",
}


def encoder_rr_assignment(
    e_urls: list[str], start: int, count: int
) -> tuple[list[str], int]:
    """Assign `count` items to encoder URLs starting from cursor `start`.

    Returns the per-item URL list and the cursor value the next call should
    start from, so the assignment is contiguous across calls instead of
    restarting at e_urls[0] every time.
    """
    urls = [e_urls[(start + i) % len(e_urls)] for i in range(count)]
    next_start = (start + count) % len(e_urls)
    return urls, next_start


def validate_ec_consumer_routing(
    prefill_urls: list[str], consumer_addrs: list[str]
) -> None:
    """Reject the topology whose EC destination cannot be routed safely."""
    if prefill_urls and consumer_addrs:
        raise ValueError(
            "Mooncake EC consumer routing supports E+PD only; disable independent "
            "prefill or omit --ec-consumer-zmq-addrs."
        )


# Diagnostic switch: forward the original request to the decoder so the
# only difference from the rewrite path is the rewrite itself.
NO_REWRITE = False
# Maximum images per encoder subrequest; 0 leaves batches unlimited.
ENCODER_MAX_BATCH_SIZE = int(os.getenv("ENCODER_MAX_BATCH_SIZE", "0"))
if ENCODER_MAX_BATCH_SIZE < 0:
    raise ValueError("ENCODER_MAX_BATCH_SIZE must be non-negative")

# Decode-side retries for a retryable internal error (`finish_reason="error"`,
# e.g. an encoder embedding the connector could not deliver). Re-issuing runs
# the encode again, which produces a fresh transfer.
DECODE_RETRIES = 1


# Grid metadata reported by the encoder instance, keyed by item index.
# Empty when the encoder did not report any (then nothing is rewritten).
def content_uuid(item: dict) -> str:
    """Cache key for a multimodal item, derived from its content.

    Must be content-derived, not request-derived: the EC cache is keyed by this
    value, so a per-request key (a request id, say) would make every request a
    miss and throw away cross-request reuse of already-encoded media -- while
    the unmodified path, which hashes the content, would keep it. That asymmetry
    silently biases any comparison between the two.
    """
    url = (
        item.get("image_url") or item.get("audio_url") or item.get("video_url") or {}
    ).get("url") or ""
    payload = url or json.dumps(item, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def rewrite_for_decode(req_data: dict, item_meta: dict[int, dict]) -> dict:
    """Replace each media item with a metadata-only reference for the decoder.

    The decoder does not need the pixels: the encoder instance already produced
    the embedding and published it through the EC connector under the same uuid.
    Sending only the grid lets the decoder size the placeholder range without
    re-running the media transform.

    `item_meta` holds what the encoder reported for each item (its cache key and
    the grid its processor actually produced), so the grid is never re-derived
    here -- a second derivation could disagree with the encoder's.
    """
    mm_items = extract_mm_items(req_data)
    # Audio processors may consume placeholders from other modalities too.
    # Keep the whole request raw if any item cannot be rewritten safely.
    control_fields = {"mm_hash", "ec_mm_hash", "transfer_id"}
    raw_audio_request = any(
        item["type"] in {"audio_url", "input_audio"} for item in mm_items
    ) and any(
        not item_meta.get(i, {}).get("mm_hash")
        or not (item_meta.get(i, {}).keys() - control_fields)
        for i in range(len(mm_items))
    )
    rewritten = 0
    transfer_items = []
    idx = 0
    new_messages = []
    for msg in req_data.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue
        new_content = []
        for item in content:
            if item.get("type") not in MM_TYPES:
                new_content.append(item)
                continue
            meta = dict(item_meta.get(idx) or {})
            idx += 1
            item_uuid = meta.pop("mm_hash", None)
            ec_mm_hash = meta.pop("ec_mm_hash", None) or item_uuid
            transfer_id = meta.pop("transfer_id", None)
            if raw_audio_request:
                new_content.append(
                    {
                        **item,
                        "uuid": item_uuid or item.get("uuid") or content_uuid(item),
                    }
                )
                if transfer_id is not None:
                    transfer_items.append(
                        {"mm_hash": ec_mm_hash, "transfer_id": transfer_id}
                    )
                continue
            # Whatever keys the encoder reported are the metadata its model
            # declared as needed to size the placeholder range; the proxy does
            # not need to know their names.
            # Downstream stacks per item; keep the existing per-item vector shape.
            metadata = {
                k: [
                    x
                    for item in v
                    for x in (item if isinstance(item, list) else [item])
                ]
                for k, v in meta.items()
            }
            if not metadata or not item_uuid:
                # Nothing to size the placeholder range with. A processor cache
                # hit is not a cause on its own: with the default `lru` type the
                # engine restores the item before the scheduler reports it. It
                # goes missing when the encode request failed, or under
                # `--mm-processor-cache-type shm`, where a hit replaces the item
                # with its shared-memory address and only the worker restores
                # it. Send the media so the decoder can derive the grid itself.
                new_content.append(item)
                continue
            embeds_type = EMBEDS_TYPES[item["type"]]
            new_content.append(
                {"type": embeds_type, embeds_type: metadata, "uuid": item_uuid}
            )
            if transfer_id is not None:
                transfer_items.append(
                    {"mm_hash": ec_mm_hash, "transfer_id": transfer_id}
                )
            rewritten += 1
        new_messages.append({**msg, "content": new_content})

    if not rewritten and not raw_audio_request:
        return req_data
    if rewritten:
        logger.info("Rewrote %d media item(s) as metadata references", rewritten)
    rewritten_request = {**req_data, "messages": new_messages}
    if transfer_items:
        ec_transfer_params = dict(req_data.get("ec_transfer_params") or {})
        ec_transfer_params["ec_items"] = transfer_items
        rewritten_request["ec_transfer_params"] = ec_transfer_params
    return rewritten_request


def extract_mm_items(request_data: dict) -> list[dict]:
    """Return *all* image/audio/video items that appear anywhere in `messages`.

    Each returned dict looks like:
        { "type": "image_url", "image_url": {...} }
    """
    items: list[dict] = []
    for msg in request_data.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if item.get("type") in MM_TYPES:
                items.append(item)
    return items


async def fanout_encoder_primer(
    orig_request: dict,
    e_urls: list[str],
    req_id: str,
    consumer_zmq: str | None = None,
) -> tuple[dict[int, dict], dict[str, Any]]:
    """1. Group images by encoder, retaining per-item round-robin assignment.
    2. Send them concurrently to the encode cluster.
    3. Raise if any of them fails.

    Returns, per item index, the metadata the encoder reported in
    `ec_transfer_params`: its EC cache key and the grid its processor produced.
    The proxy still supplies the uuid so both sides key the cache the same way;
    the grid can only come from the encoder, which is the side that computed it.

    Also returns the connector handles to put on the decode body, as a fresh
    mapping. `orig_request` is left untouched so a retry re-encodes from the
    original request instead of carrying the previous attempt's handles.
    """
    logger.info("[%s] Processing multimodal items...", req_id)

    mm_items = extract_mm_items(orig_request)
    if not mm_items:
        logger.info("[%s] No multimodal items, skipping encoder", req_id)
        return {}, {}  # nothing to do

    logger.info("[%s] got %d multimodal items...", req_id, len(mm_items))

    tasks = []
    item_uuids: dict[int, str] = {}
    item_transfer_ids: dict[int, str] = {}
    item_meta: dict[int, dict] = {}
    transfer_items: dict[int, dict[str, str]] = {}
    ec_params: dict[str, Any] = {}

    # Round-robin over encode servers to distribute load a bit. The cursor
    # persists across requests so fan-out doesn't restart at e_urls[0] every
    # time (which would hot-spot the first encoder for single-item requests).
    global encoder_rr_idx
    async with encoder_rr_lock:
        url_cycle, encoder_rr_idx = encoder_rr_assignment(
            e_urls, encoder_rr_idx, len(mm_items)
        )

    groups: list[tuple[str, list[int]]] = []
    image_groups: dict[str, list[int]] = {}
    for idx, (item, target_url) in enumerate(zip(mm_items, url_cycle)):
        if item["type"] == "image_url":
            indices = image_groups.get(target_url)
            if indices is None or (
                ENCODER_MAX_BATCH_SIZE and len(indices) >= ENCODER_MAX_BATCH_SIZE
            ):
                indices = []
                image_groups[target_url] = indices
                groups.append((target_url, indices))
            indices.append(idx)
        else:
            groups.append((target_url, [idx]))

    for target_url, indices in groups:
        # Derive a *child* request id:  <parent>:<index>:<random-short>
        child_req_id = f"{req_id}:{indices[0]}:{uuid.uuid4().hex[:6]}"
        headers = {"x-request-id": child_req_id, "Content-Type": "application/json"}

        # With --no-rewrite the decoder still receives the raw image and derives
        # the cache key by hashing it, so the encoder must do the same -- passing
        # a uuid here would make the two disagree and silently defeat the EC
        # transfer, leaving the decoder to encode the image itself.
        content = []
        for idx in indices:
            item = mm_items[idx]
            item_uuid = None if NO_REWRITE else (item.get("uuid") or content_uuid(item))
            if item_uuid is not None:
                item_uuids[idx] = item_uuid
            item_transfer_ids[idx] = uuid.uuid4().hex
            content.append(item if item_uuid is None else {**item, "uuid": item_uuid})

        encoder_req = {
            "model": orig_request.get("model"),
            "messages": [
                {
                    "role": "user",
                    "content": content,
                },
            ],
            # No max_tokens cap: the encoder instance never samples, it finishes
            # once the prompt is encoded and its embeddings are published.
            "stream": False,
        }
        for key in (
            "mm_processor_kwargs",
            "media_io_kwargs",
            "priority",
            "session_id",
        ):
            if key in orig_request:
                encoder_req[key] = orig_request[key]
        if consumer_zmq is not None:
            # The engine may rehash UUIDs with processing options. Match by
            # position instead; batches contain only images in input order.
            encoder_req["ec_transfer_params"] = {
                "consumer_zmq": consumer_zmq,
                "ec_items": [
                    {"transfer_id": item_transfer_ids[idx]} for idx in indices
                ],
            }
        tasks.append(
            encode_session.post(
                f"{target_url}/v1/chat/completions",
                data=msgspec.json.encode(encoder_req),
                headers=headers,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Fail fast if any sub-request failed
    for (_, indices), r in zip(groups, results):
        idx = indices[0]
        if isinstance(r, Exception):
            logger.error(
                "[%s] Encoder request #%d raised exception: %s",
                req_id,
                idx,
                r,
                exc_info=r,
            )
            raise HTTPException(
                status_code=502, detail=f"Encoder request failed: {str(r)}"
            )
        if r.status != 200:
            try:
                detail = await r.text()
            except Exception:
                detail = "<unable to read body>"
            logger.error(
                "[%s] Encoder request #%d returned status %s: %s",
                req_id,
                idx,
                r.status,
                detail,
            )
            raise HTTPException(
                status_code=r.status,
                detail=f"Encoder request failed: {detail}",
            )

        # The encoder reports each mm_hash's metadata (e.g. the grid) here,
        # keyed by the same uuid this proxy assigned above.
        try:
            params = msgspec.json.decode(await r.read()).get("ec_transfer_params") or {}
        except Exception:
            logger.warning("[%s] Could not read encoder metadata #%d", req_id, idx)
            params = {}
        if params:
            by_index = {}
            for mm_hash, reported in params.items():
                if len(indices) == 1:
                    break
                for local_index in reported.get("item_indices", []):
                    if type(local_index) is not int or not 0 <= local_index < len(
                        indices
                    ):
                        raise HTTPException(502, "Invalid encoder item index")
                    if local_index in by_index:
                        raise HTTPException(502, "Duplicate encoder item index")
                    by_index[local_index] = (mm_hash, reported)
            for local_index, idx in enumerate(indices):
                matched = by_index.get(local_index)
                if matched is None:
                    # Compatibility with encoders without position metadata.
                    mm_hash = item_uuids.get(idx)
                    if mm_hash in params:
                        matched = (mm_hash, params[mm_hash])
                    elif len(indices) == 1 and len(params) == 1:
                        matched = next(iter(params.items()))
                    elif (
                        mm_items[idx]["type"] == "video_url"
                        and (orig_request.get("mm_processor_kwargs") or {}).get(
                            "use_audio_in_video"
                        )
                        and len(params) == 2
                        and consumer_zmq is None
                        and all(
                            entry.keys() <= {"metadata", "item_indices"}
                            for entry in params.values()
                        )
                    ):
                        # Keep raw video for its audio/video features when no transfer
                        # handles need to be forwarded.
                        continue
                    else:
                        raise HTTPException(502, "Encoder metadata cannot be matched")
                ec_mm_hash, reported = matched
                metadata = reported.get("metadata") or {}
                item_meta[idx] = {
                    **metadata,
                    "mm_hash": item_uuids.get(idx, ec_mm_hash),
                    "ec_mm_hash": ec_mm_hash,
                }
                if idx in item_transfer_ids:
                    item_meta[idx]["transfer_id"] = item_transfer_ids[idx]
                # Whatever the encoder reported alongside `metadata` is the
                # connector's own handle on the published embedding (for NIXL,
                # peer_host/peer_port/size_bytes). The decoder's connector
                # looks it up by mm_hash on the request, so carry it through.
                ec_params[ec_mm_hash] = reported
                if NO_REWRITE and consumer_zmq is not None:
                    transfer_items[idx] = {
                        "mm_hash": ec_mm_hash,
                        "transfer_id": item_transfer_ids[idx],
                    }

    if transfer_items:
        ec_params["ec_items"] = [transfer_items[idx] for idx in sorted(transfer_items)]

    logger.info(
        "[%s] All %d encoder requests completed successfully", req_id, len(groups)
    )
    return item_meta, ec_params


async def maybe_prefill(
    req_data: dict,
    p_url: str,
    req_id: str,
    dp_rank: int | None = None,
) -> dict:
    """- Do prefill-only task if p_url exist;
    - Return a new body carrying kv transfer params (for nixl connector)
    - Else, skip and return the original request data for decode

    `req_data` is never mutated: a decode retry re-enters this function with the
    same body, and one attempt's `remote_block_ids` must not reach the next.
    """
    if p_url:
        logger.info("[%s] Processing through prefill: %s", req_id, p_url)

        prefill_response = await process_prefill_stage(req_data, p_url, req_id, dp_rank)
        # for nixl connector to facilitate kv transfer...
        prefill_response_json = msgspec.json.decode(await prefill_response.read())
        kv_transfer_params = prefill_response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            return {**req_data, "kv_transfer_params": kv_transfer_params}

    return req_data


async def process_prefill_stage(
    req_data: dict,
    p_url: str,
    req_id: str,
    dp_rank: int | None = None,
) -> dict:
    """Process request through Prefill stage and return kv_transfer_params"""
    logger.info("[%s] Sending prefill request to: %s", req_id, p_url)

    prefill_request = req_data.copy()
    prefill_request["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    prefill_request["stream"] = False
    prefill_request["max_tokens"] = 1
    if "max_completion_tokens" in prefill_request:
        prefill_request["max_completion_tokens"] = 1
    if "stream_options" in prefill_request:
        del prefill_request["stream_options"]

    headers = {"x-request-id": req_id, "Content-Type": "application/json"}
    if dp_rank is not None:
        headers["X-data-parallel-rank"] = str(dp_rank)
    try:
        prefill_response = await prefill_session.post(
            f"{p_url}/v1/chat/completions",
            data=msgspec.json.encode(prefill_request),
            headers=headers,
        )
        prefill_response.raise_for_status()

        if prefill_response.status != 200:
            error_text = await prefill_response.text()
            logger.error(
                "[%s] Prefill request failed with status %d: %s",
                req_id,
                prefill_response.status,
                error_text,
            )
            raise HTTPException(
                status_code=prefill_response.status,
                detail={"error": "Prefill request failed", "message": error_text},
            )
        logger.info("[%s] Prefill request completed successfully", req_id)

        return prefill_response

    except Exception as e:
        logger.error("Prefill processing failed: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Prefill processing error", "message": str(e)},
        ) from e


###############################################################################
# Middleware for request/response logging
###############################################################################


async def log_requests(request: Request, call_next):
    """Middleware to log all incoming requests and responses"""
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))

    # Log incoming request
    logger.info(
        ">>> [%s] %s %s from %s",
        req_id,
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )

    try:
        # Process request
        response = await call_next(request)

        # Log response
        logger.info(
            "<<< [%s] %s %s completed with status %d",
            req_id,
            request.method,
            request.url.path,
            response.status_code,
        )

        return response
    except Exception as e:
        # Log errors
        logger.exception(
            "!!! [%s] %s %s failed with error: %s",
            req_id,
            request.method,
            request.url.path,
            str(e),
        )
        raise


###############################################################################
# FastAPI lifecycle
###############################################################################


@app.on_event("startup")
async def on_startup() -> None:
    global encode_session, prefill_session, decode_session
    timeout = aiohttp.ClientTimeout(total=100_000)
    # vLLM closes an idle keep-alive connection after
    # VLLM_HTTP_TIMEOUT_KEEP_ALIVE seconds (5 by default), while aiohttp keeps
    # pooling it for 15. Reusing one it has already closed fails the request
    # with ServerDisconnectedError, and the server logs nothing at all: it
    # closed the socket before the request arrived. Retire ours first.
    server_keep_alive = float(os.getenv("VLLM_HTTP_TIMEOUT_KEEP_ALIVE", "5"))
    connector = aiohttp.TCPConnector(
        limit=0,
        **(
            {"keepalive_timeout": server_keep_alive / 2}
            if server_keep_alive > 0
            else {"force_close": True}
        ),
    )
    encode_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    prefill_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    decode_session = aiohttp.ClientSession(timeout=timeout, connector=connector)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global encode_session, prefill_session, decode_session
    if encode_session:
        await encode_session.close()
    if prefill_session:
        await prefill_session.close()
    if decode_session:
        await decode_session.close()


###############################################################################
# Core forwarding
###############################################################################


async def prepare_for_decode(
    req_data: dict,
    req_id: str,
    e_urls: list[str],
    p_url: str,
    consumer_zmq: str | None,
    prefill_dp_rank: int | None = None,
) -> tuple[dict, float, float]:
    """Encode, rewrite and prefill, returning the body to send to decode.

    `req_data` is left untouched so a retry starts from the original media
    rather than from a body whose images are already metadata references.
    """
    _t0 = time.perf_counter()
    item_meta, ec_params = await fanout_encoder_primer(
        req_data, e_urls, req_id, consumer_zmq
    )
    _t1 = time.perf_counter()
    prepared = req_data if NO_REWRITE else rewrite_for_decode(req_data, item_meta)
    if ec_params:
        # A fresh body every time: `rewrite_for_decode` hands back `req_data`
        # itself when it rewrote nothing, and this attempt's handles must not
        # outlive it into a retry.
        handles = dict(prepared.get("ec_transfer_params") or {})
        handles.update(ec_params)
        prepared = {**prepared, "ec_transfer_params": handles}
    _t2 = time.perf_counter()
    prepared = await maybe_prefill(prepared, p_url, req_id, prefill_dp_rank)
    return prepared, _t1 - _t0, _t2 - _t1


async def forward_non_stream(
    req_data: dict,
    req_id: str,
    e_urls: list[str],
    p_url: str,
    d_url: str,
    consumer_zmq: str | None,
    dp_rank: int | None = None,
    prefill_dp_rank: int | None = None,
) -> Response:
    try:
        for attempt in range(DECODE_RETRIES + 1):
            _t0 = time.perf_counter()
            prepared, encode_s, rewrite_s = await prepare_for_decode(
                req_data, req_id, e_urls, p_url, consumer_zmq, prefill_dp_rank
            )
            _t2 = time.perf_counter()

            logger.info("[%s] Forwarding to decode: %s", req_id, d_url)
            headers = {"x-request-id": req_id, "Content-Type": "application/json"}
            if dp_rank is not None:
                headers["X-data-parallel-rank"] = str(dp_rank)

            async with decode_session.post(
                f"{d_url}/v1/chat/completions",
                data=msgspec.json.encode(prepared),
                headers=headers,
            ) as resp:
                if resp.status >= 400:
                    detail = await resp.text()
                    # 500 is the decoder's retryable internal error, which
                    # includes an encoder embedding it could not obtain. Redoing
                    # the encode publishes the item again.
                    if resp.status == 500 and attempt < DECODE_RETRIES:
                        logger.warning(
                            "[%s] Decode returned 500, re-encoding and retrying "
                            "(attempt %d/%d): %s",
                            req_id,
                            attempt + 1,
                            DECODE_RETRIES,
                            detail[:200],
                        )
                        continue
                    logger.error(
                        "[%s] Decode request returned status %s: %s",
                        req_id,
                        resp.status,
                        detail,
                    )
                    raise HTTPException(status_code=resp.status, detail=detail)
                out = await resp.read()
                _t3 = time.perf_counter()
                logger.info(
                    "STAGE %s encode=%.1f rewrite=%.1f decode=%.1f total=%.1f "
                    "attempt=%d",
                    "no-rewrite" if NO_REWRITE else "rewrite",
                    encode_s * 1e3,
                    rewrite_s * 1e3,
                    (_t3 - _t2) * 1e3,
                    (_t3 - _t0) * 1e3,
                    attempt,
                )
                return Response(
                    content=out,
                    status_code=resp.status,
                    headers={
                        "Content-Type": resp.headers.get(
                            "Content-Type", "application/json"
                        )
                    },
                )
        raise HTTPException(status_code=500, detail="Decode failed after re-encoding")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[%s] Error in forward_non_stream: %s", req_id, str(e))
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}") from e


async def forward_stream(
    req_data: dict,
    req_id: str,
    e_urls: list[str],
    p_url: str,
    d_url: str,
    consumer_zmq: str | None,
    dp_rank: int | None = None,
    prefill_dp_rank: int | None = None,
) -> AsyncIterator[bytes]:
    try:
        for attempt in range(DECODE_RETRIES + 1):
            _t0 = time.perf_counter()
            prepared, encode_s, rewrite_s = await prepare_for_decode(
                req_data, req_id, e_urls, p_url, consumer_zmq, prefill_dp_rank
            )
            _t2 = time.perf_counter()

            logger.info("[%s] Starting streaming from decode: %s", req_id, d_url)
            headers = {"x-request-id": req_id, "Content-Type": "application/json"}
            if dp_rank is not None:
                headers["X-data-parallel-rank"] = str(dp_rank)

            _first = None
            async with decode_session.post(
                f"{d_url}/v1/chat/completions",
                data=msgspec.json.encode(prepared),
                headers=headers,
            ) as resp:
                # Retry only before the first chunk: once anything reached the
                # client the response cannot be replaced.
                if resp.status == 500 and attempt < DECODE_RETRIES:
                    detail = await resp.text()
                    logger.warning(
                        "[%s] Decode returned 500 before streaming, re-encoding "
                        "and retrying (attempt %d/%d): %s",
                        req_id,
                        attempt + 1,
                        DECODE_RETRIES,
                        detail[:200],
                    )
                    continue
                resp.raise_for_status()
                async for chunk in resp.content.iter_any():
                    if chunk:
                        if _first is None:
                            _first = time.perf_counter()
                        yield chunk
            _t3 = time.perf_counter()

            logger.info(
                "STAGE %s encode=%.1f rewrite=%.2f decode_ttfb=%.1f "
                "decode_total=%.1f attempt=%d",
                "no-rewrite" if NO_REWRITE else "rewrite",
                encode_s * 1e3,
                rewrite_s * 1e3,
                ((_first or _t3) - _t2) * 1e3,
                (_t3 - _t2) * 1e3,
                attempt,
            )
            logger.info("[%s] Streaming completed", req_id)
            return

    except HTTPException:
        logger.exception("[%s] HTTPException in forward_stream", req_id)
        raise
    except Exception as e:
        logger.exception("[%s] Error in forward_stream: %s", req_id, str(e))
        raise HTTPException(
            status_code=500, detail=f"Proxy streaming error: {str(e)}"
        ) from e


###############################################################################
# Public routes
###############################################################################


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        req_data = msgspec.json.decode(await request.body())
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))

        e_urls = app.state.e_urls  # we want the full list for fan-out
        p_url = random.choice(app.state.p_urls) if app.state.p_urls else None
        decode_index = random.randrange(len(app.state.d_urls))
        d_url = app.state.d_urls[decode_index]
        dp_size = app.state.ec_consumer_dp_size
        # Round-robin the replica, then name it to both halves: the decoder
        # honours the rank header instead of its own balancer, and the encoder
        # pushes to that replica's control channel. Choosing once here means a
        # decode retry re-encodes to the same replica.
        dp_rank = next(app.state.replica_counter) % dp_size if dp_size > 1 else None
        ec_index = decode_index * dp_size + (dp_rank or 0)
        consumer_zmq = app.state.d_ec_urls[ec_index] if app.state.d_ec_urls else None

        is_streaming = req_data.get("stream", False)

        if is_streaming:
            return StreamingResponse(
                forward_stream(
                    req_data, req_id, e_urls, p_url, d_url, consumer_zmq, dp_rank
                ),
                media_type="text/event-stream",
            )
        return await forward_non_stream(
            req_data, req_id, e_urls, p_url, d_url, consumer_zmq, dp_rank
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in chat_completions endpoint: %s", str(e))
        raise HTTPException(
            status_code=500, detail=f"Request processing error: {str(e)}"
        ) from e


@app.get("/v1/models")
async def list_models():
    async with decode_session.get(f"{app.state.d_urls[0]}/v1/models") as resp:
        resp.raise_for_status()
        return await resp.json()


@app.get("/health")
async def health_check():
    async def healthy(urls):
        if not urls:
            return "empty"
        for u in urls:
            try:
                async with encode_session.get(f"{u}/health") as resp:
                    resp.raise_for_status()
            except Exception:
                return "unhealthy"
        return "healthy"

    e_status, p_status, d_status = await asyncio.gather(
        healthy(app.state.e_urls), healthy(app.state.p_urls), healthy(app.state.d_urls)
    )

    overall_healthy = all(
        status != "unhealthy" for status in (e_status, p_status, d_status)
    )

    status_code = 200 if overall_healthy else 503

    return JSONResponse(
        {
            "proxy": "healthy",
            "encode_cluster": e_status,
            "prefill_cluster": p_status,
            "decode_cluster": d_status,
        },
        status_code=status_code,
    )


###############################################################################
# Simple profiler fan-out (unchanged except for sessions)
###############################################################################


async def _post_if_available(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    headers: dict,
) -> dict | None:
    """POST `payload` to `url`.

    Returns
    -------
    • The decoded JSON body on success (2xx)
    • None if the endpoint does not exist (404)
    • Raises for anything else.

    """
    try:
        resp = await session.post(url, json=payload, headers=headers)
        if resp.status == 404:  # profiling disabled on that server
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        resp.raise_for_status()
        return await resp.json(content_type=None)
    except aiohttp.ClientResponseError as exc:
        # Pass 404 through the branch above, re-raise everything else
        if exc.status == 404:
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        raise
    except Exception:
        # Network errors etc.: propagate
        raise


async def _profile_cmd(cmd: str, payload: dict, e_url: str, p_url: str, d_url: str):
    """Fire & forget to both clusters, tolerate 404."""
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}

    encode_task = _post_if_available(
        encode_session, f"{e_url}/{cmd}_profile", payload, headers
    )
    prefill_task = (
        _post_if_available(prefill_session, f"{p_url}/{cmd}_profile", payload, headers)
        if p_url is not None
        else asyncio.sleep(0)
    )
    decode_task = _post_if_available(
        decode_session, f"{d_url}/{cmd}_profile", payload, headers
    )

    encode_res, prefill_res, decode_res = await asyncio.gather(
        encode_task, prefill_task, decode_task
    )

    # If *all* clusters said “I don’t have that route”, surface an error
    if encode_res is prefill_res is decode_res is None:
        raise HTTPException(
            status_code=503,
            detail="Profiling endpoints are disabled on all clusters",
        )

    return {
        "encode": encode_res,  # may be None
        "prefill": prefill_res,  # may be None
        "decode": decode_res,  # may be None
    }


@app.post("/start_profile")
async def start_profile(request: Request):
    body = await request.json()
    # TODO: handle multi urls properly
    e_url = random.choice(app.state.e_urls)
    p_url = random.choice(app.state.p_urls) if app.state.p_urls else None
    d_url = random.choice(app.state.d_urls)
    return await _profile_cmd("start", body, e_url, p_url, d_url)


@app.post("/stop_profile")
async def stop_profile(request: Request):
    body = await request.json()
    # TODO: handle multi urls properly
    e_url = random.choice(app.state.e_urls)
    p_url = random.choice(app.state.p_urls) if app.state.p_urls else None
    d_url = random.choice(app.state.d_urls)
    return await _profile_cmd("stop", body, e_url, p_url, d_url)


def build_app(config: EPDProxyConfig | None = None) -> FastAPI:
    config = config or EPDProxyConfig()
    registry = InstanceRegistry(
        probe_interval=config.probe_interval,
        probe_timeout=config.probe_timeout,
        fail_threshold=config.fail_threshold,
        evicted_ttl=config.evicted_ttl,
    )
    proxy = EPDProxy(registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await on_startup()
        try:
            registry.start_probing()
            yield
        finally:
            await registry.stop_probing()
            await on_shutdown()

    app = FastAPI(lifespan=lifespan)
    app.state.proxy = proxy
    app.state.registry = registry

    @app.get("/instances")
    async def list_instances():
        return registry.status()

    @app.post("/instances", dependencies=[Depends(require_admin_key)])
    async def register_instance(body: InstanceRegistration):
        if body.ec_zmq_addrs and body.role not in (
            InstanceRole.PREFILL,
            InstanceRole.PREFILL_DECODE,
        ):
            raise HTTPException(400, "Only EC consumers accept Mooncake addresses")
        try:
            created = registry.register(
                InstanceRecord(
                    body.role,
                    str(body.url).rstrip("/"),
                    body.ec_zmq_addrs,
                    body.dp_size,
                )
            )
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return {"registered": created}

    @app.delete("/instances", dependencies=[Depends(require_admin_key)])
    async def unregister_instance(url: AnyHttpUrl):
        return {"removed": registry.unregister(str(url).rstrip("/"))}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        req_data = msgspec.json.decode(await request.body())
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        route = proxy.route(len(extract_mm_items(req_data)))
        args = (
            req_data,
            req_id,
            route.encoder_urls,
            route.prefill.url if route.prefill else None,
            route.decode.url,
            route.consumer_zmq,
        )
        ranks = {
            "dp_rank": route.dp_rank if route.prefill is None else None,
            "prefill_dp_rank": route.dp_rank if route.prefill is not None else None,
        }
        if req_data.get("stream", False):
            return StreamingResponse(
                forward_stream(*args, **ranks), media_type="text/event-stream"
            )
        return await forward_non_stream(*args, **ranks)

    @app.get("/v1/models")
    async def list_models():
        decode = registry.pick(InstanceRole.PREFILL_DECODE) or registry.pick(
            InstanceRole.DECODE
        )
        if decode is None:
            raise HTTPException(
                status_code=503, detail="No decode instance is registered"
            )
        async with decode_session.get(f"{decode.url}/v1/models") as resp:
            resp.raise_for_status()
            return await resp.json()

    @app.get("/health")
    async def health():
        status = registry.status()
        # An empty roster is not unhealthy: the proxy is meant to come up
        # before anything registers with it.
        return JSONResponse({"proxy": "healthy", "instances": status})

    @app.post("/start_profile")
    async def start_profile(request: Request):
        return await _profile(registry, "start", await request.json())

    @app.post("/stop_profile")
    async def stop_profile(request: Request):
        return await _profile(registry, "stop", await request.json())

    return app


async def _profile(registry: InstanceRegistry, cmd: str, payload: dict) -> dict:
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}
    targets = {role.value: registry.pick(role) for role in InstanceRole}
    results = await asyncio.gather(
        *(
            _post_if_available(
                decode_session, f"{record.url}/{cmd}_profile", payload, headers
            )
            for record in targets.values()
            if record is not None
        )
    )
    reachable = [result for result in results if result is not None]
    if not reachable:
        raise HTTPException(
            status_code=503,
            detail="Profiling endpoints are disabled on every instance",
        )
    live = [name for name, record in targets.items() if record is not None]
    return dict(zip(live, results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dynamic-registration",
        action="store_true",
        help="Enable launcher-managed HTTP registration for E+PD or E+P+D.",
    )
    parser.add_argument("--probe-interval", type=float, default=DEFAULT_PROBE_INTERVAL)
    parser.add_argument("--probe-timeout", type=float, default=DEFAULT_PROBE_TIMEOUT)
    parser.add_argument("--fail-threshold", type=int, default=DEFAULT_FAIL_THRESHOLD)
    parser.add_argument("--evicted-ttl", type=float, default=DEFAULT_EVICTED_TTL)
    parser.add_argument(
        "--log-requests",
        action="store_true",
        help=(
            "Log every request in and out, and raise the log level to DEBUG. "
            "Off by default: the proxy is on the request path."
        ),
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Forward images to the decoder unchanged (for stage-timing A/B).",
    )
    parser.add_argument(
        "--encode-servers-urls",
        default="",
        help='Comma-separated encode URLs ("http://e1:8001,http://e2:8001")',
    )
    parser.add_argument(
        "--prefill-servers-urls",
        default="none",
        help=(
            'Comma-separated prefill URLs ("http://p1:8003,http://p2:8004") '
            'to enable E->P->D, set "disable" or "none" to enable E->PD'
        ),
    )
    parser.add_argument(
        "--decode-servers-urls",
        default="",
        help='Comma-separated decode URLs ("http://d1:8005,http://d2:8006")',
    )
    parser.add_argument(
        "--decode-retries",
        type=int,
        default=1,
        help=(
            "Re-encode and re-send when decode returns 500, which is its "
            "retryable internal error (an undeliverable encoder embedding "
            "among them). 0 disables."
        ),
    )
    parser.add_argument(
        "--ec-consumer-zmq-addrs",
        default="",
        help=(
            "Comma-separated Mooncake EC consumer control addresses, aligned "
            "with --decode-servers-urls. Required for Mooncake EC consumers and "
            "supported only in E+PD mode. With --ec-consumer-dp-size > 1, list "
            "each server's replicas consecutively: s0r0,s0r1,s1r0,s1r1."
        ),
    )
    parser.add_argument(
        "--ec-consumer-dp-size",
        type=int,
        default=1,
        help=(
            "Data-parallel replicas per EC consumer. The proxy picks a replica "
            "round-robin and names it to both halves of the request, because an "
            "encoder push has to land where the request will run."
        ),
    )

    args = parser.parse_args()
    if args.fail_threshold < 1:
        parser.error("--fail-threshold must be at least 1")
    if args.log_requests:
        logging.getLogger().setLevel(logging.DEBUG)
        app.middleware("http")(log_requests)
    NO_REWRITE = args.no_rewrite
    DECODE_RETRIES = max(0, args.decode_retries)
    if args.dynamic_registration:
        if not os.getenv("ADMIN_API_KEY"):
            parser.error("--dynamic-registration requires ADMIN_API_KEY")
        if (
            args.encode_servers_urls
            or args.prefill_servers_urls.lower() not in ("disable", "none", "")
            or args.decode_servers_urls
            or args.ec_consumer_zmq_addrs
        ):
            parser.error(
                "With --dynamic-registration, instances register through /instances"
            )
        app = build_app(
            EPDProxyConfig(
                probe_interval=args.probe_interval,
                probe_timeout=args.probe_timeout,
                fail_threshold=args.fail_threshold,
                evicted_ttl=args.evicted_ttl,
            )
        )
        if args.log_requests:
            app.middleware("http")(log_requests)
    else:
        if not args.encode_servers_urls or not args.decode_servers_urls:
            parser.error(
                "Static routing requires --encode-servers-urls "
                "and --decode-servers-urls"
            )
        app.state.e_urls = [
            u.strip() for u in args.encode_servers_urls.split(",") if u.strip()
        ]
        app.state.d_urls = [
            u.strip() for u in args.decode_servers_urls.split(",") if u.strip()
        ]
        app.state.d_ec_urls = [
            u.strip() for u in args.ec_consumer_zmq_addrs.split(",") if u.strip()
        ]
        if args.ec_consumer_dp_size < 1:
            parser.error("--ec-consumer-dp-size must be at least 1")
        app.state.ec_consumer_dp_size = args.ec_consumer_dp_size
        app.state.replica_counter = itertools.count()
        expected = len(app.state.d_urls) * args.ec_consumer_dp_size
        if app.state.d_ec_urls and len(app.state.d_ec_urls) != expected:
            parser.error(
                "--ec-consumer-zmq-addrs must contain one address per consumer "
                f"replica: expected {expected} "
                f"({len(app.state.d_urls)} servers x "
                f"{args.ec_consumer_dp_size} replicas), "
                f"got {len(app.state.d_ec_urls)}"
            )
        # handle prefill instances
        if args.prefill_servers_urls.lower() in ("disable", "none", ""):
            app.state.p_urls = []
            logger.info("Disaggregated prefill disabled. Running E + PD...")
        else:
            app.state.p_urls = [
                u.strip() for u in args.prefill_servers_urls.split(",") if u.strip()
            ]
            logger.info("Disaggregated prefill phase is enabled. Running E + P + D...")
        try:
            validate_ec_consumer_routing(app.state.p_urls, app.state.d_ec_urls)
        except ValueError as exc:
            parser.error(str(exc))

        logger.info("Proxy listening on %s:%s", args.host, args.port)
        logger.info("Encode servers: %s", app.state.e_urls)
        logger.info("Prefill instances %s", app.state.p_urls)
        logger.info("Decode servers: %s", app.state.d_urls)
        if app.state.ec_consumer_dp_size > 1:
            logger.info(
                "EC consumer replicas per server: %d (control addresses: %s)",
                app.state.ec_consumer_dp_size,
                app.state.d_ec_urls,
            )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        loop="uvloop",
        access_log=True,
    )
