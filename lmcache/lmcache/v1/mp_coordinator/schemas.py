# SPDX-License-Identifier: Apache-2.0
"""Shared request/response schemas for the mp coordinator REST API.

These Pydantic models are the wire contract between the coordinator and mp
servers. The coordinator uses them to validate requests and shape responses; an
mp server (when it registers) imports the same models to build its request
bodies and parse replies, so both sides agree on the schema in one place.

This module holds HTTP models only.
"""

# Standard
from enum import Enum
from typing import Annotated
import base64

# Third Party
from pydantic import BaseModel, Field, StringConstraints, field_validator
import numpy as np

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey  # noqa: F401  re-exported
from lmcache.v1.distributed.api import Tier
from lmcache.v1.mp_coordinator.api import CacheEventBatch
from lmcache.v1.mp_coordinator.key_directory import Placement


def encode_tokens(tokens: "list[int] | np.ndarray") -> str:
    """Encode token ids into a compact base64 wire string.

    Token ids fit in ``uint32``, so a little-endian ``uint32`` buffer is far
    smaller than a JSON integer list and decodes in one ``np.frombuffer`` call.

    Args:
        tokens: Token ids (a ``list[int]`` or any array castable to ``uint32``).

    Returns:
        Base64 of the little-endian ``uint32`` token buffer.
    """
    arr = np.ascontiguousarray(np.asarray(tokens, dtype="<u4"))
    return base64.b64encode(arr.tobytes()).decode("ascii")


def decode_tokens(tokens_b64: str) -> np.ndarray:
    """Decode a base64 token string produced by :func:`encode_tokens`.

    Args:
        tokens_b64: Base64 of a little-endian ``uint32`` token buffer.

    Returns:
        A ``uint64`` array of token ids (widened so it feeds the hashers
        directly).

    Raises:
        ValueError: If ``tokens_b64`` is not valid base64 or not a multiple of
            4 bytes.
    """
    try:
        raw = base64.b64decode(tokens_b64, validate=True)
    except Exception as exc:
        raise ValueError(f"tokens_b64 is not valid base64: {exc}") from exc
    if len(raw) % 4 != 0:
        raise ValueError(
            f"tokens_b64 byte length {len(raw)} is not a multiple of 4 "
            "(malformed uint32 token buffer)"
        )
    return np.frombuffer(raw, dtype="<u4").astype(np.uint64)


class RegisterRequest(BaseModel):
    """Body of a ``POST /instances`` registration request.

    Attributes:
        instance_id: Identifier of the mp server. Optional -- if empty (or
            whitespace-only), the coordinator generates one and returns it.
        ip: IP/host of the mp server's HTTP server. Whitespace is stripped and a
            blank value is rejected, since the coordinator calls this address.
        http_port: Port of the mp server's HTTP server, which the coordinator
            calls to push work to this instance.
        metadata: Free-form registration hints.
        p2p_advertised_url: URL the instance advertises for peer-to-peer
            transfers. Optional -- empty when the instance does not participate
            in P2P.
        mq_port: Port of the instance's ZMQ message-queue server that P2P peers
            send lookup/unlock RPCs to, reachable at the instance's ``ip``.
            Optional -- 0 when P2P is disabled.
    """

    instance_id: Annotated[str, StringConstraints(strip_whitespace=True)] = ""
    ip: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    http_port: int = Field(ge=1, le=65535)
    metadata: dict[str, str] = Field(default_factory=dict)
    p2p_advertised_url: Annotated[str, StringConstraints(strip_whitespace=True)] = ""
    mq_port: int = Field(default=0, ge=0, le=65535)


class RegisterResponse(BaseModel):
    """Reply to a successful ``POST /instances``.

    Attributes:
        instance_id: The registered instance's id.
        re_registered: ``True`` if this replaced an existing registration.
    """

    instance_id: str
    re_registered: bool


class HeartbeatResponse(BaseModel):
    """Reply to a successful ``PUT /instances/{id}/heartbeat``.

    Attributes:
        instance_id: The instance whose heartbeat was recorded.
    """

    instance_id: str


# -- Quota management --------------------------------------------------------


class SetQuotaRequest(BaseModel):
    """Body of ``PUT /quota/{cache_salt}``.

    Attributes:
        limit_gb: Non-negative byte budget in GiB.
        tier: Cache tier the quota applies to (only ``l2`` is supported today).
    """

    limit_gb: float = Field(ge=0.0)
    tier: Tier = Tier.L2


class QuotaResponse(BaseModel):
    """Reply to ``PUT`` or ``DELETE /quota/{cache_salt}``.

    Attributes:
        cache_salt: The tenant identifier (``_default`` for empty salt).
        limit_gb: The current limit in GiB (0.0 after deletion).
        status: ``"ok"`` or ``"removed"`` or ``"not_found"``.
    """

    cache_salt: str
    limit_gb: float
    status: str


class QuotaConfigRequest(BaseModel):
    """Body of ``PUT /quota/config``.

    Attributes:
        default_limit_gb: Byte budget in GiB applied to salts with no
            explicit quota entry. ``None`` (default) leaves unquota'd
            salts exempt from eviction.
        tier: Cache tier the config applies to (only ``l2`` today).
    """

    default_limit_gb: float | None = Field(default=None, ge=0.0)
    tier: Tier = Tier.L2


class QuotaConfigResponse(BaseModel):
    """Reply to ``GET`` / ``PUT /quota/config``.

    Attributes:
        default_limit_gb: Current default limit in GiB, or ``None``
            when unquota'd salts are exempt from eviction.
    """

    default_limit_gb: float | None


# -- Usage tracking ----------------------------------------------------------


class EventType(str, Enum):
    """Cache events reported by an MP server."""

    STORE = "store"
    LOOKUP = "lookup"
    DELETE = "delete"


class UsageEvent(BaseModel):
    """A single cache event reported by an MP server.

    Attributes:
        type: The event type.
        key: The cache key this event applies to.
        bytes: Bytes stored (``store`` only; ``0`` for other types).
    """

    type: EventType
    key: EncodedObjectKey
    bytes: int = Field(ge=0)


class ReportUsageRequest(BaseModel):
    """Body of ``POST /quota/events``.

    Attributes:
        instance_id: Identifier of the MP server that produced this batch.
        seq: Monotonically increasing sequence number scoped to this
            ``instance_id``. Starts at 1 for the first flush after the
            server starts.
        events: Batch of store/lookup events to record.
        tier: Cache tier the events apply to (only ``l2`` is supported today).
    """

    instance_id: str
    seq: int = Field(ge=1)
    events: list[UsageEvent]
    tier: Tier = Tier.L2


class ReportUsageResponse(BaseModel):
    """Reply to ``POST /quota/events``.

    Attributes:
        recorded: Number of events processed.
    """

    recorded: int


class StatusResponse(BaseModel):
    """Combined quota and usage for a single ``cache_salt``.

    Attributes:
        cache_salt: The tenant identifier.
        quota_limit_gb: The byte budget in GiB (0.0 if no quota set).
        quota_exists: Whether an explicit quota is registered.
        usage_gb: Current usage in GiB.
    """

    cache_salt: str
    quota_limit_gb: float
    quota_exists: bool
    usage_gb: float


class StatusListResponse(BaseModel):
    """Reply to ``GET /quota``.

    Attributes:
        total_gb: Aggregate usage in GiB.
        by_cache_salt: Per-tenant breakdown with quota and usage.
    """

    total_gb: float
    by_cache_salt: list[StatusResponse]


# -- Key directory -----------------------------------------------------------


class DirectoryEventsRequest(BaseModel):
    """Body of ``POST /directory/events``.

    Attributes:
        batches: Event batches to apply, in emission order per instance.
    """

    batches: list[CacheEventBatch] = Field(default_factory=list)

    @field_validator("batches")
    @classmethod
    def _validate_batches(cls, value: list[CacheEventBatch]) -> list[CacheEventBatch]:
        """Enforce encoding-level constraints on every entry.

        Args:
            value: The hydrated batches.

        Returns:
            The unchanged batches once every constraint holds.

        Raises:
            ValueError: If an entry's key cannot convert into an
                ``ObjectKey`` or its ``content_hash_hex`` is not valid
                hex (surfaced as 422).
        """
        for batch in value:
            for entry in batch.entries:
                entry.key.to_object_key()
                bytes.fromhex(entry.content_hash_hex)
        return value


class DirectoryEventsResponse(BaseModel):
    """Reply to ``POST /directory/events``.

    Attributes:
        applied: Batches applied to the directory.
        duplicates: Batches dropped as already-applied replays.
        stale: Batches dropped for carrying an outdated incarnation.
    """

    applied: int = 0
    duplicates: int = 0
    stale: int = 0


class DirectoryLookupRequest(BaseModel):
    """Body of ``POST /directory/lookup``.

    Attributes:
        keys: The keys to resolve to placements.
    """

    keys: list[EncodedObjectKey] = Field(default_factory=list)


class DirectoryKeyPlacements(BaseModel):
    """Placements for one requested key.

    Attributes:
        key: The requested key, echoed back.
        placements: Known placements; empty when the directory knows
            nothing about the key.
    """

    key: EncodedObjectKey
    placements: list[Placement] = Field(default_factory=list)


class DirectoryLookupResponse(BaseModel):
    """Reply to ``POST /directory/lookup``.

    Attributes:
        results: One entry per requested key, in request order.
    """

    results: list[DirectoryKeyPlacements] = Field(default_factory=list)


class TokenPlacementLookupRequest(BaseModel):
    """Body of ``POST /directory/lookup_tokens``.

    Resolves ``token_ids`` to the object keys of their complete chunks
    (the same fan-out the pin APIs use) and returns each key's placements.

    Attributes:
        model_name: Model whose rank fan-out to use when resolving keys.
        world_size: World size selecting the per-rank fan-out.
        token_ids: The token sequence to resolve.
        cache_salt: Per-tenant isolation salt applied to the produced keys.
    """

    model_name: str
    world_size: int = Field(ge=1)
    token_ids: list[int] = Field(default_factory=list)
    cache_salt: str = ""


class TokenPlacementLookupResponse(BaseModel):
    """Reply to ``POST /directory/lookup_tokens``.

    Attributes:
        chunks: Number of complete chunks the token sequence resolved to.
        results: One entry per resolved key (``chunks`` x per-rank
            fan-out), each echoing the key with its known placements.
    """

    chunks: int = 0
    results: list[DirectoryKeyPlacements] = Field(default_factory=list)


# -- Global CacheBlend fingerprint directory ------------------------------


class StoreRangeModel(BaseModel):
    """Wire form of one published stored token range (see ``StoreRange``).

    The coordinator chunks ``tokens`` at its chunk size and hashes each chunk;
    chunk ``i`` maps to ``object_keys[i]`` at ``old_st_base + i * chunk_size``.

    Attributes:
        model_scope: Reuse scope (the model name).
        tokens: The stored tokens (``token_ids[start:end]``).
        object_keys: Shared-L2 storage key (hex) per chunk, in order.
        old_st_base: Token position of the range's first token.
    """

    model_scope: str
    tokens: list[int] = Field(default_factory=list)
    object_keys: list[str] = Field(default_factory=list)
    old_st_base: int = Field(ge=0)


class BlendFingerprintRequest(BaseModel):
    """Body of ``POST /blend/fingerprints``: register stored ranges.

    Attributes:
        ranges: Stored token ranges to register (idempotent).
    """

    ranges: list[StoreRangeModel] = Field(default_factory=list)


class BlendFingerprintResponse(BaseModel):
    """Reply to ``POST /blend/fingerprints``.

    Attributes:
        inserted: Number of fingerprints newly registered.
    """

    inserted: int


class BlendEvictRequest(BaseModel):
    """Body of ``DELETE /blend/fingerprints``: evict by storage key.

    Attributes:
        object_keys: ``object_key`` values to evict.
    """

    object_keys: list[str] = Field(default_factory=list)


class BlendEvictResponse(BaseModel):
    """Reply to ``DELETE /blend/fingerprints``.

    Attributes:
        removed: Number of fingerprint entries evicted.
    """

    removed: int


class BlendMatchRequest(BaseModel):
    """Body of ``POST /blend/match``.

    Attributes:
        model_scope: Scope to match within.
        tokens_b64: The request tokens, packed via :func:`encode_tokens`
            (base64 little-endian ``uint32``).
    """

    model_scope: str
    tokens_b64: str = ""

    @field_validator("tokens_b64")
    @classmethod
    def _validate_tokens_b64(cls, value: str) -> str:
        """Reject a malformed token buffer at request validation.

        Without this, ``decode_tokens`` would raise ``ValueError`` inside the
        route handler, which FastAPI surfaces as a 500 (server error) for what
        is really bad client input. Validating here returns a 422 instead.

        Args:
            value: The base64 ``tokens_b64`` field.

        Returns:
            The unchanged value once it is confirmed decodable.

        Raises:
            ValueError: If ``value`` is not valid base64 or not a whole number
                of ``uint32`` tokens (surfaced by FastAPI as 422).
        """
        decode_tokens(value)
        return value


class GlobalMatchModel(BaseModel):
    """Wire form of one matched chunk (see ``GlobalMatch``).

    Attributes:
        object_key: Shared-L2 storage key of the matched chunk.
        old_st: Token position in the stored sequence (re-RoPE source).
        cur_st: Token position in the request (re-RoPE target).
    """

    object_key: str
    old_st: int
    cur_st: int


class BlendMatchResponse(BaseModel):
    """Reply to ``POST /blend/match``.

    Attributes:
        matches: Matched chunks, ascending by ``cur_st``.
    """

    matches: list[GlobalMatchModel]


class PrefetchRequest(BaseModel):
    """Body of ``POST /cache/prefetches`` on the coordinator.

    Asks the coordinator to warm one MP server's L1 with the chunks of a token
    sequence. The caller describes content by ``token_ids`` -- the unit the
    cache speaks -- not by internal cache keys, which it cannot construct. The
    coordinator forwards the request verbatim to that server's own
    ``POST /cache/prefetches``, which hashes the tokens and expands them into the
    per-rank keys.

    Attributes:
        instance_id: Identifier of the target MP server (must be registered).
        model_name: Model whose layout the target uses to allocate L1 buffers.
        world_size: World size selecting the layout and the per-rank fan-out.
        token_ids: Prompt tokens whose complete chunks should be warmed.
        cache_salt: Per-tenant isolation salt applied to the produced keys.
    """

    instance_id: str
    model_name: str
    world_size: int = Field(ge=1)
    token_ids: list[int] = Field(default_factory=list)
    cache_salt: str = ""


class PrefetchResponse(BaseModel):
    """Reply to ``POST /cache/prefetches`` on the coordinator.

    Attributes:
        instance_id: The target MP server the prefetch was submitted to.
        request_id: The server's job id to poll via
            ``GET /cache/prefetches/{instance_id}/{request_id}``. Empty when
            ``status`` is ``"noop"`` (nothing to warm).
        chunks: Number of whole chunks submitted to warm.
        status: ``"submitted"`` (a job is in flight) or ``"noop"`` (the
            sequence was shorter than one chunk).
    """

    instance_id: str
    request_id: str = ""
    chunks: int = 0
    status: str


class PinRequest(BaseModel):
    """Body of ``POST`` / ``DELETE /cache/pins`` on the coordinator.

    Pinning protects the resolved keys from L2 eviction until unpinned. The
    coordinator resolves ``token_ids`` to keys locally; L2 pins are fleet-wide
    (per ``cache_salt``), so no target instance is needed.

    Attributes:
        model_name: Model whose rank fan-out to use when resolving keys.
        world_size: World size selecting the per-rank fan-out.
        token_ids: Prompt tokens whose complete chunks should be (un)pinned.
        cache_salt: Per-tenant isolation salt applied to the produced keys.
    """

    model_name: str
    world_size: int = Field(ge=1)
    token_ids: list[int] = Field(default_factory=list)
    cache_salt: str = ""


class PinResponse(BaseModel):
    """Reply to ``POST`` / ``DELETE /cache/pins`` on the coordinator.

    Attributes:
        requested: Number of whole chunks the token sequence resolved to.
        affected: Number of L2 keys pinned (on pin) or unpinned (on unpin);
            disambiguated by ``status``.
        status: ``"pinned"`` / ``"unpinned"``.
    """

    requested: int = 0
    affected: int = 0
    status: str


class DeleteRequest(BaseModel):
    """Body of ``POST /cache/delete`` on the coordinator.

    Attributes:
        instance_id: Identifier of the target MP server (must be registered).
        model_name: Model whose layout the target uses to resolve keys.
        world_size: World size selecting the layout and the per-rank fan-out.
        token_ids: Prompt tokens whose complete chunks should be deleted.
        cache_salt: Per-tenant isolation salt applied to the produced keys.
        tier: Which tier(s) to delete: ``l1`` (L1 only), ``l2`` (L2 only), or
            ``all`` (both). ``l1`` never touches L2 and vice versa.
        force: When True, delete even locked/pinned keys -- bypasses L1
            locks/pins on the node and the coordinator's L2 pin filter.
    """

    instance_id: str
    model_name: str
    world_size: int = Field(ge=1)
    token_ids: list[int] = Field(default_factory=list)
    cache_salt: str = ""
    tier: Tier = Tier.ALL
    force: bool = False


class DeleteResponse(BaseModel):
    """Reply to ``POST /cache/delete`` on the coordinator.

    Attributes:
        instance_id: The target MP server the request was dispatched to.
        requested: Number of whole chunks the token sequence resolved to.
        affected: Total keys removed across the tiers acted on -- L1 keys deleted
            by the node plus L2 keys deleted by the coordinator. A chunk resident
            in both tiers (``tier=all``) contributes to both, so ``affected`` may
            exceed ``requested`` (which counts chunks, not per-tier keys).
        skipped: Total keys refused because they were locked/pinned (non-force
            only) -- L1 keys the node refused plus L2 keys held back for an L2 pin.
        status: ``"deleted"`` / ``"noop"``.
    """

    instance_id: str
    requested: int = 0
    affected: int = 0
    skipped: int = 0
    status: str
