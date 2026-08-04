# SPDX-License-Identifier: Apache-2.0
"""HTTP wire models for the MP server's ``/cache/*`` endpoints.

These describe request bodies only -- the HTTP boundary. FastAPI validates them
(field types -> 422) before a handler runs; semantic validation and responses
belong to the ``cache_control`` services. ``Tier`` is imported from the domain
(it is cache vocabulary, not an HTTP type).
"""

# Standard
from dataclasses import dataclass, field

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey, Tier


@dataclass(frozen=True)
class DeleteObjectsRequest:
    """Wire body for ``DELETE /cache/objects``.

    Key-addressed delete for L1, L2, or both. ``keys`` field types are validated
    by FastAPI (422); their ``ObjectKey`` invariants and the batch cap are
    enforced by the service. ``tier`` selects the tier(s) -- ``l1`` / ``l2`` /
    ``all`` (default ``l2`` for backward compatibility with the eviction loop).
    ``force`` deletes L1 keys even if read/write-locked. ``adapter`` selects an
    L2 adapter, else the primary (ignored when ``tier`` is ``l1``).
    """

    keys: list[EncodedObjectKey]
    tier: Tier = Tier.L2
    force: bool = False
    adapter: str | None = None


@dataclass(frozen=True)
class PrefetchRequest:
    """Wire body for ``POST /cache/prefetches``.

    Callers describe content by ``token_ids``; the server resolves the per-rank
    keys itself. ``source_tier`` / ``target_tier`` are request data (today only
    ``l2`` -> ``l1``).
    """

    model_name: str
    world_size: int
    token_ids: list[int]
    cache_salt: str = ""
    source_tier: Tier = Tier.L2
    target_tier: Tier = Tier.L1


@dataclass(frozen=True)
class ClearRequest:
    """Wire body for ``POST /cache/clear``.

    ``tier`` is request data (today only ``l1``). ``force=true`` means active
    locks may be ignored if the implementation supports forced cleanup.
    """

    tier: Tier = Tier.L1
    force: bool = True


@dataclass(frozen=True)
class ChecksumRequest:
    """Wire body for ``POST /cache/checksums``.

    ``block_ids`` are the target GPU block IDs; ``chunk_size`` blocks are hashed
    per chunk. ``instance_id`` selects the GPU context; ``layerwise`` returns
    per-layer digests.
    """

    block_ids: list[int] = field(default_factory=list)
    chunk_size: int = 0
    instance_id: int = 0
    layerwise: bool = False
