# SPDX-License-Identifier: Apache-2.0
"""
Defines the data structures that will be used by the
distributed storage manager public functions

Could be implemented by native code in the future
"""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING
import enum

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

logger = init_logger(__name__)


class Tier(str, enum.Enum):
    """A cache tier.

    Subclasses ``str`` so it validates from / compares equal to the bare wire
    value (``Tier.L2 == "l2"``) and serializes as that value. ``ALL`` is only
    valid for operations that explicitly support multiple tiers.
    """

    L1 = "l1"
    L2 = "l2"
    ALL = "all"


class L1BackendType(str, enum.Enum):
    """The storage medium backing the L1 tier (a closed set, unlike L2
    backends, which are an open adapter-type registry).

    Subclasses ``str`` so it compares equal to and serializes as the bare
    wire value (``L1BackendType.DRAM == "dram"``).
    """

    DRAM = "dram"
    DEVDAX = "devdax"
    GDS = "gds"


class TrimPolicy(enum.Enum):
    """How to pick the retained subset of found keys for a prefetch.

    PREFIX retains the longest contiguous run from index 0; SEGMENTED_PREFIX
    keeps the keys that loaded when an L2 hit failed to load into L1 mid-prefix
    (gaps and all); SPARSE retains every found key for an intentional scatter.
    """

    PREFIX = enum.auto()
    SEGMENTED_PREFIX = enum.auto()
    SPARSE = enum.auto()


class PrefetchMode(enum.Enum):
    """The intent of a prefetch request.

    ``LOOKUP`` -- prefetch for an imminent reader: loaded keys are pinned for
    the requesting workers, and whether they persist or are dropped after use
    follows the configured prefetch policy.

    ``WARM`` -- speculative pre-warm with no imminent reader: loaded keys are
    retained and left unpinned (immediately resident and evictable), so a later
    lookup can hit them.
    """

    LOOKUP = enum.auto()
    WARM = enum.auto()


@dataclass(frozen=True)
class ObjectKey:
    """
    The unique identifier for an object in the distributed storage manager
    """

    chunk_hash: bytes
    """ Content hash of this particular chunk """

    model_name: str
    """ Name of the model this chunk belongs to.

    Invariant: must not contain ``@``. The L2 adapters use ``@`` as the
    field separator in serialized keys/filenames and rely on this
    invariant for unambiguous parsing. HuggingFace model IDs use
    alphanumerics + ``/-_.`` so this rejects nothing that appears in
    practice.
    """

    kv_rank: int
    """ The rank that uniquely identifies the slice of the KV cache """

    object_group_id: int = 0
    """ Index of the object group this chunk belongs to. """

    cache_salt: str = ""
    """ Per-user isolation salt. Same content from different users with
    different cache_salt values produces different ObjectKeys, giving
    strict per-user cache isolation. Defaults to empty string, in which
    case serialized keys and filenames match the pre-cache_salt shape
    (no trailing salt field) — no migration is needed for un-salted
    deployments.

    Invariant: must not contain ``@``, ``/``, ``\\``, or NUL. The L2
    adapters use ``@`` as the field separator; ``/`` and ``\\`` are
    filesystem path separators (FS adapter embeds the salt into
    filenames); NUL terminates C strings (C++ connector). Max length
    128 to stay well within ``NAME_MAX`` (255) after the model, rank,
    hash, and extension are added.
    """

    _SALT_FORBIDDEN_CHARS = frozenset("@/\\\x00")
    _SALT_MAX_LEN = 128

    def __post_init__(self) -> None:
        if "@" in self.model_name:
            raise ValueError(
                f"model_name must not contain '@' (got {self.model_name!r})"
            )
        if self.object_group_id < 0:
            raise ValueError(
                f"object_group_id must be >= 0 (got {self.object_group_id})"
            )
        bad = self._SALT_FORBIDDEN_CHARS & set(self.cache_salt)
        if bad:
            raise ValueError(
                f"cache_salt must not contain {bad!r} (got {self.cache_salt!r})"
            )
        if len(self.cache_salt) > self._SALT_MAX_LEN:
            raise ValueError(
                f"cache_salt exceeds max length {self._SALT_MAX_LEN} "
                f"(got {len(self.cache_salt)})"
            )

    def to_encoded_object_key(self) -> "EncodedObjectKey":
        """Return the JSON-safe :class:`EncodedObjectKey` projection."""
        return EncodedObjectKey(
            chunk_hash_hex=self.chunk_hash.hex(),
            model_name=self.model_name,
            kv_rank=self.kv_rank,
            object_group_id=self.object_group_id,
            cache_salt=self.cache_salt,
        )

    @staticmethod
    def IntHash2Bytes(chunk_hash: int) -> bytes:
        # NOTE: this is only used by tests
        return chunk_hash.to_bytes(4, byteorder="big")

    @staticmethod
    def Bytes2IntHash(chunk_hash: bytes) -> int:
        # NOTE: this is only used by tests
        return int.from_bytes(chunk_hash, byteorder="big") & ((1 << 64) - 1)

    @staticmethod
    def ComputeKVRank(
        world_size: int,
        global_rank: int,
        local_world_size: int,
        local_rank: int,
    ) -> int:
        """
        Compute the kv_rank from world_size and worker_id

        Args:
            world_size (int): The total number of workers (include TP + PP)
            global_rank (int): The global worker id (from 0 to world_size - 1)
            local_world_size (int): The local world size (for local node),
                should NOT be greater than 8
            local_rank (int): The local world rank (for local node)

        Returns:
            The special KV rank (bitmap) used by the objectkey

        Example:
            In the case of TP=4, PP=2, the TP worker 1 on node 1 has:
            - world_size = 8
            - global_rank = 5
            - local_world_size = 4
            - local_rank = 1

            The output KV rank is the bitmap:
            +--head--+
            |00000000|
            |00000000|
            |00000000|
            |00000000| layers
            |00001100|
            |00001100|
            |00001100|
            |00001100|
            +--------+
        """
        # TODO(ApostaC): in the long run, we want to have the above bitmap based
        # representation for asymmetric parallelism (e.g., sharing across different
        # TP/PP settings).
        # For now, let's have a simple implementation that just
        # differentiate between different parallel setups

        # For each number, we use 8-bit, and pack them together
        return (
            (world_size << 24)
            | (global_rank << 16)
            | (local_world_size << 8)
            | local_rank
        )


@dataclass(frozen=True)
class EncodedObjectKey:
    """JSON-safe wire form of :class:`ObjectKey` — ``chunk_hash`` is
    hex-encoded; other fields are preserved verbatim."""

    chunk_hash_hex: str
    """Hex-encoded ``ObjectKey.chunk_hash``."""

    model_name: str
    kv_rank: int

    object_group_id: int = 0
    """Defaults to ``0`` so pre-``object_group_id`` wire payloads still
    deserialize."""

    cache_salt: str = ""

    def to_object_key(self) -> ObjectKey:
        """Recover the corresponding :class:`ObjectKey`.

        Raises:
            ValueError: ``chunk_hash_hex`` is not valid hex, or one of
                :class:`ObjectKey`'s field invariants is violated.
        """
        return ObjectKey(
            chunk_hash=bytes.fromhex(self.chunk_hash_hex),
            model_name=self.model_name,
            kv_rank=self.kv_rank,
            object_group_id=self.object_group_id,
            cache_salt=self.cache_salt,
        )


@dataclass(frozen=True)
class KeyEntry:
    """One entry in a :class:`KeyListPage` including the encoded object
    key and its object size."""

    key: EncodedObjectKey
    size_bytes: int


@dataclass(frozen=True)
class KeyListPage:
    """A page of keys returned by ``L2AdapterInterface.list_l2_keys``."""

    entries: tuple[KeyEntry, ...]
    """The keys in the current page."""

    next_page_token: str | None
    """``None`` means this is the last page. Otherwise pass the token
    verbatim to the next call to fetch the next page."""


@dataclass(frozen=True)
class MemoryLayoutDesc:
    """
    Describes the layout of a memory object
    """

    shapes: list[torch.Size]
    dtypes: list[torch.dtype]

    def __post_init__(self):
        if len(self.shapes) != len(self.dtypes):
            raise ValueError(
                "MemoryLayoutDesc: shapes and dtype must have the same length"
            )


@dataclass(frozen=True)
class AttnWindowDesc:
    """Per-object-group cross-chunk attention windows, in LMCache chunks.

    ``num_chunks_in_sw[g]`` is the number of trailing prefix chunks that must
    be present for object group ``g`` to serve a cache hit. ``-1`` means full
    attention (the whole prefix); ``w >= 1`` is a sliding window of ``w``
    chunks (mamba is ``1``).
    """

    num_chunks_in_sw: list[int]

    def __post_init__(self) -> None:
        for w in self.num_chunks_in_sw:
            if w == 0 or w < -1:
                raise ValueError(
                    "AttnWindowDesc: each window must be -1 (full attention) "
                    f"or >= 1 chunk, got {w}"
                )

    @property
    def num_object_groups(self) -> int:
        """Number of object groups this descriptor covers."""
        return len(self.num_chunks_in_sw)

    def is_full_attention(self, object_group_idx: int) -> bool:
        """Whether the object group depends on the entire prefix.

        Args:
            object_group_idx: 0-based object group index.

        Returns:
            True if the group attends to the whole prefix, False if it uses a
            bounded sliding window.
        """
        return self.num_chunks_in_sw[object_group_idx] < 0


DEFAULT_ATTN_WINDOW_DESC = AttnWindowDesc(num_chunks_in_sw=[-1])
"""A single full-attention object group; the default when no per-object-group
windows are supplied."""


@dataclass(frozen=True)
class PrefetchRequestSpec:
    """Immutable inputs of a single L2 prefetch request.

    Bundles the caller-supplied arguments that travel together into the
    prefetch controller's submission queue. See
    ``PrefetchController._start_lookup_phase`` for per-field semantics.

    Attributes:
        keys: Object keys to prefetch; order defines the prefix.
        layout_desc: Memory layout for L1 write-buffer allocation.
        extra_count: Extra read locks per key beyond the default 1.
        policy: Retained-subset policy (see :class:`TrimPolicy`).
        attn_desc: Cross-chunk attention windows, in object-group order.
        mode: Prefetch intent (see :class:`PrefetchMode`).
    """

    keys: list[ObjectKey]
    layout_desc: MemoryLayoutDesc
    extra_count: int = 0
    policy: TrimPolicy = TrimPolicy.PREFIX
    attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC
    mode: PrefetchMode = PrefetchMode.LOOKUP


@dataclass(frozen=True)
class PrefetchHandle:
    """Opaque handle returned by ``StorageManager.submit_prefetch_task``.

    Carries the bookkeeping needed to later query lookup / prefetch status
    without exposing controller internals.
    """

    prefetch_request_id: int
    """Opaque ID for tracking L2 prefetch in the controller.
    -1 if no L2 request was submitted."""

    external_request_id: str
    """Request ID from the caller for end-to-end tracing."""

    l1_found_indices: tuple[int, ...]
    """Original-key indices found (read-locked) in L1 at submission time."""

    total_requested_keys: int
    """Total number of keys originally requested (the result-bitmap size)."""

    submit_time: float
    """Monotonic timestamp when the prefetch task was submitted."""

    l2_orig_indices: tuple[int, ...] = ()
    """Original-key index of each key submitted to L2; maps the controller's
    local result bitmap back to original positions."""


def ipc_key_to_object_keys(
    ipc_key: "IPCCacheServerKey",
    chunk_hashes: list[bytes],
    object_group_ids: list[int],
) -> list[list[ObjectKey]]:
    """
    Convert a single IPCCacheServerKey and its chunk hashes to per-object-group
    lists of ObjectKey.

    When the ipc_key's worker_id is None, each chunk hash is exploded into
    multiple ObjectKeys (one per worker in world_size).

    ``cache_salt`` is read directly from ``ipc_key`` so the produced
    ObjectKeys are per-user isolated whenever the sender set a non-empty
    salt. There is intentionally no separate ``cache_salt`` parameter —
    duplicating the source of truth would risk silent isolation bugs
    where a caller passes ``ipc_key`` but forgets the salt.

    Args:
        ipc_key: The IPC key providing model_name, world_size, worker_id,
            and cache_salt.
        chunk_hashes: List of chunk hash bytes, one per chunk.
        object_group_ids: Object group ids to produce keys for.

    Returns:
        list[list[ObjectKey]]: The i-th element is the list of ObjectKeys
        for ``object_group_ids[i]``.
    """
    cache_salt = ipc_key.cache_salt

    # The (chunk_hash, kv_rank) expansion is independent of the object group,
    # so compute it once and reuse it for every group.
    if ipc_key.worker_id is None:
        # For look up request, we want to expand to all workers
        # TODO (ApostaC): include local world size/rank info
        # in the future once it's in IPCCacheServerKey
        kv_ranks = [
            ObjectKey.ComputeKVRank(
                world_size=ipc_key.world_size,
                global_rank=worker_id,
                local_world_size=ipc_key.world_size,
                local_rank=worker_id,
            )
            for worker_id in range(ipc_key.world_size)
        ]
    else:
        kv_ranks = [
            ObjectKey.ComputeKVRank(
                world_size=ipc_key.world_size,
                global_rank=ipc_key.worker_id,
                local_world_size=ipc_key.world_size,
                local_rank=ipc_key.worker_id,
            )
        ]

    return [
        [
            ObjectKey(
                chunk_hash=chunk_hash,
                model_name=ipc_key.model_name,
                kv_rank=kv_rank,
                object_group_id=object_group_id,
                cache_salt=cache_salt,
            )
            for chunk_hash in chunk_hashes
            for kv_rank in kv_ranks
        ]
        for object_group_id in object_group_ids
    ]
