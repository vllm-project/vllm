# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake-store backend for the multi-modal processor cache.

The `lru` and `shm` backends keep processed items inside the process (or the
node) that produced them. This backend keeps them in a Mooncake object store
shared by every process pointed at the same cluster, so a processed item
survives an API-server restart and is reusable by other API-server processes
and other engine instances.

Two objects are written per multi-modal item, so that P0 never reads the large
one:

    <prefix>/meta/<mm_hash>    `(version, item_size, prompt_updates)`
    <prefix>/kwargs/<mm_hash>  msgpack `MultiModalKwargsItem`

P0 needs the prompt updates to expand placeholders and nothing else; the
tensors are only needed in P1.

Remote state is resolved in exactly one place, `is_cached()`: it confirms the
kwargs object exists *and* pulls the metadata object, and only reports a hit
once it holds both. `get_and_update_item()` never touches the network, because
by then the caller has already skipped processing on the strength of
`is_cached()` and can no longer recover from a miss.

Shadow entries are re-verified against the store every `shadow_ttl_s` seconds.
Without that, an entry the store evicted long ago would keep reporting a local
hit until a request actually failed in P1.

A shadow entry therefore only ever means "the store can serve this". An item
P0 has just published is not that yet -- its write is still in flight on the
writer thread -- so entries carry a publish state and only satisfy a lookup
once the write is confirmed.

Prompt updates are encoded through an explicit schema rather than `pickle`,
both because vLLM forbids `pickle` and because this data crosses a network. An
update whose `is_embed` or target is an opaque callable cannot be expressed in
that schema, so such items are kept local instead of being published.
"""

import enum
import json
import os
import struct
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from msgspec import msgpack
from typing_extensions import override

from vllm.logger import init_logger
from vllm.utils.cache import CacheInfo, LRUCache
from vllm.utils.mem_constants import GiB_bytes

from .cache import (
    BaseMultiModalProcessorCache,
    BaseMultiModalReceiverCache,
    MultiModalCache,
    MultiModalCacheMissError,
    MultiModalProcessorCacheInItem,
    MultiModalProcessorCacheOutItem,
)
from .inputs import MultiModalKwargsItem

if TYPE_CHECKING:
    from vllm.config import ModelConfig

    from .processing.processor import ResolvedPromptUpdate

logger = init_logger(__name__)

DEFAULT_KEY_PREFIX = "vllm/mm-processor-cache"
DEFAULT_SHADOW_TTL_S = 30.0


@dataclass
class MooncakeProcessorCacheOptions:
    """Options read from the optional `mm_processor_cache` section of the JSON
    file pointed to by `MOONCAKE_CONFIG_PATH`.

    The remaining connection settings are shared with the Mooncake KV
    connector and come from `MooncakeStoreConfig`.
    """

    key_prefix: str = DEFAULT_KEY_PREFIX
    shadow_ttl_s: float = DEFAULT_SHADOW_TTL_S
    global_segment_size: int = 0
    """Bytes this process contributes to the cluster. Defaults to 0: the
    frontend and engine processes are clients, and capacity is expected to come
    from the ranks (or the standalone `mooncake_client`) that already host the
    pool."""
    tenant_id: str | None = None
    """Overrides the KV connector's tenant so that multi-modal objects can be
    accounted for, and evicted, separately from KV blocks."""

    @staticmethod
    def load() -> "MooncakeProcessorCacheOptions":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError(
                "mm_processor_cache_type='mooncake' requires the environment "
                "variable 'MOONCAKE_CONFIG_PATH' to be set."
            )

        with open(config_path) as f:
            section: dict[str, Any] = json.load(f).get("mm_processor_cache", {})

        return MooncakeProcessorCacheOptions(
            key_prefix=section.get("key_prefix", DEFAULT_KEY_PREFIX),
            shadow_ttl_s=float(section.get("shadow_ttl_s", DEFAULT_SHADOW_TTL_S)),
            global_segment_size=int(section.get("global_segment_size", 0)),
            tenant_id=section.get("tenant_id"),
        )


_META_VERSION = 1

_PARTS_HEADER = struct.Struct("<I")
"""Size of the encoded part-length array that prefixes a kwargs payload."""


class _Unshareable(Exception):
    """A prompt update holds something the wire schema cannot express."""


def _encode_prompt_updates(
    prompt_updates: Sequence["ResolvedPromptUpdate"],
) -> list[list[Any]]:
    """Encode prompt updates as plain data.

    Raises:
        _Unshareable: An update carries an opaque callable, either as its
            target (`PromptIndex`) or as a custom `is_embed`.
    """
    from .processing.processor import _SelectTokenId, _SelectTokenIds

    encoded = list[list[Any]]()
    for update in prompt_updates:
        if not isinstance(update.target, list):
            raise _Unshareable(f"target of type {type(update.target).__name__}")

        is_embed = update.content.is_embed
        if is_embed is None:
            embed: list[Any] | None = None
        elif isinstance(is_embed, _SelectTokenId):
            embed = ["token_id", [is_embed.embed_token_id]]
        elif isinstance(is_embed, _SelectTokenIds):
            embed = ["token_ids", list(is_embed.embed_token_ids)]
        else:
            raise _Unshareable(f"is_embed of type {type(is_embed).__name__}")

        encoded.append(
            [
                update.modality,
                update.item_idx,
                update.mode.value,
                update.target,
                update.content.full,
                embed,
            ]
        )

    return encoded


def _decode_prompt_updates(
    encoded: list[list[Any]],
) -> list["ResolvedPromptUpdate"]:
    from .processing.processor import (
        PromptUpdateDetails,
        ResolvedPromptUpdate,
        UpdateMode,
    )

    decoded = list[ResolvedPromptUpdate]()
    for modality, item_idx, mode, target, full, embed in encoded:
        if embed is None:
            content = PromptUpdateDetails.from_seq(full)
        elif embed[0] == "token_id":
            content = PromptUpdateDetails.select_token_id(full, embed[1][0])
        elif embed[0] == "token_ids":
            content = PromptUpdateDetails.select_token_ids(full, embed[1])
        else:
            raise ValueError(f"Unknown is_embed encoding: {embed[0]!r}")

        decoded.append(
            ResolvedPromptUpdate(
                modality=modality,
                item_idx=item_idx,
                mode=UpdateMode(mode),
                target=target,
                content=content,
            )
        )

    return decoded


class MooncakeProcessorStore:
    """Mooncake client for processed multi-modal items.

    One instance per process, shared by the sender and receiver caches (only
    one of which exists in any given process).
    """

    _instance: "MooncakeProcessorStore | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_or_create(cls) -> "MooncakeProcessorStore":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()

            return cls._instance

    def __init__(
        self,
        store: Any | None = None,
        options: MooncakeProcessorCacheOptions | None = None,
    ) -> None:
        from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

        self.options = (
            options if options is not None else (MooncakeProcessorCacheOptions.load())
        )
        # Two clients, not one. The buffers `get_batch` returns are not stable
        # against concurrent operations on the same client: a multi-key
        # `get_batch` racing a write comes back short or shifted for some of its
        # keys. Reads run on the caller's thread and writes on the writer
        # thread, so a client each removes the race without serializing reads
        # behind megabyte-sized writes.
        self._store = store if store is not None else self._connect("read")
        self._write_store = store if store is not None else self._connect("write")

        # `MsgpackEncoder` is not thread-safe; it is only ever used from the
        # single writer thread. The decoder only runs in P1, on the engine's
        # own thread.
        self._encoder = MsgpackEncoder()
        self._decoder = MsgpackDecoder(MultiModalKwargsItem, share_mem=False)
        self._writer = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mm-mooncake-put",
        )
        # Cleared for good if this model's prompt updates turn out not to be
        # expressible on the wire.
        self._shareable = True

    def _connect(self, role: str) -> Any:
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/"
                "build.md to use mm_processor_cache_type='mooncake'."
            ) from e

        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
            MooncakeStoreConfig,
        )
        from vllm.utils.network_utils import get_ip

        store_config = MooncakeStoreConfig.load_from_config()
        tenant_id = self.options.tenant_id or store_config.tenant_id

        store = MooncakeDistributedStore()
        ret = store.setup(
            rdma_utils.get_requester_local_hostname(get_ip()),
            store_config.metadata_server,
            self.options.global_segment_size if role == "read" else 0,
            store_config.local_buffer_size,
            store_config.protocol,
            store_config.device_name,
            store_config.master_server_address,
            tenant_id=tenant_id,
        )
        if ret != 0:
            raise RuntimeError(
                f"Failed to initialize the Mooncake store for the multi-modal "
                f"processor cache (setup returned {ret})."
            )

        logger.info(
            "Multi-modal processor cache backed by Mooncake "
            "(%s client, key_prefix=%s, shadow_ttl_s=%s, "
            "global_segment_size=%d, tenant_id=%s)",
            role,
            self.options.key_prefix,
            self.options.shadow_ttl_s,
            self.options.global_segment_size,
            tenant_id,
        )

        return store

    @property
    def shadow_ttl_s(self) -> float:
        return self.options.shadow_ttl_s

    def _meta_key(self, mm_hash: str) -> str:
        return f"{self.options.key_prefix}/meta/{mm_hash}"

    def _kwargs_key(self, mm_hash: str) -> str:
        return f"{self.options.key_prefix}/kwargs/{mm_hash}"

    def probe(
        self,
        mm_hashes: list[str],
    ) -> dict[str, tuple[int, Sequence["ResolvedPromptUpdate"]]]:
        """Return `(item_size, prompt_updates)` for each hash fully present in
        the store.

        A hash is reported only when its kwargs object exists *and* its
        metadata could be read, so that the caller can commit to skipping
        processing. Store errors degrade to a miss rather than propagating.
        """
        if not mm_hashes:
            return {}

        # One request repeats an item whenever the same image appears twice in
        # a prompt, so `mm_hashes` can hold duplicates. Batch lookups must see
        # each key once: a repeated key in a `get_batch` comes back correct for
        # one occurrence and shifted for the other.
        mm_hashes = list(dict.fromkeys(mm_hashes))

        try:
            exists = self._store.batch_is_exist(
                [self._kwargs_key(mm_hash) for mm_hash in mm_hashes]
            )
        except Exception as e:
            logger.warning_once(
                "Mooncake batch_is_exist failed (%s); treating multi-modal "
                "items as uncached.",
                e,
            )
            return {}

        present = [mm_hash for mm_hash, state in zip(mm_hashes, exists) if state == 1]
        if not present:
            return {}

        try:
            blobs = self._store.get_batch(
                [self._meta_key(mm_hash) for mm_hash in present]
            )
        except Exception as e:
            logger.warning_once(
                "Mooncake get_batch failed for multi-modal metadata (%s); "
                "treating the items as uncached.",
                e,
            )
            return {}

        found: dict[str, tuple[int, Sequence[ResolvedPromptUpdate]]] = {}
        for mm_hash, blob in zip(present, blobs):
            if not blob:
                continue

            try:
                version, item_size, encoded_updates = msgpack.decode(blob)
                if version != _META_VERSION:
                    raise ValueError(f"unsupported metadata version {version}")

                prompt_updates = _decode_prompt_updates(encoded_updates)
            except Exception:
                # Log enough to diagnose the object without the store, then
                # drop it: the next publish upserts a fresh pair, whereas
                # leaving it would fail every probe for as long as it lives.
                logger.warning(
                    "Discarding unreadable Mooncake metadata for mm_hash %s "
                    "(%d bytes, starts with %s).",
                    mm_hash,
                    len(blob),
                    bytes(blob[:32]).hex(),
                    exc_info=True,
                )
                self.drop_meta(mm_hash)
                continue

            found[mm_hash] = (item_size, prompt_updates)

        return found

    def _drop(self, mm_hash: str, *, meta_only: bool) -> None:
        """Remove objects that could not be read back.

        Publishing uses `put`, which declines to overwrite, so an unreadable
        object has to be removed before it can be replaced. Runs on the writer
        thread to keep store writes off the caller's path.
        """
        keys = [self._meta_key(mm_hash)]
        if not meta_only:
            keys.append(self._kwargs_key(mm_hash))

        def remove() -> None:
            for key in keys:
                try:
                    self._write_store.remove(key)
                except Exception:
                    logger.debug("Failed to drop %s.", key, exc_info=True)

        self._writer.submit(remove)

    def drop_meta(self, mm_hash: str) -> None:
        """Remove an undecodable metadata object."""
        self._drop(mm_hash, meta_only=True)

    def drop(self, mm_hash: str) -> None:
        """Remove both objects of an item that could not be retrieved."""
        self._drop(mm_hash, meta_only=False)

    def put(
        self,
        mm_hash: str,
        item: MultiModalKwargsItem,
        prompt_updates: Sequence["ResolvedPromptUpdate"],
        item_size: int,
        on_done: Callable[[bool], None] | None = None,
    ) -> None:
        """Publish an item to the store, off the caller's thread.

        Best effort: a dropped write only costs a later miss, so failures are
        logged rather than raised. `on_done` is called on the writer thread
        with whether the item can now be read back, so that callers can hold
        off on promising it to P1 until then.
        """
        if not self._shareable:
            if on_done is not None:
                on_done(False)
            return

        def publish() -> None:
            ok = self._put_blocking(mm_hash, item, prompt_updates, item_size)
            if on_done is not None:
                on_done(ok)

        self._writer.submit(publish)

    def _put_blocking(
        self,
        mm_hash: str,
        item: MultiModalKwargsItem,
        prompt_updates: Sequence["ResolvedPromptUpdate"],
        item_size: int,
    ) -> bool:
        """Write both objects, returning whether the item is now readable."""
        try:
            meta = msgpack.encode(
                (_META_VERSION, item_size, _encode_prompt_updates(prompt_updates))
            )
        except _Unshareable as e:
            # Models are free to build `is_embed` as a closure, which cannot be
            # expressed on the wire. Such items stay local-only instead of
            # failing the request.
            logger.warning_once(
                "This model's prompt updates cannot be shared through Mooncake "
                "(%s); processed multi-modal items will only be cached "
                "in-process.",
                e,
            )
            # A property of the model, not of this item: stop trying.
            self._shareable = False
            return False
        except Exception:
            logger.warning_once(
                "Failed to encode multi-modal prompt updates; processed items "
                "will only be cached in-process.",
            )
            return False

        try:
            bufs = self._encoder.encode(item)
            lengths = msgpack.encode([len(buf) for buf in bufs])
            header = _PARTS_HEADER.pack(len(lengths)) + lengths
            # `put`, not `upsert`. Neither primitive is safe on its own:
            # `put` declines to overwrite an existing key (so a damaged object
            # would be permanent), while a losing `upsert` has already
            # repointed the key in its start phase before it reports the
            # conflict -- leaving a key that exists but cannot be read, which
            # the readers below would then promise to the engine. `put` never
            # damages a key, and repair comes from removing the bad object
            # first (see `drop_meta` and the receiver cache).
            # kwargs first: a reader that sees the metadata can then assume the
            # kwargs object was written too.
            ret = self._write_store.put_parts(self._kwargs_key(mm_hash), header, *bufs)
            if ret != 0:
                logger.debug(
                    "Mooncake put_parts returned %d for mm_hash %s.", ret, mm_hash
                )
                return False

            ret = self._write_store.put(self._meta_key(mm_hash), meta)
            if ret != 0:
                logger.debug(
                    "Mooncake put returned %d for the metadata of mm_hash %s.",
                    ret,
                    mm_hash,
                )
                return False
        except Exception:
            logger.debug(
                "Failed to publish mm_hash %s to Mooncake.",
                mm_hash,
                exc_info=True,
            )
            return False

        return True

    def get_kwargs(self, mm_hash: str) -> MultiModalKwargsItem | None:
        """Read back a processed item, or `None` if it is not retrievable."""
        try:
            data = self._store.get(self._kwargs_key(mm_hash))
        except Exception:
            logger.warning(
                "Mooncake get failed for mm_hash %s.", mm_hash, exc_info=True
            )
            return None

        if not data:
            return None

        try:
            view = memoryview(data)
            head = _PARTS_HEADER.size
            (lengths_size,) = _PARTS_HEADER.unpack_from(view)
            len_arr: list[int] = msgpack.decode(view[head : head + lengths_size])

            bufs = []
            start = head + lengths_size
            for length in len_arr:
                bufs.append(view[start : start + length])
                start += length

            return self._decoder.decode(bufs)
        except Exception:
            logger.warning(
                "Discarding undecodable Mooncake payload for mm_hash %s.",
                mm_hash,
                exc_info=True,
            )
            return None

    def flush(self) -> None:
        """Block until every submitted write has been attempted."""
        self._writer.submit(lambda: None).result()

    def close(self) -> None:
        self._writer.shutdown(wait=True)

        seen = set()
        for store in (self._store, self._write_store):
            if id(store) in seen:
                continue
            seen.add(id(store))
            try:
                store.close()
            except Exception:
                logger.debug("Error closing a Mooncake client.", exc_info=True)


class _PublishState(enum.Enum):
    """Whether the store can serve the item this shadow entry describes."""

    IN_FLIGHT = enum.auto()
    """P0 submitted the write but it has not been confirmed."""

    FAILED = enum.auto()
    """The write failed. Equivalent to having no entry at all; it lingers only
    because the writer thread cannot safely remove one, and the owning thread
    reaps it on the next lookup."""

    STORED = enum.auto()
    """Confirmed readable, either by `probe` or by a completed write."""


@dataclass
class _ShadowEntry:
    item_size: int
    prompt_updates: Sequence["ResolvedPromptUpdate"]
    verified_at: float
    state: _PublishState = _PublishState.STORED


class MooncakeProcessorSenderCache(BaseMultiModalProcessorCache):
    """
    The cache which is used on P0 when the backing store is Mooncake.

    How to update each item:

    - If it is shadowed locally as stored and was verified recently, clear the
      input to avoid unnecessary IPC.

    - If it is in the store, shadow its metadata and clear the input.

    - Otherwise, publish it to the store and return the input, so that P1 gets
      the data over IPC on the cold path instead of reading back what P0 just
      wrote.

    Only the metadata is kept in P0, as with
    [`MultiModalProcessorSenderCache`][vllm.multimodal.cache.MultiModalProcessorSenderCache],
    but the shadow can also go stale against the store, so entries carry the
    time they were last verified.

    Publishing is asynchronous, so an entry does not become usable the moment
    it is added: until the write is confirmed, further requests for the same
    hash take the cold path (see `_PublishState`). Clearing the input any
    earlier would tell P1 to read an object whose write is still queued, and
    P1 cannot recover from that except by failing the request.
    """

    def __init__(
        self,
        model_config: "ModelConfig",
        store: MooncakeProcessorStore | None = None,
    ) -> None:
        super().__init__()

        mm_config = model_config.get_multimodal_config()

        self._store = store or MooncakeProcessorStore.get_or_create()
        self._shadow = LRUCache[str, _ShadowEntry](
            GiB_bytes * mm_config.mm_processor_cache_gb,
            getsizeof=lambda entry: entry.item_size,
        )

        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)

    def _peek(self, mm_hash: str) -> _ShadowEntry | None:
        """Read a shadow entry without updating the eviction order."""
        return self._shadow.cache.get(mm_hash)

    @override
    def is_cached(self, mm_hashes: list[str]) -> list[bool]:
        now = time.monotonic()
        ttl = self._store.shadow_ttl_s

        unverified = []
        for mm_hash in mm_hashes:
            entry = self._peek(mm_hash)
            if entry is None:
                unverified.append(mm_hash)
            elif entry.state is _PublishState.IN_FLIGHT:
                # Probing now would miss and drop the entry, losing the record
                # that a write for this hash is already on its way.
                continue
            elif entry.state is _PublishState.FAILED:
                # Stands in for a removal the writer thread could not do. This
                # is the owning thread, so reap it and treat it as absent: the
                # store may hold a copy published by another process.
                self._shadow.pop(mm_hash, None)
                unverified.append(mm_hash)
            elif now - entry.verified_at > ttl:
                unverified.append(mm_hash)

        if unverified:
            found = self._store.probe(unverified)

            for mm_hash in unverified:
                meta = found.get(mm_hash)
                if meta is None:
                    # Either never stored, or evicted remotely while we still
                    # had a shadow entry for it.
                    self._shadow.pop(mm_hash, None)
                    continue

                item_size, prompt_updates = meta
                self._shadow.put_if_fits(
                    mm_hash,
                    _ShadowEntry(item_size, prompt_updates, now),
                )

        return [self._is_usable(self._peek(mm_hash)) for mm_hash in mm_hashes]

    @staticmethod
    def _is_usable(entry: _ShadowEntry | None) -> bool:
        """Whether P1 can be told to read this item from the store."""
        return entry is not None and entry.state is _PublishState.STORED

    def _publish_done(self, mm_hash: str, ok: bool) -> None:
        """Record the outcome of a publish. Runs on the writer thread.

        Only ever assigns to fields of an entry that is already in the shadow.
        Inserting or removing would race the eviction order with the thread
        that owns it; a single field write cannot, and a stale read there costs
        at most one extra cold path or probe.
        """
        entry = self._peek(mm_hash)
        if entry is None or entry.state is not _PublishState.IN_FLIGHT:
            return

        if ok:
            entry.verified_at = time.monotonic()
            entry.state = _PublishState.STORED
        else:
            entry.state = _PublishState.FAILED

    @override
    def is_cached_item(self, mm_hash: str) -> bool:
        return self.is_cached([mm_hash])[0]

    @override
    def get_and_update_item(
        self,
        mm_item: MultiModalProcessorCacheInItem,
        mm_hash: str,
    ) -> MultiModalProcessorCacheOutItem:
        self._total += 1

        entry = self._peek(mm_hash)
        if self._is_usable(entry):
            assert entry is not None
            self._hits += 1
            self._shadow.touch(mm_hash)
            return None, entry.prompt_updates

        assert mm_item is not None, f"Expected a cached item for {mm_hash=}"
        item, prompt_updates = mm_item

        if entry is None or entry.state is _PublishState.FAILED:
            item_size = MultiModalCache.get_item_size(item)
            # Shadowed as in-flight, not as stored: until the write lands,
            # another request for this hash must keep taking the cold path
            # rather than promise P1 an object that is not there yet. The entry
            # still goes in, so that request does not resubmit the same write.
            self._shadow.put_if_fits(
                mm_hash,
                _ShadowEntry(
                    item_size,
                    prompt_updates,
                    time.monotonic(),
                    _PublishState.IN_FLIGHT,
                ),
            )
            self._store.put(
                mm_hash,
                item,
                prompt_updates,
                item_size,
                on_done=lambda ok: self._publish_done(mm_hash, ok),
            )

        return mm_item

    @override
    def touch_sender_cache_item(self, mm_hash: str) -> None:
        self._shadow.touch(mm_hash)

    @override
    def invalidate(self, mm_hash: str) -> None:
        self._shadow.pop(mm_hash, None)

    @override
    def clear_cache(self) -> None:
        # The store is shared with other processes, so only the local shadow
        # is dropped.
        self._shadow.clear()

    @override
    def close(self) -> None:
        self._store.close()

    @override
    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        info = CacheInfo(hits=self._hits, total=self._total)

        if delta:
            info_delta = info - self._last_info
            self._last_info = info
            info = info_delta

        return info


class MooncakeProcessorReceiverCache(BaseMultiModalReceiverCache):
    """
    The cache which is used on P1 when the backing store is Mooncake.

    How to update each item:

    - If the item is in the local cache, replace the input with the cached one.

    - If the input carries data (cold path), store and return it.

    - Otherwise read it from the store; failing that, the P0 shadow has drifted
      and a retryable error is raised.

    P1 never writes to the store: P0 publishes each item as it is processed.
    """

    def __init__(
        self,
        model_config: "ModelConfig",
        store: MooncakeProcessorStore | None = None,
    ) -> None:
        super().__init__()

        mm_config = model_config.get_multimodal_config()

        self._store = store or MooncakeProcessorStore.get_or_create()
        self._cache = MultiModalCache.get_lru_cache(
            mm_config.mm_processor_cache_gb,
            MultiModalKwargsItem,
        )

    @override
    def get_and_update_item(
        self,
        mm_item: MultiModalKwargsItem | None,
        mm_hash: str,
    ) -> MultiModalKwargsItem:
        if (cached_item := self._cache.get(mm_hash)) is not None:
            return cached_item

        if mm_item is None:
            mm_item = self._store.get_kwargs(mm_hash)

        if mm_item is None:
            # P0 sent data=None trusting its shadow, but the item is not
            # retrievable. Drop the pair: the key may exist while its data does
            # not, in which case `put` would decline to republish it and every
            # later request for this item would fail the same way.
            self._store.drop(mm_hash)

            # Raise a retryable error so P0 drops the stale entry and the
            # client resends the data.
            raise MultiModalCacheMissError([mm_hash])

        self.cache_if_fits(self._cache, mm_hash, mm_item)

        return mm_item

    @override
    def touch_receiver_cache_item(
        self,
        mm_hash: str,
        mm_item: MultiModalKwargsItem | None = None,
    ) -> None:
        self._cache.touch(mm_hash)

    @override
    def clear_cache(self) -> None:
        self._cache.clear()
