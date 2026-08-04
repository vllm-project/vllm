# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import Any, Callable

# Third Party
import msgspec
import torch

# First Party
from lmcache.v1.platform.base.ipc_wrapper import (  # noqa: E402,F401
    DeviceIPCWrapper,
)

"""
Defines the types and the customized encoder/decoders for inter-process
communications.

Key Types:
- IPCCacheServerKey: Token-based cache key
  - Contains token_ids, start, end, request_id (all required)
  - Converted to ObjectKey for storage operations via ipc_key_to_object_keys()
"""


@dataclass(order=True, frozen=True)
class IPCCacheServerKey:
    """Cache key for the IPC (multiprocess) protocol.

    This key type is sent by the client over ZMQ (serialized via msgspec).

    The client sends token_ids, start, end, and request_id (all required).
    The server computes chunk hashes via TokenHasher and converts to
    ObjectKey for storage operations using ipc_key_to_object_keys().

    The request_id field is for session tracking and is NOT included
    in equality/hash comparisons (two keys with same content but different
    request_ids are considered equal for cache purposes).
    """

    model_name: str
    world_size: int
    worker_id: int | None

    token_ids: tuple[int, ...]  # frozen tuple for hashability
    start: int
    end: int

    # === Session tracking (not part of cache identity) ===
    request_id: str = field(compare=False)

    # === Per-user isolation salt (part of cache identity) ===
    # msgspec encodes dataclasses as maps, so forward wire compatibility
    # works by field name: an old payload without ``cache_salt`` decodes
    # on new code using the default "". Placing the field last is a style
    # choice — all defaulted fields must come after non-defaulted ones.
    #
    # Invariant: must not contain ``@``, ``/``, ``\``, or NUL, and
    # must be <= 128 chars — same rationale as ObjectKey (see
    # ObjectKey.cache_salt). Validated in __post_init__.
    cache_salt: str = ""

    # Duplicated from ObjectKey — cannot import ObjectKey here due to
    # circular dependency (api.py imports IPCCacheServerKey).
    _SALT_FORBIDDEN_CHARS = frozenset("@/\\\x00")
    _SALT_MAX_LEN = 128

    def __post_init__(self) -> None:
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

    # Helper function for unit tests only
    @classmethod
    def from_token_ids(
        cls,
        model_name: str,
        world_size: int,
        worker_id: int | None,
        token_ids: list[int],
        start: int = 0,
        end: int = 0,
        request_id: str = "",
        cache_salt: str = "",
    ) -> "IPCCacheServerKey":
        """Create a key from token ids. Only used by the tests."""
        return cls(
            model_name=model_name,
            world_size=world_size,
            worker_id=worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
        )

    def no_worker_id_version(self) -> "IPCCacheServerKey":
        """Create a copy with worker_id=None for lookup requests."""
        return IPCCacheServerKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=None,
            token_ids=self.token_ids,
            start=self.start,
            end=self.end,
            request_id=self.request_id,
            cache_salt=self.cache_salt,
        )


# Type exports
KVCache = list[DeviceIPCWrapper]


class RegisterEngineDrivenContextPayload(msgspec.Struct):
    """Payload for the REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT protocol message.

    Attributes:
        instance_id: Worker instance identifier (typically PID).
        model_name: Model name associated with this worker.
        world_size: Worker world size used in cache keys.
        block_size: Tokens per paged block.
        num_layers: Number of model layers.
        hidden_dim_size: Flattened hidden dimension per token.
        dtype_str: Torch dtype name (e.g. ``"float16"``).
        use_mla: Whether the worker KV format is MLA.
    """

    instance_id: int
    model_name: str
    world_size: int
    block_size: int
    num_layers: int
    hidden_dim_size: int
    dtype_str: str
    use_mla: bool


@dataclass
class CustomizedSerdeConfig:
    serializer: Callable[[Any], bytes]
    deserializer: Callable[[bytes], Any]
    code: int


@dataclass
class BlockAllocationRecord:
    """A single per-request GPU block allocation delta from vLLM."""

    req_id: str
    new_block_ids: list[int]
    new_token_ids: list[int]


@dataclass
class CBMatchResult:
    """Result of a sub-sequence match from BlendTokenRangeMatcher.

    Attributes:
        old_st: Start position in the originally registered (stored) sequence.
        old_ed: End position in the originally registered (stored) sequence.
        cur_st: Start position in the query sequence where the match was found.
        cur_ed: End position in the query sequence where the match was found.
        hash: Token hash bytes (from registration) used as the storage key.
    """

    old_st: int
    old_ed: int
    cur_st: int
    cur_ed: int
    hash: bytes


@dataclass
class CBUnifiedLookupResult:
    """Resolved payload of ``CB_UNIFIED_LOOKUP``: prefix lookup + non-prefix
    fingerprint match, reconciled in one RPC. The RPC returns ``None`` (not this)
    while either leg's KV is still loading into L1; this type is sent only once
    both are resident.

    Attributes:
        prefix_coverage_tokens: Contiguous prefix-cache coverage (L1+L2) in
            tokens — what the standard LOOKUP would report.
        non_prefix_segments: Fingerprint matches outside the prefix coverage
            (cur_st order), each carrying ``(old_st, old_ed, cur_st, cur_ed,
            hash)``. Already sparse-prefetched, so the retrieve set equals the
            prefetched set. Includes fleet-coordinator (shared-L2) matches:
            those are merged in before the sparse prefetch -- prefix-covered and
            locally-duplicated ones dropped -- so they ride the identical
            prefetch + retrieve path and need no separate handling.
        segmented_prefix_segments: Post-gap chunks retained by the
            ``SEGMENTED_PREFIX`` prefix leg (beyond ``count_leading_ones``) — at
            their original positions (``old_st == cur_st``), so the connector
            tags them ``prefix`` (pure load, no recompute) and only the gap is
            recomputed. Sourced from the prefix bitmap, not the fingerprint
            matcher; empty when ``SEGMENTED_PREFIX`` is off.
    """

    prefix_coverage_tokens: int
    non_prefix_segments: list[CBMatchResult]
    segmented_prefix_segments: list[CBMatchResult] = field(default_factory=list)


_CUSTOMERIZED_SERIALIZERS = {
    DeviceIPCWrapper: CustomizedSerdeConfig(
        serializer=DeviceIPCWrapper.Serialize,
        deserializer=DeviceIPCWrapper.Deserialize,
        code=1,
    ),
}


def get_customized_encoder(type: Any) -> msgspec.msgpack.Encoder:
    # TODO: `type` is not used here
    def enc_hook(obj: Any) -> Any:
        for supported_type, cfg in _CUSTOMERIZED_SERIALIZERS.items():
            if isinstance(obj, supported_type):
                data = cfg.serializer(obj)
                return msgspec.msgpack.Ext(cfg.code, data)
        if isinstance(obj, torch.dtype):
            return str(obj).removeprefix("torch.")
        if isinstance(obj, torch.Size):
            return list(obj)
        raise TypeError(f"Unsupported type for serialization: {type(obj)}")

    return msgspec.msgpack.Encoder(enc_hook=enc_hook)


def get_customized_decoder(type: Any) -> msgspec.msgpack.Decoder:
    def ext_hook(code: int, data: bytes) -> Any:
        for cfg in _CUSTOMERIZED_SERIALIZERS.values():
            if cfg.code == code:
                return cfg.deserializer(data)
        raise TypeError(f"Unsupported ext code for deserialization: {code}")

    def dec_hook(expected_type: type, obj: Any) -> Any:
        if expected_type is torch.dtype:
            return getattr(torch, obj)
        if expected_type is torch.Size:
            return torch.Size(obj)
        if isinstance(obj, expected_type):
            return obj
        raise NotImplementedError(
            f"Unsupported type for deserialization: {expected_type}"
        )

    return msgspec.msgpack.Decoder(ext_hook=ext_hook, dec_hook=dec_hook, type=type)
