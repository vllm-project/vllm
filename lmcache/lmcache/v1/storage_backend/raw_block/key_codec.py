# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Literal
import hashlib
import urllib.parse

# First Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey, parse_cache_key
from lmcache.v1.distributed.api import ObjectKey

RawBlockKeyNamespace = Literal["legacy", "object"]

_KEY_SEP = "@"
_UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class RawBlockKeySpec:
    """Encoded raw-block key plus stable slot-header identity."""

    encoded: str
    slot_identity: int


def object_key_to_string(key: ObjectKey) -> str:
    """Serialize an ObjectKey using raw-block's reversible key shape.

    Args:
        key: Object key supplied by the MP storage layer.

    Returns:
        A stable string containing model name, KV rank, object group id,
        chunk hash, and optional cache salt.

    Raises:
        AttributeError: If ``key`` does not expose the ObjectKey fields.
    """
    safe_model = urllib.parse.quote(key.model_name, safe="")
    base = (
        f"{safe_model}{_KEY_SEP}{key.kv_rank:#010x}"
        f"{_KEY_SEP}{key.object_group_id:x}{_KEY_SEP}{key.chunk_hash.hex()}"
    )
    if key.cache_salt:
        return f"{base}{_KEY_SEP}{key.cache_salt}"
    return base


def decode_object_key(encoded: str) -> ObjectKey:
    """Deserialize an ObjectKey from raw-block's reversible encoding.

    Args:
        encoded: Encoded key string produced by ``object_key_to_string``.

    Returns:
        The reconstructed ``ObjectKey``.

    Raises:
        ValueError: If the encoded string has an unexpected shape or invalid
            hexadecimal chunk hash.
    """
    parts = encoded.split(_KEY_SEP)
    if len(parts) == 4:
        safe_model, kv_rank_str, object_group_str, chunk_hash_hex = parts
        cache_salt = ""
    elif len(parts) == 5:
        safe_model, kv_rank_str, object_group_str, chunk_hash_hex, cache_salt = parts
    else:
        raise ValueError(f"Invalid raw-block ObjectKey encoding: {encoded!r}")

    return ObjectKey(
        chunk_hash=bytes.fromhex(chunk_hash_hex),
        model_name=urllib.parse.unquote(safe_model),
        kv_rank=int(kv_rank_str, 16),
        object_group_id=int(object_group_str, 16),
        cache_salt=cache_salt,
    )


def encode_object_key(key: ObjectKey) -> RawBlockKeySpec:
    """Encode an MP ObjectKey for raw-block storage.

    Args:
        key: Object key supplied by the MP storage layer.

    Returns:
        Encoded key and deterministic slot-header identity.
    """
    encoded = object_key_to_string(key)
    return RawBlockKeySpec(
        encoded=encoded,
        slot_identity=_object_slot_identity(encoded),
    )


def decode_legacy_key(encoded: str) -> CacheEngineKey | LayerCacheEngineKey:
    """Deserialize a legacy non-MP cache key string.

    Args:
        encoded: String produced by ``CacheEngineKey.to_string`` or
            ``LayerCacheEngineKey.to_string``.

    Returns:
        Parsed legacy cache key.

    Raises:
        TypeError: If parsing returns an unsupported key type.
    """
    parsed = parse_cache_key(encoded)
    if not isinstance(parsed, (CacheEngineKey, LayerCacheEngineKey)):
        raise TypeError(
            "parse_cache_key returned unsupported key type "
            f"{type(parsed).__name__} for {encoded!r}"
        )
    return parsed


def encode_legacy_key(key: CacheEngineKey | LayerCacheEngineKey) -> RawBlockKeySpec:
    """Encode a legacy non-MP cache key for raw-block storage.

    Args:
        key: Legacy cache key from the non-MP storage plugin path.

    Returns:
        Encoded key and slot-header identity derived from the chunk hash.
    """
    return RawBlockKeySpec(
        encoded=key.to_string(),
        slot_identity=int(key.chunk_hash) & _UINT64_MASK,
    )


def slot_identity_from_encoded_key(
    encoded: str,
    namespace: RawBlockKeyNamespace,
) -> int:
    """Return the stable slot-header identity for an encoded key.

    Args:
        encoded: Encoded raw-block key string.
        namespace: Key namespace used when the key was encoded.

    Returns:
        Unsigned 64-bit identity stored in the per-slot header.

    Raises:
        ValueError: If ``namespace`` is unsupported.
        TypeError: If legacy key parsing returns an unsupported key type.
    """
    if namespace == "legacy":
        key = decode_legacy_key(encoded)
        return int(key.chunk_hash) & _UINT64_MASK
    if namespace == "object":
        return _object_slot_identity(encoded)
    raise ValueError(f"Unsupported raw-block key namespace: {namespace!r}")


def _object_slot_identity(encoded: str) -> int:
    """Return the stable 64-bit slot identity for an encoded ObjectKey.

    Args:
        encoded: Raw-block ObjectKey string produced by ``object_key_to_string``.

    Returns:
        Unsigned 64-bit identity stored in the raw-block slot header.
    """
    digest = hashlib.blake2b(encoded.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)
