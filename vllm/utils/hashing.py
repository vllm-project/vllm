# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import pickle
from _hashlib import HASH, UnsupportedDigestmodError
from collections.abc import Callable, Sequence
from typing import Any

import cbor2
import msgspec

try:
    # It is important that this remains an optional dependency.
    # It would not be allowed in environments with strict security controls,
    # so it's best not to have it installed when not in use.
    import xxhash as _xxhash

    if not hasattr(_xxhash, "xxh3_128_digest"):
        _xxhash = None
except ImportError:  # pragma: no cover
    _xxhash = None


def sha256(input: Any) -> bytes:
    """Hash any picklable Python object using SHA-256.

    The input is serialized using pickle before hashing, which allows
    arbitrary Python objects to be used. Note that this function does
    not use a hash seed—if you need one, prepend it explicitly to the input.

    Args:
        input: Any picklable Python object.

    Returns:
        Bytes representing the SHA-256 hash of the serialized input.
    """
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(input_bytes).digest()


def sha256_cbor(input: Any) -> bytes:
    """Hash objects using CBOR serialization and SHA-256.

    This option is useful for non-Python-dependent serialization and hashing.

    Args:
        input: Object to be serialized and hashed. Supported types include
            basic Python types and complex structures like lists, tuples, and
            dictionaries.
            Custom classes must implement CBOR serialization methods.

    Returns:
        Bytes representing the SHA-256 hash of the CBOR serialized input.
    """
    input_bytes = cbor2.dumps(input, canonical=True)
    return hashlib.sha256(input_bytes).digest()


_MSGPACK_ENCODER = msgspec.msgpack.Encoder()


def sha256_msgpack(input: Any) -> bytes:
    """Hash objects using MessagePack serialization and SHA-256."""
    input_bytes = _MSGPACK_ENCODER.encode(input)
    return hashlib.sha256(input_bytes).digest()


def _xxhash_digest(input_bytes: bytes | bytearray | memoryview) -> bytes:
    if _xxhash is None:
        raise ModuleNotFoundError(
            "xxhash is required for the 'xxhash' prefix caching hash algorithms. "
            "Install it via `pip install xxhash`."
        )
    return _xxhash.xxh3_128_digest(input_bytes)


def xxhash(input: Any) -> bytes:
    """Hash picklable objects using xxHash."""
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return _xxhash_digest(input_bytes)


def xxhash_cbor(input: Any) -> bytes:
    """Hash objects serialized with CBOR using xxHash."""
    input_bytes = cbor2.dumps(input, canonical=True)
    return _xxhash_digest(input_bytes)


def xxhash_msgpack(input: Any) -> bytes:
    """Hash objects serialized with MessagePack using xxHash."""
    input_bytes = _MSGPACK_ENCODER.encode(input)
    return _xxhash_digest(input_bytes)


_MSGPACK_HASH_FUNCTIONS = frozenset({sha256_msgpack, xxhash_msgpack})


def hash_block_token_sequence(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: bytes,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> bytes:
    """Hash a prefix-cache block and its parent hash.

    Existing algorithms preserve the legacy tuple-based object shape exactly.
    MessagePack algorithms avoid the tuple conversion because MessagePack
    encodes tuples and lists as the same array type.
    """
    if hash_function in _MSGPACK_HASH_FUNCTIONS:
        return hash_function((parent_block_hash, curr_block_token_ids, extra_keys))

    return hash_function((parent_block_hash, tuple(curr_block_token_ids), extra_keys))


def get_block_hash_fn(
    hash_function: Callable[[Any], bytes],
) -> Callable[[bytes, Sequence[int], tuple[Any, ...] | None], bytes]:
    """Return a block-hash function specialized for ``hash_function``."""
    if hash_function in _MSGPACK_HASH_FUNCTIONS:

        def msgpack_block_hash(
            parent_block_hash: bytes,
            curr_block_token_ids: Sequence[int],
            extra_keys: tuple[Any, ...] | None = None,
        ) -> bytes:
            return hash_function((parent_block_hash, curr_block_token_ids, extra_keys))

        return msgpack_block_hash

    def tuple_block_hash(
        parent_block_hash: bytes,
        curr_block_token_ids: Sequence[int],
        extra_keys: tuple[Any, ...] | None = None,
    ) -> bytes:
        return hash_function(
            (parent_block_hash, tuple(curr_block_token_ids), extra_keys)
        )

    return tuple_block_hash


def get_hash_fn_by_name(hash_fn_name: str) -> Callable[[Any], bytes]:
    """Get a hash function by name, or raise an error if the function is not found.

    Args:
        hash_fn_name: Name of the hash function.

    Returns:
        A hash function.
    """
    if hash_fn_name == "sha256":
        return sha256
    if hash_fn_name == "sha256_cbor":
        return sha256_cbor
    if hash_fn_name == "sha256_msgpack":
        return sha256_msgpack
    if hash_fn_name == "xxhash":
        return xxhash
    if hash_fn_name == "xxhash_cbor":
        return xxhash_cbor
    if hash_fn_name == "xxhash_msgpack":
        return xxhash_msgpack

    raise ValueError(f"Unsupported hash function: {hash_fn_name}")


def safe_hash(data: bytes, usedforsecurity: bool = True) -> HASH:
    """Hash for configs, defaulting to md5 but falling back to sha256
    in FIPS constrained environments.

    Args:
        data: bytes
        usedforsecurity: Whether the hash is used for security purposes

    Returns:
        Hash object
    """
    try:
        return hashlib.md5(data, usedforsecurity=usedforsecurity)
    except (UnsupportedDigestmodError, ValueError):
        return hashlib.sha256(data)
