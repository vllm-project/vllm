# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import pickle
from _hashlib import HASH, UnsupportedDigestmodError
from collections.abc import Callable
from typing import Any

import cbor2


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
