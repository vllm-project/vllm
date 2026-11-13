# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Persistent cache identity and full-block disk layout for encoder cache.

Each ``.ec`` file contains every complete CPU block allocated to one encoder
cache item, including any final-block padding. Compatibility is isolated by a
stable namespace directory, while the caller validates the expected allocated
file size before loading. There is no per-file header or namespace
``config.json``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

EC_FILE_FORMAT_VERSION = 1
EC_FILE_SUFFIX = ".ec"

_HEX_DIGEST_LENGTH = hashlib.sha256().digest_size * 2
_LOWERCASE_HEX_DIGITS = frozenset("0123456789abcdef")


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != _HEX_DIGEST_LENGTH or not set(value) <= _LOWERCASE_HEX_DIGITS:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ECCacheNamespace:
    """Stable factors that determine encoder-output byte compatibility.

    The digest excludes process- and request-local identity so compatible model
    configurations can reuse files across restarts. Model identity and effective
    configuration are captured by ``encoder_config_hash``.
    """

    encoder_config_hash: str

    def __post_init__(self) -> None:
        _validate_digest(self.encoder_config_hash, "encoder_config_hash")

    def digest(self) -> str:
        factors = asdict(self)
        factors["format_version"] = EC_FILE_FORMAT_VERSION
        return _sha256_json(factors)


def build_ec_cache_key(
    namespace_digest: str,
    identifier: str,
    modality: str,
    num_tokens: int,
    file_size_bytes: int,
) -> str:
    """Return a filesystem-safe digest for one encoder-cache item."""
    _validate_digest(namespace_digest, "namespace_digest")
    if not identifier:
        raise ValueError("identifier must not be empty")
    if not modality:
        raise ValueError("modality must not be empty")
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if file_size_bytes <= 0:
        raise ValueError("file_size_bytes must be positive")
    return _sha256_json(
        {
            "identifier": identifier,
            "modality": modality,
            "namespace_digest": namespace_digest,
            "num_tokens": num_tokens,
            "file_size_bytes": file_size_bytes,
        }
    )


def get_ec_entry_path(
    root_dir: Path,
    namespace_digest: str,
    cache_key: str,
) -> Path:
    """Map a cache key to a versioned, hash-prefix-sharded raw tensor file."""
    _validate_digest(namespace_digest, "namespace_digest")
    _validate_digest(cache_key, "cache_key")
    return (
        get_ec_namespace_path(root_dir, namespace_digest)
        / cache_key[:3]
        / cache_key[3:5]
        / f"{cache_key}{EC_FILE_SUFFIX}"
    )


def get_ec_namespace_path(root_dir: Path, namespace_digest: str) -> Path:
    """Return the versioned directory containing one cache namespace."""
    _validate_digest(namespace_digest, "namespace_digest")
    return root_dir / f"v{EC_FILE_FORMAT_VERSION}" / namespace_digest


def get_ec_temporary_path(
    entry_path: Path, operation_token: str, owner_id: str
) -> Path:
    """Return a temporary path in the namespace's per-owner staging directory."""
    if not operation_token:
        raise ValueError("operation_token must not be empty")
    token_digest = hashlib.sha256(operation_token.encode("utf-8")).hexdigest()[:16]
    if len(owner_id) != 32 or not set(owner_id) <= _LOWERCASE_HEX_DIGITS:
        raise ValueError("owner_id must be 32 lowercase hexadecimal characters")
    return (
        entry_path.parents[2]
        / ".tmp"
        / owner_id
        / f".{entry_path.name}.{owner_id}.{token_digest}.tmp"
    )
