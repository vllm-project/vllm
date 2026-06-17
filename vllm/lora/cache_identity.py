# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content-derived identity for a runtime LoRA adapter, used to version
prefix-cache block hashes across same-name reloads (issue #42125).

For a readable LOCAL adapter, the identity digests the on-disk contents (the
selected weight file + canonicalized ``adapter_config.json``) plus
loader-affecting request fields, so reloading different contents under the same
``lora_name`` yields a different key -- deterministically across API-server
replicas, unlike a per-process counter. The key is filled before block hashes
are computed (serving load + ``InputProcessor.process_inputs`` backstop).

Limitations: adapters whose files are not readable where the key is computed
(Hugging Face Hub ids, relative/remote paths, shared-nothing multi-node) get a
path-only identity and are NOT content-versioned. There is no guard against the
adapter bytes changing between when the key is computed and when the worker
loads them (the TOCTOU race is out of scope), and external KV connectors do not
consume these keys.
"""

import contextlib
import hashlib
import json
import os

from vllm.logger import init_logger

logger = init_logger(__name__)

# Bump when the manifest layout changes so old/new keys never alias.
_VERSION = b"vllm-lora-effective-id-v1"

# Weight file names in the loader's selection order
# (see LoRAModel.from_local_checkpoint).
_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt")
_TENSORIZER_WEIGHT = "adapter_model.tensors"
_CONFIG_NAME = "adapter_config.json"

_READ_CHUNK = 1 << 20


def _frame(hasher: "hashlib._Hash", label: bytes, length: int) -> None:
    # Length-prefixed framing so fields concatenate unambiguously
    # (avoids ("ab","c") vs ("a","bc") collisions -- no raw concat).
    hasher.update(len(label).to_bytes(8, "big"))
    hasher.update(label)
    hasher.update(length.to_bytes(8, "big"))


def _add_bytes(hasher: "hashlib._Hash", label: bytes, payload: bytes) -> None:
    _frame(hasher, label, len(payload))
    hasher.update(payload)


def _add_file(hasher: "hashlib._Hash", label: bytes, path: str) -> None:
    size = os.path.getsize(path)
    _frame(hasher, label, size)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_READ_CHUNK), b""):
            hasher.update(chunk)


def _select_sources(
    lora_path: str, tensorizer_config_dict: dict | None
) -> tuple[str, str | None]:
    """Approximate LoRAModel.from_local_checkpoint's source selection for LOCAL
    files: a tensorizer dir if configured, else safetensors/.bin/.pt by the
    loader's precedence. Remote tensorizer URIs and HF-Hub ids are not resolved
    here -- they have no local file and degrade to a path-only identity.

    Returns (config_path, weight_path); weight_path is None if no known local
    weight file is present.
    """
    if tensorizer_config_dict:
        tdir = tensorizer_config_dict.get("tensorizer_dir") or lora_path
        return (
            os.path.join(tdir, _CONFIG_NAME),
            os.path.join(tdir, _TENSORIZER_WEIGHT),
        )
    config_path = os.path.join(lora_path, _CONFIG_NAME)
    for name in _WEIGHT_NAMES:
        candidate = os.path.join(lora_path, name)
        if os.path.isfile(candidate):
            return config_path, candidate
    return config_path, None


def compute_lora_cache_key(
    lora_path: str,
    *,
    is_3d_lora_weight: bool = False,
    tensorizer_config_dict: dict | None = None,
) -> str:
    """Deterministic content identity for a local adapter.

    Hashes the selected weight file + canonicalized ``adapter_config.json`` +
    loader-affecting fields. Degrades to a path-only identity when no local
    source is readable (HF-Hub ids, relative/remote paths). This is not a
    guarantee about the bytes the worker ultimately loads -- there is no TOCTOU
    re-verification.
    """
    lora_path = os.path.expanduser(lora_path)
    hasher = hashlib.sha256()
    _add_bytes(hasher, b"version", _VERSION)
    _add_bytes(hasher, b"is_3d_lora_weight", b"\x01" if is_3d_lora_weight else b"\x00")
    if tensorizer_config_dict is not None:
        canonical = json.dumps(
            tensorizer_config_dict, sort_keys=True, separators=(",", ":")
        )
        _add_bytes(hasher, b"tensorizer_config", canonical.encode("utf-8"))

    config_path, weight_path = _select_sources(lora_path, tensorizer_config_dict)
    hashed_any = False
    try:
        # Only hash absolute local paths: a relative path can resolve to a
        # different location on the worker (different CWD), so it is treated as
        # path-only here to avoid "API hashes A, worker loads B".
        if os.path.isabs(config_path) and os.path.isfile(config_path):
            # Canonicalize JSON so reformatting alone does not change the key;
            # a non-JSON config is hashed as raw bytes.
            with open(config_path, "rb") as f:
                cfg = f.read()
            with contextlib.suppress(ValueError, TypeError):
                cfg = json.dumps(
                    json.loads(cfg), sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            _add_bytes(hasher, b"adapter_config.json", cfg)
            hashed_any = True
        if (
            weight_path is not None
            and os.path.isabs(weight_path)
            and os.path.isfile(weight_path)
        ):
            _add_file(
                hasher,
                b"weights:" + os.path.basename(weight_path).encode(),
                weight_path,
            )
            hashed_any = True
    except OSError:
        hashed_any = False
    if not hashed_any:
        # Nothing readable here (e.g. a Hugging Face Hub id or a path resolved
        # elsewhere) -> path-only identity. A same-path content overwrite cannot
        # be distinguished in this degraded case; surface it so operators know
        # the adapter is not content-versioned.
        logger.debug(
            "LoRA prefix-cache identity for %r is path-only (no readable adapter "
            "files at this location); same-path content swaps are not detected.",
            lora_path,
        )
        _add_bytes(hasher, b"path-fallback", lora_path.encode("utf-8"))
    return hasher.hexdigest()


def is_content_versioned(
    lora_path: str, *, tensorizer_config_dict: dict | None = None
) -> bool:
    """Whether ``compute_lora_cache_key`` hashes on-disk content for this path.

    True for an absolute local path with a readable ``adapter_config.json`` or
    weight file; False (path-only identity) for HF-Hub ids and relative/remote
    paths. Callers use this to warn when an in-place reload would be keyed by
    path only and thus cannot detect a same-path content swap.
    """
    lora_path = os.path.expanduser(lora_path)
    config_path, weight_path = _select_sources(lora_path, tensorizer_config_dict)
    return any(
        p is not None and os.path.isabs(p) and os.path.isfile(p)
        for p in (config_path, weight_path)
    )


def ensure_lora_cache_key(lora_request) -> None:
    """Fill ``lora_request.lora_cache_key`` in place if unset.

    Single source of truth for the prefix-cache identity so every fill site
    (serving load, ``InputProcessor.process_inputs`` backstop) hashes the same
    inputs. No-op when the request is None or already carries a key.
    """
    if lora_request is None or lora_request.lora_cache_key is not None:
        return
    lora_request.lora_cache_key = compute_lora_cache_key(
        lora_request.lora_path,
        is_3d_lora_weight=lora_request.is_3d_lora_weight,
        tensorizer_config_dict=lora_request.tensorizer_config_dict,
    )
