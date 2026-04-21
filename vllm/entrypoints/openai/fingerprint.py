# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the ``system_fingerprint`` string returned by the OpenAI-compatible
server.

Four modes, configured via ``--fingerprint-mode``:

* ``full`` (default): ``vllm-<version>[-<parallelism>]-<hash8>`` — encodes
  server version, any non-trivial parallelism degree (tp/pp/dp/ep), and an
  8-char prefix of ``vllm_config.compute_hash()`` (covers model identity,
  quant config, speculative, attention backend, etc.).
* ``hash``: ``vllm-<version>-<hash8>`` — parallelism stripped.
* ``custom``: user-provided literal via ``--fingerprint-value``.
* ``none``: the field is omitted (serialized as ``null``).

The string is computed once at serving init and cached, so per-request cost
is a single attribute read.
"""

from __future__ import annotations

from typing import Any, Literal

FingerprintMode = Literal["full", "hash", "custom", "none"]

_DEFAULT_MODE: FingerprintMode = "full"
_CUSTOM_VALUE: str | None = None

# Sentinel distinguishes "cached None" (mode=none / custom unset) from "miss".
_MISS: Any = object()
_CACHE: dict[tuple[int, str, str | None], str | None] = {}


def set_default_fingerprint_mode(
    mode: FingerprintMode,
    custom_value: str | None = None,
) -> None:
    """Configure the fingerprint mode for subsequent ``get_system_fingerprint``
    calls. Called once at server startup."""
    global _DEFAULT_MODE, _CUSTOM_VALUE
    _DEFAULT_MODE = mode
    _CUSTOM_VALUE = custom_value
    _CACHE.clear()


def get_system_fingerprint(vllm_config: Any) -> str | None:
    """Return the cached fingerprint for ``vllm_config``, or ``None`` when the
    mode is ``none`` (or ``custom`` with no value)."""
    key = (id(vllm_config), _DEFAULT_MODE, _CUSTOM_VALUE)
    cached = _CACHE.get(key, _MISS)
    if cached is not _MISS:
        return cached
    fp = build_system_fingerprint(vllm_config, _DEFAULT_MODE, _CUSTOM_VALUE)
    _CACHE[key] = fp
    return fp


def build_system_fingerprint(
    vllm_config: Any,
    mode: FingerprintMode = "full",
    custom_value: str | None = None,
) -> str | None:
    if mode == "none":
        return None
    if mode == "custom":
        return custom_value

    from vllm import __version__ as vllm_version

    try:
        hash8 = vllm_config.compute_hash()[:8]
    except Exception:
        hash8 = "nohash"

    if mode == "hash":
        return f"vllm-{vllm_version}-{hash8}"

    # mode == "full"
    parts: list[str] = [f"vllm-{vllm_version}"]
    pc = getattr(vllm_config, "parallel_config", None)
    if pc is not None:
        if getattr(pc, "tensor_parallel_size", 1) > 1:
            parts.append(f"tp{pc.tensor_parallel_size}")
        if getattr(pc, "pipeline_parallel_size", 1) > 1:
            parts.append(f"pp{pc.pipeline_parallel_size}")
        if getattr(pc, "data_parallel_size", 1) > 1:
            parts.append(f"dp{pc.data_parallel_size}")
        if getattr(pc, "enable_expert_parallel", False):
            parts.append("ep")
    parts.append(hash8)
    return "-".join(parts)
