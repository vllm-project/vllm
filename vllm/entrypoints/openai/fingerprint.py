# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the ``system_fingerprint`` string returned by the OpenAI-compatible
server.

The fingerprint is computed once at serving initialization and stamped on
non-streaming responses. Its goals:

* Make it obvious that vLLM is serving the request (``vllm-`` prefix).
* Surface the bits of config that most often change response behavior: model
  identity (via config hash), dtype + kv-cache dtype, quantization, parallelism
  degrees, and the hardware family.
* Stay short so it fits the spirit of OpenAI's ``fp_<hex>`` field.
"""

from __future__ import annotations

import re
from typing import Any, Literal

FingerprintMode = Literal["full", "hash"]

# Set by the API server at startup via ``set_default_fingerprint_mode``.
# Kept as a module-level default so that serving classes can construct their
# fingerprint without each caller having to thread a kwarg through subclasses.
_DEFAULT_MODE: FingerprintMode = "full"

# Keyed by ``(id(vllm_config), mode)`` so that multiple serving classes sharing
# the same engine reuse a single computed string.
_CACHE: dict[tuple[int, str], str] = {}


def set_default_fingerprint_mode(mode: FingerprintMode) -> None:
    global _DEFAULT_MODE
    _DEFAULT_MODE = mode
    _CACHE.clear()


def get_system_fingerprint(
    vllm_config: Any,
    mode: FingerprintMode | None = None,
) -> str:
    """Return the cached fingerprint string for ``vllm_config``."""
    resolved_mode: FingerprintMode = mode if mode is not None else _DEFAULT_MODE
    key = (id(vllm_config), resolved_mode)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    s = build_system_fingerprint(vllm_config, resolved_mode)
    _CACHE[key] = s
    return s


_TORCH_DTYPE_ALIASES = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "float64": "fp64",
    "float": "fp32",
    "half": "fp16",
}


def _short_dtype(dtype: Any) -> str:
    s = str(dtype)
    if s.startswith("torch."):
        s = s[len("torch.") :]
    return _TORCH_DTYPE_ALIASES.get(s, s)


_DEVICE_NAME_RE = re.compile(r"[A-Za-z0-9]+")
_DEVICE_VENDOR_PREFIXES = (
    "NVIDIA ",
    "AMD Instinct ",
    "AMD ",
    "Intel ",
)


def _short_device_name(name: str | None) -> str:
    """Reduce a verbose device name to a short token (e.g. ``A100``, ``H100``,
    ``MI300X``). Returns ``"unknown"`` if we cannot parse a token."""
    if not name:
        return "unknown"
    n = name.strip()
    for prefix in _DEVICE_VENDOR_PREFIXES:
        if n.startswith(prefix):
            n = n[len(prefix) :]
            break
    # Strip the common "-SXM4-80GB" / " 80GB HBM3" style suffixes by keeping
    # only the first alphanumeric token.
    m = _DEVICE_NAME_RE.match(n)
    return m.group(0) if m else "unknown"


def build_system_fingerprint(
    vllm_config: Any,
    mode: FingerprintMode = "full",
) -> str:
    """Compute the ``system_fingerprint`` string.

    Call once at serving startup and cache the result; every response that
    needs a fingerprint just reads the cached string.
    """
    # compute_hash() is already used for the torch.compile cache key, so it
    # already covers model identity, revision, quant_config, speculative,
    # attention backend, cache, LoRA, etc. Use the first 8 hex chars.
    try:
        cfg_hash = vllm_config.compute_hash()[:8]
    except Exception:
        cfg_hash = "nohash"

    if mode == "hash":
        return f"vllm-{cfg_hash}"

    from vllm import __version__ as vllm_version
    from vllm.platforms import current_platform

    try:
        hw = _short_device_name(current_platform.get_device_name(0))
    except Exception:
        hw = getattr(current_platform, "device_type", None) or "unknown"

    parts: list[str] = [f"vllm-{vllm_version}", hw]

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

    mc = getattr(vllm_config, "model_config", None)
    if mc is not None:
        parts.append(_short_dtype(mc.dtype))
        if getattr(mc, "quantization", None):
            parts.append(str(mc.quantization))

    cc = getattr(vllm_config, "cache_config", None)
    if cc is not None:
        cache_dtype = getattr(cc, "cache_dtype", "auto")
        if cache_dtype and cache_dtype != "auto":
            parts.append(f"kv{cache_dtype}")

    parts.append(cfg_hash)
    return "-".join(parts)
