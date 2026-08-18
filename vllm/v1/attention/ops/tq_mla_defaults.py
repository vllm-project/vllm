# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production defaults for TurboQuant MLA (driven by ``--kv-cache-dtype``).

``turboquant_*_nc`` presets automatically enable the validated production
bundle: FWHT store, k_pe 4-bit, fused sparse prefill (gather+flash+dedup),
and adaptive decode splits.  Environment variables remain **opt-out** regression
anchors only (set ``=0`` to disable a feature).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )

NC_PRODUCTION_CACHE_DTYPES = frozenset(
    {
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    }
)


def is_nc_production_cache_dtype(cache_dtype: str) -> bool:
    return cache_dtype in NC_PRODUCTION_CACHE_DTYPES


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw == "1"


def default_store_fwht(cache_dtype: str, tq_config: TurboQuantConfig) -> bool:
    default = is_nc_production_cache_dtype(cache_dtype) and not tq_config.key_fp8
    return _env_bool("VLLM_TQ_MLA_STORE_FWHT", default)


def default_kpe_4bit(cache_dtype: str, tq_config: TurboQuantConfig) -> bool:
    default = is_nc_production_cache_dtype(cache_dtype) and not tq_config.key_fp8
    return _env_bool("VLLM_TQ_KPE_4BIT", default)


def default_kpe_fp8(
    cache_dtype: str,
    tq_config: TurboQuantConfig,
    *,
    kpe_4bit: bool,
) -> bool:
    if kpe_4bit:
        return False
    default = not is_nc_production_cache_dtype(cache_dtype)
    return _env_bool("VLLM_TQ_KPE_FP8", default)


def resolve_kpe_layout(
    cache_dtype: str,
    tq_config: TurboQuantConfig,
    rope_dim: int,
) -> tuple[bool, bool, int]:
    """Return ``(kpe_4bit, kpe_fp8, k_pe_bytes)`` for cache spec / impl init."""
    from vllm.v1.attention.ops.triton_turboquant_mla_store import kpe_packed_bytes

    kpe_4bit = default_kpe_4bit(cache_dtype, tq_config)
    kpe_fp8 = default_kpe_fp8(cache_dtype, tq_config, kpe_4bit=kpe_4bit)
    k_pe_bytes = kpe_packed_bytes(
        rope_dim,
        kpe_4bit=kpe_4bit,
        kpe_fp8=kpe_fp8,
    )
    return kpe_4bit, kpe_fp8, k_pe_bytes
