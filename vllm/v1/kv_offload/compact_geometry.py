# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport-neutral compact geometry types and builders.

Owns the pure ``CompactGroupCharge``, layer/block-size helpers, and
``build_compact_group_charges`` that the scheduler and offloading
config layer rely on at import time.

Exists outside ``cpu/`` so that core config and interface modules can
reference compact geometry without pulling in CPU-specific runtime
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

# Re-export KVCacheConfig so that cpu.compact_accounting and other callers
# can import it through the neutral module without a separate import edge.
__all__ = [
    "CompactGroupCharge",
    "CompactTransportSignature",
    "KVCacheConfig",
    "build_compact_group_charges",
]


# ---------------------------------------------------------------------------
# Pure helpers (no CPU runtime dependency)
# ---------------------------------------------------------------------------


def _layer_specs(spec: KVCacheSpec) -> dict[str, KVCacheSpec]:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.kv_cache_specs
    return {}


def _real_page_size_bytes(spec: KVCacheSpec) -> int:
    if isinstance(spec, AttentionSpec):
        return spec.real_page_size_bytes
    if isinstance(spec, MambaSpec):
        # Mamba exposes padding through page_size_bytes; reconstruct the real
        # state payload by removing that optional page-level padding.
        if spec.page_size_padded is None:
            return spec.page_size_bytes
        return sum(
            prod(shape) * dtype.itemsize
            for shape, dtype in zip(spec.shapes, spec.dtypes)
        )
    return spec.page_size_bytes


# ---------------------------------------------------------------------------
# Transport signature types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactGroupCharge:
    """Scheduler-safe compact payload charge derived from canonical specs."""

    group_idx: int
    native_block_tokens: int
    compact_real_bytes_per_rank: int
    compact_real_bytes_server: int


@dataclass(frozen=True)
class CompactTransportSignature:
    """Transport-neutral signature stamped on every KVCacheConfig.

    Carries exactly two fields so that the cross-worker config boundary
    transports only neutral geometry metadata, never rich charge details
    (``native_block_tokens``, ``compact_real_bytes_server``) that are
    internal to the CPU accounting module.
    """

    group_idx: int
    compact_bytes_per_native_block_per_worker: int


# ---------------------------------------------------------------------------
# Builder (no worker state required)
# ---------------------------------------------------------------------------


def build_compact_group_charges(
    kv_cache_config: KVCacheConfig,
    *,
    world_size: int,
    block_size_factor: int,
) -> tuple[CompactGroupCharge, ...]:
    """Derive group payload charges without worker physical packed tensors."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if block_size_factor <= 0:
        raise ValueError("block_size_factor must be positive")

    charges: list[CompactGroupCharge] = []
    for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
        inner_specs = _layer_specs(group.kv_cache_spec)
        real_bytes = sum(
            _real_page_size_bytes(inner_specs.get(layer_name, group.kv_cache_spec))
            for layer_name in group.layer_names
        )
        payload_per_rank = real_bytes * block_size_factor
        if payload_per_rank <= 0:
            raise ValueError(f"KV group {group_idx} has non-positive compact payload")
        charges.append(
            CompactGroupCharge(
                group_idx=group_idx,
                native_block_tokens=group.kv_cache_spec.block_size,
                compact_real_bytes_per_rank=payload_per_rank,
                compact_real_bytes_server=payload_per_rank * world_size,
            )
        )
    if not charges:
        raise ValueError("compact accounting requires at least one KV group")
    return tuple(charges)
