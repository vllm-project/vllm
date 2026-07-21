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
from vllm.v1.kv_offload.config import (
    CompactGroupSliceConfig,
    CompactSliceConfig,
)

# Re-export KVCacheConfig so that cpu.compact_accounting and other callers
# can import it through the neutral module without a separate import edge.
__all__ = [
    "CompactGroupCharge",
    "CompactTransportSignature",
    "KVCacheConfig",
    "build_compact_group_charges",
    "build_compact_slice_accounting",
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
        real_bytes = 0
        for layer_name in group.layer_names:
            if inner_specs:
                spec = inner_specs.get(layer_name)
                if spec is None:
                    raise ValueError(
                        f"KV group {group_idx} layer {layer_name!r} not found "
                        f"in UniformTypeKVCacheSpecs"
                    )
            else:
                spec = group.kv_cache_spec
            real_bytes += _real_page_size_bytes(spec)
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


def build_compact_slice_accounting(
    kv_cache_config: KVCacheConfig,
    *,
    world_size: int,
) -> tuple[CompactGroupSliceConfig, ...]:
    """Build worker-local packed slice geometry through canonical accounting.

    This runs only on rich per-worker configs, before the scheduler collapses
    ``UniformTypeKVCacheSpecs`` to representative specs. Aggregate group
    charges remain a separate scheduler-safe transport contract.
    """
    if kv_cache_config.compact_aggregate_signature is None:
        raise ValueError("compact aggregate signature is required")
    if not kv_cache_config.kv_cache_tensors or not any(
        tensor.block_stride for tensor in kv_cache_config.kv_cache_tensors
    ):
        raise ValueError("compact slice accounting requires packed tensor metadata")
    if any(
        not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_config.kv_cache_groups
    ):
        raise ValueError(
            "compact slice accounting requires rich UniformTypeKVCacheSpecs"
        )

    # Lazy import preserves the neutral import boundary while reusing the one
    # canonical physical accounting implementation.
    from vllm.v1.kv_offload.cpu.compact_accounting import (
        build_compact_layout_accounting,
    )

    accounting = build_compact_layout_accounting(
        kv_cache_config,
        world_size=world_size,
        block_size_factor=1,
        cpu_budget_bytes=1,
    )
    group_slices = tuple(
        CompactGroupSliceConfig(
            group_idx=group.group_idx,
            slices=tuple(
                CompactSliceConfig(
                    offset_bytes=slice_.offset_bytes,
                    real_bytes_per_gpu_block=slice_.real_bytes_per_gpu_block,
                    padded_bytes_per_gpu_block=slice_.padded_bytes_per_gpu_block,
                    layer_name=slice_.layer_name,
                )
                for slice_ in group.slices
            ),
            compact_real_bytes_per_rank=group.compact_real_bytes_per_rank,
            compact_padded_bytes_per_rank=group.compact_padded_bytes_per_rank,
        )
        for group in accounting.groups
    )

    signature = kv_cache_config.compact_aggregate_signature
    assert signature is not None
    if len(group_slices) != len(signature):
        raise ValueError(
            f"compact slice group count {len(group_slices)} does not match "
            f"aggregate signature count {len(signature)}"
        )
    for group_slice, charge in zip(group_slices, signature):
        if group_slice.group_idx != charge.group_idx:
            raise ValueError(
                f"compact slice group {group_slice.group_idx} does not match "
                f"aggregate signature group {charge.group_idx}"
            )
        if (
            group_slice.compact_real_bytes_per_rank
            != charge.compact_bytes_per_native_block_per_worker
        ):
            raise ValueError(
                f"Group {group_slice.group_idx} slice accounting "
                f"{group_slice.compact_real_bytes_per_rank} != transported charge "
                f"{charge.compact_bytes_per_native_block_per_worker}"
            )
    return group_slices
