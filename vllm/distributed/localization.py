# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-device locality-domain contexts, streams, and memory pools.

Initialize before weight allocation and CUDA graph capture; all layers share
these resources. Enable with VLLM_USE_LOCALITY_DOMAINS. Optional
VLLM_MOE_LOCALIZE_BACKFILL adds SMs outside each domain's local allocation.
Requires upstream PyTorch LocalizedAllocator, SMPartition, and
``torch.cuda.execute_on_streams`` APIs (nightly from 2026-10-08, 2.16), FlashInfer
with locality-domain MoE support (main after flashinfer-ai/flashinfer#6488; no
0.7.x release has it), CUDA driver and cuda.bindings 13.4+, and a GPU with
multiple locality domains. When any of these is missing, localization logs why
and is disabled; the ordinary full-width MoE path runs instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class LocalizedResources:
    """Shared resources with aligned context, stream, pool, and SM-count lists.

    The first ``num_split`` entries are compute domains. Remaining entries are
    auxiliary resources, which have no memory pool.
    """

    num_split: int = field(repr=False)
    # Keep contexts alive: streams hold weak references and are destroyed with them.
    contexts: list[Any] = field(repr=False)
    streams: list[torch.cuda.Stream] = field(repr=False)
    pools: list[Any | None] = field(repr=False)  # None where no allocator is needed
    sm_per_resource: list[int] = field(default_factory=list, repr=False)

    @property
    def num_resources(self) -> int:
        return len(self.streams)

    @property
    def split_streams(self) -> list[torch.cuda.Stream]:
        """Compute streams in shard order, excluding the auxiliary memset stream."""
        return self.streams[: self.num_split]

    @property
    def aux_stream(self) -> torch.cuda.Stream | None:
        """The auxiliary stream, if present."""
        return (
            self.streams[self.num_split]
            if self.num_resources > self.num_split
            else None
        )

    @property
    def split_sm_counts(self) -> list[int]:
        return self.sm_per_resource[: self.num_split]


# Retain resources for the process lifetime; initialize before model execution.
_LOCALIZED_RESOURCES: dict[int, LocalizedResources] = {}


def localization_enabled() -> bool:
    return envs.VLLM_USE_LOCALITY_DOMAINS


def _locality_contexts(device_index: int, num_domains: int, *, backfill: bool):
    """Split compute domains and retain the disjoint remainder for memset."""
    from torch.cuda.green_contexts import GreenContext, SMPartition

    resource = SMPartition.from_device(device_index)
    target_sm_count = resource.sm_count // num_domains if backfill else 0
    if backfill and target_sm_count <= 0:
        raise RuntimeError("insufficient resources for locality-domain backfill")
    partitions, remainder = resource.split(
        num_sms=target_sm_count,
        locality_domain_ids=list(range(num_domains)),
        backfill=backfill,
    )
    contexts = [GreenContext(sm_partition=partition) for partition in partitions]
    if backfill:
        sm_counts = [context.sm_count for context in contexts]
        expected_sm_counts = [target_sm_count] * num_domains
        if sm_counts != expected_sm_counts:
            raise RuntimeError(
                "backfilled partitions do not match the requested allocation"
            )
    return contexts, remainder


def _reserved_remainder_stream(remainder):
    """Create a memset stream from the compute split's remaining SMs."""
    if remainder is None:
        return None, None, "no remainder SMs (dies consume the full device)", 0
    try:
        from torch.cuda.green_contexts import GreenContext

        ctx = GreenContext(sm_partition=remainder)
        sm_count = ctx.sm_count
        return (
            ctx.Stream(),
            ctx,
            "reserved remainder green context",
            sm_count,
        )
    except Exception as e:  # noqa: BLE001 - any failure -> full-device fallback
        return (
            None,
            None,
            f"remainder unavailable ({type(e).__name__}: {str(e)[:60]})",
            0,
        )


def localized_moe_unsupported_reason() -> str | None:
    """Explain why the installed torch/FlashInfer cannot run the localized MoE.

    Returns None when both provide the APIs FlashInfer's locality-domain path
    calls. Checked before any hardware is touched so an unsupported stack never
    shards weights it cannot use.
    """
    from vllm.utils.flashinfer import has_flashinfer_localized_moe

    if not hasattr(torch.cuda, "execute_on_streams"):
        return (
            "torch.cuda.execute_on_streams is missing; FlashInfer's localized "
            "MoE needs a PyTorch nightly from 2026-10-08 or later (2.16)"
        )
    if not has_flashinfer_localized_moe():
        return (
            "flashinfer.cute_dsl_fused_moe_nvfp4 does not accept the localized_* "
            "arguments; needs FlashInfer main after flashinfer-ai/flashinfer#6488 "
            "(not in any 0.7.x release)"
        )
    return None


def maybe_init_localized_resources(
    device: torch.device,
) -> LocalizedResources | None:
    """Create the per-device green contexts / streams / pools. Idempotent.

    Returns None (and logs why) when localization is off or unavailable, so the
    caller can fall back to the ordinary non-localized path.
    """
    if not localization_enabled():
        return None

    idx = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    if idx in _LOCALIZED_RESOURCES:
        return _LOCALIZED_RESOURCES[idx]

    try:
        from torch.cuda import LocalizedAllocator
        from torch.cuda.green_contexts import (
            get_num_locality_domains,
            is_localization_supported,
        )
    except ImportError as e:
        logger.warning(
            "[locality domains] localization requested but torch lacks the API: %s", e
        )
        return None

    reason = localized_moe_unsupported_reason()
    if reason is not None:
        logger.warning(
            "[locality domains] localization requested but unsupported by the "
            "installed stack: %s; using the full-width MoE path",
            reason,
        )
        return None

    try:
        if not is_localization_supported(idx):
            logger.warning(
                "[locality domains] is_localization_supported(%d) is False; "
                "check cuda.bindings >= 13.4 and driver >= 13.4",
                idx,
            )
            return None

        num_domains = get_num_locality_domains(idx)
        if num_domains <= 1:
            logger.warning(
                "[locality domains] device %d does not support domain splitting",
                idx,
            )
            return None

        backfill = False
        if envs.VLLM_MOE_LOCALIZE_BACKFILL:
            try:
                contexts, remainder = _locality_contexts(
                    idx, num_domains, backfill=True
                )
                backfill = True
            except Exception as e:  # noqa: BLE001 - optional optimization
                logger.warning(
                    "[locality domains] backfill unavailable (%s: %s); "
                    "using node-local compute partitions",
                    type(e).__name__,
                    e,
                )
        if not backfill:
            contexts, remainder = _locality_contexts(idx, num_domains, backfill=False)
        streams = [c.Stream() for c in contexts]
        # MemPool binds to the current device, which may differ from idx.
        with torch.accelerator.device_index(idx):
            pools: list[Any | None] = [
                torch.cuda.MemPool(
                    allocator=LocalizedAllocator(i, device=idx).allocator()
                )
                for i in range(num_domains)
            ]
        total_sm = torch.cuda.get_device_properties(idx).multi_processor_count
        sm_per_resource = [context.sm_count for context in contexts]
        logger.info(
            "[locality domains] using %s compute partitions",
            "backfilled" if backfill else "node-local",
        )

        # Use remainder SMs for memset, falling back to a full-device stream.
        aux_stream, aux_ctx, memset_reason, aux_sms = _reserved_remainder_stream(
            remainder
        )
        if aux_stream is None:
            aux_stream = torch.cuda.Stream(device=idx)
            aux_ctx = None
            memset_reason = f"full-device memset stream ({memset_reason})"
            aux_sms = total_sm
        logger.info("[locality domains] output-memset stream: %s", memset_reason)
        contexts.append(aux_ctx)
        streams.append(aux_stream)
        pools.append(None)
        sm_per_resource.append(aux_sms)

        res = LocalizedResources(
            num_split=num_domains,
            contexts=contexts,
            streams=streams,
            pools=pools,
            sm_per_resource=sm_per_resource,
        )
    except Exception as e:  # noqa: BLE001 - never break startup over this
        logger.warning(
            "[locality domains] failed to initialize localization: %s: %s",
            type(e).__name__,
            e,
        )
        return None

    _LOCALIZED_RESOURCES[idx] = res
    return res


def get_localized_resources(
    device_index: int | None = None,
) -> LocalizedResources | None:
    """Look up resources created by init_device. None => not localized."""
    if device_index is None:
        if not torch.cuda.is_available():
            return None
        device_index = torch.accelerator.current_device_index()
    return _LOCALIZED_RESOURCES.get(device_index)
