# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-device locality-domain contexts, streams, and memory pools.

Initialize before weight allocation and CUDA graph capture; all layers share
these resources. Enable with VLLM_USE_LOCALITY_DOMAINS. Optional
VLLM_MOE_LOCALIZE_BACKFILL adds SMs outside each domain's local allocation.
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

    num_split: int
    # Keep contexts alive: streams hold weak references and are destroyed with them.
    contexts: list[Any] = field(repr=False)
    streams: list[torch.cuda.Stream] = field(repr=False)
    pools: list[Any | None] = field(repr=False)  # None where no allocator is needed
    sm_per_resource: list[int] = field(default_factory=list)

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

    def __str__(self) -> str:
        return (
            f"LocalizedResources(num_split={self.num_split}, "
            f"num_resources={self.num_resources}, "
            f"sm_per_resource={self.sm_per_resource}, "
            f"pools={sum(p is not None for p in self.pools)})"
        )


# Retain resources for the process lifetime; initialize before model execution.
_LOCALIZED_RESOURCES: dict[int, LocalizedResources] = {}


def localization_enabled() -> bool:
    return envs.VLLM_USE_LOCALITY_DOMAINS


def _backfilled_contexts(device_index: int, num_domains: int):
    """Create equal-sized locality-domain groups that consume all usable SMs."""
    from torch.cuda._utils import _check_cuda_bindings as _chk
    from torch.cuda._utils import _cuda_bindings_driver as _drv
    from torch.cuda.green_contexts import GreenContext

    device = _chk(_drv.cuDeviceGet(device_index))
    sm_resource = _chk(
        _drv.cuDeviceGetDevResource(
            device, _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )
    target_sm_count = int(sm_resource.sm.smCount) // num_domains
    if target_sm_count <= 0:
        raise RuntimeError(
            f"cannot split {sm_resource.sm.smCount} SMs across {num_domains} domains"
        )

    flags = (
        _drv.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_BACKFILL
        | _drv.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
    )
    params = []
    for domain_id in range(num_domains):
        p = _drv.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
        p.smCount = target_sm_count
        # Use the default co-scheduled group size.
        p.coscheduledSmCount = 0
        p.flags = flags
        p.localityDomainId = domain_id
        params.append(p)

    split, remaining = _chk(
        _drv.cuDevSmResourceSplit(num_domains, sm_resource, 0, params)
    )
    sm_counts = [int(resource.sm.smCount) for resource in split]
    expected_sm_counts = [target_sm_count] * num_domains
    if sm_counts != expected_sm_counts:
        raise RuntimeError(
            f"requested backfilled partitions {expected_sm_counts}, got {sm_counts}"
        )

    def _make_context(resource):
        desc = _chk(_drv.cuDevResourceGenerateDesc([resource], 1))
        green_ctx = _chk(
            _drv.cuGreenCtxCreate(
                desc,
                device,
                _drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
            )
        )
        return GreenContext(green_context_obj=green_ctx)

    contexts = [_make_context(resource) for resource in split]
    rem_sms = int(remaining.sm.smCount)
    remainder_ctx = _make_context(remaining) if rem_sms > 0 else None
    return contexts, sm_counts, remainder_ctx, rem_sms


def _reserved_remainder_stream(device_index: int, num_domains: int):
    """Create a memset stream on SMs outside the compute domains.

    Repeat the locality split because PyTorch discards its remainder resource.

    Returns:
        (stream, context, reason, sm_count). On failure or no remaining SMs,
        returns (None, None, reason, 0) for the caller's full-device fallback.
    """
    try:
        from torch.cuda._utils import _check_cuda_bindings as _chk
        from torch.cuda._utils import _cuda_bindings_driver as _drv
        from torch.cuda.green_contexts import GreenContext

        device = _chk(_drv.cuDeviceGet(device_index))
        sm_resource = _chk(
            _drv.cuDeviceGetDevResource(
                device, _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
            )
        )
        flag = (
            _drv.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
        )
        params = []
        for domain_id in range(num_domains):
            p = _drv.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
            p.smCount = 0  # discovery mode: each domain gets its node-local SMs
            p.coscheduledSmCount = 0
            p.flags = flag
            p.localityDomainId = domain_id
            params.append(p)
        _split, remaining = _chk(
            _drv.cuDevSmResourceSplit(num_domains, sm_resource, 0, params)
        )
        rem_sms = int(remaining.sm.smCount)
        if rem_sms <= 0:
            return None, None, "no remainder SMs (dies consume the full device)", 0
        desc = _chk(_drv.cuDevResourceGenerateDesc([remaining], 1))
        green_ctx = _chk(
            _drv.cuGreenCtxCreate(
                desc, device, _drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
            )
        )
        ctx = GreenContext(green_context_obj=green_ctx)
        return ctx.Stream(), ctx, f"reserved {rem_sms}-SM remainder green ctx", rem_sms
    except Exception as e:  # noqa: BLE001 - any failure -> full-device fallback
        return (
            None,
            None,
            f"remainder unavailable ({type(e).__name__}: {str(e)[:60]})",
            0,
        )


def _node_local_sm_count(device_index: int) -> int | None:
    """Return the node-local SM count, or None if the query fails."""
    try:
        from cuda.bindings import driver as cuda

        rc, sm_count = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_MULTIPROCESSOR_COUNT,
            device_index,
        )
        total = torch.cuda.get_device_properties(device_index).multi_processor_count
        if rc != cuda.CUresult.CUDA_SUCCESS or not (0 < sm_count <= total):
            logger.warning(
                "[locality domains] node-local SM query unusable "
                "(rc=%s value=%s total=%d); "
                "falling back to total // num_domains",
                rc,
                sm_count,
                total,
            )
            return None
        logger.info("[locality domains] node-local SMs=%d of %d total", sm_count, total)
        return sm_count
    except Exception as e:  # noqa: BLE001 - any failure -> documented fallback
        logger.warning(
            "[locality domains] node-local SM query failed (%s: %s); falling back to "
            "total // num_domains",
            type(e).__name__,
            e,
        )
        return None


def maybe_init_localized_resources(
    device: torch.device,
) -> LocalizedResources | None:
    """Create the per-device green contexts / streams / pools. Idempotent.

    Returns None (and logs why) when localization is off or unavailable, so the
    caller can fall back to the ordinary non-localized path.
    """
    # Surface allocation fallbacks, which can affect weight locality and performance.
    import warnings as _warnings

    _warnings.filterwarnings("always", message=".*Localized CUDA allocation failed.*")

    if not localization_enabled():
        return None

    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx in _LOCALIZED_RESOURCES:
        return _LOCALIZED_RESOURCES[idx]

    try:
        from torch.cuda import LocalizedMemPool
        from torch.cuda.green_contexts import (
            GreenContext,
            get_num_locality_domains,
            is_localization_supported,
        )
    except ImportError as e:
        logger.warning(
            "[locality domains] localization requested but torch lacks the API: %s", e
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
                "[locality domains] device %d reports %d locality domains",
                idx,
                num_domains,
            )
            return None

        backfill = False
        rem_ctx = None
        rem_sms = 0
        if envs.VLLM_MOE_LOCALIZE_BACKFILL:
            try:
                (
                    contexts,
                    compute_sm_counts,
                    rem_ctx,
                    rem_sms,
                ) = _backfilled_contexts(idx, num_domains)
                backfill = True
            except Exception as e:  # noqa: BLE001 - optional optimization
                logger.warning(
                    "[locality domains] backfill unavailable (%s: %s); "
                    "using node-local compute partitions",
                    type(e).__name__,
                    e,
                )
        if not backfill:
            contexts = [
                GreenContext(locality_domain_id=i, device_id=idx)
                for i in range(num_domains)
            ]
        streams = [c.Stream() for c in contexts]
        pools: list[Any | None] = [
            LocalizedMemPool(i, device=idx) for i in range(num_domains)
        ]
        total_sm = torch.cuda.get_device_properties(idx).multi_processor_count
        # Size persistent grids to each partition; total SMs include the remainder.
        if backfill:
            sm_per_resource = compute_sm_counts
            logger.info(
                "[locality domains] backfill compute partitions: %s SMs",
                compute_sm_counts,
            )
        else:
            sm_per_die = _node_local_sm_count(idx) or (total_sm // num_domains)
            sm_per_resource = [sm_per_die] * num_domains

        # Use remainder SMs for memset, falling back to a full-device stream.
        if backfill:
            if rem_ctx is not None:
                aux_ctx, aux_stream, aux_sms = rem_ctx, rem_ctx.Stream(), rem_sms
                memset_reason = f"reserved {aux_sms}-SM backfill remainder green ctx"
            else:
                aux_stream, aux_ctx, aux_sms = None, None, 0
                memset_reason = "no remainder after backfill"
        else:
            aux_stream, aux_ctx, memset_reason, aux_sms = _reserved_remainder_stream(
                idx, num_domains
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
        device_index = torch.cuda.current_device()
    return _LOCALIZED_RESOURCES.get(device_index)
