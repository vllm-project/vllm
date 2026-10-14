# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared per-device locality-domain contexts, streams, and pools.

Initialize before weight allocation and CUDA graph capture, and retain
contexts for the lifetime of their streams. Enable with
VLLM_USE_LOCALITY_DOMAINS.
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
    """Per-device locality-domain resources. Every list is INDEX-ALIGNED and has
    ``len() == num_resources``:

        ``[0, num_split)``             compute partitions; index ``i`` is exactly
                                       what the kernels take as ``partition_id``
                                       / ``domain_id``.
        ``[num_split, num_resources)`` auxiliary resources -- currently just the
                                       memset context, which is a green context
                                       and a stream like the split ones but owns
                                       no allocator.

    Modelling the auxiliary resource as ordinary list entries rather than
    special-case fields keeps one construction path and leaves room for further
    auxiliary contexts without a schema change.
    """

    num_split: int
    # Retained on purpose: torch maps stream -> context through a
    # WeakValueDictionary and GreenContext.__del__ destroys the streams it
    # handed out, so dropping this list yields "CUDA error: context is
    # destroyed" on first use. Applies to the auxiliary context too.
    contexts: list[Any] = field(repr=False)
    streams: list[torch.cuda.Stream] = field(repr=False)
    pools: list[Any | None] = field(repr=False)  # None where no allocator is needed
    sm_per_resource: list[int] = field(default_factory=list)

    @property
    def num_resources(self) -> int:
        return len(self.streams)

    @property
    def split_streams(self) -> list[torch.cuda.Stream]:
        """The compute partitions only -- what FlashInfer's localized_streams takes.

        It validates ``len(localized_streams) == len(localized_weights)``, so the
        auxiliary stream must NOT be included here.
        """
        return self.streams[: self.num_split]

    @property
    def aux_stream(self) -> torch.cuda.Stream | None:
        """The auxiliary stream, or None if the split consumed the whole device."""
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


# Strong references, process lifetime, keyed by device index. A plain dict
# rather than functools.cache: creation is EAGER (init_device, where the config
# lives) while reads happen later from inside the model. A cache would silently
# construct on a late first read -- i.e. make streams at the worst moment.
_LOCALIZED_RESOURCES: dict[int, LocalizedResources] = {}


def localization_enabled() -> bool:
    return envs.VLLM_USE_LOCALITY_DOMAINS


def _reserved_remainder_stream(device_index: int, num_domains: int):
    """Create an auxiliary stream from resources outside the compute domains."""
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


_CU_DEVICE_ATTRIBUTE_MEMORY_NODE_MULTIPROCESSOR_COUNT = 157


def _node_local_sm_count(device_index: int) -> int | None:
    """SMs guaranteed node-local to one die, or None if unavailable.

    Returns None on any failure so the caller falls back to total // num_domains;
    a wrong-but-plausible number here would silently mis-size every localized
    grid, which is the bug this exists to avoid.
    """
    import ctypes

    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
        val = ctypes.c_int(0)
        rc = libcuda.cuDeviceGetAttribute(
            ctypes.byref(val),
            ctypes.c_int(_CU_DEVICE_ATTRIBUTE_MEMORY_NODE_MULTIPROCESSOR_COUNT),
            ctypes.c_int(device_index),
        )
        total = torch.cuda.get_device_properties(device_index).multi_processor_count
        if rc != 0 or not (0 < val.value <= total):
            logger.warning(
                "[locality domains] node-local SM query unusable "
                "(rc=%d value=%d total=%d); "
                "falling back to total // num_domains",
                rc,
                val.value,
                total,
            )
            return None
        logger.info(
            "[locality domains] node-local SMs=%d of %d total", val.value, total
        )
        return val.value
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
            # Most likely cuda-bindings < 13.4: torch's gate needs BOTH the
            # driver and the cuda.bindings package at >= 13.4.
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

        # ---- split resources: one green context / stream / pool per die ----
        contexts = [
            GreenContext(locality_domain_id=i, device_id=idx)
            for i in range(num_domains)
        ]
        streams = [c.Stream() for c in contexts]
        pools: list[Any | None] = [
            LocalizedMemPool(i, device=idx) for i in range(num_domains)
        ]
        total_sm = torch.cuda.get_device_properties(idx).multi_processor_count
        sm_per_die = _node_local_sm_count(idx) or (total_sm // num_domains)
        sm_per_resource = [sm_per_die] * num_domains

        # ---- auxiliary resource: the memset context ------------------------
        # Appended to the SAME lists, so there is one construction path and one
        # index space. It owns no allocator, hence a None pool.
        aux_stream, aux_ctx, memset_reason, aux_sms = _reserved_remainder_stream(
            idx, num_domains
        )
        if aux_stream is None:
            aux_stream = torch.cuda.Stream(device=idx)
            aux_ctx = None
            memset_reason = f"full-device memset stream ({memset_reason})"
            aux_sms = total_sm
        # Harness scripts assert on this exact line -- it is the only remaining
        # evidence that the memset context was created, so keep it verbatim.
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
