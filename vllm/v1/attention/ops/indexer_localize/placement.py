# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Locality-domain resources and KV cache placement of the die-local indexer.

The KV cache backing allocation (one buffer for every layer and cache group)
is a VMM range whose physical chunks alternate between the two locality
domains in runs close to 32 MiB (whole KV blocks, and whole VMM granules when
both fit): each granule takes the domain `(midpoint // run_bytes) % 2`, and a
page the domain of the granule holding its midpoint. Block ids, slot mappings
and cache-insert kernels are unchanged.
"""

import math
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.distributed.localization import get_locality_domain_contexts
from vllm.logger import init_logger
from vllm.utils.extensible_tensor import ExtensibleTensor, GranulePlacement

logger = init_logger(__name__)

NUM_DOMAINS = 2
_TARGET_RUN_BYTES = 32 << 20


def unsupported_config_reason(vllm_config: Any) -> str | None:
    """Why this configuration cannot use the die-local indexer, or None.

    Models opt in with a `supports_localized_indexer` class attribute.
    """
    from vllm.model_executor.model_loader import get_model_cls

    model_config = vllm_config.model_config
    parallel = vllm_config.parallel_config
    if not getattr(get_model_cls(model_config), "supports_localized_indexer", False):
        return f"architecture {model_config.architectures}"
    if (
        parallel.decode_context_parallel_size > 1
        or parallel.prefill_context_parallel_size > 1
    ):
        return "decode or prefill context parallelism"
    if vllm_config.kv_transfer_config is not None:
        return "a KV connector registers or exports the KV cache"
    return None


@dataclass(frozen=True)
class DomainResources:
    """Green-context streams of a device's locality domains (the contexts stay
    alive in `get_locality_domain_contexts`'s cache)."""

    device: int
    streams: list[torch.cuda.Stream]
    sms: list[int]
    total_sms: int


_RESOURCES: dict[int, DomainResources | None] = {}


def _unsupported_device_reason(device_index: int) -> str | None:
    if not hasattr(torch.cuda, "execute_on_streams"):
        return "torch.cuda.execute_on_streams is missing"
    try:
        import cuda.bindings.driver  # noqa: F401
        from torch.cuda.green_contexts import (
            get_num_locality_domains,
            is_localization_supported,
        )
    except ImportError as e:
        return f"missing API ({e})"
    if not is_localization_supported(device_index):
        return "locality domains are not supported by the driver"
    num_domains = get_num_locality_domains(device_index)
    if num_domains != NUM_DOMAINS:
        return f"the GPU has {num_domains} locality domains, not {NUM_DOMAINS}"
    return None


def get_domain_resources(device: torch.device) -> DomainResources | None:
    """Per-domain green-context streams of `device`, created once.

    The first call must come before CUDA graph capture (it comes from the KV
    cache allocation). Returns None, after one warning, when the device or
    the installed stack cannot split the decode scan by locality domain.
    """
    idx = device.index
    if idx not in _RESOURCES:
        res = None
        reason = _unsupported_device_reason(idx)
        if reason is None:
            domains = get_locality_domain_contexts(idx)
            res = DomainResources(
                device=idx,
                streams=list(domains.streams),
                sms=[int(n) for n in domains.sm_counts],
                total_sms=torch.cuda.get_device_properties(idx).multi_processor_count,
            )
        else:
            logger.warning(
                "VLLM_INDEXER_LOCALIZE is set but unavailable on device %d (%s); "
                "using the stock sparse indexer.",
                idx,
                reason,
            )
        _RESOURCES[idx] = res
    return _RESOURCES[idx]


# (base, end, run_bytes, granule) of each localized VMM range. The tensors own
# the memory.
_BUFFERS: list[tuple[int, int, int, int]] = []


def _forget(base: int) -> None:
    _BUFFERS[:] = [buf for buf in _BUFFERS if buf[0] != base]


def find_buffer(ptr: int) -> tuple[int, int, int, int] | None:
    """The localized allocation holding device address `ptr`, if any."""
    for buf in _BUFFERS:
        if buf[0] <= ptr < buf[1]:
            return buf
    return None


def _run_bytes(block_bytes: int, granule: int) -> int:
    """A run close to 32 MiB of whole blocks and, when the two fit together
    (Kimi K3.1's blocks do not), whole granules (then every page's granule
    lies in its own run)."""
    unit = math.lcm(block_bytes, granule)
    if unit > 4 * _TARGET_RUN_BYTES:
        unit = block_bytes
    return max(1, round(_TARGET_RUN_BYTES / unit)) * unit


def _place_by_domain(buffer: ExtensibleTensor, run_bytes: int) -> None:
    """Create each granule `buffer` commits from now on in the locality domain
    of its midpoint's run, with cuda-bindings' `cuMemCreate` (it resolves the
    driver entry point that accepts locality-domain locations).
    """
    from cuda.bindings import driver as cu

    props = []
    for domain in range(NUM_DOMAINS):
        prop = cu.CUmemAllocationProp()
        prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = (
            cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
        )
        prop.location.localized.deviceId = int(buffer.device.index)
        prop.location.localized.localityDomainId = domain
        props.append(prop)
    granule = buffer.granularity

    def domain_of(g: int) -> int:
        return ((g * granule + granule // 2) // run_bytes) % NUM_DOMAINS

    def create(nbytes: int, domain: int) -> int:
        err, handle = cu.cuMemCreate(nbytes, props[domain], 0)
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(
                f"cuMemCreate in locality domain {domain} failed: {err!r}"
            )
        return int(handle)

    buffer.set_granule_placement(GranulePlacement(domain_of, create))
    base = buffer.base_ptr
    _BUFFERS.append((base, base + buffer.capacity_bytes, run_bytes, granule))
    weakref.finalize(buffer, _forget, base)
    logger.info(
        "KV cache placed in %.1f MiB runs alternating between %d locality "
        "domains (%.2f GiB reserved).",
        run_bytes / (1 << 20),
        NUM_DOMAINS,
        buffer.capacity_bytes / (1 << 30),
    )


class _Owner:
    """Keeps a plain localized allocation mapped while a tensor views it."""

    def __init__(self, buffer: ExtensibleTensor, size: int):
        self.buffer = buffer
        self.__cuda_array_interface__ = {
            "shape": (size,),
            "typestr": "|i1",
            "data": (buffer.base_ptr, False),
            "version": 2,
        }


def localized_kv_allocate(
    kv_cache_config: Any,
    device: torch.device,
    allocate: Callable[[int], torch.Tensor] | None,
) -> Callable[[int], torch.Tensor] | None:
    """The `allocate` callable for `allocate_kv_cache` with domain placement.

    `allocate` is the extensible KV cache's (its chunks are then committed per
    domain) or None (one localized allocation of the whole cache). Returns
    `allocate` unchanged when placement is unavailable.
    """
    if device.type != "cuda":  # the CPU worker allocates its cache here too
        return allocate
    # The engine allocates under its config; without one (direct callers)
    # only the device decides.
    config = get_current_vllm_config_or_none()
    reason = None
    if config is not None:
        reason = unsupported_config_reason(config)
        if (
            reason is None
            and allocate is None
            and config.model_config.enable_cumem_allocator
        ):
            reason = "the CuMem allocator (sleep mode) manages the KV cache"
    if reason is not None:
        logger.warning(
            "VLLM_INDEXER_LOCALIZE: %s; the KV cache keeps the stock placement "
            "and the sparse indexer the stock path.",
            reason,
        )
        return allocate
    if get_domain_resources(device) is None:
        return allocate
    if allocate is None:
        # Both models force block-outermost layouts: one block stride.
        block_bytes = kv_cache_config.kv_cache_tensors[0].block_stride

        def allocate_localized(size: int) -> torch.Tensor:
            buffer = ExtensibleTensor(size, device=device)
            _place_by_domain(buffer, _run_bytes(block_bytes, buffer.granularity))
            buffer.resize_segments_([size], zero_new=True)
            return torch.as_tensor(_Owner(buffer, size), device=device)

        return allocate_localized
    # The extensible KV cache's allocate, the only one the model runner passes.
    kv_cache = allocate.__self__  # type: ignore[attr-defined]
    buffer = kv_cache.buffer
    _place_by_domain(buffer, _run_bytes(kv_cache.bytes_per_block, buffer.granularity))
    return allocate


def page_domain_table(
    kv: torch.Tensor, device: torch.device | str | None = None
) -> torch.Tensor | None:
    """The planned domain (int8) of every page of an indexer cache view, on
    `device` (default: the view's device).

    `kv` is `[num_pages, rows, bytes]`, page `p` starting at
    `data_ptr + p * stride(0)` bytes. None if not in a localized allocation.
    """
    buf = find_buffer(kv.data_ptr())
    if buf is None:
        return None
    base, _, run_bytes, granule = buf
    page_stride = kv.stride(0) * kv.element_size()
    page_bytes = math.prod(kv.shape[1:]) * kv.element_size()
    p = torch.arange(kv.shape[0], device=device or kv.device, dtype=torch.int64)
    mid = (kv.data_ptr() - base) + p * page_stride + page_bytes // 2
    # The domain the granule holding the page's midpoint was created in.
    gmid = (mid // granule) * granule + granule // 2
    return ((gmid // run_bytes) % NUM_DOMAINS).to(torch.int8)


def page_domain_plan(kv: torch.Tensor) -> tuple[int, int]:
    """`(q0, k)` with `table[b] == ((q0 + b) // k) % 2` for every page `b`
    of `page_domain_table(kv)`, or `(0, 0)`: the build kernel then reads the
    table. With `run` a whole number `k` of page strides and the first page's
    midpoint at `q0 * stride + r0` (`r0 < stride`), page `b`'s midpoint lies
    in run `(q0 + b) // k`; the table holds the domain of the granule around
    that midpoint, which differs where a granule straddles two runs, so the
    plan is verified against the table on the host.
    """
    buf = find_buffer(kv.data_ptr())
    table = page_domain_table(kv, device="cpu")
    if buf is None or table is None:
        return 0, 0
    base, _, run_bytes, _ = buf
    stride = kv.stride(0) * kv.element_size()
    page_bytes = math.prod(kv.shape[1:]) * kv.element_size()
    k, q0 = run_bytes // stride, ((kv.data_ptr() - base) + page_bytes // 2) // stride
    b = torch.arange(table.numel(), dtype=torch.int64)
    if not torch.equal((((q0 + b) // k) % NUM_DOMAINS).to(torch.int8), table):
        return 0, 0
    return q0, k
