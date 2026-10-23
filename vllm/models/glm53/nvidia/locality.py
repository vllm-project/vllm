# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA locality domains: the SM map of each domain and per-domain word pools."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.models.glm53.nvidia.ops.cute_dsl import _rt as rt

logger = logging.getLogger(__name__)

WORD_BYTES = torch.int32.itemsize
#: SM the driver does not attribute to a domain.
UNATTRIBUTED = -1


def _value(operation: str, response: tuple[object, ...]) -> object:
    """Check a ``cuda.bindings.driver`` response and return its value, if any."""
    from cuda.bindings import driver

    if response[0] != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"locality domains: {operation} failed: {response[0]!r}")
    if len(response) == 1:
        return None
    return response[1] if len(response) == 2 else response[1:]


def _location(device: int, domain: int | None) -> object:
    """A device memory location, localized to ``domain`` unless it is None."""
    from cuda.bindings import driver

    location = driver.CUmemLocation()
    if domain is None:
        location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        location.id = device
    else:
        location.type = (
            driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
        )
        location.localized.deviceId = device
        location.localized.localityDomainId = domain
    return location


class _Chunk:
    """One localized ``cuMemCreate`` allocation with a bump pointer."""

    def __init__(self, device: int, domain: int | None, minimum: int) -> None:
        """Create, map and zero a chunk of at least ``minimum`` bytes in ``domain``
        (None: anywhere on the device).
        """
        from cuda.bindings import driver

        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location = _location(device, domain)
        minimum_granularity = (
            driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
        granularity = int(
            _value(
                "cuMemGetAllocationGranularity",
                driver.cuMemGetAllocationGranularity(prop, minimum_granularity),
            )
        )
        self.size: int = -(-minimum // granularity) * granularity
        # cuda-bindings handle types are generated at runtime (no static stubs).
        self._handle: object = _value(
            "cuMemCreate", driver.cuMemCreate(self.size, prop, 0)
        )
        self.address: int = int(
            _value(
                "cuMemAddressReserve",
                driver.cuMemAddressReserve(self.size, granularity, 0, 0),
            )
        )
        _value("cuMemMap", driver.cuMemMap(self.address, self.size, 0, self._handle, 0))
        access = driver.CUmemAccessDesc()
        access.location = _location(device, None)
        access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _value(
            "cuMemSetAccess",
            driver.cuMemSetAccess(self.address, self.size, [access], 1),
        )
        _value("cuMemsetD8", driver.cuMemsetD8(self.address, 0, self.size))
        self.used: int = 0

    def release(self) -> None:
        """Unmap and free the chunk."""
        from cuda.bindings import driver

        _value("cuMemUnmap", driver.cuMemUnmap(self.address, self.size))
        _value("cuMemRelease", driver.cuMemRelease(self._handle))
        _value("cuMemAddressFree", driver.cuMemAddressFree(self.address, self.size))


class Mirrored:
    """A pool of words with a second copy."""

    def __init__(self, device: int, first: int | None, half_words: int) -> None:
        """Create the two halves of a pool."""
        from cuda.bindings import driver

        self.half_words: int = half_words
        self.used: int = 0
        size = WORD_BYTES * half_words
        halves = []
        for domain in (None, None) if first is None else (first, 1 - first):
            prop = driver.CUmemAllocationProp()
            prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            prop.location = _location(device, domain)
            handle = _value("cuMemCreate", driver.cuMemCreate(size, prop, 0))
            halves.append(handle)
        self.address: int = int(
            _value("cuMemAddressReserve", driver.cuMemAddressReserve(2 * size, 0, 0, 0))
        )
        access = driver.CUmemAccessDesc()
        access.location = _location(device, None)
        access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        for index, handle in enumerate(halves):
            _value(
                "cuMemMap",
                driver.cuMemMap(self.address + index * size, size, 0, handle, 0),
            )
        _value(
            "cuMemSetAccess",
            driver.cuMemSetAccess(self.address, 2 * size, [access], 1),
        )
        _value("cuMemsetD8", driver.cuMemsetD8(self.address, 0, 2 * size))
        self._size: int = size
        self._handles: list[object] = halves

    def release(self) -> None:
        """Unmap and free both halves."""
        from cuda.bindings import driver

        for index, handle in enumerate(self._handles):
            _value(
                "cuMemUnmap",
                driver.cuMemUnmap(self.address + index * self._size, self._size),
            )
            _value("cuMemRelease", driver.cuMemRelease(handle))
        _value(
            "cuMemAddressFree", driver.cuMemAddressFree(self.address, 2 * self._size)
        )


class _Interface:
    """``__cuda_array_interface__`` of ``words`` int32 values at ``address``."""

    def __init__(self, address: int, words: int) -> None:
        """Describe the ``words`` int32 values at device ``address``."""
        self.__cuda_array_interface__: dict[str, object] = {
            "shape": (words,),
            "typestr": "<i4",
            "data": (address, False),
            "version": 3,
            "strides": None,
            "stream": None,
        }


@dataclass(frozen=True)
class AllocationMark:
    """The allocation state of the pools at one moment."""

    #: Used bytes of each chunk, by domain.
    chunks: dict[int, tuple[int, ...]]
    #: Used words of each mirrored pool, by (first domain, half words).
    mirrored: dict[tuple[int, int], int]


def unsupported_reason(driver: object, cu_device: object) -> tuple[str | None, int]:
    """Check by feature whether locality domains can be used."""
    needed = (
        ("CUdevice_attribute", "CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT"),
        ("CUmemLocationType", "CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN"),
        ("CUdevSmResourceGroup_flags", "CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID"),
        ("CUpointer_attribute", "CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL"),
    )
    for enum, member in needed:
        if not hasattr(getattr(driver, enum, None), member):
            return f"cuda-bindings has no {enum}.{member}", 1
    for call in ("cuDevSmResourceSplit", "cuGreenCtxCreate"):
        if not hasattr(driver, call):
            return f"cuda-bindings has no {call}", 1
    response = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, cu_device
    )
    if response[0] != driver.CUresult.CUDA_SUCCESS:
        return f"the driver does not report a domain count ({response[0]!r})", 1
    count = int(response[1])
    if count < 2:
        return f"the device reports {count} locality domain(s), need 2", count
    return None, count


class LocalityDomains:
    """The locality layout of one device and its word pools."""

    #: Words are handed out on 16-byte boundaries unless the caller asks for more.
    ALIGN_WORDS: ClassVar[int] = 4

    def __init__(self, device: torch.device) -> None:
        """Detect locality support on a device."""
        from cuda.bindings import driver

        self._device: torch.device = device
        self._ordinal: int = int(device.index)  # type: ignore[arg-type]
        self.sms: int = torch.cuda.get_device_properties(device).multi_processor_count
        # Compiled CuTe recorders by grid; JIT executors carry no static type.
        self._compiled: dict[int, Callable[..., None]] = {}
        #: The domain count (1 when the placement is off).
        self.count: int = 1
        with torch.cuda.device(device):
            # An allocation makes PyTorch's primary context current here.
            torch.zeros(1, device=device)
            _value("cuInit", driver.cuInit(0))
            self._cu_device: object = _value(
                "cuDeviceGet", driver.cuDeviceGet(self._ordinal)
            )
            reason, self.count = unsupported_reason(driver, self._cu_device)
            sm_domain: tuple[int, ...] = ()
            if reason is None:
                try:
                    sm_domain = self._enumerate()
                except (AttributeError, RuntimeError) as error:
                    reason = f"the SM domains cannot be enumerated ({error})"
        #: Whether memory is homed by locality domain (else: plain device memory).
        self.active: bool = reason is None
        if reason is not None:
            logger.info("GLM-5.3 locality placement off: %s", reason)
            self.count, sm_domain = 1, (0,) * self.sms
        self.sm_domain: tuple[int, ...] = sm_domain
        self._chunks: dict[int, list[_Chunk]] = {d: [] for d in range(self.count)}
        self._mirrored: dict[tuple[int, int], Mirrored] = {}

    def _home(self, domain: int) -> int | None:
        """Return ``domain`` where memory is homed by domain, else None."""
        return domain if self.active else None

    def _record(self, grid: int, out: torch.Tensor, stream: object) -> None:
        """Run the CuTe SM recorder with ``grid`` CTAs on ``stream``, then wait."""
        import cutlass
        import cutlass.cute as cute

        pointer = cute.runtime.make_ptr(
            cutlass.Uint32, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )
        launch = self._compiled.get(grid)
        if launch is None:
            launch = cute.compile(rt.LocalityProbe(grid).record_sm, pointer, stream)
            self._compiled[grid] = launch
        launch(pointer, stream)
        _value("cuStreamSynchronize", self._synchronize(stream))

    @staticmethod
    def _synchronize(stream: object) -> tuple[object, ...]:
        """Synchronize ``stream`` and return the driver's response."""
        from cuda.bindings import driver

        return driver.cuStreamSynchronize(stream)

    def _enumerate(self) -> tuple[int, ...]:
        """Return the SM ids of each domain."""
        from cuda.bindings import driver

        resource = _value(
            "cuDeviceGetDevResource",
            driver.cuDeviceGetDevResource(
                self._cu_device, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
            ),
        )
        primary = _value(
            "cuDevicePrimaryCtxRetain", driver.cuDevicePrimaryCtxRetain(self._cu_device)
        )
        table = [UNATTRIBUTED] * self.sms
        group_flags = driver.CUdevSmResourceGroup_flags
        by_domain = group_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
        for domain in range(self.count):
            params = driver.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
            params.smCount = 0  # every SM of the domain
            params.flags = int(by_domain)
            params.localityDomainId = domain
            split = _value(
                "cuDevSmResourceSplit",
                driver.cuDevSmResourceSplit(1, resource, 0, [params]),
            )
            group = split[0][0]
            count = int(group.sm.smCount)
            desc = _value(
                "cuDevResourceGenerateDesc",
                driver.cuDevResourceGenerateDesc([group], 1),
            )
            green = _value(
                "cuGreenCtxCreate",
                driver.cuGreenCtxCreate(
                    desc,
                    self._cu_device,
                    driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
                ),
            )
            stream = _value(
                "cuGreenCtxStreamCreate",
                driver.cuGreenCtxStreamCreate(
                    green, int(driver.CUstream_flags.CU_STREAM_NON_BLOCKING), 0
                ),
            )
            # PyTorch allocates only while the primary context is current.
            out = torch.zeros(count, dtype=torch.int32, device=self._device)
            try:
                context = _value("cuCtxFromGreenCtx", driver.cuCtxFromGreenCtx(green))
                _value("cuCtxSetCurrent", driver.cuCtxSetCurrent(context))
                # A kernel module is per context: compile and load under this one.
                self._compiled.pop(count, None)
                self._record(count, out, stream)
            finally:
                _value("cuCtxSetCurrent", driver.cuCtxSetCurrent(primary))
                _value("cuStreamDestroy", driver.cuStreamDestroy(stream))
                _value("cuGreenCtxDestroy", driver.cuGreenCtxDestroy(green))
                self._compiled.pop(count, None)
            for sm in out.cpu().tolist():
                table[int(sm)] = domain
        return tuple(table)

    def grid_sms(self, grid: int) -> list[int]:
        """Return the SM each CTA of an idle one-CTA-per-SM launch ran on."""
        from cuda.bindings import driver

        with torch.cuda.device(self._device):
            stream = driver.CUstream(
                torch.cuda.current_stream(self._device).cuda_stream
            )
            out = torch.zeros(grid, dtype=torch.int32, device=self._device)
            self._record(grid, out, stream)
            return [int(sm) for sm in out.cpu().tolist()]

    def grid_domains(self, grid: int) -> list[int]:
        """Return the domain of each CTA of a one-CTA-per-SM launch."""
        return [self.sm_domain[sm] for sm in self.grid_sms(grid)]

    def majority(self, domains: Sequence[int]) -> int:
        """Return the domain most of ``domains`` name (ties go to the lower one)."""
        votes = [domains.count(domain) for domain in range(self.count)]
        return max(range(self.count), key=lambda domain: (votes[domain], -domain))

    def words(
        self, domain: int, count: int, align_words: int = ALIGN_WORDS
    ) -> torch.Tensor:
        """Return ``count`` zeroed int32 words homed on ``domain``."""
        if not 0 <= domain < self.count:
            raise ValueError(f"domain {domain} out of range for {self.count} domains")
        align = WORD_BYTES * align_words
        size = -(-WORD_BYTES * count // 16) * 16
        chunks = self._chunks[domain]
        for chunk in chunks:
            if -(-chunk.used // align) * align + size <= chunk.size:
                break
        else:
            with torch.cuda.device(self._device):
                chunk = _Chunk(self._ordinal, self._home(domain), size)
            chunks.append(chunk)
        chunk.used = -(-chunk.used // align) * align
        address = chunk.address + chunk.used
        chunk.used += size
        return torch.as_tensor(_Interface(address, count), device=self._device).zero_()

    def copy_bits(self, preferred: int) -> list[int]:
        """Return each SM's copy index."""
        return [int(d not in (preferred, UNATTRIBUTED)) for d in self.sm_domain]

    def mirrored_words(self, first: int, half_words: int, count: int) -> torch.Tensor:
        """Return ``count`` zeroed words of the device's :class:`Mirrored` pool."""
        pool = self._mirrored.get((first, half_words))
        if pool is None:
            with torch.cuda.device(self._device):
                pool = self._mirrored[first, half_words] = Mirrored(
                    self._ordinal, self._home(first), half_words
                )
        offset = -(-pool.used // self.ALIGN_WORDS) * self.ALIGN_WORDS
        if offset + count > half_words:
            raise RuntimeError("the mirrored pool is full")
        pool.used = offset + count
        words = torch.as_tensor(
            _Interface(pool.address + WORD_BYTES * offset, count), device=self._device
        )
        self.mirror_of(words, half_words).zero_()
        return words.zero_()

    def mark(self) -> AllocationMark:
        """Return the pools' allocation state, to give back to :meth:`release`."""
        return AllocationMark(
            chunks={d: tuple(c.used for c in cs) for d, cs in self._chunks.items()},
            mirrored={key: pool.used for key, pool in self._mirrored.items()},
        )

    def release(self, mark: AllocationMark) -> None:
        """Return the pools to ``mark``: everything allocated since is free for
        reuse.
        """
        for domain, chunks in self._chunks.items():
            saved = mark.chunks.get(domain, ())
            for index, chunk in enumerate(chunks):
                chunk.used = saved[index] if index < len(saved) else 0
        for key, pool in self._mirrored.items():
            pool.used = mark.mirrored.get(key, 0)

    def mirror_of(self, words: torch.Tensor, half_words: int) -> torch.Tensor:
        """Return the second copy of a pooled tensor."""
        return torch.as_tensor(
            _Interface(words.data_ptr() + WORD_BYTES * half_words, words.numel()),
            device=self._device,
        )

    def close(self) -> None:
        """Release every pooled chunk; tensors from :meth:`words` become invalid."""
        with torch.cuda.device(self._device):
            for pool in self._mirrored.values():
                pool.release()
            self._mirrored.clear()
            for chunks in self._chunks.values():
                for chunk in chunks:
                    chunk.release()
                chunks.clear()


_BY_DEVICE: dict[int, LocalityDomains] = {}


def domains_for(device: torch.device) -> LocalityDomains:
    """Return the process-wide domains of ``device``; pooled tensors borrow them."""
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    if index not in _BY_DEVICE:
        _BY_DEVICE[index] = LocalityDomains(torch.device("cuda", index))
    return _BY_DEVICE[index]
