# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs

try:
    import torch.distributed._symmetric_memory as torch_symm_mem
except ImportError:
    torch_symm_mem = None

_SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))
_VECTOR_BYTES = 16
_ALIGNMENT = 128
_CacheKey = tuple[int, torch.device, int, int]


def _doorbell_num_bytes(world_size: int) -> int:
    return _ALIGNMENT + 2 * world_size * 8


def _align(value: int) -> int:
    return (value + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _slot_num_bytes(max_num_bytes: int, world_size: int) -> int:
    return _align(_align(max_num_bytes) + _doorbell_num_bytes(world_size))


def _all_ranks(group: ProcessGroup, value: bool) -> bool:
    available = torch.tensor(value, dtype=torch.int32, device="cpu")
    dist.all_reduce(available, op=dist.ReduceOp.MIN, group=group)
    return bool(available.item())


class _CollectiveUnavailable(RuntimeError):
    """A rank-synchronized resource failure that can safely fall back."""


class LowSMAllReduce:
    """Cached low-SM BF16 all-reduce with bounded result storage.

    Each slot is single-flight: its returned tensor must be consumed before
    reuse. Ranks must initialize matching contracts and call slots with
    matching sizes and collective order.
    """

    _instances: ClassVar[dict[_CacheKey, "LowSMAllReduce | None"]] = {}

    @classmethod
    def initialize(
        cls,
        *,
        group: ProcessGroup,
        device: torch.device,
        max_num_bytes: int,
        num_slots: int = 1,
    ) -> "LowSMAllReduce | None":
        device = torch.device(device)
        world_size = dist.get_world_size(group)
        capability = (
            torch.cuda.get_device_capability(device) if device.type == "cuda" else None
        )
        valid_contract = max_num_bytes >= _VECTOR_BYTES and num_slots > 0
        contract_capacity = max_num_bytes if valid_contract else 0
        contract_slots = num_slots if valid_contract else 0
        locally_eligible = (
            valid_contract
            and torch_symm_mem is not None
            and torch_symm_mem._should_use_implicit_mempool()
            and envs.VLLM_ALLREDUCE_USE_SYMM_MEM
            and not envs.VLLM_BATCH_INVARIANT
            and hasattr(torch.ops._C, "low_sm_all_reduce_")
            and world_size > 1
            and capability in _SUPPORTED_CAPABILITIES
        )
        preflight = torch.tensor(
            [
                int(locally_eligible),
                contract_capacity,
                -contract_capacity,
                contract_slots,
                -contract_slots,
            ],
            dtype=torch.int64,
            device="cpu",
        )
        dist.all_reduce(preflight, op=dist.ReduceOp.MIN, group=group)
        if (
            not preflight[0].item()
            or preflight[1].item() != -preflight[2].item()
            or preflight[3].item() != -preflight[4].item()
        ):
            return None

        key: _CacheKey = (id(group), device, max_num_bytes, num_slots)
        if key in cls._instances:
            return cls._instances[key]
        try:
            cls._instances[key] = cls(
                device=device,
                group=group,
                max_num_bytes=max_num_bytes,
                num_slots=num_slots,
            )
        except _CollectiveUnavailable:
            cls._instances[key] = None
        return cls._instances[key]

    def __init__(
        self,
        *,
        device: torch.device,
        group: ProcessGroup,
        max_num_bytes: int,
        num_slots: int,
    ) -> None:
        assert torch_symm_mem is not None
        self.device = device
        self.max_num_bytes = max_num_bytes
        self.num_slots = num_slots
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self._slot_num_bytes = _slot_num_bytes(max_num_bytes, self.world_size)
        storage_num_bytes = self._slot_num_bytes * num_slots

        with torch.accelerator.device_index(device.index):
            storage = None
            allocation_error = None
            try:
                storage = torch_symm_mem.empty(
                    storage_num_bytes,
                    device=device,
                    dtype=torch.uint8,
                )
            except RuntimeError as error:
                allocation_error = error

        if not _all_ranks(group, storage is not None):
            raise _CollectiveUnavailable(
                "symmetric-memory allocation failed on at least one rank"
            ) from allocation_error
        assert storage is not None

        self._doorbell_num_bytes = _doorbell_num_bytes(self.world_size)
        self._doorbell_slot_offset = _align(max_num_bytes)
        with torch.accelerator.device_index(device.index):
            self._storage = storage
            # Rendezvous lazily creates peer resources. Keep them out of an
            # enclosing model-weight pool so sleep/wake preserves their
            # addresses together with the symmetric allocation.
            with torch.cuda.use_mem_pool(torch_symm_mem.get_mem_pool(device)):
                self._handle = torch_symm_mem.rendezvous(storage, group.group_name)
                resource_error = None
                peer_ptrs_resource = None
                try:
                    multicast_ptr = int(self._handle.multicast_ptr or 0)
                    peer_ptrs_resource = self._handle.buffer_ptrs_dev
                    peer_ptrs = (
                        peer_ptrs_resource.data_ptr()
                        if isinstance(peer_ptrs_resource, torch.Tensor)
                        else int(peer_ptrs_resource or 0)
                    )
                except (RuntimeError, TypeError, ValueError) as error:
                    resource_error = error
                    multicast_ptr = 0
                    peer_ptrs = 0
            for slot in range(num_slots):
                offset = slot * self._slot_num_bytes + self._doorbell_slot_offset
                storage[offset : offset + self._doorbell_num_bytes].zero_()
            torch.accelerator.synchronize()
        if not _all_ranks(group, multicast_ptr != 0 and peer_ptrs != 0):
            raise _CollectiveUnavailable(
                "NVLS resources are not available on every rank"
            ) from resource_error
        self._peer_ptrs_resource = peer_ptrs_resource
        self._peer_ptrs = peer_ptrs
        self._multicast_ptr = multicast_ptr

    def supports(self, input_: torch.Tensor) -> bool:
        num_bytes = input_.nbytes
        return (
            input_.device == self.device
            and input_.dtype == torch.bfloat16
            and input_.is_contiguous()
            and 0 < num_bytes <= self.max_num_bytes
            and num_bytes % _VECTOR_BYTES == 0
        )

    def __call__(self, input_: torch.Tensor, *, slot: int = 0) -> torch.Tensor:
        if not 0 <= slot < self.num_slots:
            raise ValueError(f"Low-SM all-reduce slot {slot} is out of range.")
        if not self.supports(input_):
            raise ValueError(
                "Low-SM all-reduce requires a nonempty, contiguous BF16 input "
                f"on {self.device}, divisible by 16 bytes and within its "
                f"{self.max_num_bytes}-byte capacity."
            )
        if input_.untyped_storage().data_ptr() == self._storage.data_ptr():
            raise ValueError("Input must not alias the low-SM result storage.")

        num_bytes = input_.nbytes
        slot_offset = slot * self._slot_num_bytes
        output = self._storage[slot_offset : slot_offset + num_bytes]
        output = output.view(torch.bfloat16).view_as(input_)
        # CUDA VMM permits virtual aliases at operation boundaries. This
        # unicast copy completes before the native op reads via its multicast
        # alias on the same stream.
        output.copy_(input_)
        doorbell_offset = slot_offset + self._doorbell_slot_offset
        doorbells = self._storage[
            doorbell_offset : doorbell_offset + self._doorbell_num_bytes
        ]
        torch.ops._C.low_sm_all_reduce_(
            output,
            self._multicast_ptr + slot_offset,
            doorbells,
            self._peer_ptrs,
            doorbell_offset,
            self.rank,
            self.world_size,
        )
        return output
