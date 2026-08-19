# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.distributed.device_communicators.all_reduce_utils import (
    SYMM_MEM_ALL_REDUCE_MAX_SIZES,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

from dataclasses import dataclass
from typing import Any

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    symm_mem_available = True
except ImportError:
    torch_symm_mem = None  # type: ignore[assignment]
    symm_mem_available = False

logger = init_logger(__name__)


@dataclass
class SymmMemPeerAllocation:
    """Rendezvoused symmetric buffer plus per-rank peer pointers."""

    storage: torch.Tensor
    handle: Any
    peer_ptrs: torch.Tensor
    world_size: int
    rank: int
    multicast_ptr: int
    buffer_size: int
    _peer_views: list[torch.Tensor]

    def peer_ptrs_for_view(self, local_view: torch.Tensor) -> torch.Tensor:
        offset_bytes = local_view.data_ptr() - self.storage.data_ptr()
        if offset_bytes < 0 or offset_bytes >= self.buffer_size:
            raise ValueError(
                "Symmetric-memory view offset "
                f"{offset_bytes} is outside buffer size {self.buffer_size}"
            )
        view_ptrs = self.peer_ptrs + offset_bytes
        if bool((view_ptrs == 0).any().item()):
            raise RuntimeError("Symmetric-memory view produced a null peer pointer")
        return view_ptrs

    def multicast_ptr_for_view(self, local_view: torch.Tensor) -> int:
        if self.multicast_ptr == 0:
            return 0
        return self.multicast_ptr + (local_view.data_ptr() - self.storage.data_ptr())

    def close(self) -> None:
        self.peer_ptrs = torch.empty(0, dtype=torch.int64, device=self.storage.device)
        self.handle = None
        self.storage = torch.empty(0, dtype=self.storage.dtype, device="cpu")


def allocate_symm_mem_peer(
    shape: tuple[int, ...] | int,
    dtype: torch.dtype,
    device: torch.device,
    group: ProcessGroup,
) -> SymmMemPeerAllocation:
    if not symm_mem_available or torch_symm_mem is None:
        raise RuntimeError("torch.distributed._symmetric_memory is unavailable")
    storage = torch_symm_mem.empty(shape, dtype=dtype, device=device)
    storage.zero_()
    torch.accelerator.synchronize()
    handle = torch_symm_mem.rendezvous(storage, group.group_name)
    if handle is None:
        raise RuntimeError("symmetric-memory rendezvous returned None")
    handle.barrier()
    view_shape = (shape,) if isinstance(shape, int) else tuple(shape)
    peer_views = [
        handle.get_buffer(peer, list(view_shape), dtype, 0)
        for peer in range(group.size())
    ]
    peer_ptrs = torch.tensor(
        [int(view.data_ptr()) for view in peer_views],
        dtype=torch.int64,
        device=storage.device,
    )
    if bool((peer_ptrs == 0).any().item()):
        raise RuntimeError("symmetric-memory rendezvous produced a null peer pointer")
    if int(peer_ptrs[group.rank()].item()) != int(storage.data_ptr()):
        raise RuntimeError("Local symmetric-memory pointer does not match storage")
    multicast_ptr = 0
    try:
        multicast_ptr = int(handle.multicast_ptr or 0)
    except Exception:
        multicast_ptr = 0
    buffer_size = int(getattr(handle, "buffer_size", storage.nbytes))
    return SymmMemPeerAllocation(
        storage=storage,
        handle=handle,
        peer_ptrs=peer_ptrs,
        world_size=group.size(),
        rank=group.rank(),
        multicast_ptr=multicast_ptr,
        buffer_size=buffer_size,
        _peer_views=peer_views,
    )


class SymmMemCommunicator:
    _WORLD_SIZES_MULTIMEM = {
        "9.0": [4, 6, 8],
        "10.0": [6, 8],
        "10.3": [6, 8],
        "10.7": [6, 8],  # sm_107 (Rubin): reuse 10.3 thresholds
    }

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        # add options for testing
        force_multimem: bool | None = None,
        max_size_override: int | None = None,
    ):
        self.disabled = True

        if not symm_mem_available:
            return

        if not current_platform.is_cuda():
            logger.warning("SymmMemCommunicator: symmetric memory is not available.")
            return
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        torch.accelerator.set_device_index(device)
        self.dtype = torch.bfloat16
        self.device = device
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        capability = current_platform.get_device_capability()
        if capability is None:
            logger.warning(
                "SymmMemCommunicator: device capability is unknown, "
                "communicator is not available."
            )
            return
        self.device_capability = capability.as_version_str()
        if self.device_capability not in SYMM_MEM_ALL_REDUCE_MAX_SIZES:
            logger.warning(
                "SymmMemCommunicator: Device capability %s not supported, "
                "communicator is not available.",
                self.device_capability,
            )
            return
        if self.world_size not in SYMM_MEM_ALL_REDUCE_MAX_SIZES[self.device_capability]:
            logger.warning(
                "SymmMemCommunicator: World size %d not supported, "
                "communicator is not available.",
                self.world_size,
            )
            return
        # Use override max_size if provided, otherwise use default
        if max_size_override is not None:
            self.max_size = max_size_override
            logger.info(
                "SymmMemCommunicator: Using override max_size: %s bytes",
                self.max_size,
            )
        else:
            self.max_size = SYMM_MEM_ALL_REDUCE_MAX_SIZES[self.device_capability][
                self.world_size
            ]
        try:
            self.buffer = torch_symm_mem.empty(
                self.max_size // self.dtype.itemsize,
                device=self.device,
                dtype=self.dtype,
            )
            handle = torch_symm_mem.rendezvous(self.buffer, self.group.group_name)
        except RuntimeError as e:
            logger.warning_once(
                "SymmMemCommunicator: symmetric memory initialization failed: %s "
                "Communicator is not available. To suppress this warning set "
                "VLLM_ALLREDUCE_USE_SYMM_MEM=0",
                str(e),
            )
            return
        if handle.multicast_ptr == 0:
            logger.warning(
                "SymmMemCommunicator: symmetric memory "
                "multicast operations are not supported."
            )
            return
        self.force_multimem = force_multimem
        self.disabled = False
        if envs.VLLM_BATCH_INVARIANT:
            self.disabled = True

    def should_use_symm_mem(self, inp: torch.Tensor):
        if self.disabled:
            return False
        if inp.dtype != self.dtype:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 4 != 0:
            return False
        return inp_size <= self.max_size

    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        if not self.should_use_symm_mem(inp):
            return None
        if out is None:
            out = torch.empty_like(inp)
        self.buffer[: inp.numel()].copy_(inp.view(-1))

        # Determine which algorithm to use
        use_multimem = False
        if self.force_multimem is not None:
            # Test override: use forced setting
            use_multimem = self.force_multimem
        else:
            # Normal logic: use multimem for supported world sizes
            use_multimem = (
                self.world_size in self._WORLD_SIZES_MULTIMEM[self.device_capability]
            )

        if use_multimem:
            torch.ops.symm_mem.multimem_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        out.copy_(self.buffer[: inp.numel()].view(out.shape))
        return out
