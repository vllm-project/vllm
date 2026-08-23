# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.all_reduce_utils import (
    CUSTOM_ALL_REDUCE_MAX_SIZES,
    gpu_p2p_access_check,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

try:
    ops.meta_size()
    custom_ar = True
except Exception:
    # For CPUs
    custom_ar = False

try:
    import torch.distributed._symmetric_memory as torch_symm_mem
except ImportError:
    torch_symm_mem = None

logger = init_logger(__name__)


def _can_p2p(rank: int, world_size: int) -> bool:
    for i in range(world_size):
        if i == rank:
            continue
        if envs.VLLM_SKIP_P2P_CHECK:
            logger.debug("Skipping P2P check and trusting the driver's P2P report.")
            # can_device_access_peer takes visible device ordinals, while
            # rank and i are logical local IDs.
            return torch.cuda.can_device_access_peer(
                current_platform.logical_device_id_to_visible_device_id(rank),
                current_platform.logical_device_id_to_visible_device_id(i),
            )
        if not gpu_p2p_access_check(rank, i):
            return False
    return True


from vllm.distributed.utils import is_weak_contiguous  # noqa: E402


class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8, 16]
    _DEFAULT_ALL_GATHER_MAX_SIZE = 2 * 1024 * 1024
    _DEFAULT_MNNVL_ALL_GATHER_MAX_SIZES = {
        2: 8 * 1024 * 1024,
        4: 4 * 1024 * 1024,
        6: 2 * 1024 * 1024,
        8: 2 * 1024 * 1024,
        16: 2 * 1024 * 1024,
    }
    _DEFAULT_REDUCE_SCATTER_MAX_SIZE = 16 * 1024 * 1024
    _DEFAULT_MNNVL_REDUCE_SCATTER_MAX_SIZE = 16 * 1024 * 1024

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size=8192 * 1024,
        max_all_gather_size=_DEFAULT_ALL_GATHER_MAX_SIZE,
        max_mnnvl_all_gather_size=None,
        max_reduce_scatter_size=_DEFAULT_REDUCE_SCATTER_MAX_SIZE,
        max_mnnvl_reduce_scatter_size=_DEFAULT_MNNVL_REDUCE_SCATTER_MAX_SIZE,
        symm_mem_enabled=False,
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bound to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self._ptr = 0
        self.disabled = True
        self.mnnvl_buffer = None
        self.mnnvl_handle = None
        self.mnnvl_peer_buffers: list[torch.Tensor] | None = None
        self.mnnvl_multicast_ptr = 0
        self.mnnvl_buffer_size = 0
        self.mnnvl_lamport_ag_local_ptr = 0
        self.mnnvl_lamport_ag_multicast_ptr = 0
        self.mnnvl_lamport_rs_local_ptr = 0
        self.mnnvl_lamport_epochs = None
        self.mnnvl_lamport_ag_epoch_ptr = 0
        self.mnnvl_lamport_rs_epoch_ptr = 0
        self.mnnvl_only = False

        if not custom_ar:
            # disable because of missing custom allreduce library
            # e.g. in a non-GPU environment
            logger.info_once(
                "Custom allreduce is disabled because "
                "of missing custom allreduce library"
            )
            return

        self.group = group

        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "CustomAllreduce should be attached to a non-NCCL group."
        )

        same_node = all(in_the_same_node_as(group, source_rank=0))
        self.mnnvl_only = not same_node

        rank = dist.get_rank(group=self.group)
        self.rank = rank
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning_once(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size,
                str(CustomAllreduce._SUPPORTED_WORLD_SIZES),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        device_capability = current_platform.get_device_capability()
        if (
            current_platform.is_cuda()
            and symm_mem_enabled
            and device_capability is not None
        ):
            device_capability_str = device_capability.as_version_str()
            if (
                device_capability_str in CUSTOM_ALL_REDUCE_MAX_SIZES
                and world_size in CUSTOM_ALL_REDUCE_MAX_SIZES[device_capability_str]
            ):
                max_size = min(
                    CUSTOM_ALL_REDUCE_MAX_SIZES[device_capability_str][world_size],
                    max_size,
                )
        # device.index is a visible ordinal, not a logical local ID.
        fully_connected = False
        if same_node:
            physical_device_id = (
                current_platform.visible_device_id_to_physical_device_id(device.index)
            )
            tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
            gather_list = [
                torch.tensor([0], dtype=torch.int, device="cpu")
                for _ in range(world_size)
            ]
            dist.all_gather(gather_list, tensor, group=self.group)
            physical_device_ids = [t.item() for t in gather_list]
            assert current_platform.is_cuda_alike()
            fully_connected = current_platform.is_fully_connected(physical_device_ids)
        if same_node and world_size > 2 and not fully_connected:
            logger.warning(
                "Custom allreduce is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_custom_all_reduce=True explicitly."
            )
            return
        # test P2P capability, this checks software/cudaruntime support
        # this is expensive to compute at the first time
        # then we cache the result
        # On AMD GPU, p2p is always enabled between XGMI connected GPUs
        if (
            same_node
            and not current_platform.is_rocm()
            and not _can_p2p(rank, world_size)
        ):
            logger.warning(
                "Custom allreduce is disabled because your platform lacks "
                "GPU P2P capability or P2P test failed. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly."
            )
            return

        self.disabled = False
        # Buffers memory are owned by this Python class and passed to C++.
        # Metadata composes of two parts: metadata for synchronization and a
        # temporary buffer for storing intermediate allreduce results.
        if same_node:
            self.meta_ptrs = self.create_shared_buffer(
                ops.meta_size() + max_size, group=group, uncached=True
            )
        else:
            meta_ptr, _ = ops.allocate_shared_buffer_and_handle(ops.meta_size())
            self.meta_ptrs = [meta_ptr] * world_size
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before the operation is performed
        legacy_buffer_size = max(max_size, max_all_gather_size, max_reduce_scatter_size)
        if same_node:
            self.buffer_ptrs = self.create_shared_buffer(
                legacy_buffer_size,
                group=group,
            )
        else:
            buffer_ptr, _ = ops.allocate_shared_buffer_and_handle(legacy_buffer_size)
            self.buffer_ptrs = [buffer_ptr] * world_size
        # This stores tuples of pointers to IPC buffers from all ranks.
        # Each registered tuple contains at most 16 addresses.
        # Allocating 8MB is enough for 65536 such tuples. The largest model uses
        # fewer than 10000 registered tuples.
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.max_size = max_size
        self.max_all_gather_size = max_all_gather_size
        if max_mnnvl_all_gather_size is None:
            max_mnnvl_all_gather_size = self._DEFAULT_MNNVL_ALL_GATHER_MAX_SIZES[
                world_size
            ]
        self.max_mnnvl_all_gather_size = max_mnnvl_all_gather_size
        self.max_reduce_scatter_size = max_reduce_scatter_size
        self.max_mnnvl_reduce_scatter_size = max_mnnvl_reduce_scatter_size
        self.rank = rank
        self.world_size = world_size
        self.fully_connected = fully_connected
        self._ptr = ops.init_custom_ar(
            self.meta_ptrs, self.rank_data, rank, self.fully_connected
        )
        ops.register_buffer(self._ptr, self.buffer_ptrs)
        self._init_mnnvl_buffer(
            max(
                max_mnnvl_all_gather_size * world_size,
                max_mnnvl_reduce_scatter_size,
            )
        )
        if not same_node and not self.mnnvl_multicast_ptr:
            logger.warning(
                "Custom collectives are disabled because this multi-node "
                "group does not support MNNVL multicast."
            )
            self.close()
            self.disabled = True

    def _init_mnnvl_buffer(self, stage_size: int) -> None:
        if torch_symm_mem is None or not current_platform.is_cuda():
            return
        try:
            buffer_size = stage_size * 6
            buffer = torch_symm_mem.empty(
                buffer_size, dtype=torch.uint8, device=self.device
            )
            handle = torch_symm_mem.rendezvous(buffer, self.group.group_name)
            if handle.multicast_ptr == 0:
                return
            peer_buffers = [
                handle.get_buffer(
                    peer,
                    (buffer_size,),
                    torch.uint8,
                    storage_offset=0,
                )
                for peer in range(self.world_size)
            ]
            ptrs = [peer_buffer.data_ptr() for peer_buffer in peer_buffers]
            lamport_ag_offset = 0
            lamport_rs_offset = stage_size * 3
            lamport_ag_ptrs = [ptr + lamport_ag_offset for ptr in ptrs]
            lamport_rs_ptrs = [ptr + lamport_rs_offset for ptr in ptrs]
            ops.register_buffer(self._ptr, lamport_ag_ptrs)
            ops.register_buffer(self._ptr, lamport_rs_ptrs)

            buffer.view(torch.int32).fill_(-2147483648)
            epochs = torch.zeros(
                (2, 32),
                dtype=torch.int32,
                device=self.device,
            )
            torch.accelerator.synchronize()
            dist.barrier(group=self.group)

            self.mnnvl_buffer = buffer
            self.mnnvl_handle = handle
            self.mnnvl_peer_buffers = peer_buffers
            self.mnnvl_multicast_ptr = handle.multicast_ptr
            self.mnnvl_buffer_size = stage_size
            self.mnnvl_lamport_ag_local_ptr = lamport_ag_ptrs[self.rank]
            self.mnnvl_lamport_ag_multicast_ptr = (
                handle.multicast_ptr + lamport_ag_offset
            )
            self.mnnvl_lamport_rs_local_ptr = lamport_rs_ptrs[self.rank]
            self.mnnvl_lamport_epochs = epochs
            self.mnnvl_lamport_ag_epoch_ptr = epochs[0].data_ptr()
            self.mnnvl_lamport_rs_epoch_ptr = epochs[1].data_ptr()
        except RuntimeError as error:
            logger.debug("MNNVL AG/RS initialization failed: %s", error)

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def register_graph_buffers(self):
        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
        logger.debug("Registering %d cuda graph addresses", len(offset))
        # We cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.
        all_data: list[list[list[int] | None]]
        all_data = [[None, None] for _ in range(dist.get_world_size(group=self.group))]
        all_data[self.rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )
        # Unpack list of tuples to tuple of lists.
        handles = cast(list[list[int]], [d[0] for d in all_data])
        offsets = cast(list[list[int]], [d[1] for d in all_data])
        ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled or self.world_size > 8:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        if self.world_size == 2 or self.fully_connected:
            return inp_size < self.max_size
        return False

    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor = None, registered: bool = False
    ):
        """Performs an out-of-place all reduce.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            ops.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            ops.all_reduce(
                self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
            )
        return out

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, registered=True)
            else:
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                return torch.empty_like(input)
        else:
            # Note: outside of cuda graph context, custom allreduce incurs a
            # cost of cudaMemcpy, which should be small (<=1% of overall
            # latency) compared to the performance gain of using custom kernels
            return self.all_reduce(input, registered=False)

    def should_custom_all_gather(self, inp: torch.Tensor) -> bool:
        if self.disabled or not current_platform.is_cuda():
            return False
        if self.world_size == 16 and not self.mnnvl_only:
            return False
        inp_size = inp.nbytes
        if inp.dtype not in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ):
            return False
        max_size = (
            self.max_mnnvl_all_gather_size
            if self.mnnvl_multicast_ptr
            else self.max_all_gather_size
        )
        return (
            0 < inp_size <= max_size
            and inp_size % 16 == 0
            and is_weak_contiguous(inp)
            and (self.fully_connected or bool(self.mnnvl_multicast_ptr))
        )

    def custom_all_gather(self, inp: torch.Tensor) -> torch.Tensor | None:
        if not self.should_custom_all_gather(inp):
            return None
        out_shape = (inp.shape[0] * self.world_size,) + inp.shape[1:]
        if self.mnnvl_multicast_ptr:
            logger.info_once(
                "Using the MNNVL Lamport all-gather kernel.",
                scope="global",
            )
            out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
            ops.mnnvl_lamport_all_gather(
                self._ptr,
                inp,
                out,
                self.mnnvl_lamport_ag_local_ptr,
                self.mnnvl_lamport_ag_multicast_ptr,
                self.mnnvl_lamport_ag_epoch_ptr,
                self.mnnvl_buffer_size,
            )
        else:
            out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
            ops.custom_all_gather(
                self._ptr,
                inp,
                out,
                self.buffer_ptrs[self.rank],
                self.max_all_gather_size,
            )
        return out

    def should_custom_reduce_scatter(self, inp: torch.Tensor) -> bool:
        if self.disabled or not current_platform.is_cuda():
            return False
        if self.world_size == 16 and not self.mnnvl_only:
            return False
        inp_size = inp.nbytes
        if inp.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return False
        if inp.shape[0] % self.world_size != 0:
            return False
        output_size = inp_size // self.world_size
        max_size = (
            self.max_mnnvl_reduce_scatter_size
            if self.mnnvl_multicast_ptr
            else self.max_reduce_scatter_size
        )
        return (
            0 < inp_size <= max_size
            and output_size % 16 == 0
            and is_weak_contiguous(inp)
            and (self.fully_connected or bool(self.mnnvl_multicast_ptr))
        )

    def custom_reduce_scatter(self, inp: torch.Tensor) -> torch.Tensor | None:
        if not self.should_custom_reduce_scatter(inp):
            return None
        out_shape = (inp.shape[0] // self.world_size,) + inp.shape[1:]
        out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
        if self.mnnvl_multicast_ptr:
            logger.info_once(
                "Using the MNNVL Lamport reduce-scatter kernel.",
                scope="global",
            )
            ops.mnnvl_lamport_reduce_scatter(
                self._ptr,
                inp,
                out,
                self.mnnvl_lamport_rs_local_ptr,
                self.mnnvl_lamport_rs_epoch_ptr,
                self.mnnvl_buffer_size,
            )
        else:
            ops.custom_reduce_scatter(
                self._ptr,
                inp,
                out,
                self.buffer_ptrs[self.rank],
                self.max_reduce_scatter_size,
            )
        return out

    def close(self):
        if not self.disabled and self._ptr:
            if ops is not None:
                ops.dispose(self._ptr)
            self._ptr = 0
            self.free_shared_buffer(self.meta_ptrs, rank=self.rank)
            self.free_shared_buffer(self.buffer_ptrs, rank=self.rank)
            self.mnnvl_peer_buffers = None
            self.mnnvl_handle = None
            self.mnnvl_buffer = None
            self.mnnvl_lamport_epochs = None

    def __del__(self):
        self.close()

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int,
        group: ProcessGroup | None = None,
        uncached: bool | None = False,
    ) -> list[int]:
        pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: list[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer)  # type: ignore
            else:
                pointers.append(ops.open_mem_handle(h))
        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: list[int],
        group: ProcessGroup | None = None,
        rank: int | None = None,
    ) -> None:
        if rank is None:
            rank = dist.get_rank(group=group)
        if ops is not None:
            ops.free_shared_buffer(pointers[rank])
