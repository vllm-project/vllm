# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)

# Global instance — one per worker process, initialized once
_aiter_allreduce: Optional["AiterAllreduce"] = None

_AR_MAX_SIZE = 8192 * 1024 * 8 * 2


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


def get_aiter_allreduce() -> Optional["AiterAllreduce"]:
    return _aiter_allreduce


def initialize_aiter_allreduce(
    world_size: int, rank: int, group: ProcessGroup, device: torch.device
) -> None:
    """Initialize the aiter fused AR+RMSNorm instance if not already done.

    Called by RocmAiterAllReduceFusionPass at model init time.
    The instance owns the aiter C++ ptr, staging buffer, and capture state.
    """
    global _aiter_allreduce
    if _aiter_allreduce is not None:
        return
    try:
        _aiter_allreduce = AiterAllreduce(world_size, rank, group, device)
        logger.debug(
            "Initialized aiter allreduce: world_size=%d, rank=%d",
            world_size,
            rank,
        )
    except Exception as e:
        logger.warning("Failed to initialize aiter allreduce: %s", e)
        _aiter_allreduce = None


def destroy_aiter_allreduce() -> None:
    global _aiter_allreduce
    if _aiter_allreduce is not None:
        _aiter_allreduce.close()
        _aiter_allreduce = None


class AiterAllreduce:
    """Self-contained instance for aiter's fused allreduce+RMSNorm kernel.

    Owns:
      - aiter C++ custom_ar ptr (_ptr)
      - local staging buffer (input_buffer) — IPC-registered with all ranks
      - CUDA graph capture state (_IS_CAPTURING)

    Intentionally separate from vLLM's CustomAllreduce so that vLLM's CA
    is used for regular (non-fused) allreduce while this object is used
    exclusively for the fused AR+RMS path.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        group: ProcessGroup,
        device: torch.device,
    ) -> None:
        import aiter as aiter_ops

        self.group = group
        self.rank = rank
        self.world_size = world_size

        self.max_size = _AR_MAX_SIZE
        self.group = group
        self._IS_CAPTURING = False
        self._ptr = 0
        self.device = device

        fully_connected = True
        if world_size > 2 and not fully_connected:
            logger.warning(
                "Custom allreduce is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_custom_all_reduce=True explicitly."
            )
            return
        self.disabled = False
        self.fully_connected = fully_connected
        # buffers memory are owned by this Python class and passed to C++
        # meta data composes of two parts: meta data for synchronization
        # (256 bytes) and a temporary buffer for storing intermediate
        # allreduce results.
        # if current_platform.is_rocm():
        self.meta_size = aiter_ops.meta_size()
        self.meta = aiter_ops.allocate_meta_buffer(
            aiter_ops.meta_size() + self.max_size
        )
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.input_buffer = torch.empty(
            self.max_size, dtype=torch.uint8, device=self.device
        )
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.world_size = world_size
        handle = aiter_ops.get_meta_buffer_ipc_handle(self.meta)
        shard_data = (
            handle,  # ipc handle to base ptr
            0,  # offset of base ptr
        )
        handles, offsets = self._gather_ipc_meta(shard_data)

        self._ptr = aiter_ops.init_custom_ar(
            self.meta, self.rank_data, handles, offsets, self.rank, self.fully_connected
        )
        # Register both input and output buffers
        self.register_input_buffer(self.input_buffer)

    def _get_ipc_meta(self, inp: torch.Tensor):
        import aiter as aiter_ops

        # if current_platform.is_rocm():
        if 1:
            # _share_cuda_() doesn't accept meta buffer not allocated from
            # PyTorch cache allocator, use direct HIP call to get IPC handle
            handle = aiter_ops.get_meta_buffer_ipc_handle(inp)
            shard_data = (
                handle,  # ipc handle to base ptr
                0,  # offset of base ptr
            )
        else:
            data = inp.untyped_storage()._share_cuda_()
            shard_data = (
                data[1],  # ipc handle to base ptr
                data[3],  # offset of base ptr
            )
        return self._gather_ipc_meta(shard_data)

    def _gather_ipc_meta(self, shard_data):
        # Note: don't use `[[None]] * self.world_size` here
        # because it will create a list of the same reference
        all_data: list[list[Any]] = [[None] for i in range(self.world_size)]
        all_data[self.rank][0] = shard_data

        ranks = dist.get_process_group_ranks(group=self.group)
        ranks.sort()
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )

        # we cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.

        handles = []
        offsets = []
        for i in range(len(all_data)):
            handles.append(all_data[i][0][0])  # type: ignore
            offsets.append(all_data[i][0][1])  # type: ignore
        return handles, offsets

    def register_input_buffer(self, inp: torch.Tensor):
        import aiter as aiter_ops

        handles, offsets = self._get_ipc_meta(inp)
        aiter_ops.register_input_buffer(self._ptr, inp, handles, offsets)

    def register_graph_buffers(self):
        import aiter as aiter_ops

        handle, offset = aiter_ops.get_graph_buffer_ipc_meta(self._ptr)
        handles, offsets = self._gather_ipc_meta((handle, offset))
        logger.info("Registering %d cuda graph addresses", len(offset))
        aiter_ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
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
            return inp_size <= (self.max_size / 2)
        return False

    @contextmanager
    def capture(self):
        """Context manager for CUDA graph capture.

        Sets _IS_CAPTURING so the fused op knows to use registered=True,
        then calls register_graph_buffers after capture completes.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if self._ptr:
                self.register_graph_buffers()

    def __del__(self):
        self.close()

    def close(self) -> None:
        if self._ptr:
            try:
                import aiter as aiter_ops

                aiter_ops.dispose(self._ptr)
            except Exception:
                pass
            self._ptr = 0
