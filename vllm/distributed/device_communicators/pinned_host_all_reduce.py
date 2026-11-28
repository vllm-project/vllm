# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from multiprocessing import shared_memory
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.shm_broadcast import check_shm_free_space
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

try:
    ops.pinned_host_ar_buffer_size(16, 16, 16)
    _ops_available = True
except Exception:
    _ops_available = False


class PinnedHostAllReduce:
    """All-reduce for two GPUs on one node without peer access.

    Both ranks map one shared-memory segment as pinned, device-mapped host
    memory and exchange their inputs through it inside a single kernel (plus a
    device-to-host copy for large inputs), with no host synchronization, so
    calls can be captured in CUDA graphs. Inputs must be contiguous float16,
    bfloat16 or float32 tensors of a multiple of 16 bytes, up to `max_size`.
    Both ranks add in float32 and round once, which matches NCCL's two-rank sum
    bitwise.

    Calls share one device-side sequence number and one set of buffers, so all
    calls on an instance must be ordered on one stream (as captured graphs and
    eager calls on the model stream are).
    """

    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
    # Mode thresholds, tuned on RTX 3080 pairs on PCIe Gen4 without P2P.
    _LL_MAX_SIZE = 16 * 1024
    _FLAG_MAX_SIZE = 48 * 1024
    _MAX_SIZE = 2 * 1024 * 1024

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_size: int = _MAX_SIZE,
        start_seq: int = 0,
    ) -> None:
        """Collective over `group`, a non-NCCL group of two ranks.

        `start_seq` is the sequence number before the first call; tests set it
        to cross the 32-bit boundary.
        """
        self.disabled = True
        self._ptr = 0
        self._shm: shared_memory.SharedMemory | None = None
        self._view: ctypes.Array | None = None
        if not _ops_available or not current_platform.is_cuda():
            return
        if dist.get_world_size(group) != 2:
            return
        if not all(in_the_same_node_as(group, source_rank=0)):
            logger.warning(
                "Pinned-host all-reduce is disabled because the two ranks are "
                "not on the same node."
            )
            return

        self.max_size = max_size
        self.rank = dist.get_rank(group)
        size = ops.pinned_host_ar_buffer_size(
            self._LL_MAX_SIZE, self._FLAG_MAX_SIZE, max_size
        )
        name: list[str | None] = [None]
        if self.rank == 0:
            try:
                check_shm_free_space(
                    size, allocation_name="pinned-host all-reduce buffer"
                )
                self._shm = shared_memory.SharedMemory(create=True, size=size)
                name[0] = self._shm.name
            except Exception as e:
                logger.warning("Pinned-host all-reduce is disabled: %s", e)
        try:
            dist.broadcast_object_list(
                name, src=dist.get_global_rank(group, 0), group=group
            )
            if name[0] is None:
                return

            ok = True
            try:
                if self._shm is None:
                    # Python tracks shared memory it did not create; see shm_broadcast.
                    with patch(
                        "multiprocessing.resource_tracker.register",
                        lambda *args, **kwargs: None,
                    ):
                        self._shm = shared_memory.SharedMemory(name=name[0])
                assert self._shm.buf is not None
                self._view = (ctypes.c_char * size).from_buffer(self._shm.buf)
                with torch.accelerator.device_index(device.index):
                    self._ptr = ops.init_pinned_host_ar(
                        ctypes.addressof(self._view),
                        size,
                        self.rank,
                        self._LL_MAX_SIZE,
                        self._FLAG_MAX_SIZE,
                        max_size,
                        start_seq,
                    )
            except Exception as e:
                logger.warning("Pinned-host all-reduce is disabled: %s", e)
                ok = False

            # Also a barrier: the segment can be unlinked once both ranks mapped it.
            flag = torch.tensor([int(ok)], dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
        finally:
            # Rank 0 unlinks even if setup fails before the barrier.
            if self.rank == 0 and self._shm is not None:
                self._shm.unlink()
        if not flag.item():
            self.close()
            return
        self.disabled = False

    def should_use(self, inp: torch.Tensor) -> bool:
        # Must not depend on the address: both ranks have to take the same path.
        if self.disabled:
            return False
        nbytes = inp.numel() * inp.element_size()
        return (
            inp.dtype in self._SUPPORTED_DTYPES
            and inp.is_contiguous()
            and 0 < nbytes <= self.max_size
            and nbytes % 16 == 0
        )

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        # The kernels need 16-byte aligned tensors. Fresh allocations (the
        # output, a clone) are aligned, so only an offset input view is copied.
        if inp.data_ptr() % 16:
            inp = inp.clone()
        out = torch.empty_like(inp)
        ops.pinned_host_all_reduce(self._ptr, inp, out)
        return out

    def close(self) -> None:
        self.disabled = True
        if self._ptr:
            ops.dispose_pinned_host_ar(self._ptr)
            self._ptr = 0
        if self._shm is not None:
            self._view = None
            self._shm.close()
            self._shm = None

    def __del__(self):
        self.close()
