# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Standalone workspace allocator for Lamport-based AllReduce CUDA kernels.

Compatible with MiniMaxReduceRMSKernel's LamportComm protocol.
No dependency on TensorRT-LLM internals -- only requires PyTorch and cuda-python.

Workspace memory layout (device-side void* array, N = world_size):
    [0     .. N-1]   : ipc_buffers   (placeholder zeros, unused by MiniMax kernel)
    [N     .. 2N-1]  : ipc_barriers  (placeholder zeros, unused by MiniMax kernel)
    [2N    .. 3N-1]  : lamport_bufs  — IPC triple-buffer pointer per rank
    [3N]             : flag_buffer   → int32[3] = {counter, unused, lamport_flag}
    [3N+1]           : layout_buffer → int64[2] = {clear_size, comm_size}

Usage:
    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # Option 1: class-based (explicit lifecycle)
    ws = LamportWorkspace(rank, world_size, comm_size=2 * 1024 * 1024)
    workspace_tensor = ws.workspace   # torch.Tensor(int64, CUDA)

    # Option 2: cached function (matches TRT-LLM's get_allreduce_workspace API)
    workspace_tensor = get_allreduce_workspace(rank, world_size)

    # Pass to kernel
    output = your_minimax_kernel(input, weights, workspace_tensor, rank, world_size, eps)
"""

import array
import struct
import sys
import threading

import torch

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

_ALIGN = 1 << 21  # 2 MiB — CUDA IPC allocation alignment


# ---------------------------------------------------------------------------
# CUDA helpers
# ---------------------------------------------------------------------------


def _check(error):
    """Raise on CUDA runtime error."""
    success = getattr(cudart.cudaError_t, "cudaSuccess", None) or cudart.cudaError_t(0)
    if error != success:
        raise RuntimeError(f"CUDA runtime error: {error}")


def _cuda_malloc(size: int):
    aligned = ((size + _ALIGN - 1) >> 21) << 21
    err, ptr = cudart.cudaMalloc(aligned)
    _check(err)
    return ptr, aligned


def _cuda_free(ptr: int):
    if ptr:
        _check(cudart.cudaFree(ptr)[0])


def _cuda_memset_zero(ptr: int, size: int):
    _check(cudart.cudaMemset(ptr, 0, size)[0])


def _cuda_memcpy_d2d(dst: int, src: int, size: int):
    _check(
        cudart.cudaMemcpy(
            dst, src, size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        )[0]
    )


# ---------------------------------------------------------------------------
# IPC buffer
# ---------------------------------------------------------------------------


class IpcBuffer:
    """
    Allocates CUDA device memory and exchanges IPC handles with all ranks
    so that every rank holds a valid device pointer to every other rank's buffer.
    """

    def __init__(self, rank: int, world_size: int, size: int, process_group=None):
        self.rank = rank
        self.world_size = world_size
        self.peer_ptrs: list[int] = [0] * world_size
        self.local_ptr: int = 0
        self._alive = False

        if size <= 0:
            return

        self.local_ptr, _ = _cuda_malloc(size)
        _cuda_memset_zero(self.local_ptr, size)
        self._alive = True

        # --- exchange IPC handles via torch.distributed ---
        err, local_handle = cudart.cudaIpcGetMemHandle(self.local_ptr)
        _check(err)

        all_handles: list[bytes | None] = [None] * world_size
        torch.distributed.all_gather_object(
            all_handles, bytes(local_handle.reserved), group=process_group
        )

        for r in range(world_size):
            if r == rank:
                self.peer_ptrs[r] = self.local_ptr
            else:
                handle = cudart.cudaIpcMemHandle_t()
                handle.reserved = all_handles[r]
                err, ptr = cudart.cudaIpcOpenMemHandle(
                    handle, cudart.cudaIpcMemLazyEnablePeerAccess
                )
                _check(err)
                self.peer_ptrs[r] = ptr

    def serialize(self) -> list[int]:
        """Return peer pointers as a list of int64 values (one per rank)."""
        raw = b""
        for ptr in self.peer_ptrs:
            raw += struct.pack("P", ptr)
        return array.array("Q", raw).tolist()

    def cleanup(self):
        if not self._alive:
            return
        self._alive = False
        for r in range(self.world_size):
            if self.peer_ptrs[r] == 0:
                continue
            if r == self.rank:
                _cuda_free(self.peer_ptrs[r])
            else:
                try:
                    _check(cudart.cudaIpcCloseMemHandle(self.peer_ptrs[r])[0])
                except RuntimeError:
                    pass
            self.peer_ptrs[r] = 0
        self.local_ptr = 0

    def __del__(self):
        if not sys.is_finalizing():
            self.cleanup()


# ---------------------------------------------------------------------------
# Lamport negative-zero initialization
# ---------------------------------------------------------------------------


def _lamport_fill_neg_zero(device_ptr: int, size_bytes: int):
    """
    Fill device memory with IEEE-754 negative zero (-0.0f = 0x80000000).
    This is the "slot empty" sentinel for the Lamport protocol: the kernel
    spin-waits until a value is *not* negative zero.
    """
    if size_bytes == 0 or device_ptr == 0:
        return
    n_floats = size_bytes // 4
    # torch preserves -0.0 in IEEE-754
    fill = torch.full((n_floats,), -0.0, dtype=torch.float32, device="cuda")
    _cuda_memcpy_d2d(device_ptr, fill.data_ptr(), size_bytes)
    del fill


# ---------------------------------------------------------------------------
# LamportWorkspace — the main class
# ---------------------------------------------------------------------------


class LamportWorkspace:
    """
    Self-contained workspace for Lamport-based cross-GPU AllReduce.

    Parameters
    ----------
    rank : int
        Local rank (0-based).
    world_size : int
        Total number of ranks in the TP group.
    comm_size : int
        Size in bytes of *one* Lamport buffer slot. The total IPC allocation
        per rank is ``3 * comm_size`` (triple-buffering). Must be large enough
        to hold the per-slot data written by the kernel.  Use
        ``compute_comm_size_for_minimax()`` for a safe default.
    process_group : optional
        ``torch.distributed`` process group for IPC handle exchange.
        ``None`` uses the default group.
    """

    def __init__(self, rank: int, world_size: int, comm_size: int, process_group=None):
        assert world_size >= 2, "Lamport workspace requires at least 2 ranks"
        assert comm_size > 0, "comm_size must be positive"

        self.rank = rank
        self.world_size = world_size
        self.comm_size = comm_size

        # 1) Lamport triple-buffer (the only IPC memory the kernel reads/writes)
        lamport_total = 3 * comm_size
        self._lamport = IpcBuffer(rank, world_size, lamport_total, process_group)
        _lamport_fill_neg_zero(self._lamport.local_ptr, lamport_total)

        # 2) flag_buffer on device: int32[3] = {counter, unused, lamport_flag}
        #    counter  — used for block-level sync inside the kernel
        #    unused   — reserved (index 1)
        #    lamport_flag — triple-buffer rotation index (0 → 1 → 2 → 0 …)
        self._flag_buf = torch.zeros(3, dtype=torch.int32, device="cuda")

        # 3) layout_buffer on device: int64[2] = {clear_size, comm_size}
        #    clear_size — bytes to clear from *previous* slot (set by kernel)
        #    comm_size  — size of one triple-buffer slot
        self._layout_buf = torch.tensor(
            [0, comm_size], dtype=torch.int64, device="cuda"
        )

        # 4) Assemble device-side void* pointer array
        N = world_size
        ptrs: list[int] = []
        ptrs += [0] * N  # [0   .. N-1]   ipc_buffers  (placeholder)
        ptrs += [0] * N  # [N   .. 2N-1]  ipc_barriers (placeholder)
        ptrs += self._lamport.serialize()  # [2N  .. 3N-1]  lamport peer ptrs
        ptrs.append(self._flag_buf.data_ptr())  # [3N]           flag_buffer
        ptrs.append(self._layout_buf.data_ptr())  # [3N+1]       layout_buffer

        self._workspace = torch.tensor(ptrs, dtype=torch.int64, device="cuda")

    @property
    def workspace(self) -> torch.Tensor:
        """Device tensor (int64) that can be passed to the kernel as ``void** workspace``."""
        return self._workspace

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_comm_size_for_minimax(
        max_tokens: int,
        world_size: int,
        fused_qk: bool = True,
    ) -> int:
        """
        Return a safe ``comm_size`` (in bytes) for MiniMaxReduceRMSKernel.

        The kernel stores per-token variance scalars in the Lamport buffer:
          - single-matrix path: ``world_size × max_tokens × 4`` bytes per slot
          - fused Q+K path:     ``world_size × 2 × ceil(max_tokens/4) × 16`` bytes per slot

        The returned value is rounded up to 2 MiB alignment.
        """
        if fused_qk:
            groups = (max_tokens + 3) // 4
            slot_bytes = world_size * 2 * groups * 16  # 16 = sizeof(float4)
        else:
            slot_bytes = world_size * max_tokens * 4  # 4  = sizeof(float)
        return ((slot_bytes + _ALIGN - 1) >> 21) << 21

    def cleanup(self):
        if hasattr(self, "_lamport"):
            self._lamport.cleanup()

    def __del__(self):
        if not sys.is_finalizing():
            self.cleanup()

    def __repr__(self):
        return (
            f"LamportWorkspace(rank={self.rank}, world_size={self.world_size}, "
            f"comm_size={self.comm_size})"
        )


# ---------------------------------------------------------------------------
# Cached convenience function (mirrors TRT-LLM's get_allreduce_workspace)
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_workspace_cache: dict = {}


def get_allreduce_workspace(
    rank: int,
    world_size: int,
    comm_size: int | None = None,
    max_tokens: int = 16384,
    process_group=None,
) -> torch.Tensor:
    """
    Return a cached workspace tensor for the given (rank, world_size) pair.

    On first call the workspace is allocated and IPC handles are exchanged;
    subsequent calls with the same arguments return the cached tensor.

    Parameters
    ----------
    rank, world_size : int
        TP rank and TP size.
    comm_size : int, optional
        Explicit slot size in bytes.  If ``None``, computed automatically
        from ``max_tokens`` and ``world_size`` (fused Q+K path).
    max_tokens : int
        Maximum number of tokens per batch (used when ``comm_size is None``).
    process_group : optional
        ``torch.distributed`` process group.
    """
    key = (rank, world_size)
    with _cache_lock:
        if key not in _workspace_cache:
            if comm_size is None:
                comm_size = LamportWorkspace.compute_comm_size_for_minimax(
                    max_tokens, world_size, fused_qk=True
                )
            ws = LamportWorkspace(rank, world_size, comm_size, process_group)
            _workspace_cache[key] = ws
        return _workspace_cache[key].workspace
