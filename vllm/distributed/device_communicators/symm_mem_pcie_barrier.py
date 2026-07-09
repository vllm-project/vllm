# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCIe-safe replacement for torch symmetric memory's group barrier.

torch's ``_SymmetricMemory.barrier(channel)`` synchronizes the group with
CAS exchanges on peer-mapped signal pads. On platforms without native P2P
atomics (``cudaDevP2PAttrNativeAtomicSupported == 0``, e.g. PCIe-only
multi-GPU boxes) those CAS ops are not atomic: barrier tokens get lost or
duplicated under PCIe load and the group wedges permanently.

This module swaps the barrier for a protocol that only needs primitives
such platforms do support:

  * sender:   ``cuStreamWriteValue32`` of the instance's sequence number
              into a **per-instance ring slot** of the receiver's signal
              pad — a plain posted P2P write (with the default preceding
              memory barrier, giving release semantics for prior P2P
              traffic);
  * receiver: ``cuStreamWaitValue32(EQ)`` polling the matching slot of
              its *own* pad — local memory only.

No remote read-modify-write anywhere. Slot index is
``(channel, seq % RING, sender)``, so concurrent same-channel instances
issued from different streams (the pipelined fused ops interleave a
compute and a backend stream) touch different slots and cannot corrupt
each other regardless of device-side execution order; EQ matching makes
a slot's stale previous-lap value simply not match. This also removes the
multi-stream instance-interleaving hazard of the CAS design
(pytorch/pytorch#189228). Ring-slot reuse is safe for ``RING >= 2``: a
rank can only issue instance ``K+RING`` after completing ``K+RING-1``,
which requires every peer to have written ``K+RING-1`` and therefore to
have passed its wait for instance ``K``.

The patch is process-global and a drop-in barrier for every handle; the
pipelined fused ops (``fused_matmul_reduce_scatter``,
``fused_all_gather_matmul`` and their scaled variants) synchronize
exclusively through ``barrier(channel)`` plus stream-ordered plain
writes, so patching the barrier makes the whole family PCIe-safe.

Known constraints:

  * **No CUDA graph capture.** The sequence number is baked into the
    recorded write, so a replayed barrier would trivially satisfy its own
    wait and synchronize nothing (the stock CAS barrier replays
    correctly). Calling the patched barrier during capture raises. The
    fused ops only run at sequence-parallel sizes, which sit above vLLM's
    cudagraph capture ceiling, so this path is structurally unreachable
    today.
  * **Sequence state is keyed by the handle's signal-pad address.** If an
    allocation is freed and a new symm-mem handle reuses the same VA with
    a zeroed pad, call :func:`reset_pcie_barrier_state`. torch's fused
    ops DO re-allocate their workspace when an op requests more space —
    unsafely (see the workspace-guard notes below); the installed guard
    floors the first allocation and keeps retired workspaces alive, so
    VA reuse does not occur.
"""

import os
import threading

import torch
from torch._C._distributed_c10d import _SymmetricMemory

from vllm.logger import init_logger

logger = init_logger(__name__)

# torch's get_symm_mem_workspace() re-allocates and re-rendezvous's the
# fused-op workspace whenever an op requests more than the current size,
# dropping the old workspace tensor with NO synchronization. Device-side
# work still targeting the old workspace (parked barrier waits on its
# signal pads, in-flight P2P chunk copies from peers) then polls freed /
# recyclable memory and can park its stream forever. Two defenses:
#   * floor the first allocation high enough that growth never happens
#     in practice;
#   * if growth does happen, drain the device and keep the old tensor
#     alive forever (bounded leak, rare) so parked ops keep polling
#     stable memory.
_WORKSPACE_FLOOR = int(
    os.getenv("VLLM_SYMM_MEM_WORKSPACE_FLOOR_MB", "256")
) * 1024 * 1024
_workspace_graveyard: list = []
_orig_get_workspace = None

_MAX_CHANNELS = 8
_RING = 4
_lock = threading.Lock()
# own-pad VA -> {channel: last sequence number}
_seq_state: dict[int, dict[int, int]] = {}
_installed = False
_orig_barrier = None
_drv = None


def _pcie_safe_barrier(self, channel: int = 0, timeout_ms: int = 0) -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "The PCIe-safe symm-mem barrier cannot be captured in a CUDA "
            "graph: its baked sequence number would satisfy its own wait "
            "on replay without synchronizing the group."
        )
    world = self.world_size
    rank = self.rank
    pad_ptrs = self.signal_pad_ptrs
    if channel >= _MAX_CHANNELS:
        raise ValueError(f"channel {channel} >= {_MAX_CHANNELS}")
    if self.signal_pad_size < _MAX_CHANNELS * _RING * world * 4:
        raise RuntimeError(
            f"signal pad too small: {self.signal_pad_size} < "
            f"{_MAX_CHANNELS * _RING * world * 4}"
        )

    own_key = pad_ptrs[rank]
    with _lock:
        chans = _seq_state.setdefault(own_key, {})
        seq = chans.get(channel, 0) + 1
        chans[channel] = seq

    stream = _drv.CUstream(torch.cuda.current_stream().cuda_stream)
    base = 4 * world * (channel * _RING + seq % _RING)
    eq = int(_drv.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ)
    for peer in range(world):
        (err,) = _drv.cuStreamWriteValue32(
            stream,
            _drv.CUdeviceptr(pad_ptrs[peer] + base + 4 * rank),
            seq,
            0,
        )
        if int(err) != 0:
            raise RuntimeError(f"cuStreamWriteValue32 failed: {err}")
    for peer in range(world):
        (err,) = _drv.cuStreamWaitValue32(
            stream,
            _drv.CUdeviceptr(pad_ptrs[rank] + base + 4 * peer),
            seq,
            eq,
        )
        if int(err) != 0:
            raise RuntimeError(f"cuStreamWaitValue32 failed: {err}")


def reset_pcie_barrier_state() -> None:
    """Forget all per-handle sequence numbers (see module docstring)."""
    with _lock:
        _seq_state.clear()


def _guarded_get_workspace(group_name, min_size):
    import torch.distributed._symmetric_memory as tsm

    tensor = tsm._group_name_to_workspace_tensor.get(group_name)
    size = tensor.numel() * tensor.element_size() if tensor is not None else 0
    need = max(min_size, _WORKSPACE_FLOOR)
    if tensor is not None and size < need:
        logger.warning(
            "symm-mem workspace grows %d -> %d bytes; draining the device "
            "and retiring the old workspace without freeing it",
            size,
            need,
        )
        torch.cuda.synchronize()
        _workspace_graveyard.append(tensor)
    return _orig_get_workspace(group_name, need)


def _install_workspace_guard() -> None:
    global _orig_get_workspace
    import torch.distributed._symmetric_memory as tsm

    if _orig_get_workspace is not None:
        return
    _orig_get_workspace = tsm.get_symm_mem_workspace
    tsm.get_symm_mem_workspace = _guarded_get_workspace


def install_pcie_safe_barrier() -> None:
    """Replace ``_SymmetricMemory.barrier`` process-wide. Idempotent."""
    global _installed, _orig_barrier, _drv
    if _installed:
        return
    from cuda.bindings import driver as drv

    _drv = drv
    _orig_barrier = _SymmetricMemory.barrier
    _SymmetricMemory.barrier = _pcie_safe_barrier
    _install_workspace_guard()
    _installed = True
    logger.info_once(
        "torch symm-mem group barrier replaced with the PCIe-safe "
        "stream-memops protocol (no native P2P atomics on this platform); "
        "fused-op workspace floored at %d MiB with growth guard."
        % (_WORKSPACE_FLOOR // (1024 * 1024))
    )
