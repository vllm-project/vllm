# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symmetric-memory rendezvous that spends NVLink multicast objects only where a
kernel stores through the multicast address.

Every `symm_mem.rendezvous` of torch's CUDA backend binds the allocation to a new
multicast object when the platform supports multicast. On a GB300 NVL72 rack these
objects come from one pool of about 820 (measured, driver 580) that every process
of the rack's NVLink domain shares, whatever the buffer sizes; a deployment that
finds the pool drained fails at start with `CUDA_ERROR_OUT_OF_MEMORY` from
`cuMulticastBindMem`. The Kimi K3 mailboxes written by peer stores (the DCP
partials, the top-k exchange, the EP dispatch) only read `buffer_ptrs`;
`peer_rendezvous` maps them without a multicast object. The peer mapping is the
same either way (fabric handles across the NVLink domain), so the kernels and their
numerics are unchanged.

torch reads `TORCH_SYMM_MEM_DISABLE_MULTICAST` at every rendezvous (torch 2.14;
only "1" disables, "0" binds like an unset variable); `peer_rendezvous` sets it to
"1" for its own call only, under a lock, and restores the previous value ("0" where
it was unset: the variable is added to the environment once and later only changes
value, so no other thread's `getenv` races a reallocation of the environment). It
is not a user switch.
"""

import os
import threading

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)

_DISABLE_MULTICAST = "TORCH_SYMM_MEM_DISABLE_MULTICAST"
_LOCK = threading.Lock()


def peer_rendezvous(tensor: torch.Tensor, group):
    """`symm_mem.rendezvous(tensor, group)` without a multicast object: the handle
    maps every peer (`buffer_ptrs`, `get_buffer`); `multicast_ptr` is 0.
    Collective over `group` (a process group or its name), like the rendezvous."""
    import torch.distributed._symmetric_memory as symm_mem

    with _LOCK:
        previous = os.environ.get(_DISABLE_MULTICAST)
        os.environ[_DISABLE_MULTICAST] = "1"
        try:
            return symm_mem.rendezvous(tensor, group)
        finally:
            os.environ[_DISABLE_MULTICAST] = "0" if previous is None else previous


def _process_group(group) -> dist.ProcessGroup:
    if isinstance(group, str):
        from torch.distributed.distributed_c10d import _resolve_process_group

        return _resolve_process_group(group)
    return group


def multicast_rendezvous(tensor: torch.Tensor, group, what: str):
    """`(handle, multicast address)` of `symm_mem.rendezvous(tensor, group)`, the
    address 0 on every rank of `group` when the multicast binding failed on any
    of them (torch then skips it on that rank only), so every rank takes the same
    path. Collective over `group`."""
    import torch.distributed._symmetric_memory as symm_mem

    handle = symm_mem.rendezvous(tensor, group)
    mc = int(handle.multicast_ptr or 0)
    flag = torch.tensor([1 if mc else 0], dtype=torch.int32, device=tensor.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=_process_group(group))
    if not flag.item():
        logger.warning_once(
            "%s: no NVLink multicast object on some rank of its group (the rack's "
            "multicast pool is exhausted or the platform has none).",
            what,
        )
        return handle, 0
    return handle, mc
