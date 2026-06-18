"""Broadcast prompt_embeds via a filename-based shared-memory handle instead of
pickling the tensor bytes into the executor's shm broadcast MessageQueue.

The EngineCore -> TP-workers broadcast (the SchedulerOutput) normally pickles each
`NewRequestData.prompt_embeds` through torch's default tensor reduction
(`torch.save`). For large embeds tensor at TP=4 that means a full serialize on the
sender + a memcpy into the shm ring buffer + a `torch.load` deserialize on EVERY worker.
This increases GPU-idle time for the prompt_embeds prefill path.

`SharedTensorHandle` replaces the tensor with a small picklable stand-in:
    - The storage is moved once to a /dev/shm file and only the rebuild handle (manager
      socket + storage name + size) is broadcast.
    - Every TP worker mmaps the same file. Zero serialize, zero per-worker copy.

Enable with `VLLM_SHM_EMBEDS=1`, requires `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
"""

import os
from contextlib import contextmanager

import torch

_ENABLED = os.environ.get("VLLM_SHM_EMBEDS", "0") == "1"


def enabled() -> bool:
    return _ENABLED


@contextmanager
def _file_system_sharing():
    """Temporarily force filename-based tensor sharing, then restore the default.

    Scoped to just the reduce_storage() call rather than mutating the process-wide
    strategy permanently: the rebuild handle is self-describing (a filename), so it
    stays valid after the strategy is restored, and we avoid making unrelated tensor
    sharing in this process inherit file_system.
    `file_system` is required here because it is 1->N safe, the default file_descriptor
    passes an fd over a socket that only ONE reader can claim, which breaks a broadcast
    to N workers.
    """
    mp = torch.multiprocessing
    prev = mp.get_sharing_strategy()
    mp.set_sharing_strategy("file_system")
    try:
        yield
    finally:
        mp.set_sharing_strategy(prev)


class SharedTensorHandle:
    """Picklable stand-in for a CPU tensor backed by shared memory.

    Pickles only the storage filename handle + shape/dtype/stride/offset metadata,
    never the tensor data. `tensor()` then rebuilds an mmap view of the shared file.
    """

    __slots__ = ("storage_fn", "storage_args", "dtype", "shape", "stride", "offset")

    def __init__(self, storage_fn, storage_args, dtype, shape, stride, offset):
        self.storage_fn = storage_fn
        self.storage_args = storage_args
        self.dtype = dtype
        self.shape = shape
        self.stride = stride
        self.offset = offset

    def tensor(self) -> torch.Tensor:
        storage = self.storage_fn(*self.storage_args)
        t = torch.empty(0, dtype=self.dtype)
        t.set_(storage, self.offset, self.shape, self.stride)
        return t


def shared_handle(t: torch.Tensor):
    """Return (SharedTensorHandle, keepalive_tensor) for a CPU tensor.

    The keepalive tensor owns the shared storage and MUST be held alive until every
    reader has rebuilt, or the shm manager unlinks the backing file early.
    """
    from torch.multiprocessing.reductions import reduce_storage

    src = t.detach().contiguous()
    if src.is_shared():
        # Tensor came in already shared (e.g. via the client->EngineCore tensor-IPC
        # hop, which uses fd-based sharing). Clone to a clean, non-shared storage so
        # reduce_storage performs a single file_system share.
        src = src.clone()
    with _file_system_sharing():
        # Create a new shared-memory file and corresponding handle tuple.
        storage_fn, storage_args = reduce_storage(src.untyped_storage())
    handle = SharedTensorHandle(
        storage_fn,
        storage_args,
        src.dtype,
        tuple(src.shape),
        tuple(src.stride()),
        src.storage_offset(),
    )
    return handle, src


def externalize_prompt_embeds(args: tuple) -> list:
    """Replace prompt_embeds tensors in execute_model `args` with shared-mem handles.

    - Mutates the SchedulerOutput in `args` in place.
    - Returns a keepalive list of the shared source tensors.
    - The caller MUST hold these alive until all workers have rebuilt (i.e. until
      the rpc response returns) so the shm manager does not unlink the files early.
    """
    if not _ENABLED:
        return []
    keepalive: list = []
    for a in args:
        new_reqs = getattr(a, "scheduled_new_reqs", None)
        if not new_reqs:
            continue
        for req in new_reqs:
            # Externalize the prompt_embeds CPU tensor, skip None / non-tensors.
            # (CUDA tensors would need the cuda-IPC path, not file_system shm.)
            match getattr(req, "prompt_embeds", None):
                case torch.Tensor() as t if not t.is_cuda:
                    handle, src = shared_handle(t)
                    # Replace the tensor with the handle to the shared memory.
                    req.prompt_embeds = handle
                    keepalive.append(src)
                case _:
                    # No prompt_embeds or not a CPU tensor, leave as-is.
                    continue
    return keepalive


def internalize_prompt_embeds(args: tuple) -> None:
    """Rebuild SharedTensorHandle stand-ins into mmap tensors (worker side)."""
    for a in args:
        new_reqs = getattr(a, "scheduled_new_reqs", None)
        if not new_reqs:
            continue
        for req in new_reqs:
            h = getattr(req, "prompt_embeds", None)
            if isinstance(h, SharedTensorHandle):
                req.prompt_embeds = h.tensor()
