# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Node-local backing storage for immutable, broadcast CPU inputs.

Each reader maps before acknowledging. The last reader unlinks the file; live
views keep its pages alive independently of the writer and the message ring.
PyTorch has no read-only tensor type, so private copy-on-write mappings protect
other readers from accidental in-place writes without copying on reads.
"""

import mmap
import os
import shutil
import struct
import tempfile

import torch

_HEADER = struct.Struct("Q")
_DATA_OFFSET = mmap.ALLOCATIONGRANULARITY


def rebuild_shared_tensor(
    path: str, shape: tuple[int, ...], dtype_str: str
) -> torch.Tensor:
    import fcntl

    with open(path, "r+b", buffering=0) as file:
        # torch.from_file closes its fd after mmap, unlike Python mmap on 3.12.
        # Thousands of live images must not consume thousands of file descriptors.
        storage = torch.from_file(
            path,
            shared=False,
            size=os.fstat(file.fileno()).st_size,
            dtype=torch.uint8,
            device="cpu",
        )
        result = storage[_DATA_OFFSET:].view(getattr(torch, dtype_str)).view(shape)
        fcntl.flock(file, fcntl.LOCK_EX)
        try:
            remaining = _HEADER.unpack(file.read(_HEADER.size))[0]
            if remaining == 0:
                raise RuntimeError("Shared input was acknowledged too many times")
            if remaining == 1:
                os.unlink(path)
            else:
                file.seek(0)
                file.write(_HEADER.pack(remaining - 1))
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)
    return result


class SharedTensorStore:
    """One writer, a fixed number of local readers, one file per tensor."""

    def __init__(self, directory: str, num_readers: int):
        if num_readers < 1:
            raise ValueError("Shared inputs require at least one reader")
        import fcntl

        directory = os.path.abspath(directory)
        self.num_readers = num_readers
        self._directory = tempfile.TemporaryDirectory(prefix="vllm-mm-", dir=directory)
        # Publish the lease only after locking: another starting executor must
        # not mistake this newly-created directory for an abandoned one.
        pending = os.path.join(self._directory.name, ".owner-pending")
        self._lease = open(pending, "xb")  # noqa: SIM115 (store-lifetime lease)
        fcntl.flock(self._lease, fcntl.LOCK_EX)
        os.rename(pending, os.path.join(self._directory.name, ".owner"))
        self._reap_abandoned(directory)

    @staticmethod
    def _reap_abandoned(directory: str) -> None:
        import fcntl

        with os.scandir(directory) as entries:
            for entry in entries:
                if not entry.name.startswith("vllm-mm-") or not entry.is_dir(
                    follow_symlinks=False
                ):
                    continue
                try:
                    with open(os.path.join(entry.path, ".owner"), "rb") as lease:
                        try:
                            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            continue
                        shutil.rmtree(entry.path)
                except FileNotFoundError:
                    # Another reaper won, or the owner has not published yet.
                    continue

    def reduce_tensor(self, tensor: torch.Tensor):
        if (
            tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or tensor.requires_grad
            or tensor.numel() * tensor.element_size() < 1024 * 1024
        ):
            return None
        try:
            raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
        except RuntimeError:
            return None
        # Write through the file API so tmpfs exhaustion raises ENOSPC rather
        # than causing SIGBUS in an mmap write. Never silently fall back to N copies.
        with tempfile.NamedTemporaryFile(
            prefix="tensor-", dir=self._directory.name, delete=False
        ) as file:
            path = file.name
            try:
                file.write(_HEADER.pack(self.num_readers))
                file.seek(_DATA_OFFSET)
                file.write(memoryview(raw))
                file.flush()
            except BaseException:
                os.unlink(path)
                raise
        return rebuild_shared_tensor, (
            path,
            tuple(tensor.shape),
            str(tensor.dtype).removeprefix("torch."),
        )

    def close(self) -> None:
        # Also removes unacknowledged files after an executor/reader failure.
        self._directory.cleanup()
        self._lease.close()
