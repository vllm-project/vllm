# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded host-side encoder state retained for deferred audio alignment."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypedDict

import torch

AUDIO_ALIGNMENT_HANDLE_KEY = "_vllm_audio_alignment_handle"
DEFAULT_AUDIO_ALIGNMENT_MAX_BYTES = 16 * 1024**3


class AudioAlignmentCapture(TypedDict):
    tensors: dict[str, torch.Tensor]
    duration_s: float


@dataclass
class _Capture:
    value: AudioAlignmentCapture
    num_bytes: int
    ready: torch.cuda.Event | None = None

    def wait(self) -> None:
        if self.ready is not None:
            self.ready.synchronize()
            self.ready = None


class AudioAlignmentStore:
    """Retain one compact capture per audio, shared by requests and encoder hits.

    Encoder-cache residency is a reference: dropping an unowned capture while
    its embedding remains cached would break a later request's encoder hit.
    Request references survive completion and preemption until explicitly
    released by the offline caller. The budget includes alignment workspaces.
    """

    def __init__(self, max_bytes: int = DEFAULT_AUDIO_ALIGNMENT_MAX_BYTES):
        if max_bytes <= 0:
            raise ValueError("Audio alignment max_bytes must be positive")
        self.max_bytes = max_bytes
        self._requests: dict[str, tuple[str, ...]] = {}
        self._owners: dict[str, set[str]] = {}
        self._resident: set[str] = set()
        self._captures: dict[str, _Capture] = {}
        self._errors: dict[str, str] = {}
        self._capture_bytes = 0
        self._workspace_bytes = 0

    def register(self, handle: str, hashes: Sequence[str]) -> None:
        if not handle or not hashes:
            raise ValueError("Audio alignment requires a handle and audio hashes")
        audio_hashes = tuple(hashes)
        previous = self._requests.get(handle)
        if previous is not None:
            if previous != audio_hashes:
                raise ValueError(
                    "Audio alignment handle already identifies other audio"
                )
            return
        self._requests[handle] = audio_hashes
        for mm_hash in audio_hashes:
            self._owners.setdefault(mm_hash, set()).add(handle)

    def needs_capture(self, mm_hash: str) -> bool:
        return (
            bool(self._owners.get(mm_hash))
            and mm_hash not in self._captures
            and mm_hash not in self._errors
        )

    def mark_cached(self, mm_hash: str) -> None:
        """Record encoding even when an earlier retained capture is reused."""
        if (
            mm_hash in self._captures
            or mm_hash in self._errors
            or self._owners.get(mm_hash)
        ):
            self._resident.add(mm_hash)

    def put(
        self,
        mm_hash: str,
        tensors: Mapping[str, torch.Tensor],
        duration_s: float,
    ) -> bool:
        """Copy a capture into independent host buffers on the encoder stream.

        A budget refusal is recorded for the owning requests instead of raising
        from the encoder and terminating unrelated generation requests.
        """
        if mm_hash in self._captures:
            self._resident.add(mm_hash)
            return True
        if not self._owners.get(mm_hash):
            return False
        self._resident.add(mm_hash)
        num_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        if self._capture_bytes + self._workspace_bytes + num_bytes > self.max_bytes:
            self._errors[mm_hash] = "capture_budget_exceeded"
            return False

        cuda_devices = {t.device for t in tensors.values() if t.is_cuda}
        if len(cuda_devices) > 1:
            raise ValueError("One audio alignment capture must use one CUDA device")
        host_tensors = {
            name: torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=tensor.is_cuda,
            )
            for name, tensor in tensors.items()
        }
        ready = None
        try:
            for name, tensor in tensors.items():
                host_tensors[name].copy_(tensor.detach(), non_blocking=tensor.is_cuda)
        except BaseException:
            # A partially copied pinned buffer must outlive its queued copy.
            if cuda_devices:
                device = next(iter(cuda_devices))
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(device))
                ready.synchronize()
            raise
        if cuda_devices:
            device = next(iter(cuda_devices))
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))
        self._captures[mm_hash] = _Capture(
            value={"tensors": host_tensors, "duration_s": float(duration_s)},
            num_bytes=num_bytes,
            ready=ready,
        )
        self._errors.pop(mm_hash, None)
        self._capture_bytes += num_bytes
        return True

    def _hash_for_handle(self, handle: str) -> str | None:
        hashes = self._requests.get(handle)
        if hashes is None:
            return None
        if len(hashes) != 1:
            raise ValueError("Audio alignment currently supports one audio per request")
        return hashes[0]

    def get(self, handle: str) -> AudioAlignmentCapture | None:
        mm_hash = self._hash_for_handle(handle)
        capture = self._captures.get(mm_hash) if mm_hash is not None else None
        if capture is None:
            return None
        capture.wait()
        return capture.value

    def error(self, handle: str) -> str | None:
        mm_hash = self._hash_for_handle(handle)
        return self._errors.get(mm_hash) if mm_hash is not None else None

    def release(self, handles: Sequence[str]) -> None:
        for handle in handles:
            for mm_hash in self._requests.pop(handle, ()):
                owners = self._owners.get(mm_hash)
                if owners is not None:
                    owners.discard(handle)
                    if not owners:
                        del self._owners[mm_hash]
                self._discard_unreferenced(mm_hash)

    def evict(self, mm_hash: str) -> None:
        self._resident.discard(mm_hash)
        self._discard_unreferenced(mm_hash)

    def _discard_unreferenced(self, mm_hash: str) -> None:
        if mm_hash in self._resident or self._owners.get(mm_hash):
            return
        capture = self._captures.pop(mm_hash, None)
        if capture is not None:
            capture.wait()
            self._capture_bytes -= capture.num_bytes
        self._errors.pop(mm_hash, None)

    @contextmanager
    def reserve_workspace(self, num_bytes: int) -> Iterator[None]:
        if num_bytes < 0:
            raise ValueError("Audio alignment workspace size must be nonnegative")
        if self._capture_bytes + self._workspace_bytes + num_bytes > self.max_bytes:
            raise ValueError("Audio alignment capture and workspace exceed the budget")
        self._workspace_bytes += num_bytes
        try:
            yield
        finally:
            self._workspace_bytes -= num_bytes

    def reset(self) -> None:
        for capture in self._captures.values():
            capture.wait()
        self._captures.clear()
        self._requests.clear()
        self._owners.clear()
        self._resident.clear()
        self._errors.clear()
        self._capture_bytes = 0

    def stats(self) -> dict[str, int]:
        return {
            "captures": len(self._captures),
            "handles": len(self._requests),
            "encoder_cache_entries": len(self._resident),
            "bytes": self._capture_bytes,
            "workspace_bytes": self._workspace_bytes,
            "limit_bytes": self.max_bytes,
        }
