# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk-backed weight and scale planes for TP-local HY4 Flash Engram."""

import os

import torch

from vllm.utils.gpu_sync_debug import gpu_sync_allowed


class _CheckpointPlane:
    def __init__(self, tensor: torch.Tensor):
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("Engram disk offload requires a CPU safetensors view")
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        with open("/proc/self/maps") as maps:
            for line in maps:
                fields = line.rstrip("\n").split(maxsplit=5)
                lo, hi = (int(bound, 16) for bound in fields[0].split("-"))
                if not (lo <= ptr < hi and ptr + size <= hi):
                    continue
                if len(fields) != 6 or not fields[5].endswith(".safetensors"):
                    break
                self.offset = ptr - lo + int(fields[2], 16)
                # Keep this descriptor for the life of the plane.
                self.file = open(fields[5], "rb", buffering=0)  # noqa: SIM115
                stat = os.fstat(self.file.fileno())
                major, minor = (int(part, 16) for part in fields[3].split(":"))
                if (
                    stat.st_ino != int(fields[4])
                    or (os.major(stat.st_dev), os.minor(stat.st_dev)) != (major, minor)
                    or stat.st_size < self.offset + size
                ):
                    self.file.close()
                    raise ValueError("Engram checkpoint file changed while loading")
                self.shape = tensor.shape[1:]
                self.dtype = tensor.dtype
                return
        raise ValueError("Engram disk offload requires a file-backed safetensors view")

    def read(self, ids: list[int]) -> torch.Tensor:
        rows = torch.empty((len(ids), *self.shape), dtype=self.dtype, device="cpu")
        if ids:
            byte_rows = rows.view(torch.uint8).reshape(len(ids), -1).numpy()
            row_bytes = byte_rows.shape[1]
            fd = self.file.fileno()
            for row, row_id in zip(byte_rows, ids):
                offset = self.offset + row_id * row_bytes
                if os.preadv(fd, [row], offset) != row_bytes:
                    raise ValueError(
                        f"Engram checkpoint {self.file.name} truncated at {offset}"
                    )
        return rows


class FlashDiskTable:
    """Read distinct local buckets, retaining packed FP8 bytes and row scales."""

    def __init__(self):
        self.planes: dict[str, _CheckpointPlane] = {}

    def bind(self, name: str, tensor: torch.Tensor) -> None:
        if name in self.planes:
            raise ValueError(f"Engram disk plane {name} is already loaded")
        self.planes[name] = _CheckpointPlane(tensor)

    def gather(
        self, hashes: torch.Tensor, start: int, end: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        # The eager disk path intentionally synchronizes to get this step's ids.
        with gpu_sync_allowed():
            cpu_hashes = hashes.to("cpu")
        live = (cpu_hashes >= start) & (cpu_hashes < end)
        ids, inverse = (cpu_hashes[live] - start).unique(return_inverse=True)
        compact = torch.full_like(cpu_hashes, -1)
        compact[live] = inverse.to(compact.dtype)
        row_ids = ids.tolist()
        weight = self.planes["weight"].read(row_ids)
        scale = self.planes.get("weight_scale_inv")
        scales = scale.read(row_ids) if scale is not None else None
        with gpu_sync_allowed():
            return (
                weight.to(hashes.device),
                scales.to(hashes.device) if scales is not None else None,
                compact.to(hashes.device),
            )
