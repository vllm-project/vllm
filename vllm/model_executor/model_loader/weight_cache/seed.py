# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Daemon-to-daemon seeding for the IPC weight cache."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch

from vllm.logger import init_logger

PEER_IPC_SEED_SOURCE = "peer_ipc"
RDMA_SEED_SOURCE = "rdma"
logger = init_logger(__name__)


def build_manifest(entries: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Describe every exported tensor so a mirror can allocate it."""
    return {
        name: {
            "shape": list(entry.shape),
            "dtype": entry.dtype.removeprefix("torch."),
            "is_param": entry.kind == "param",
        }
        for name, entry in entries.items()
    }


def manifest_dtype(dtype_name: str) -> torch.dtype:
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(
            f"Weight-cache seed manifest uses unsupported dtype {dtype_name!r}"
        )
    return dtype


def manifest_nbytes(manifest: Mapping[str, Mapping[str, Any]]) -> int:
    total = 0
    for metadata in manifest.values():
        numel = 1
        for dimension in metadata["shape"]:
            numel *= dimension
        total += numel * manifest_dtype(metadata["dtype"]).itemsize
    return total


class WeightCacheSeedSource(ABC):
    name: str

    @abstractmethod
    def prepare_seed(
        self,
        entries: Mapping[str, Any],
        state_tensors: Mapping[str, torch.Tensor],
        gpu_id: int | None,
    ) -> dict[str, Any]:
        """Prepare source-side transfer metadata."""

    @abstractmethod
    def fill(
        self,
        manifest: Mapping[str, Mapping[str, Any]],
        seed: Mapping[str, Any],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Copy source weights into the mirror and return local tensors."""

    def close(self) -> None:
        return None


class PeerIpcSeedSource(WeightCacheSeedSource):
    """Copy tensors through CUDA IPC handles on the same host."""

    name = PEER_IPC_SEED_SOURCE

    def prepare_seed(self, entries, state_tensors, gpu_id):
        return {
            "backend": self.name,
            "entries": dict(entries),
            "source_device_index": torch.accelerator.current_device_index(),
            "source_gpu_id": gpu_id,
        }

    def fill(self, manifest, seed, device):
        source_index = seed["source_device_index"]
        source_gpu_id = seed.get("source_gpu_id")
        if source_gpu_id is not None:
            from vllm.model_executor.model_loader.weight_cache.protocol import (
                get_physical_device_id,
            )

            if get_physical_device_id(source_index) != source_gpu_id:
                raise RuntimeError(
                    "Source CUDA IPC handle uses a device that is not visible at "
                    "the same logical index in the mirror; use matching "
                    "CUDA_VISIBLE_DEVICES mappings."
                )
        source_entries = seed["entries"]
        result: dict[str, torch.Tensor] = {}
        by_entry: dict[int, torch.Tensor] = {}
        started = time.perf_counter()
        for name, metadata in manifest.items():
            entry = source_entries.get(name)
            if entry is None:
                raise RuntimeError(f"Source daemon omitted manifest entry {name!r}")
            cached = by_entry.get(id(entry))
            if cached is not None:
                result[name] = cached
                continue
            source = entry.rebuild(source_index, retarget=False)
            expected_dtype = manifest_dtype(metadata["dtype"])
            expected_shape = torch.Size(metadata["shape"])
            if source.shape != expected_shape or source.dtype != expected_dtype:
                raise RuntimeError(
                    f"Source entry {name!r} does not match its manifest: "
                    f"{source.shape}/{source.dtype} vs "
                    f"{expected_shape}/{expected_dtype}"
                )
            target = torch.empty_like(source, device=device)
            target.copy_(source, non_blocking=True)
            result[name] = target
            by_entry[id(entry)] = target
        torch.accelerator.synchronize()
        total_bytes = manifest_nbytes(manifest)
        elapsed = time.perf_counter() - started
        logger.info(
            "Copied %.2f GiB from peer weight-cache daemon in %.2fs (%.1f GiB/s)",
            total_bytes / 1024**3,
            elapsed,
            total_bytes / 1024**3 / max(elapsed, 1e-9),
        )
        return result


class RdmaSeedSource(WeightCacheSeedSource):
    """Pull weights through the Mooncake TransferEngine."""

    name = RDMA_SEED_SOURCE

    def __init__(self, protocol: str = "rdma"):
        self.protocol = protocol
        self._engine = None
        self._source_ptrs: list[int] = []
        self._source_session: str | None = None

    def _get_engine(self):
        if self._engine is None:
            from mooncake.engine import TransferEngine

            from vllm.utils.network_utils import get_ip

            engine = TransferEngine()
            if engine.initialize(get_ip(), "P2PHANDSHAKE", self.protocol, "") != 0:
                raise RuntimeError("Mooncake TransferEngine initialization failed")
            self._engine = engine
            self._source_session = f"{get_ip()}:{engine.get_rpc_port()}"
        return self._engine

    def prepare_seed(self, entries, state_tensors, gpu_id):
        engine = self._get_engine()
        regions: dict[str, tuple[int, int]] = {}
        unique: dict[int, int] = {}
        for name in entries:
            tensor = state_tensors[name]
            if not tensor.is_contiguous():
                raise RuntimeError(f"Weight-cache tensor {name!r} is not contiguous")
            pointer, length = tensor.data_ptr(), tensor.nbytes
            regions[name] = (pointer, length)
            unique[pointer] = max(unique.get(pointer, 0), length)
        pointers, lengths = list(unique), list(unique.values())
        if engine.batch_register_memory(pointers, lengths) != 0:
            raise RuntimeError("Mooncake failed to register weight-cache tensors")
        self._source_ptrs = pointers
        return {
            "backend": self.name,
            "session": self._source_session,
            "regions": regions,
        }

    def fill(self, manifest, seed, device):
        from mooncake.engine import TransferEngine

        from vllm.utils.network_utils import get_ip

        session = seed.get("session")
        if not session:
            raise RuntimeError("Source daemon did not publish a Mooncake session")
        engine = TransferEngine()
        if engine.initialize(get_ip(), "P2PHANDSHAKE", self.protocol, "") != 0:
            raise RuntimeError("Mooncake TransferEngine initialization failed")
        result: dict[str, torch.Tensor] = {}
        local_ptrs: list[int] = []
        remote_ptrs: list[int] = []
        lengths: list[int] = []
        by_region: dict[tuple[int, int, str], torch.Tensor] = {}
        for name, metadata in manifest.items():
            remote_ptr, remote_length = seed["regions"][name]
            dtype = manifest_dtype(metadata["dtype"])
            shape = torch.Size(metadata["shape"])
            key = (remote_ptr, remote_length, str(shape))
            if key in by_region:
                result[name] = by_region[key]
                continue
            target = torch.empty(shape, dtype=dtype, device=device)
            if target.nbytes != remote_length:
                raise RuntimeError(f"Weight-cache region size mismatch for {name!r}")
            result[name] = by_region[key] = target
            local_ptrs.append(target.data_ptr())
            remote_ptrs.append(remote_ptr)
            lengths.append(target.nbytes)
        if engine.batch_register_memory(local_ptrs, lengths) != 0:
            raise RuntimeError("Mooncake failed to register mirror tensors")
        try:
            ret = engine.batch_transfer_sync_read(
                session, local_ptrs, remote_ptrs, lengths
            )
        finally:
            engine.batch_unregister_memory(local_ptrs)
        if ret != 0:
            raise RuntimeError(f"Mooncake weight-cache read failed with status {ret}")
        return result

    def close(self) -> None:
        if self._engine is not None and self._source_ptrs:
            self._engine.batch_unregister_memory(self._source_ptrs)
            self._source_ptrs = []


def get_seed_source(name: str) -> WeightCacheSeedSource:
    if name == PEER_IPC_SEED_SOURCE:
        return PeerIpcSeedSource()
    if name == RDMA_SEED_SOURCE:
        return RdmaSeedSource()
    raise ValueError(f"Unknown weight-cache seed backend {name!r}")
