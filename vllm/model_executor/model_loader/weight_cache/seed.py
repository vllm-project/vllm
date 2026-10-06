# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Daemon-to-daemon seeding for the IPC weight cache.

A second replica's daemons can fill their shards from an already-loaded
replica instead of reading the checkpoint again. Two movers are available:
``peer_ipc`` copies through CUDA IPC handles on the same host, and ``rdma``
pulls through the Mooncake TransferEngine across hosts.

Both move storages rather than tensors. Post-processing leaves strided views
that share one allocation (MLA's ``W_UV`` and ``W_UK_T`` are transposed
slices of ``kv_b_proj``), so the mirror copies each storage once and rebuilds
every tensor on it at the source's offset and strides. Values, layout and
aliasing then match the source, and so does the memory footprint.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.protocol import TensorEntry
from vllm.platforms import current_platform

PEER_IPC_SEED_SOURCE = "peer_ipc"
RDMA_SEED_SOURCE = "rdma"

logger = init_logger(__name__)


def build_manifest(
    tensors: Mapping[str, torch.Tensor], kinds: Mapping[str, str]
) -> tuple[dict[str, Any], list[torch.UntypedStorage]]:
    """Describe ``tensors`` as views over their unique storages.

    Returns:
        The JSON-safe manifest a mirror rebuilds the tensors from, and the
        storages it indexes, in order, for the mover to publish.

    """
    storages: list[torch.UntypedStorage] = []
    index_by_storage: dict[tuple[torch.device, int], int] = {}
    views: dict[str, dict[str, Any]] = {}
    for name, tensor in tensors.items():
        storage = tensor.untyped_storage()
        index = index_by_storage.setdefault(
            (tensor.device, storage.data_ptr()), len(storages)
        )
        if index == len(storages):
            storages.append(storage)
        views[name] = {
            "storage": index,
            "offset": tensor.storage_offset(),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "is_param": kinds[name] == "param",
        }
    manifest = {
        "storages": [
            {"nbytes": storage.nbytes(), "device": storage.device.type}
            for storage in storages
        ],
        "tensors": views,
    }
    return manifest, storages


def manifest_dtype(dtype_name: str) -> torch.dtype:
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(
            f"Weight-cache seed manifest uses unsupported dtype {dtype_name!r}"
        )
    return dtype


def manifest_nbytes(manifest: Mapping[str, Any]) -> int:
    return sum(storage["nbytes"] for storage in manifest["storages"])


def storage_bytes(storage: torch.UntypedStorage) -> torch.Tensor:
    """A flat byte tensor covering the whole storage."""
    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)


def _view_extent(view: Mapping[str, Any], itemsize: int) -> int:
    """Bytes from the start of its storage that a strided view reaches."""
    if 0 in view["shape"]:
        return 0
    last = view["offset"] + sum(
        (size - 1) * stride
        for size, stride in zip(view["shape"], view["stride"], strict=True)
    )
    return (last + 1) * itemsize


def rebuild_tensors(
    manifest: Mapping[str, Any], buffers: list[torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Recreate every manifest tensor on the mirror's copies of the storages.

    ``set_`` silently grows a storage too small for the view, which would
    move the buffer away from the other views on it, so a manifest from a
    remote peer is checked against the buffers first.
    """
    tensors: dict[str, torch.Tensor] = {}
    for name, view in manifest["tensors"].items():
        index = view["storage"]
        if not isinstance(index, int) or not 0 <= index < len(buffers):
            raise RuntimeError(f"Seed manifest entry {name!r} names no storage")
        buffer = buffers[index]
        dtype = manifest_dtype(view["dtype"])
        if _view_extent(view, dtype.itemsize) > buffer.nbytes:
            raise RuntimeError(f"Seed manifest entry {name!r} overruns its storage")
        tensors[name] = torch.empty(0, dtype=dtype, device=buffer.device).set_(
            buffer.untyped_storage(), view["offset"], view["shape"], view["stride"]
        )
    return tensors


def _allocate_storage(storage: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    """A fresh byte buffer for one manifest storage, on the source's device type."""
    target = torch.device("cpu") if storage["device"] == "cpu" else device
    return torch.empty(storage["nbytes"], dtype=torch.uint8, device=target)


class WeightCacheSeedSource(ABC):
    name: str
    is_node_local: bool
    """Whether the mover can only reach a peer on the same host."""

    @abstractmethod
    def prepare_seed(
        self,
        storages: list[torch.UntypedStorage],
        gpu_uuid: str | None,
    ) -> dict[str, Any]:
        """Publish how a mirror can read each storage, in manifest order."""

    @abstractmethod
    def copy_storages(
        self,
        storages: list[Mapping[str, Any]],
        seed: Mapping[str, Any],
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Copy every source storage into a fresh local byte buffer."""

    def fill(
        self,
        manifest: Mapping[str, Any],
        seed: Mapping[str, Any],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Copy the source's storages and rebuild its tensors on them."""
        started = time.perf_counter()
        buffers = self.copy_storages(manifest["storages"], seed, device)
        if len(buffers) != len(manifest["storages"]):
            raise RuntimeError("Seed backend did not copy every manifest storage")
        tensors = rebuild_tensors(manifest, buffers)
        total_gib = manifest_nbytes(manifest) / 1024**3
        elapsed = time.perf_counter() - started
        logger.info(
            "Mirrored %.2f GiB in %d storage(s) over %s in %.2fs (%.1f GiB/s)",
            total_gib,
            len(buffers),
            self.name,
            elapsed,
            total_gib / max(elapsed, 1e-9),
        )
        return tensors

    def release(self) -> None:
        """Drop source-side state for the weights the daemon is releasing."""
        return None

    def close(self) -> None:
        self.release()


class PeerIpcSeedSource(WeightCacheSeedSource):
    """Copy storages through CUDA IPC handles on the same host."""

    name = PEER_IPC_SEED_SOURCE
    is_node_local = True

    def prepare_seed(self, storages, gpu_uuid):
        return {
            "backend": self.name,
            "storages": [
                TensorEntry.from_tensor(storage_bytes(storage), "buffer")
                for storage in storages
            ],
            "source_device_index": torch.accelerator.current_device_index(),
            "source_gpu_uuid": gpu_uuid,
        }

    def copy_storages(self, storages, seed, device):
        source_index = seed["source_device_index"]
        source_gpu_uuid = seed.get("source_gpu_uuid")
        # The handle names a device by index, so the mirror must see the same
        # physical GPU at that index or it would open a handle on the wrong
        # device.
        if (
            source_gpu_uuid is not None
            and current_platform.get_device_uuid(source_index) != source_gpu_uuid
        ):
            raise RuntimeError(
                f"Source CUDA IPC handle names device index {source_index}, "
                "which is a different physical GPU here; give the source and "
                "the mirror matching CUDA_VISIBLE_DEVICES mappings."
            )
        entries = seed["storages"]
        if len(entries) != len(storages):
            raise RuntimeError("Source daemon omitted manifest storages")
        buffers: list[torch.Tensor] = []
        for storage, entry in zip(storages, entries):
            source = entry.rebuild(source_index, retarget=False)
            if source.nbytes != storage["nbytes"]:
                raise RuntimeError(
                    f"Source storage of {source.nbytes} bytes does not match "
                    f"its manifest size {storage['nbytes']}"
                )
            buffer = _allocate_storage(storage, device)
            buffer.copy_(source, non_blocking=True)
            buffers.append(buffer)
        torch.accelerator.synchronize()
        return buffers


class RdmaSeedSource(WeightCacheSeedSource):
    """Pull storages through the Mooncake TransferEngine.

    The seed it publishes is plain JSON (a session string and integer
    regions), so it is the only mover usable on the remote control plane.
    """

    name = RDMA_SEED_SOURCE
    is_node_local = False

    def __init__(self, protocol: str = "rdma"):
        self.protocol = protocol
        self._engine = None
        self._source_session: str | None = None
        # Mooncake refuses to register memory twice, so each storage is
        # registered once and serves every later mirror.
        self._registered: dict[int, int] = {}

    def _initialize_engine(self):
        from mooncake.engine import TransferEngine

        from vllm.utils.network_utils import get_ip

        engine = TransferEngine()
        if engine.initialize(get_ip(), "P2PHANDSHAKE", self.protocol, "") != 0:
            raise RuntimeError("Mooncake TransferEngine initialization failed")
        return engine, f"{get_ip()}:{engine.get_rpc_port()}"

    def _get_engine(self):
        if self._engine is None:
            self._engine, self._source_session = self._initialize_engine()
        return self._engine

    def prepare_seed(self, storages, gpu_uuid):
        engine = self._get_engine()
        regions = [[storage.data_ptr(), storage.nbytes()] for storage in storages]
        new = [
            (pointer, nbytes)
            for pointer, nbytes in regions
            if nbytes and pointer not in self._registered
        ]
        if new:
            pointers, lengths = (list(column) for column in zip(*new))
            if engine.batch_register_memory(pointers, lengths) != 0:
                raise RuntimeError("Mooncake failed to register weight-cache storages")
            self._registered.update(new)
        return {
            "backend": self.name,
            "session": self._source_session,
            "regions": regions,
        }

    def copy_storages(self, storages, seed, device):
        session = seed.get("session")
        if not session:
            raise RuntimeError("Source daemon did not publish a Mooncake session")
        regions = seed.get("regions")
        if not isinstance(regions, list) or len(regions) != len(storages):
            raise RuntimeError("Source daemon published no region per storage")
        buffers = [_allocate_storage(storage, device) for storage in storages]
        transfers = []
        for buffer, (remote_pointer, remote_nbytes) in zip(buffers, regions):
            if remote_nbytes != buffer.nbytes:
                raise RuntimeError("Weight-cache storage size mismatch")
            if buffer.nbytes:
                transfers.append((buffer.data_ptr(), remote_pointer, buffer.nbytes))
        if not transfers:
            return buffers
        local_pointers, remote_pointers, lengths = (
            list(column) for column in zip(*transfers)
        )
        engine, _ = self._initialize_engine()
        if engine.batch_register_memory(local_pointers, lengths) != 0:
            raise RuntimeError("Mooncake failed to register mirror storages")
        try:
            ret = engine.batch_transfer_sync_read(
                session, local_pointers, remote_pointers, lengths
            )
        finally:
            engine.batch_unregister_memory(local_pointers)
        if ret != 0:
            raise RuntimeError(f"Mooncake weight-cache read failed with status {ret}")
        return buffers

    def release(self) -> None:
        if self._engine is not None and self._registered:
            self._engine.batch_unregister_memory(list(self._registered))
            self._registered.clear()


def get_seed_source(name: str) -> WeightCacheSeedSource:
    if name == PEER_IPC_SEED_SOURCE:
        return PeerIpcSeedSource()
    if name == RDMA_SEED_SOURCE:
        return RdmaSeedSource()
    raise ValueError(f"Unknown weight-cache seed backend {name!r}")
