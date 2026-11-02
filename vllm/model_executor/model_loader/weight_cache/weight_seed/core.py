# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Daemon-to-daemon weight shard seeding for the IPC weight cache.

A second replica's daemons can fill their shards from an already-loaded
replica instead of reading the checkpoint again. The two movers live in
:mod:`.peer_ipc` (CUDA IPC, same host) and :mod:`.rdma` (Mooncake
TransferEngine, cross-host); this core holds the mover interface and the
mover registry, with the manifest helpers in :mod:`.utils`.

Both move storages rather than tensors. Post-processing leaves strided views
that share one allocation (MLA's ``W_UV`` and ``W_UK_T`` are transposed
slices of ``kv_b_proj``), so the mirror copies each storage once and rebuilds
every tensor on it at the source's offset and strides. Values, layout and
aliasing then match the source, and so does the memory footprint.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.weight_seed.utils import (
    allocate_storage,
    manifest_nbytes,
    rebuild_tensors,
)

SeedBackend = Literal["peer_ipc", "rdma"]

PEER_IPC_BACKEND: SeedBackend = "peer_ipc"
RDMA_BACKEND: SeedBackend = "rdma"

logger = init_logger(__name__)


def parse_seed_backend(value: Any) -> SeedBackend:
    """Narrow a wire or config value to a known seed backend, or raise."""
    if value in (PEER_IPC_BACKEND, RDMA_BACKEND):
        return value
    raise ValueError(f"Unknown weight-cache seed backend {value!r}")


class WeightMover(ABC):
    """One mover that copies weight storages from a source daemon."""

    name: SeedBackend
    is_node_local: bool
    """Whether the mover can only reach a peer on the same host."""

    @abstractmethod
    def prepare_seed(self, storages: list[torch.UntypedStorage]) -> dict[str, Any]:
        """Publish how a mirror can read each storage, in manifest order."""

    @abstractmethod
    def copy_into(
        self,
        storages: list[Mapping[str, Any]],
        seed: Mapping[str, Any],
        buffers: list[torch.Tensor],
    ) -> None:
        """Copy every source storage into its local byte buffer."""

    def fill(
        self,
        manifest: Mapping[str, Any],
        seed: Mapping[str, Any],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Copy the source's storages and rebuild its tensors on them."""
        started = time.perf_counter()
        storages = manifest["storages"]
        buffers = [allocate_storage(storage, device) for storage in storages]
        self.copy_into(storages, seed, buffers)
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


def get_mover(name: SeedBackend, *, device_index: int | None = None) -> WeightMover:
    """The mover for a backend; ``device_index`` pins the peer_ipc device."""
    if name == PEER_IPC_BACKEND:
        from vllm.model_executor.model_loader.weight_cache.weight_seed.peer_ipc import (
            PeerIpcMover,
        )

        return PeerIpcMover(device_index)
    if name == RDMA_BACKEND:
        from vllm.model_executor.model_loader.weight_cache.weight_seed.rdma import (
            RdmaMover,
        )

        return RdmaMover()
    raise ValueError(f"Unknown weight-cache seed backend {name!r}")
