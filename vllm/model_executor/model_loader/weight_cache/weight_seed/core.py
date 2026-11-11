# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight movers: copy whole storages so views keep their sharing and footprint."""

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


def acknowledge_lease(transport: Any, lease_id: str | None = None) -> None:
    """Tell the source that this lease completed successfully.

    `lease_id=None` is the legacy command for an old source without ID leases.
    """
    if transport is None:
        return
    request: dict[str, Any] = {"kind": "weights", "acknowledge": True}
    if lease_id is not None:
        request["lease_id"] = lease_id
    response = transport.request(request)
    if response is None:
        raise RuntimeError("Could not acknowledge weight-cache copy")
    if response.get("status") != "ok":
        raise RuntimeError(
            f"Source daemon rejected weight-cache acknowledgement: "
            f"{response.get('message', response)}"
        )


def cancel_lease(transport: Any, lease_id: str | None = None) -> None:
    """Cancel a lease after confirming its mirror or copy has stopped.

    This is also the explicit recovery operation after a mirror process crash.
    Cancellation never happens automatically on a timeout; callers must ensure
    the target can no longer read the source before releasing the source lease.
    `lease_id=None` uses the acknowledgement understood by old sources.
    """
    if lease_id is None:
        acknowledge_lease(transport)
        return
    if transport is None:
        return
    request: dict[str, Any] = {"kind": "weights", "cancel": True}
    if lease_id is not None:
        request["lease_id"] = lease_id
    response = transport.request(request)
    if response is None:
        raise RuntimeError("Could not cancel weight-cache copy")
    if response.get("status") != "ok":
        raise RuntimeError(
            f"Source daemon rejected weight-cache cancellation: "
            f"{response.get('message', response)}"
        )


def get_active_lease_ids(transport: Any) -> list[str]:
    """List source IDs available for explicit post-crash cancellation.

    Orchestration should confirm the mirror has stopped, then pass each ID to
    `cancel_lease`. A source send failure does not release its lease implicitly.
    """
    if transport is None:
        raise RuntimeError("Could not query active weight-cache leases")
    response = transport.request({"kind": "weights", "list_leases": True})
    if response is None:
        raise RuntimeError("Could not query active weight-cache leases")
    if response.get("status") != "ok":
        raise RuntimeError(
            f"Source daemon rejected weight-cache lease query: "
            f"{response.get('message', response)}"
        )
    payload = response.get("payload")
    lease_ids = payload.get("lease_ids") if isinstance(payload, dict) else None
    if not isinstance(lease_ids, list) or any(
        not isinstance(lease_id, str) for lease_id in lease_ids
    ):
        raise RuntimeError("Source daemon returned malformed active lease IDs")
    return lease_ids


class WeightMover(ABC):
    """One mover that copies weight storages from a source daemon."""

    name: SeedBackend

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
    """The mover for a backend; `device_index` pins the peer_ipc device."""
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
