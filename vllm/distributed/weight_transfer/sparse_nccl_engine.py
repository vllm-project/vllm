# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse NCCL weight transfer engine.

Sparse patches use checkpoint names, shapes, and flat indices. The model's native
weight loader maps them to rank-local runtime parameters, including TP shards and
packed parameters.
"""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
from typing_extensions import Self

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    WeightTransferEngine,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.nccl_common import (
    NCCLWeightTransferInitInfo,
    worker_init_payload,
    worker_init_process_group,
)
from vllm.distributed.weight_transfer.nccl_common import (
    trainer_init as open_trainer_endpoint,
)
from vllm.model_executor.model_loader.checkpoint_weight_patch import (
    CheckpointWeightPatch,
    load_checkpoint_weight_patches,
)

__all__ = [
    "SparseWeightPatch",
    "SparseNCCLTrainerInitInfo",
    "SparseNCCLWeightTransferUpdateInfo",
    "SparseNCCLWeightTransferEngine",
    "SparseNCCLTrainerWeightTransferEngine",
]


@dataclass
class SparseWeightPatch:
    """A sparse patch in checkpoint coordinates."""

    name: str
    indices: torch.Tensor
    values: torch.Tensor
    full_shape: tuple[int, ...]
    """Full checkpoint shape."""


@dataclass
class SparseNCCLTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for the sparse NCCL weight transfer backend.

    Same rendezvous shape as the dense NCCL backend (the sender opens its
    endpoint as NCCL rank 0), but with no packed wire params: sparse transfers
    are never packed. `backend` is the factory dispatch key."""

    backend: ClassVar[str] = "sparse_nccl"

    master_address: str
    master_port: int
    world_size: int


@dataclass
class SparseNCCLWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for the sparse NCCL weight transfer backend."""

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    num_updates_list: list[int]
    """Number of sparse entries to receive for each parameter in ``names``."""

    def __post_init__(self) -> None:
        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {len(self.names)}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: "
                f"got {len(self.shapes)} and {len(self.names)}"
            )
        if len(self.num_updates_list) == 0:
            raise ValueError("`num_updates_list` cannot be empty for sparse updates")
        if len(self.num_updates_list) != num_params:
            raise ValueError(
                f"`num_updates_list` should be of the same size as `names`: "
                f"got {len(self.num_updates_list)} and {len(self.names)}"
            )
        if any(num_updates < 0 for num_updates in self.num_updates_list):
            raise ValueError("Sparse `num_updates_list` entries must be non-negative")


class SparseNCCLWeightTransferEngine(
    WeightTransferEngine[NCCLWeightTransferInitInfo, SparseNCCLWeightTransferUpdateInfo]
):
    """
    Sparse weight transfer engine using NCCL.

    Receives checkpoint-coordinate patches broadcast from the trainer and applies
    them through the model's native weight loader. Sparse updates modify initialized
    model tensors in place, so the layerwise reload lifecycle is not used.
    """

    init_info_cls = NCCLWeightTransferInitInfo
    update_info_cls = SparseNCCLWeightTransferUpdateInfo
    supports_draft_weight_update = False

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self.model_update_group: PyNcclCommunicator | None = None

    def init_transfer_engine(self, init_info: NCCLWeightTransferInitInfo) -> None:
        """Initialize the NCCL process group with the trainer."""
        self.model_update_group = worker_init_process_group(
            init_info, self.parallel_config
        )

    def start_weight_update(self) -> None:
        """No-op: sparse patches are applied in place, no layerwise reload."""
        pass

    def finish_weight_update(self) -> None:
        """No-op: sparse patches are applied in place, no layerwise reload."""
        pass

    def receive_weights(self, update_info: SparseNCCLWeightTransferUpdateInfo) -> None:
        """Receive sparse flat-index patches from the trainer and apply them."""
        if self.model_update_group is None:
            raise RuntimeError(
                "NCCL weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )

        # Receive on the communicator's device rather than relying on the
        # ambient current device in the RPC thread.
        device = self.model_update_group.device
        with torch.accelerator.device_index(device.index):
            stream = torch.cuda.current_stream(device=device)
            patches = []
            for name, dtype_name, shape, num_updates in zip(
                update_info.names,
                update_info.dtype_names,
                update_info.shapes,
                update_info.num_updates_list,
                strict=True,
            ):
                dtype = getattr(torch, dtype_name)
                indices = torch.empty(num_updates, dtype=torch.int32, device=device)
                values = torch.empty(num_updates, dtype=dtype, device=device)
                self.model_update_group.broadcast(indices, src=0, stream=stream)
                self.model_update_group.broadcast(values, src=0, stream=stream)
                patches.append(
                    CheckpointWeightPatch(
                        name=name,
                        shape=tuple(shape),
                        dtype=dtype,
                        values=values,
                        indices=indices,
                    )
                )
            load_checkpoint_weight_patches(self.model, patches)

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            self.model_update_group = None


class SparseNCCLTrainerWeightTransferEngine(
    TrainerWeightTransferEngine[SparseNCCLTrainerInitInfo]
):
    """Trainer-side sparse NCCL weight transfer engine.

    Broadcasts flat-index (indices, values) patches from NCCL rank 0 while the
    inference-side `update_weights` runs concurrently on a side thread (the
    worker's recvs rendezvous inside the same NCCL broadcasts). `send_weights`
    owns a complete one-shot lifecycle. RL infrastructure that owns the generic
    client lifecycle can call `send_weight_chunk` between one start and finish.

    Sparse patches differ every round, so they are not a stable `WeightSource`:
    the engine takes no `source`, and patches are passed directly to the send
    methods. An empty patch list is a no-op.

    Only the designated trainer sender joins the transfer group; other trainer
    ranks skip sparse sends.
    """

    init_info_cls = SparseNCCLTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
        is_sender: bool = True,
    ) -> None:
        # Sparse is a delta backend: it takes per-round patches via
        # send_weights, so a `source` would silently never be sent. The
        # parameter exists only to match the base/factory signature.
        if source is not None:
            raise ValueError(
                "Sparse NCCL weight transfer takes no WeightSource; pass each "
                "round's patches to send_weights(patches) instead."
            )
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.model_update_group: PyNcclCommunicator | None = None

    @classmethod
    def trainer_init(
        cls,
        init_info: SparseNCCLTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        engine = cls(client=client, source=source, is_sender=init_info.is_sender)
        if not engine.is_sender:
            return engine

        # Workers sit at rank_offset 1, after the single trainer sender rank 0.
        # Sparse transfers are never packed, so the worker keeps the unpacked
        # defaults on its init info.
        worker_init_info = NCCLWeightTransferInitInfo(
            master_address=init_info.master_address,
            master_port=init_info.master_port,
            rank_offset=1,
            world_size=init_info.world_size,
        )

        # The inference workers block inside init_weight_transfer_engine waiting
        # for the NCCL rendezvous, so we kick that off on a side thread while we
        # open the trainer endpoint (rank 0); both sides must rendezvous together.
        with ThreadPoolExecutor(max_workers=1) as exe:
            future = exe.submit(
                engine.client.init_weight_transfer_engine,
                worker_init_payload(worker_init_info),
            )
            # Open the trainer endpoint as NCCL rank 0 on the current device
            # (the init info satisfies the helper's rendezvous protocol).
            engine.model_update_group = open_trainer_endpoint(init_info)
            future.result()  # surface any inference-side init error

        return engine

    def send_weights(self, patches: Iterable[SparseWeightPatch] | None = None) -> None:
        """Broadcast one sparse update through a one-shot lifecycle."""
        patches = self._prepare_patches(patches)
        if not patches:
            return

        self.client.start_weight_update()
        self._broadcast_chunk(patches)
        self.client.finish_weight_update()

    def send_weight_chunk(
        self, patches: Iterable[SparseWeightPatch] | None = None
    ) -> None:
        """Broadcast one chunk inside a caller-owned weight update lifecycle."""
        patches = self._prepare_patches(patches)
        if not patches:
            return
        self._broadcast_chunk(patches)

    def _prepare_patches(
        self, patches: Iterable[SparseWeightPatch] | None
    ) -> list[SparseWeightPatch]:
        if not self.is_sender:
            return []

        prepared = list(patches) if patches is not None else []
        if not prepared:
            return []
        if self.model_update_group is None:
            raise RuntimeError("trainer_init() must be called before sending weights.")
        for patch in prepared:
            self._validate_patch(patch)
        return prepared

    def _broadcast_chunk(self, patches: list[SparseWeightPatch]) -> None:
        assert self.model_update_group is not None
        device = self.model_update_group.device
        device_context = (
            torch.accelerator.device_index(device.index)
            if device.type != "cpu"
            else nullcontext()
        )
        with device_context:
            update_info = SparseNCCLWeightTransferUpdateInfo(
                names=[patch.name for patch in patches],
                dtype_names=[
                    str(patch.values.dtype).split(".")[-1] for patch in patches
                ],
                shapes=[list(patch.full_shape) for patch in patches],
                num_updates_list=[patch.indices.numel() for patch in patches],
            )

            # update_weights (workers receive) must run concurrently with the
            # trainer-side broadcasts — both rendezvous inside the same NCCL calls.
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(
                    self.client.update_weights, asdict(update_info)
                )
                # Surface an RPC that failed before the worker entered NCCL rather
                # than broadcasting to a peer that will never receive.
                if future.done():
                    future.result()
                stream = torch.cuda.current_stream()
                for patch in patches:
                    self.model_update_group.broadcast(
                        patch.indices, src=0, stream=stream
                    )
                    self.model_update_group.broadcast(
                        patch.values, src=0, stream=stream
                    )
                future.result()  # surface inference-side errors
            finally:
                # A failed broadcast can leave the RPC blocked in the matching
                # receive. Waiting for that thread would hide the original error.
                executor.shutdown(wait=False, cancel_futures=True)
            self._post_send_sync()

    @staticmethod
    def _validate_patch(patch: SparseWeightPatch) -> None:
        """Reject a malformed patch before starting the NCCL transfer."""
        if patch.full_shape is None:
            raise ValueError(f"Sparse patch requires full_shape: {patch.name}")
        if patch.indices.dtype != torch.int32:
            raise ValueError(
                f"Sparse weight updates require int32 indices: {patch.name}"
            )
        if patch.indices.ndim != 1 or patch.values.ndim != 1:
            raise ValueError(
                f"Sparse weight patches must be 1D flattened updates: {patch.name}"
            )
        if patch.indices.numel() != patch.values.numel():
            raise ValueError(
                f"`indices` and `values` must have matching lengths for {patch.name}"
            )

    def _post_send_sync(self) -> None:
        """Wait for the broadcasts to land before returning, so a caller may
        rebuild or free the patch tensors as soon as a send method returns rather
        than relying on same-stream ordering. See
        `NCCLTrainerWeightTransferEngine._post_send_sync` for why there is no
        cross-rank barrier."""
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()

    def shutdown(self) -> None:
        self.model_update_group = None
