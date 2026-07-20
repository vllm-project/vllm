# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse NCCL weight transfer engine.

A standalone engine (not a subclass of `NCCLWeightTransferEngine`) for applying
sparse, flat-index weight patches in place. It shares only NCCL process-group
initialization with the dense engine (via `nccl_common`); the update path
applies index/value patches directly to existing model parameters and never runs
layerwise reload.

MVP limitations:
* TP=1 and PP=1 only
* uses runtime/kernel-format parameter names
* not composable with checkpoint-format or packed updates
"""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, ClassVar

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
    trainer_init,
    worker_init_process_group,
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
    """A sparse in-place patch for one existing parameter."""

    name: str
    indices: torch.Tensor
    values: torch.Tensor
    full_shape: tuple[int, ...] | None = None
    """Full shape of the patched parameter. Required when the patch is sent
    via `SparseNCCLTrainerWeightTransferEngine` (it ships in the per-round
    update info); the worker-side apply path does not read it."""


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

    Receives flat-index (indices, values) patches broadcast from the trainer
    (rank 0) and applies them in place to existing model parameters. Weights are
    applied directly without layerwise reload, so `start_weight_update` and
    `finish_weight_update` are no-ops.
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
        if self.parallel_config.world_size != 1:
            raise NotImplementedError(
                "Sparse weight updates currently require TP=1 and PP=1"
            )

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

        # Use the worker's assigned device rather than the ambient current
        # device: the receive path is no longer wrapped in
        # `with torch.device(self.device)` by the caller, so the current device
        # is not guaranteed to match self.device. The recv buffers must live on
        # the same device as the NCCL communicator (created on self.device).
        device = self.device
        for name, dtype_name, num_updates in zip(
            update_info.names,
            update_info.dtype_names,
            update_info.num_updates_list,
        ):
            dtype = getattr(torch, dtype_name)
            indices = torch.empty(num_updates, dtype=torch.int32, device=device)
            values = torch.empty(num_updates, dtype=dtype, device=device)
            self.model_update_group.broadcast(
                indices, src=0, stream=torch.cuda.current_stream()
            )
            self.model_update_group.broadcast(
                values, src=0, stream=torch.cuda.current_stream()
            )
            self._apply_patch(
                SparseWeightPatch(name=name, indices=indices, values=values)
            )
            del indices
            del values

    def _apply_patch(self, patch: SparseWeightPatch) -> None:
        """Apply a single sparse flat-index patch to an existing model param."""
        param = self.model.get_parameter(patch.name)
        if not param.data.is_contiguous():
            raise NotImplementedError(
                "Sparse weight updates currently require contiguous params: "
                f"{patch.name}"
            )
        if patch.indices.dtype != torch.int32:
            raise ValueError(
                f"Sparse weight updates currently require int32 indices: {patch.name}"
            )
        if patch.indices.ndim != 1 or patch.values.ndim != 1:
            raise ValueError(
                f"Sparse weight patches must be 1D flattened updates: {patch.name}"
            )
        if patch.indices.numel() != patch.values.numel():
            raise ValueError(
                f"`indices` and `values` must have matching lengths for {patch.name}"
            )
        if patch.values.dtype != param.dtype:
            raise ValueError(
                f"Sparse values dtype {patch.values.dtype} does not match "
                f"parameter dtype {param.dtype} for {patch.name}"
            )

        flat_param = param.data.view(-1)
        flat_param.index_copy_(
            0,
            patch.indices.to(device=flat_param.device, dtype=torch.long),
            patch.values.to(device=flat_param.device),
        )

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            self.model_update_group = None

    @staticmethod
    def trainer_send_weights(*args: Any, **kwargs: Any) -> None:
        """Removed. Use the stateful `SparseNCCLTrainerWeightTransferEngine`.

        Transitional stub kept only to satisfy the (still abstract)
        `WeightTransferEngine.trainer_send_weights`; that member is dropped from
        the worker ABC once every backend has migrated to the trainer engine.
        """
        raise NotImplementedError(
            "The static sparse NCCL trainer path has been replaced by "
            "SparseNCCLTrainerWeightTransferEngine. Build it via "
            "WeightTransferTrainerFactory.trainer_init("
            "SparseNCCLTrainerInitInfo(...), client=..., source=...) and drive "
            "it with send_weights()."
        )


class SparseNCCLTrainerWeightTransferEngine(
    TrainerWeightTransferEngine[SparseNCCLTrainerInitInfo]
):
    """Trainer-side sparse NCCL weight transfer engine.

    Broadcasts flat-index (indices, values) patches from NCCL rank 0 while the
    inference-side `update_weights` runs concurrently on a side thread (the
    worker's recvs rendezvous inside the same NCCL broadcasts), then finishes
    the update. Unlike the full-resync backends, sparse patches differ every
    round (a fresh set of deltas from each optimizer step), so they are not a
    stable `WeightSource`: the engine takes no `source`, and each round's
    patches are passed straight to `send_weights(patches)`. A round with no
    patches is a no-op.

    The sparse backend assumes a single-rank trainer (matching its TP=1 / PP=1
    MVP scope), so non-sender ranks skip `send_weights` entirely.
    """

    init_info_cls = SparseNCCLTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
        is_sender: bool = True,
    ) -> None:
        # `source` is unused (sparse takes per-round patches via send_weights);
        # accepted only to match the base/factory signature.
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
                engine.client.init_weight_transfer_engine, asdict(worker_init_info)
            )
            # Open the trainer endpoint as NCCL rank 0 on the current device
            # (the shared helper accepts the rendezvous fields as a dict).
            engine.model_update_group = trainer_init(
                {
                    "master_address": init_info.master_address,
                    "master_port": init_info.master_port,
                    "world_size": init_info.world_size,
                }
            )
            future.result()  # surface any inference-side init error

        return engine

    def send_weights(self, patches: Iterable[SparseWeightPatch] | None = None) -> None:
        """Broadcast this round's sparse patches. `patches` is the per-round
        payload (sparse deltas differ every round), so it is passed here rather
        than fixed at init. Every patch must set `full_shape`."""
        if not self.is_sender:
            return

        patches = list(patches) if patches is not None else []
        if not patches:
            return

        shapes = []
        for patch in patches:
            if patch.full_shape is None:
                raise ValueError(
                    "SparseWeightPatch.full_shape must be set to send via the "
                    f"trainer engine: {patch.name}"
                )
            shapes.append(list(patch.full_shape))

        update_info = SparseNCCLWeightTransferUpdateInfo(
            names=[patch.name for patch in patches],
            dtype_names=[str(patch.values.dtype).split(".")[-1] for patch in patches],
            shapes=shapes,
            num_updates_list=[patch.indices.numel() for patch in patches],
        )

        assert self.model_update_group is not None, (
            "trainer_init() must be called before send_weights()."
        )
        self.client.start_weight_update()
        # update_weights (workers receive) must run concurrently with the
        # trainer-side broadcasts — both rendezvous inside the same NCCL calls.
        with ThreadPoolExecutor(max_workers=1) as exe:
            future = exe.submit(self.client.update_weights, asdict(update_info))
            # Cheap best-effort: if update_weights already failed (e.g. a bad
            # request rejected before any NCCL call), surface it now instead of
            # hanging in broadcast waiting for a peer that will never arrive.
            if future.done():
                future.result()
            stream = torch.cuda.current_stream()
            for patch in patches:
                self.model_update_group.broadcast(patch.indices, src=0, stream=stream)
                self.model_update_group.broadcast(patch.values, src=0, stream=stream)
            future.result()  # surface inference-side errors
        self.client.finish_weight_update()

    def shutdown(self) -> None:
        self.model_update_group = None
