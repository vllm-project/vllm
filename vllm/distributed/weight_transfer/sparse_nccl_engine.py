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

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.nccl_common import (
    NCCLWeightTransferInitInfo,
    trainer_init,
    worker_init_process_group,
)
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs

__all__ = [
    "SparseWeightPatch",
    "SparseNCCLWeightTransferUpdateInfo",
    "SparseNCCLWeightTransferEngine",
]


@dataclass
class SparseWeightPatch:
    """A sparse in-place patch for one existing parameter."""

    name: str
    indices: torch.Tensor
    values: torch.Tensor


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
    def trainer_send_weights(
        iterator: Iterator[SparseWeightPatch],
        trainer_args: dict[str, Any] | NCCLTrainerSendWeightsArgs,
    ) -> None:
        """Broadcast sparse flat-index patches from trainer to vLLM workers."""
        if isinstance(trainer_args, dict):
            args = NCCLTrainerSendWeightsArgs(**trainer_args)
        else:
            args = trainer_args

        if args.packed:
            raise ValueError(
                "Sparse NCCL updates cannot be combined with `packed=True`"
            )

        stream = args.stream or torch.cuda.current_stream()
        for patch in iterator:
            args.group.broadcast(patch.indices, src=args.src, stream=stream)
            args.group.broadcast(patch.values, src=args.src, stream=stream)

    # Trainer-side process-group setup (shared with the dense engine).
    trainer_init = staticmethod(trainer_init)
