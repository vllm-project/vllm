# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse disk weight transfer engine.

The trainer writes each round's sparse patches to one safetensors file on
storage that every inference worker can read at the same path. Workers apply the
file through the model's native weight loader, as the sparse NCCL backend does.
"""

import json
import os
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Self

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseWeightPatch
from vllm.model_executor.model_loader.checkpoint_weight_patch import (
    CheckpointWeightPatch,
    load_checkpoint_weight_patches,
)

__all__ = [
    "SparseDiskTrainerInitInfo",
    "SparseDiskTrainerWeightTransferEngine",
    "SparseDiskWeightTransferEngine",
    "SparseDiskWeightTransferInitInfo",
    "SparseDiskWeightTransferUpdateInfo",
    "load_sparse_patches",
    "save_sparse_patches",
]

PATCH_FORMAT = "vllm-sparse-patch-v1"


def save_sparse_patches(patches: Iterable[SparseWeightPatch], path: str | Path) -> None:
    """Write sparse patches to one safetensors file.

    Patch `i` is stored as tensors `{i}.indices` and `{i}.values`; names and full
    checkpoint shapes are in the file metadata. The file appears at `path` only
    once it is complete.
    """
    path = Path(path)
    tensors: dict[str, torch.Tensor] = {}
    entries = []
    for i, patch in enumerate(patches):
        tensors[f"{i}.indices"] = patch.indices.detach().cpu()
        tensors[f"{i}.values"] = patch.values.detach().cpu()
        entries.append({"name": patch.name, "shape": list(patch.full_shape)})
    partial = path.with_name(f"{path.name}.partial")
    save_file(
        tensors,
        partial,
        metadata={"format": PATCH_FORMAT, "patches": json.dumps(entries)},
    )
    os.replace(partial, path)


def load_sparse_patches(
    path: str | Path, device: torch.device | str = "cpu"
) -> list[CheckpointWeightPatch]:
    """Read a file written by `save_sparse_patches` onto `device`."""
    with safe_open(path, framework="pt", device=str(device)) as file:
        metadata = file.metadata() or {}
        if metadata.get("format") != PATCH_FORMAT:
            raise ValueError(f"Not a {PATCH_FORMAT} sparse patch file: {path}")
        patches = []
        for i, entry in enumerate(json.loads(metadata["patches"])):
            values = file.get_tensor(f"{i}.values")
            patches.append(
                CheckpointWeightPatch(
                    name=entry["name"],
                    shape=tuple(entry["shape"]),
                    dtype=values.dtype,
                    values=values,
                    indices=file.get_tensor(f"{i}.indices"),
                )
            )
    return patches


@dataclass
class SparseDiskWeightTransferInitInfo(WeightTransferInitInfo):
    """Init info for the sparse disk backend. There is nothing to rendezvous."""


@dataclass
class SparseDiskWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for the sparse disk backend."""

    patch_path: str
    """Absolute path of a patch file, readable by every inference worker."""

    def __post_init__(self) -> None:
        if not os.path.isabs(self.patch_path):
            raise ValueError(f"`patch_path` must be absolute: {self.patch_path}")


class SparseDiskWeightTransferEngine(
    WeightTransferEngine[
        SparseDiskWeightTransferInitInfo, SparseDiskWeightTransferUpdateInfo
    ]
):
    """Sparse weight transfer engine reading patch files from shared storage.

    Applies checkpoint-coordinate patches through the model's native weight
    loader. Sparse updates modify initialized model tensors in place, so the
    layerwise reload lifecycle is not used.
    """

    init_info_cls = SparseDiskWeightTransferInitInfo
    update_info_cls = SparseDiskWeightTransferUpdateInfo
    supports_draft_weight_update = False

    def init_transfer_engine(self, init_info: SparseDiskWeightTransferInitInfo) -> None:
        pass

    def start_weight_update(self) -> None:
        """No-op: sparse patches are applied in place, no layerwise reload."""
        pass

    def finish_weight_update(self) -> None:
        """No-op: sparse patches are applied in place, no layerwise reload."""
        pass

    def receive_weights(self, update_info: SparseDiskWeightTransferUpdateInfo) -> None:
        patches = load_sparse_patches(update_info.patch_path, self.device)
        load_checkpoint_weight_patches(self.model, patches)

    def shutdown(self) -> None:
        pass


@dataclass
class SparseDiskTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for the sparse disk weight transfer backend."""

    backend: ClassVar[str] = "sparse_disk"

    patch_dir: str
    """Existing directory the trainer writes patch files to. Every inference
    worker must see it at the same absolute path."""


class SparseDiskTrainerWeightTransferEngine(
    TrainerWeightTransferEngine[SparseDiskTrainerInitInfo]
):
    """Trainer-side sparse disk weight transfer engine.

    Writes each call's patches to a new file under `patch_dir`, has the inference
    workers apply it, then deletes the file. The send methods and their patch
    type match `SparseNCCLTrainerWeightTransferEngine`; only the transport
    differs, so no GPU or NCCL group is needed on the trainer.
    """

    init_info_cls = SparseDiskTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        patch_dir: str | Path,
        source: WeightSource | None = None,
        is_sender: bool = True,
    ) -> None:
        if source is not None:
            raise ValueError(
                "Sparse disk weight transfer takes no WeightSource; pass each "
                "round's patches to send_weights(patches) instead."
            )
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.patch_dir = Path(patch_dir).absolute()

    @classmethod
    def trainer_init(
        cls,
        init_info: SparseDiskTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        engine = cls(
            client=client,
            patch_dir=init_info.patch_dir,
            source=source,
            is_sender=init_info.is_sender,
        )
        if engine.is_sender:
            client.init_weight_transfer_engine(
                asdict(SparseDiskWeightTransferInitInfo())
            )
        return engine

    def send_weights(self, patches: Iterable[SparseWeightPatch] | None = None) -> None:
        """Send one sparse update through a one-shot lifecycle."""
        patches = self._prepare_patches(patches)
        if not patches:
            return

        self.client.start_weight_update()
        self._send_chunk(patches)
        self.client.finish_weight_update()

    def send_weight_chunk(
        self, patches: Iterable[SparseWeightPatch] | None = None
    ) -> None:
        """Send one chunk inside a caller-owned weight update lifecycle."""
        patches = self._prepare_patches(patches)
        if not patches:
            return
        self._send_chunk(patches)

    def _prepare_patches(
        self, patches: Iterable[SparseWeightPatch] | None
    ) -> list[SparseWeightPatch]:
        if not self.is_sender or patches is None:
            return []
        return list(patches)

    def _send_chunk(self, patches: list[SparseWeightPatch]) -> None:
        path = self.patch_dir / f"patch-{uuid.uuid4().hex}.safetensors"
        save_sparse_patches(patches, path)
        try:
            self.client.update_weights(
                asdict(SparseDiskWeightTransferUpdateInfo(patch_path=str(path)))
            )
        finally:
            path.unlink(missing_ok=True)
