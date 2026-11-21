# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse disk weight transfer engine.

The trainer publishes each version's sparse patches as one file on storage that
every inference host can read, while the engine serves. In the update, each host
applies the new files to its own writable copy of the checkpoint, and every
worker reloads the whole model from that copy.
"""

import copy
import fcntl
import json
import math
import os
import shutil
import struct
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

import torch
from safetensors import safe_open
from safetensors.torch import save

from vllm.config.weight_transfer import WeightTransferConfig
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
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

__all__ = [
    "SparseDiskTrainerInitInfo",
    "SparseDiskTrainerWeightTransferEngine",
    "SparseDiskWeightTransferEngine",
    "SparseDiskWeightTransferInitInfo",
    "SparseDiskWeightTransferUpdateInfo",
    "patch_file",
    "save_sparse_patches",
]

# Same-size integer views, so patches are copied bit for bit whatever the dtype.
_INT_VIEWS = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def patch_file(patch_dir: str | Path, version: int) -> Path:
    """The file that holds `version`'s patches."""
    return Path(patch_dir) / f"patch-{version:012d}.safetensors"


def save_sparse_patches(
    patches: Iterable[SparseWeightPatch], patch_dir: str | Path, version: int
) -> Path:
    """Write `version`'s patches, the changes from `version - 1`, to `patch_dir`.

    Patch values overwrite the checkpoint values at their flat indices. A
    version is written once: publishing it again raises.
    """
    tensors: dict[str, torch.Tensor] = {}
    entries = []
    for i, patch in enumerate(patches):
        if patch.indices.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"indices of {patch.name} must be int32 or int64")
        if patch.indices.ndim != 1 or patch.indices.shape != patch.values.shape:
            raise ValueError(
                f"indices and values of {patch.name} must be 1-D and the same length"
            )
        tensors[f"{i}.indices"] = patch.indices.detach().cpu().contiguous()
        tensors[f"{i}.values"] = patch.values.detach().cpu().contiguous()
        entries.append({"name": patch.name, "shape": list(patch.full_shape)})
    # Created exclusively and written in place, as `save_file` may rename a
    # temporary file, which object-store mounts cannot do. A version is applied
    # only after this returns, and a file cut short by a crash fails to open.
    path = patch_file(patch_dir, version)
    with open(path, "xb") as f:
        f.write(save(tensors, metadata={"patches": json.dumps(entries)}))
    return path


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


@dataclass
class SparseDiskWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side init info for the sparse disk backend."""

    patch_dir: str
    """Directory the trainer publishes patch files to, at the same path on every
    inference host. It names the update line: a copy made for another patch_dir
    is rebuilt."""
    copy_dir: str
    """Host-local directory (not a network filesystem) for the writable copy of
    the checkpoint, at the same path on every host."""
    base_version: int = 0
    """Version of the checkpoint the inference servers started from."""


@dataclass
class SparseDiskWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for the sparse disk backend."""

    version: int


class SparseDiskWeightTransferEngine(
    WeightTransferEngine[
        SparseDiskWeightTransferInitInfo, SparseDiskWeightTransferUpdateInfo
    ]
):
    """Applies published patch files to a host-local checkpoint copy, then
    reloads the full model from the copy.

    Every rank on a host shares one copy. One worker at a time checks and
    writes it, and never while a reload on the host still reads it.
    `copy_dir/state.json` records the copy's update line, version and boot, so
    a restarted server reuses it and a rebooted host, whose page cache may have
    lost writes, rebuilds it.
    """

    init_info_cls = SparseDiskWeightTransferInitInfo
    update_info_cls = SparseDiskWeightTransferUpdateInfo
    supports_draft_weight_update = False

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self._init_info: SparseDiskWeightTransferInitInfo | None = None
        # The startup checkpoint folder and the files to copy from it.
        self._source: tuple[str, list[str]] | None = None
        # Shared lock on the copy, held from `receive_weights` to the end of
        # `finish_weight_update`: layerwise reload keeps views of it until then.
        # A writer refuses rather than waits for it: ranks that update together
        # finish only after all of them have loaded.
        self._reader: int | None = None

    @property
    def _copy_dir(self) -> Path:
        assert self._init_info is not None, "init_transfer_engine() was not called"
        return Path(self._init_info.copy_dir)

    @property
    def _checkpoint_dir(self) -> Path:
        return self._copy_dir / "checkpoint"

    def init_transfer_engine(self, init_info: SparseDiskWeightTransferInitInfo) -> None:
        """Make the host copy of the startup checkpoint, unless the host has one."""
        from vllm.model_executor.model_loader import get_model_loader
        from vllm.model_executor.model_loader.default_loader import (
            DefaultModelLoader,
        )

        loader = get_model_loader(self.vllm_config.load_config)
        if type(loader) is not DefaultModelLoader:
            raise ValueError(
                "sparse_disk reloads from a copy of the checkpoint files and needs "
                f"the default loader, not load_format="
                f"{self.vllm_config.load_config.load_format!r}"
            )
        folder, files, use_safetensors, index_file = loader._prepare_weights(
            self.model_config.model,
            None,
            self.model_config.revision,
            fall_back_to_pt=getattr(self.model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(
                self.model, "allow_patterns_overrides", None
            ),
        )
        if not use_safetensors:
            raise ValueError("sparse_disk needs a safetensors checkpoint")
        names = [os.path.relpath(f, folder) for f in files]
        if os.path.isfile(os.path.join(folder, index_file)):
            names.append(index_file)
        self._source = (folder, names)
        self._init_info = init_info
        os.makedirs(init_info.copy_dir, exist_ok=True)
        self._advance_copy(None)

    def start_weight_update(self) -> None:
        from vllm.model_executor.model_loader.reload import start_reload

        start_reload(self.model)

    def finish_weight_update(self) -> None:
        from vllm.model_executor.model_loader.reload import finish_reload

        try:
            finish_reload(self.model, self.model_config, self.device)
        finally:
            self._release_reader()

    def receive_weights(self, update_info: SparseDiskWeightTransferUpdateInfo) -> None:
        """Bring the host copy to the version, then load the whole model from it."""
        from vllm.model_executor.model_loader import get_model_loader

        self._advance_copy(update_info.version)
        try:
            model_config = copy.copy(self.model_config)
            model_config.model = str(self._checkpoint_dir)
            model_config.revision = None
            loader = get_model_loader(self.vllm_config.load_config)
            self.model.load_weights(loader.get_all_weights(model_config, self.model))
        except BaseException:
            self._release_reader()
            raise

    def shutdown(self) -> None:
        self._release_reader()

    def _flock(self, name: str, operation: int) -> int:
        fd = os.open(self._copy_dir / name, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, operation)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _check_unread(self) -> None:
        """Refuse to write the copy while a reload on this host still reads it."""
        try:
            os.close(self._flock(".readers", fcntl.LOCK_EX | fcntl.LOCK_NB))
        except BlockingIOError:
            raise RuntimeError(
                f"the copy in {self._copy_dir} is still loading in another "
                "update; servers that share a copy must update together"
            ) from None

    def _release_reader(self) -> None:
        if self._reader is not None:
            os.close(self._reader)
            self._reader = None

    def _advance_copy(self, version: int | None) -> None:
        """Bring the host copy to `version`, first rebuilding it if it belongs to
        another update line or boot, or is older than the base version. With a
        version, also take the shared lock that `finish_weight_update` releases."""
        assert self._init_info is not None, "init_transfer_engine() was not called"
        # One worker at a time: the others find the copy written, and each
        # takes the shared lock before a later turn could write again.
        turn = self._flock(".lock", fcntl.LOCK_EX)
        try:
            state = self._read_state()
            if (
                state is None
                or state["patch_dir"] != self._init_info.patch_dir
                or state["boot_id"] != _boot_id()
                or state["version"] < self._init_info.base_version
            ):
                state = self._build_copy()
            if version is None:
                return
            if state["version"] > version:
                raise RuntimeError(
                    f"the copy in {self._copy_dir} is at version "
                    f"{state['version']}, newer than {version}; versions only move "
                    "forward"
                )
            # From the version after the copy's; an interrupted apply is redone
            # whole, which overwrite patches allow.
            for v in range(state["version"] + 1, version + 1):
                self._apply(v, state)
            if state["applying"] is not None:
                raise RuntimeError(
                    f"the copy in {self._copy_dir} is partly at version "
                    f"{state['applying']}; update to it or later to finish it"
                )
            if self._reader is None:  # else held since an earlier update_weights
                self._reader = self._flock(".readers", fcntl.LOCK_SH)
        finally:
            os.close(turn)

    def _read_state(self) -> dict[str, Any] | None:
        try:
            return json.loads((self._copy_dir / "state.json").read_text())
        except (FileNotFoundError, ValueError):  # cut short by a host crash
            return None

    def _write_state(self, state: dict[str, Any]) -> None:
        path = self._copy_dir / "state.json"
        tmp = path.with_name("state.json.tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, path)

    def _build_copy(self) -> dict[str, Any]:
        """Copy the startup checkpoint files; the copy is at the base version."""
        assert self._init_info is not None and self._source is not None
        self._check_unread()
        (self._copy_dir / "state.json").unlink(missing_ok=True)
        shutil.rmtree(self._checkpoint_dir, ignore_errors=True)
        folder, names = self._source
        start = time.perf_counter()
        for name in names:
            dst = self._checkpoint_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(os.path.join(folder, name), dst)
        logger.info(
            "Copied %d checkpoint files from %s to %s in %.1f s",
            len(names),
            folder,
            self._checkpoint_dir,
            time.perf_counter() - start,
        )
        state = {
            "patch_dir": self._init_info.patch_dir,
            "version": self._init_info.base_version,
            "applying": None,
            "boot_id": _boot_id(),
        }
        self._write_state(state)
        return state

    def _read_slots(self) -> dict[str, tuple[Path, int, int, str, tuple[int, ...]]]:
        """Each tensor's file, byte range, dtype name and shape in the copy, which
        another server on this host may have built from other shards."""
        slots = {}
        for file in self._checkpoint_dir.rglob("*.safetensors"):
            with open(file, "rb") as f:
                (size,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(size))
            header.pop("__metadata__", None)
            for name, info in header.items():
                start, end = (8 + size + offset for offset in info["data_offsets"])
                slots[name] = (file, start, end, info["dtype"], tuple(info["shape"]))
        return slots

    def _apply(self, version: int, state: dict[str, Any]) -> None:
        """Check every patch of the version's file, then write them all."""
        assert self._init_info is not None
        self._check_unread()
        path = patch_file(self._init_info.patch_dir, version)
        started = time.perf_counter()
        slots = self._read_slots()
        mapped: dict[Path, torch.Tensor] = {}
        with safe_open(path, framework="pt") as f:
            writes = []
            for i, entry in enumerate(json.loads(f.metadata()["patches"])):
                name = entry["name"]
                if name not in slots:
                    raise ValueError(f"{path}: {name} is not in the checkpoint")
                file, start, end, dtype, shape = slots[name]
                if tuple(entry["shape"]) != shape:
                    raise ValueError(
                        f"{path}: {name} has shape {tuple(entry['shape'])}, the "
                        f"checkpoint {shape}"
                    )
                values_dtype = f.get_slice(f"{i}.values").get_dtype()
                if values_dtype != dtype:
                    raise ValueError(
                        f"{path}: {name} values are {values_dtype}, the checkpoint "
                        f"{dtype}"
                    )
                indices = f.get_tensor(f"{i}.indices")
                values = f.get_tensor(f"{i}.values")
                numel = math.prod(shape)
                if indices.numel() and (
                    int(indices.min()) < 0 or int(indices.max()) >= numel
                ):
                    raise ValueError(f"{path}: bad indices for {name}")
                if file not in mapped:
                    mapped[file] = torch.from_file(
                        str(file),
                        shared=True,
                        size=file.stat().st_size,
                        dtype=torch.uint8,
                    )
                int_view = _INT_VIEWS[values.element_size()]
                dst = mapped[file][start:end].view(int_view)
                writes.append((dst, indices, values.view(int_view)))

            state["applying"] = version
            self._write_state(state)
            for dst, indices, src in writes:
                dst.index_put_((indices,), src)
        state["version"] = version
        state["applying"] = None
        self._write_state(state)
        logger.info(
            "Applied %s to %s in %.2f s",
            path,
            self._checkpoint_dir,
            time.perf_counter() - started,
        )


@dataclass
class SparseDiskTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for the sparse disk backend; the sender ships the
    worker-side fields at the handshake."""

    backend: ClassVar[str] = "sparse_disk"

    patch_dir: str
    """Directory to publish patch files to; every inference host must read it at
    the same path."""
    copy_dir: str
    """Host-local directory on the inference hosts for the checkpoint copy."""
    base_version: int = 0
    """Version of the checkpoint the inference servers started from; the first
    `publish` writes the next one."""


class SparseDiskTrainerWeightTransferEngine(
    TrainerWeightTransferEngine[SparseDiskTrainerInitInfo]
):
    """Trainer-side sparse disk weight transfer engine.

    `publish(patches)` writes the next version while the engine serves;
    `send_weights()`, called in the pause, updates the inference side to it.
    Patches are in checkpoint coordinates, as for sparse NCCL, and only the
    sender's are published.
    """

    init_info_cls = SparseDiskTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        patch_dir: str,
        version: int,
        source: WeightSource | None = None,
        is_sender: bool = True,
    ) -> None:
        if source is not None:
            raise ValueError(
                "Sparse disk weight transfer takes no WeightSource; pass each "
                "round's patches to publish(patches) instead."
            )
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.patch_dir = patch_dir
        # The last version published.
        self.version = version

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
            version=init_info.base_version,
            source=source,
            is_sender=init_info.is_sender,
        )
        if engine.is_sender:
            client.init_weight_transfer_engine(
                asdict(
                    SparseDiskWeightTransferInitInfo(
                        patch_dir=init_info.patch_dir,
                        copy_dir=init_info.copy_dir,
                        base_version=init_info.base_version,
                    )
                )
            )
        return engine

    def publish(self, patches: Iterable[SparseWeightPatch]) -> int:
        """Write the next version's patches and return the version."""
        if self.is_sender:
            save_sparse_patches(patches, self.patch_dir, self.version + 1)
            self.version += 1
        return self.version

    def send_weights(self) -> None:
        """Update the inference side to the last published version."""
        if not self.is_sender:
            return
        update_info = asdict(SparseDiskWeightTransferUpdateInfo(version=self.version))
        self.client.start_weight_update()
        self.client.update_weights(update_info)
        self.client.finish_weight_update()
