# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse disk weight transfer engine.

The trainer publishes each version's sparse patches as one file on storage that
every inference host can read, while the engine serves. Each worker keeps its
part of the checkpoint in host memory, as its load plan writes it; in the update
it applies the new files there and copies that whole part into the model.
"""

import bisect
import contextlib
import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
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
from vllm.utils.torch_utils import set_default_torch_num_threads

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.model_loader.reload.plan import Box

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

_FORMAT = "sparse_disk/1"
# Same-size integer views, so patches are copied bit for bit whatever the dtype.
_INT_VIEWS = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
# Most patch elements mapped to the host copy at a time.
_APPLY_CHUNK = 1 << 24


def _numel(box: "Box") -> int:
    return math.prod(b - a for a, b in box)


def patch_file(patch_dir: str | Path, version: int) -> Path:
    """The file that holds `version`'s patches."""
    return Path(patch_dir) / f"patch-{version:012d}.safetensors"


def save_sparse_patches(
    patches: Iterable[SparseWeightPatch], patch_dir: str | Path, version: int
) -> Path:
    """Write `version`'s patches, the changes from `version - 1`, to `patch_dir`.

    Patch values overwrite the checkpoint values at their flat indices, which
    must be strictly increasing; a tensor is patched at most once per version.
    Patches with values of one dtype are stored as one concatenated index and
    value tensor. A version is written once: publishing it again raises.
    """
    groups: dict[torch.dtype, list[SparseWeightPatch]] = {}
    names: set[str] = set()
    for patch in patches:
        indices = patch.indices
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"indices of {patch.name} must be int32 or int64")
        if indices.ndim != 1 or indices.shape != patch.values.shape:
            raise ValueError(
                f"indices and values of {patch.name} must be 1-D and the same length"
            )
        if indices.numel() and not (
            bool((indices[1:] > indices[:-1]).all())
            and int(indices[0]) >= 0
            and int(indices[-1]) < math.prod(patch.full_shape)
        ):
            raise ValueError(
                f"indices of {patch.name} must be strictly increasing and in range"
            )
        if patch.name in names:
            raise ValueError(f"{patch.name} is patched twice")
        names.add(patch.name)
        groups.setdefault(patch.values.dtype, []).append(patch)
    tensors: dict[str, torch.Tensor] = {}
    entries = []
    for i, group in enumerate(groups.values()):
        wide = any(math.prod(p.full_shape) > 2**31 for p in group)
        index_dtype = torch.int64 if wide else torch.int32
        tensors[f"{i}.indices"] = torch.cat(
            [p.indices.detach().cpu().to(index_dtype) for p in group]
        )
        tensors[f"{i}.values"] = torch.cat([p.values.detach().cpu() for p in group])
        entries += [
            {
                "name": p.name,
                "shape": list(p.full_shape),
                "group": i,
                "count": p.indices.numel(),
            }
            for p in group
        ]
    data = save(tensors, {"format": _FORMAT, "patches": json.dumps(entries)})
    # Created exclusively and written in place, as `save_file` may rename a
    # temporary file, which object-store mounts cannot do; opened outside the
    # try, so a published version is never removed. A version is applied only
    # after this returns, and a file cut short by a crash fails to open.
    path = patch_file(patch_dir, version)
    f = open(path, "xb")  # noqa: SIM115
    try:
        with f:
            f.write(data)
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise
    return path


@dataclass
class SparseDiskWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side init info for the sparse disk backend."""

    patch_dir: str
    """Directory the trainer publishes patch files to, at the same path on every
    inference host."""
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
    """Applies published patch files to this worker's host copy of its part of
    the checkpoint, then loads that whole part into the model.

    The first update builds the copy from the startup checkpoint, at
    `base_version`, and replays the files after it; a restarted server does the
    same. The copy holds what the load plan writes on this worker, in pinned
    memory where available.
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
        self._copy: _HostCopy | None = None
        # The version the model holds; it outlives the copy.
        self._loaded = 0

    def init_transfer_engine(self, init_info: SparseDiskWeightTransferInitInfo) -> None:
        """Check that the model can load through a load plan from its startup
        checkpoint files."""
        from vllm.model_executor.model_loader import get_model_loader
        from vllm.model_executor.model_loader.default_loader import (
            DefaultModelLoader,
        )
        from vllm.model_executor.model_loader.reload import check_plannable

        load_format = self.vllm_config.load_config.load_format
        loader = get_model_loader(self.vllm_config.load_config)
        if type(loader) is not DefaultModelLoader or load_format in (
            "fastsafetensors",
            "instanttensor",
        ):
            raise ValueError(
                "sparse_disk builds its copy from the checkpoint files on the host "
                f"and needs the default loader, not load_format={load_format!r}"
            )
        if self._init_info is None:
            self._loaded = init_info.base_version
        elif init_info.base_version != self._init_info.base_version:
            raise ValueError(
                f"this server started from version {self._init_info.base_version}, "
                f"not {init_info.base_version}; restart it from that checkpoint"
            )
        if getattr(self.model, "secondary_weights", ()):
            raise ValueError("sparse_disk does not support secondary weight sources")
        check_plannable(self.model)
        self.shutdown()
        self._init_info = init_info

    def start_weight_update(self) -> None:
        from vllm.model_executor.model_loader.reload import start_reload

        start_reload(self.model)

    def finish_weight_update(self) -> None:
        from vllm.model_executor.model_loader.reload import finish_reload

        finish_reload(self.model, self.model_config, self.device)

    def receive_weights(self, update_info: SparseDiskWeightTransferUpdateInfo) -> None:
        """Bring the host copy to the version, then load it into the model."""
        init_info = self._init_info
        if init_info is None:
            raise RuntimeError(
                "Sparse disk weight transfer engine not initialized. "
                "Call init_transfer_engine() first."
            )
        newest = max(self._loaded, self._copy.version if self._copy else 0)
        if update_info.version < newest:
            raise RuntimeError(
                f"this server has applied version {newest}, newer than "
                f"{update_info.version}; versions only move forward"
            )
        if self._copy is None:
            self._copy = _HostCopy(self, init_info.base_version)
        copy = self._copy
        with set_default_torch_num_threads():
            while copy.version < update_info.version:
                path = patch_file(init_info.patch_dir, copy.version + 1)
                started = time.perf_counter()
                try:
                    with safe_open(path, framework="pt") as f:
                        writes = copy.map(f)
                except (ValueError, SafetensorError) as e:
                    raise ValueError(f"{path}: {e}") from e
                try:
                    for dtype, positions, values in writes:  # bit for bit
                        copy.buffers[dtype].view(values.dtype)[positions] = values
                except BaseException:
                    self.shutdown()  # partly written: rebuilt at the next update
                    raise
                del writes  # before the next file is mapped
                copy.version += 1
                logger.info("Applied %s in %.2f s", path, time.perf_counter() - started)
        if self.device.type != "cpu":
            torch.accelerator.empty_cache()  # the mapping's temporaries
        copy.load(self.model)
        self._loaded = copy.version

    def shutdown(self) -> None:
        from vllm.v1.kv_offload.cpu.host_register import host_unregister

        for ptr in self._copy.registered if self._copy else ():
            host_unregister(ptr)
        self._copy = None


class _HostCopy:
    """A worker's part of the checkpoint at `version`, as its load plan writes
    it: per checkpoint dtype, one host buffer of every box this worker loads.
    A name the plan leaves to `load_weights` holds its whole tensor there."""

    def __init__(self, engine: SparseDiskWeightTransferEngine, version: int) -> None:
        from vllm.model_executor.model_loader import get_model_loader
        from vllm.model_executor.model_loader.reload import plan_loads
        from vllm.utils.platform_utils import is_pin_memory_available
        from vllm.v1.kv_offload.cpu.host_register import host_register

        started = time.perf_counter()
        self.version = version
        self.device = device = engine.device
        model, model_config = engine.model, engine.model_config
        loader = get_model_loader(engine.vllm_config.load_config)
        # what the default loader feeds the model, as at the cold start
        meta = [
            ParamMeta(n, t.dtype, tuple(t.shape))
            for n, t in loader.get_all_weights(model_config, model)
        ]
        # every one of them, to check patch files against
        self.catalog = {m.name: (m.shape, m.dtype) for m in meta}
        self.plan = plan_loads(model, meta)
        # name -> [(box, offset into its dtype's buffer)]
        self.parts: dict[str, list[tuple[Box, int]]] = {}
        sizes: dict[torch.dtype, int] = {}
        for m in meta:
            for box in self.plan.get_boxes(m.name) or ():
                offset = sizes.get(m.dtype, 0)
                self.parts.setdefault(m.name, []).append((box, offset))
                sizes[m.dtype] = offset + _numel(box)
        self.fallbacks = [m.name for m in meta if self.plan.get_dest(m.name) is None]
        self.buffers = {d: torch.empty(n, dtype=d) for d, n in sizes.items()}
        with set_default_torch_num_threads():
            for name, t in loader.get_all_weights(model_config, model):
                for box, flat in self._views(name):
                    part = t[tuple(slice(a, b) for a, b in box)]
                    flat.view(part.shape).copy_(part)
        self.pairs = [
            (dst, dict(self._views(name))[box].view(dst.shape))
            for name in self.parts
            for box, dst in self.plan.get_dest(name) or ()
        ]
        self._index_geometry()
        # last, so that a failure above leaves no registration behind
        pin = device.type != "cpu" and is_pin_memory_available()
        self.registered = [
            b.data_ptr()
            for b in self.buffers.values()
            if pin and host_register(b.data_ptr(), b.nbytes)
        ]
        logger.info(
            "Built the host copy of %d tensors (%.2f GiB, %s) in %.1f s",
            len(meta),
            sum(b.nbytes for b in self.buffers.values()) / 2**30,
            "pinned" if self.registered else "pageable",
            time.perf_counter() - started,
        )

    def _views(self, name: str) -> list[tuple["Box", torch.Tensor]]:
        parts = self.parts.get(name, ())
        flat = self.buffers[self.catalog[name][1]] if parts else None
        return [(box, flat[o : o + _numel(box)]) for box, o in parts]  # type: ignore[index]

    def _index_geometry(self) -> None:
        """Device tensors to map patch elements in bulk: per checkpoint name,
        its shape and, per slot, one of its boxes on this worker with its copy
        offset; slots past its last box are empty."""
        self.ids = {name: i for i, name in enumerate(self.catalog)}
        ndim = max([1, *(len(shape) for shape, _ in self.catalog.values())])
        slots = max([1, *map(len, self.parts.values())])
        shapes, boxes, bases = [], [], []
        for name, (shape, _) in self.catalog.items():
            parts = [
                ([(0, 1)] * (ndim - len(box)) + list(box), offset)
                for box, offset in self.parts.get(name, ())
            ]
            parts += [([(0, 0)] * ndim, 0)] * (slots - len(parts))
            shapes.append([1] * (ndim - len(shape)) + list(shape))
            boxes.append([box for box, _ in parts])
            bases.append([offset for _, offset in parts])
        box = torch.tensor(boxes, dtype=torch.long, device=self.device)
        self.geometry = (
            torch.tensor(shapes, dtype=torch.long, device=self.device).T,
            box[..., 0].permute(1, 2, 0),
            box[..., 1].permute(1, 2, 0),
            torch.tensor(bases, dtype=torch.long, device=self.device).T,
        )

    def map(self, f: Any) -> list[tuple[torch.dtype, torch.Tensor, torch.Tensor]]:
        """Check every patch of the open file `f` against the checkpoint and
        find where its values land in this copy, without touching it: (buffer
        dtype, positions in it, values as same-size integers)."""
        metadata = f.metadata() or {}
        if metadata.get("format") != _FORMAT:
            raise ValueError(f"not a {_FORMAT} patch file")
        entries = json.loads(metadata["patches"])
        if len({e["name"] for e in entries}) != len(entries):
            raise ValueError("a tensor is patched twice")
        writes = []
        for i in sorted({e["group"] for e in entries}):
            group = [e for e in entries if e["group"] == i]
            indices = f.get_tensor(f"{i}.indices")
            values = f.get_tensor(f"{i}.values")
            for e in group:
                if e["name"] not in self.catalog:
                    raise ValueError(f"{e['name']} is not in the checkpoint")
                if (tuple(e["shape"]), values.dtype) != self.catalog[e["name"]]:
                    raise ValueError(
                        f"{e['name']} is {e['shape']} {values.dtype}, "
                        f"the checkpoint {self.catalog[e['name']]}"
                    )
            counts = torch.tensor([e["count"] for e in group])
            if (
                counts.dtype != torch.int64
                or indices.dtype not in (torch.int32, torch.int64)
                or (counts < 0).any()
                or not indices.shape == values.shape == (int(counts.sum()),)
            ):
                raise ValueError(f"bad group {i}")
            names = [e["name"] for e in group]
            values = values.view(_INT_VIEWS[values.element_size()])
            for positions, src in self._map_bulk(names, counts, indices, values):
                writes.append((self.catalog[names[0]][1], positions, src))
        return writes

    def _map_bulk(
        self,
        names: list[str],
        counts: torch.Tensor,
        indices: torch.Tensor,
        values: torch.Tensor,
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        """Check and map every element, a chunk at a time on the worker's
        device, and yield the copy offset and value of each one in a box."""
        ids = torch.tensor([self.ids[n] for n in names], device=self.device)
        size, start, stop, base = (t[..., ids] for t in self.geometry)
        numel = size.prod(0)
        slots = [len(self.parts.get(n, ())) for n in names]
        boxes = torch.tensor(slots, device=self.device)
        ends = counts.cumsum(0)
        ends_list = ends.tolist()
        chunk = _chunk_elements(self.device)
        for lo in range(0, indices.numel(), chunk):
            hi = min(lo + chunk, indices.numel())
            first = max(lo - 1, 0)  # with the last element before, for the order
            a = bisect.bisect_right(ends_list, first)
            b = bisect.bisect_left(ends_list, hi) + 1
            span = ends[a:b].clamp(max=hi) - (ends[a:b] - counts[a:b]).clamp(min=first)
            k = torch.repeat_interleave(
                torch.arange(a, b, device=self.device),
                span.to(self.device),
                output_size=hi - first,
            )
            index = indices[first:hi].to(self.device).long()
            falling = (k[1:] == k[:-1]) & (index[1:] <= index[:-1])
            bad = torch.cat([k[1:][falling], k[(index < 0) | (index >= numel[k])]])
            if bad.numel():
                raise ValueError(f"bad indices for {names[int(bad.min())]}")
            k, index = k[lo - first :], index[lo - first :]
            src = values[lo:hi].to(self.device)
            for s in range(max(slots)):
                if s:  # only the names with a box in this slot
                    more = (boxes[k] > s).nonzero().view(-1)
                    k, index, src = k[more], index[more], src[more]
                rest = index
                selected = torch.ones_like(index, dtype=torch.bool)
                position = base[s, k]
                stride = torch.ones_like(index)
                for d in reversed(range(size.shape[0])):
                    extent = size[d, k]
                    coord = rest % extent
                    rest = rest // extent
                    a_d, b_d = start[s, d, k], stop[s, d, k]
                    selected &= (coord >= a_d) & (coord < b_d)
                    position += (coord - a_d) * stride
                    stride *= b_d - a_d
                chosen = selected.nonzero().view(-1)
                yield position[chosen].cpu(), src[chosen].cpu()

    def load(self, model: torch.nn.Module) -> None:
        """Copy the whole part into the model and report it loaded."""
        from vllm.model_executor.model_loader.reload.direct import mark_loaded

        for dst, src in self.pairs:
            dst.copy_(src, non_blocking=src.dtype == dst.dtype)
        for name in self.parts:
            for w in self.plan.entries[name].writes or ():
                mark_loaded(model, f"{w.module}.{w.tensor}" if w.module else w.tensor)
        if self.fallbacks:
            model.load_weights(
                [
                    (n, self._views(n)[0][1].view(self.catalog[n][0]).clone())
                    for n in self.fallbacks
                ]
            )


def _chunk_elements(device: torch.device) -> int:
    """Patch elements to map at once: about 200 bytes each of free device
    memory, within bounds."""
    from vllm.platforms import current_platform

    if device.type == "cpu":
        return _APPLY_CHUNK
    free = current_platform.free_device_memory(device)
    free += torch.accelerator.memory_reserved(device)
    free -= torch.accelerator.memory_allocated(device)
    return max(1 << 20, min(_APPLY_CHUNK, free // 200))


@dataclass
class SparseDiskTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for the sparse disk backend; the sender ships the
    worker-side fields at the handshake."""

    backend: ClassVar[str] = "sparse_disk"

    patch_dir: str
    """Directory to publish patch files to; every inference host must read it at
    the same path."""
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
                {"patch_dir": engine.patch_dir, "base_version": engine.version}
            )
        return engine

    def publish(self, patches: Iterable[SparseWeightPatch]) -> None:
        """Write the next version's patches."""
        if self.is_sender:
            save_sparse_patches(patches, self.patch_dir, self.version + 1)
            self.version += 1

    def send_weights(self) -> None:
        """Update the inference side to the last published version."""
        if not self.is_sender:
            return
        self.client.start_weight_update()
        self.client.update_weights({"version": self.version})
        self.client.finish_weight_update()
