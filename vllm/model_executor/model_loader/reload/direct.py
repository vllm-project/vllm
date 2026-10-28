# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct weight reload.

Every update is a full sync written straight into the live parameters, after
which the cold-start post-processing (PWAL) runs once:

- Cold start records each parameter's checkpoint layout at the PWAL entry and
  every parameter's and buffer's runtime layout and address after PWAL.
- `direct_start` views each parameter whose layout PWAL changed back in its
  checkpoint layout over its own storage, and tracks every `weight_loader` call.
- `direct_finish` reruns PWAL through the cold-start entry, then refuses the
  update if a parameter was never written or a tensor moved.

PWAL must write its results into the storage it was given. A result PWAL
allocates is put back where the tensor sat after cold-start PWAL (see
`place_in_runtime_layout`), so it may be larger than the checkpoint tensor as
long as it matches its cold-start layout. A failed update leaves the model
undefined; the engine must be restarted.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "DirectReloadError",
    "direct_finish",
    "direct_reload_configured",
    "direct_start",
    "place_in_runtime_layout",
    "record_checkpoint_layout",
    "record_runtime_layout",
]

_RECORD_ATTR = "_direct_reload_record"
# Tensors a checkpoint may omit; their defaults are filled by PWAL.
_OPTIONAL_TENSORS = ("k_scale", "v_scale", "q_scale", "prob_scale")

# Cold-start runtime layouts by tensor id, set only while direct_finish reruns PWAL
_runtime_layouts: dict[int, "TensorLayout"] | None = None


class DirectReloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorLayout:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    storage_offset: int

    @classmethod
    def of(cls, tensor: torch.Tensor) -> "TensorLayout":
        return cls(
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.storage_offset(),
        )

    def byte_range(self) -> tuple[int, int]:
        """Bytes of the storage this layout reads, as `[start, end)`."""
        itemsize = self.dtype.itemsize
        start = self.storage_offset * itemsize
        if 0 in self.shape:
            return start, start
        span = 1 + sum((n - 1) * s for n, s in zip(self.shape, self.stride))
        return start, start + span * itemsize

    def view(self, storage: torch.UntypedStorage, device: torch.device) -> torch.Tensor:
        return torch.empty(0, dtype=self.dtype, device=device).set_(
            storage, self.storage_offset, self.shape, self.stride
        )


@dataclass
class _CheckpointTensor:
    names: list[str]
    tensor: torch.Tensor
    layout: TensorLayout


@dataclass
class _RuntimeTensor:
    tensor: torch.Tensor
    data_ptr: int
    layout: TensorLayout


@dataclass
class _Session:
    loaders: list[tuple[torch.Tensor, Callable | None]] = field(default_factory=list)
    written: set[int] = field(default_factory=set)


@dataclass
class _Record:
    checkpoint: list[_CheckpointTensor]
    runtime: dict[str, _RuntimeTensor] | None = None
    session: _Session | None = None
    failed: str | None = None


def direct_reload_configured() -> bool:
    """Whether `VLLM_WEIGHT_RELOAD_MODE` selects direct reload."""
    return envs.VLLM_WEIGHT_RELOAD_MODE == "direct"


def record_checkpoint_layout(model: nn.Module) -> None:
    """Record every parameter's checkpoint layout, once, before cold-start PWAL."""
    if hasattr(model, _RECORD_ATTR) or not direct_reload_configured():
        return
    entries: dict[int, _CheckpointTensor] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        if param.is_meta:
            continue
        if (entry := entries.get(id(param))) is not None:
            entry.names.append(name)
        else:
            entries[id(param)] = _CheckpointTensor(
                [name], param, TensorLayout.of(param)
            )
    setattr(model, _RECORD_ATTR, _Record(list(entries.values())))


def record_runtime_layout(model: nn.Module) -> None:
    """Record every tensor's address and layout, once, after cold-start PWAL."""
    record: _Record | None = getattr(model, _RECORD_ATTR, None)
    if record is None or record.runtime is not None:
        return
    record.runtime = {
        name: _RuntimeTensor(tensor, tensor.data_ptr(), TensorLayout.of(tensor))
        for name, tensor in _named_tensors(model)
    }


def direct_start(model: nn.Module) -> None:
    """View checkpoint tensors back in checkpoint layout and track their writes."""
    record = _get_record(model)
    if record.session is not None:
        raise DirectReloadError("A direct reload is already in progress.")
    _check_supported(model)

    params = dict(model.named_parameters(remove_duplicate=False))
    restores: list[tuple[_CheckpointTensor, torch.UntypedStorage]] = []
    for entry in record.checkpoint:
        for name in entry.names:
            if params.get(name) is not entry.tensor:
                raise DirectReloadError(
                    f"{name} was replaced after cold start; post-processing must "
                    "keep parameter objects (replace_parameter)."
                )
        tensor = entry.tensor
        if TensorLayout.of(tensor) == entry.layout:
            continue
        storage = tensor.untyped_storage()
        if entry.layout.byte_range()[1] > storage.nbytes():
            raise DirectReloadError(
                f"{entry.names[0]}: its storage ({storage.nbytes()} bytes) cannot "
                f"hold the checkpoint layout {entry.layout}; post-processing "
                "freed or shrank it."
            )
        restores.append((entry, storage))
    _check_disjoint(record)

    for entry, storage in restores:
        entry.tensor.data = entry.layout.view(storage, entry.tensor.device)

    session = _Session()
    for entry in record.checkpoint:
        tensor = entry.tensor
        loader = getattr(tensor, "weight_loader", None)
        session.loaders.append((tensor, loader))
        tensor.weight_loader = _TrackedLoader(
            loader or _default_weight_loader(), session.written, id(tensor)
        )
    record.session = session
    logger.info(
        "Direct reload started: %d of %d parameters viewed back in checkpoint layout.",
        len(restores),
        len(record.checkpoint),
    )


def direct_finish(
    model: nn.Module, model_config: Any, device: torch.device | None = None
) -> None:
    """Rerun cold-start post-processing and verify the update."""
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    record = _get_record(model)
    session = record.session
    if session is None:
        raise DirectReloadError("direct_finish called without direct_start.")
    record.session = None

    torch.accelerator.synchronize()
    for tensor, loader in session.loaders:
        if loader is None:
            del tensor.weight_loader
        else:
            tensor.weight_loader = loader

    missing = [
        entry.names[0]
        for entry in record.checkpoint
        if id(entry.tensor) not in session.written
        and not entry.names[0].endswith(_OPTIONAL_TENSORS)
    ]
    if missing:
        _fail(record, f"{len(missing)} parameters were never written: {missing}")

    global _runtime_layouts
    assert record.runtime is not None
    _runtime_layouts = {id(t.tensor): t.layout for t in record.runtime.values()}
    start = time.perf_counter()
    try:
        process_weights_after_loading(
            model, model_config, device or _model_device(model)
        )
    except BaseException as e:
        _fail(record, f"post-processing raised {e!r}", cause=e)
    finally:
        _runtime_layouts = None

    moved = _moved_tensors(model, record)
    if moved:
        _fail(record, f"{len(moved)} tensors moved or changed layout: {moved}")
    logger.info(
        "Direct reload finished: post-processing took %.2f s.",
        time.perf_counter() - start,
    )


def place_in_runtime_layout(
    old: torch.Tensor, new: torch.Tensor
) -> torch.Tensor | None:
    """Copy `new` into `old`'s storage where `old` sat after cold-start PWAL.

    Called by `replace_parameter` while a direct reload reruns PWAL. The write
    covers exactly the bytes the tensor had after cold-start PWAL, so a result
    larger than the checkpoint view fits without touching other tensors.

    Returns:
        The view of `old`'s storage holding `new`, or None if `new` does not
        match the cold-start layout (the caller then falls back).

    """
    layout = None if _runtime_layouts is None else _runtime_layouts.get(id(old))
    if (
        layout is None
        or new.device != old.device
        or (tuple(new.shape), tuple(new.stride()), new.dtype)
        != (layout.shape, layout.stride, layout.dtype)
    ):
        return None
    storage = old.untyped_storage()
    if layout.byte_range()[1] > storage.nbytes():
        return None
    target = layout.view(storage, old.device)
    if new.data_ptr() != target.data_ptr():
        if new.untyped_storage().data_ptr() == storage.data_ptr():
            new = new.clone()
        target.copy_(new)
    return target


class _TrackedLoader:
    """Records that a parameter's loader ran, then calls the original.

    Compares equal to the wrapped loader, so `weight_loader ==
    default_weight_loader` checks in model code keep working.
    """

    def __init__(self, loader: Callable, written: set[int], key: int) -> None:
        self.__wrapped__ = loader
        self._written = written
        self._key = key

    def __call__(self, *args, **kwargs):
        self._written.add(self._key)
        return self.__wrapped__(*args, **kwargs)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _TrackedLoader):
            other = other.__wrapped__
        return self.__wrapped__ == other

    def __hash__(self) -> int:
        return hash(self.__wrapped__)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__wrapped__, name)


def _get_record(model: nn.Module) -> _Record:
    record: _Record | None = getattr(model, _RECORD_ATTR, None)
    if record is None or record.runtime is None:
        raise DirectReloadError(
            "Direct reload needs the model to be loaded with "
            "VLLM_WEIGHT_RELOAD_MODE=direct."
        )
    if record.failed is not None:
        raise DirectReloadError(
            f"A previous direct reload failed ({record.failed}); the model is "
            "undefined. Restart the engine."
        )
    return record


def _check_supported(model: nn.Module) -> None:
    from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
    from vllm.model_executor.layers.quantization.online.moe_base import (
        OnlineMoEMethodBase,
    )

    for name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, (OnlineLinearBase, OnlineMoEMethodBase)):
            raise NotImplementedError(
                f"{name}: direct reload does not support online quantization "
                f"({type(quant_method).__name__}); use "
                "VLLM_WEIGHT_RELOAD_MODE=layerwise."
            )
    for name, param in model.named_parameters():
        if type(param.data) is not torch.Tensor:
            raise NotImplementedError(
                f"{name}: direct reload does not support tensor subclasses "
                f"({type(param.data).__name__})."
            )


def _check_disjoint(record: _Record) -> None:
    by_storage: dict[int, list[tuple[int, int, str]]] = {}
    for entry in record.checkpoint:
        start, end = entry.layout.byte_range()
        base = entry.tensor.untyped_storage().data_ptr()
        by_storage.setdefault(base, []).append((start, end, entry.names[0]))
    for ranges in by_storage.values():
        ranges.sort()
        for (_, prev_end, prev), (start, _, name) in zip(ranges, ranges[1:]):
            if start < prev_end:
                raise DirectReloadError(
                    f"Checkpoint layouts of {prev} and {name} overlap in one "
                    "storage; post-processing packed them together."
                )


def _moved_tensors(model: nn.Module, record: _Record) -> list[str]:
    assert record.runtime is not None
    current = dict(_named_tensors(model))
    moved = []
    for name, expected in record.runtime.items():
        tensor = current.get(name)
        if (
            tensor is not expected.tensor
            or tensor.data_ptr() != expected.data_ptr
            or TensorLayout.of(tensor) != expected.layout
        ):
            moved.append(name)
    return moved


def _fail(record: _Record, reason: str, cause: BaseException | None = None):
    record.failed = reason
    raise DirectReloadError(
        f"Direct reload failed: {reason}. The model is undefined; restart the engine."
    ) from cause


def _named_tensors(model: nn.Module):
    yield from model.named_parameters(remove_duplicate=False)
    yield from model.named_buffers(remove_duplicate=False)


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _default_weight_loader() -> Callable:
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    return default_weight_loader
