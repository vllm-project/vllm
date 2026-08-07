# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-model checkpoint reload with tensor-level copy-back."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import torch

from vllm.config import ModelConfig
from vllm.model_executor.model_loader.reload.meta import (
    SKIP_MODULES,
    SKIP_TENSORS,
    materialize_meta_tensor,
    to_meta_tensor,
)
from vllm.model_executor.model_loader.reload.sanitize import (
    restore_layer_refs,
    sanitize_layer_refs,
)

__all__ = [
    "ModelwiseReloader",
    "ModelwiseReloadSession",
    "record_modelwise_reload_metadata",
]


@dataclass(frozen=True)
class _TensorMetadata:
    tensor: torch.Tensor | None
    persistent: bool = True


@dataclass(frozen=True)
class _ModelMetadata:
    parameters: dict[tuple[str, str], _TensorMetadata]
    buffers: dict[tuple[str, str], _TensorMetadata]
    restore_device: torch.device


@dataclass(frozen=True)
class _RuntimeBindings:
    parameters: dict[tuple[str, str], torch.Tensor | None]
    buffers: dict[tuple[str, str], torch.Tensor | None]
    buffer_persistence: dict[tuple[str, str], bool]


_MODEL_METADATA: WeakKeyDictionary[torch.nn.Module, _ModelMetadata] = (
    WeakKeyDictionary()
)


def _modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    return dict(model.named_modules())


def _iter_direct_tensors(
    model: torch.nn.Module,
) -> Iterator[tuple[str, torch.nn.Module, str, str, torch.Tensor]]:
    for module_name, module in model.named_modules():
        for name, tensor in module._parameters.items():
            if tensor is not None:
                yield module_name, module, "parameter", name, tensor
        for name, tensor in module._buffers.items():
            if tensor is not None:
                yield module_name, module, "buffer", name, tensor


def _clone_metadata(tensor: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
    clone = to_meta_tensor(tensor)
    return restore_layer_refs(clone, module)


def _original_weight_loader(tensor: torch.Tensor):
    loader = getattr(tensor, "weight_loader", None)
    while loader is not None and getattr(loader, "__name__", "") == (
        "online_process_loader"
    ):
        loader = loader.__wrapped__
    return loader


def _strip_online_loader(tensor: torch.Tensor) -> None:
    loader = _original_weight_loader(tensor)
    if loader is not None:
        tensor.weight_loader = loader


def record_modelwise_reload_metadata(model: torch.nn.Module) -> None:
    """Record checkpoint-format tensor bindings for whole-model reload.

    This is captured after model construction and before the initial checkpoint
    load or post-load processing changes parameter layouts.
    """
    parameters: dict[tuple[str, str], _TensorMetadata] = {}
    buffers: dict[tuple[str, str], _TensorMetadata] = {}
    captured: dict[int, _TensorMetadata] = {}

    for module_name, module in model.named_modules():
        if module.__class__.__name__ in SKIP_MODULES:
            continue
        for name, tensor in module._parameters.items():
            if name in SKIP_TENSORS:
                continue
            key = (module_name, name)
            if tensor is None:
                parameters[key] = _TensorMetadata(None)
            else:
                entry = captured.get(id(tensor))
                if entry is None:
                    entry = _TensorMetadata(
                        sanitize_layer_refs(to_meta_tensor(tensor), module)
                    )
                    captured[id(tensor)] = entry
                parameters[key] = entry
        for name, tensor in module._buffers.items():
            if name in SKIP_TENSORS:
                continue
            key = (module_name, name)
            persistent = name not in module._non_persistent_buffers_set
            # Non-persistent buffers are runtime state, not checkpoint state.
            # Keep them bound throughout a reload instead of restoring an
            # uninitialized construction-time snapshot over live caches such
            # as rotary ``cos_sin_cache``.
            if not persistent:
                continue
            if tensor is None:
                buffers[key] = _TensorMetadata(None, persistent)
                continue
            entry = captured.get(id(tensor))
            if entry is None:
                entry = _TensorMetadata(
                    sanitize_layer_refs(to_meta_tensor(tensor), module), persistent
                )
                captured[id(tensor)] = entry
            buffers[key] = entry

    _MODEL_METADATA[model] = _ModelMetadata(
        parameters=parameters,
        buffers=buffers,
        restore_device=torch.get_default_device(),
    )


def _capture_runtime_bindings(model: torch.nn.Module) -> _RuntimeBindings:
    parameters = {}
    buffers = {}
    buffer_persistence = {}
    for module_name, module in model.named_modules():
        for name, tensor in module._parameters.items():
            parameters[(module_name, name)] = tensor
        for name, tensor in module._buffers.items():
            key = (module_name, name)
            buffers[key] = tensor
            buffer_persistence[key] = name not in module._non_persistent_buffers_set
    return _RuntimeBindings(parameters, buffers, buffer_persistence)


def _clear_tensor_bindings(
    model: torch.nn.Module,
    *,
    preserve_nonpersistent_buffers: bool = False,
) -> None:
    for _, module in model.named_modules():
        if module.__class__.__name__ in SKIP_MODULES:
            continue
        for name in tuple(module._parameters):
            if name not in SKIP_TENSORS:
                delattr(module, name)
        for name in tuple(module._buffers):
            if name not in SKIP_TENSORS:
                if (
                    preserve_nonpersistent_buffers
                    and name in module._non_persistent_buffers_set
                ):
                    continue
                delattr(module, name)


def _restore_checkpoint_bindings(
    model: torch.nn.Module, metadata: _ModelMetadata
) -> None:
    modules = _modules(model)
    _clear_tensor_bindings(model, preserve_nonpersistent_buffers=True)
    restored: dict[int, torch.Tensor] = {}
    for (module_name, name), entry in metadata.parameters.items():
        module = modules[module_name]
        if entry.tensor is None:
            module.register_parameter(name, None)
            continue
        tensor = restored.get(id(entry))
        if tensor is None:
            tensor = _clone_metadata(entry.tensor, module)
            restored[id(entry)] = tensor
        _strip_online_loader(tensor)
        module.register_parameter(name, tensor)
    for (module_name, name), entry in metadata.buffers.items():
        module = modules[module_name]
        if entry.tensor is None:
            module.register_buffer(name, None, persistent=entry.persistent)
            continue
        tensor = restored.get(id(entry))
        if tensor is None:
            tensor = _clone_metadata(entry.tensor, module)
            restored[id(entry)] = tensor
        _strip_online_loader(tensor)
        module.register_buffer(name, tensor, persistent=entry.persistent)


def _materialize_checkpoint_bindings(
    model: torch.nn.Module, device: torch.device
) -> None:
    materialized: dict[int, torch.Tensor] = {}
    with device:
        for _, module, _, name, tensor in _iter_direct_tensors(model):
            if tensor.is_meta:
                value = materialized.get(id(tensor))
                if value is None:
                    value = materialize_meta_tensor(tensor)
                    materialized[id(tensor)] = value
                setattr(module, name, value)


def _restore_runtime_bindings(
    model: torch.nn.Module, runtime: _RuntimeBindings
) -> None:
    modules = _modules(model)
    _clear_tensor_bindings(model)
    for (module_name, name), tensor in runtime.parameters.items():
        modules[module_name].register_parameter(name, tensor)
    for (module_name, name), tensor in runtime.buffers.items():
        module = modules[module_name]
        persistent = runtime.buffer_persistence[(module_name, name)]
        module.register_buffer(name, tensor, persistent=persistent)


def _validate_copy_back(processed: _RuntimeBindings, runtime: _RuntimeBindings) -> None:
    findings = []
    for kind, old_bindings, new_bindings in (
        ("parameter", runtime.parameters, processed.parameters),
        ("buffer", runtime.buffers, processed.buffers),
    ):
        for key, old in old_bindings.items():
            new = new_bindings.get(key)
            if old is None:
                if new is not None:
                    findings.append(f"materialized runtime {kind} {key}")
                continue
            if new is None:
                if kind == "parameter":
                    findings.append(f"missing runtime {kind} {key}")
                continue
            if old.shape != new.shape or old.dtype != new.dtype:
                findings.append(
                    f"incompatible runtime {kind} {key}: "
                    f"{tuple(old.shape)}/{old.dtype} -> "
                    f"{tuple(new.shape)}/{new.dtype}"
                )
    if findings:
        raise ValueError(
            "Whole-model reload changed the runtime tensor schema:\n  "
            + "\n  ".join(findings[:20])
        )


def _copy_back(processed: _RuntimeBindings, runtime: _RuntimeBindings) -> None:
    for old_bindings, new_bindings in (
        (runtime.parameters, processed.parameters),
        (runtime.buffers, processed.buffers),
    ):
        for key, old in old_bindings.items():
            new = new_bindings.get(key)
            if old is not None and new is not None:
                old.data.copy_(new)


class ModelwiseReloadSession:
    """Explicit model-wide checkpoint reload transaction.

    ``start`` restores the checkpoint-format tensor schema, any number of
    ``load_weights`` calls may then stream checkpoint tensors into the model,
    and ``finish`` runs model-wide post-load processing before copying values
    into the original runtime storages. The transaction boundary, rather than
    tensor element counts, determines when post-load processing runs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_config: ModelConfig,
        target_device: torch.device,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.target_device = target_device
        self._runtime: _RuntimeBindings | None = None
        self._original_torchao = False
        self._loaded_weights: set[str] = set()
        self._loaded_weights_unknown = False

    @property
    def active(self) -> bool:
        return self._runtime is not None

    @torch.no_grad()
    def start(self) -> None:
        if self.active:
            raise RuntimeError("Model-wise reload session is already active")

        metadata = _MODEL_METADATA.get(self.model)
        if metadata is None:
            raise RuntimeError(
                "Whole-model reload metadata was not recorded during model "
                "initialization"
            )

        runtime = _capture_runtime_bindings(self.model)
        self._original_torchao = getattr(self.model, "_do_torchao_reload", False)
        self.model._do_torchao_reload = False
        try:
            _restore_checkpoint_bindings(self.model, metadata)
            _materialize_checkpoint_bindings(self.model, metadata.restore_device)
        except BaseException:
            _restore_runtime_bindings(self.model, runtime)
            self.model._do_torchao_reload = self._original_torchao
            raise

        self._runtime = runtime
        self._loaded_weights.clear()
        self._loaded_weights_unknown = False

    @torch.no_grad()
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str] | None:
        if not self.active:
            raise RuntimeError("Model-wise reload session is not active")

        loaded = self.model.load_weights(weights)
        if loaded is None:
            self._loaded_weights_unknown = True
        else:
            self._loaded_weights.update(loaded)
        return loaded

    def _restore_runtime(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        _restore_runtime_bindings(self.model, runtime)
        self.model._do_torchao_reload = self._original_torchao
        self._runtime = None

    @torch.no_grad()
    def finish(self) -> set[str] | None:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("Model-wise reload session is not active")

        try:
            from vllm.model_executor.model_loader.utils import (
                process_weights_after_loading,
            )

            process_weights_after_loading(
                self.model,
                self.model_config,
                self.target_device,
                force=True,
            )
            processed = _capture_runtime_bindings(self.model)
            _validate_copy_back(processed, runtime)
            _copy_back(processed, runtime)
        finally:
            self._restore_runtime()

        if self._loaded_weights_unknown:
            return None
        return set(self._loaded_weights)

    @torch.no_grad()
    def abort(self) -> None:
        """Restore serving bindings without committing received weights."""
        self._restore_runtime()


class ModelwiseReloader:
    """Reload checkpoint weights as one model-wide transaction.

    The serving model is temporarily rebound to its checkpoint-format tensor
    schema, loaded normally, and post-processed once at model scope. Processed
    values are then copied into the original runtime parameter and buffer
    storages. No per-layer load accounting or layerwise finalization is used.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_config: ModelConfig,
        target_device: torch.device,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.target_device = target_device

    @torch.no_grad()
    def reload(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str] | None:
        session = ModelwiseReloadSession(
            self.model,
            self.model_config,
            self.target_device,
        )
        session.start()
        try:
            session.load_weights(weights)
            return session.finish()
        except BaseException:
            session.abort()
            raise
