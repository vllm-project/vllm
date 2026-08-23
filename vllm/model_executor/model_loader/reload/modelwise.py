# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-model checkpoint reload with tensor-level copy-back."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import torch

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
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
from vllm.model_executor.model_loader.reload.sharding import (
    RankShard,
    capture_rank_sharding,
    get_rank_sharding_manifest,
    install_sharding_recorders,
)
from vllm.model_executor.model_loader.reload.source import observe_weight_sources
from vllm.model_executor.model_loader.reload.units import (
    ReloadUnit,
    ShardCoverageTracker,
    StagingSpec,
    install_trackers,
    uninstall_trackers,
)

logger = init_logger(__name__)

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


def _discover_manifest_expert_trackers(
    model: torch.nn.Module,
    metadata: _ModelMetadata,
    expected: set[RankShard],
    skip_modules: frozenset[str],
) -> list[tuple[str, torch.nn.Module, ShardCoverageTracker]]:
    """Build generic expert trackers directly from the rank manifest.

    Quant methods do not need to implement a reload-specific hook for this
    fallback. Expert loaders already expose ``shard_id`` and ``expert_id``;
    the manifest supplies the complete coverage set, while checkpoint tensor
    metadata supplies one-expert staging shapes.
    """
    modules = _modules(model)
    parameter_metadata = metadata.parameters
    candidates: dict[str, set[RankShard]] = {}
    for shard in expected:
        fragment = dict(shard.fragment.items)
        if "expert_id" not in fragment:
            continue
        target_parts = shard.target_name.rsplit(".", 1)
        if len(target_parts) != 2:
            continue
        module_name, parameter_name = target_parts
        if module_name in skip_modules or module_name not in modules:
            continue
        module = modules[module_name]
        has_expert_loader = (
            any(
                getattr(param, "weight_loader", None).__class__.__name__
                == "ReloadAwareWeightLoader"
                for param in module._parameters.values()
                if param is not None
            )
            or getattr(module, "local_num_experts", None) is not None
        )
        if has_expert_loader:
            candidates.setdefault(module_name, set()).add(shard)

    found = []
    for module_name, shards in candidates.items():
        module = modules[module_name]
        keys = set()
        staged: dict[str, StagingSpec] = {}
        for shard in shards:
            fragment = dict(shard.fragment.items)
            expert_id = int(fragment["expert_id"])
            mapper = getattr(module, "_map_global_expert_id_to_local_expert_id", None)
            local_expert = mapper(expert_id) if mapper is not None else expert_id
            if local_expert < 0:
                continue
            shard_id = str(
                fragment.get("shard_id", fragment.get("loaded_shard_id", "default"))
            )
            _, parameter_name = shard.target_name.rsplit(".", 1)
            keys.add((parameter_name, local_expert, shard_id))
            entry = parameter_metadata.get((module_name, parameter_name))
            if entry is not None and entry.tensor is not None:
                shape = tuple(entry.tensor.shape[1:])
                if not shape and shard_id in ("w1", "w3"):
                    # Per-tensor MoE scales are stored as one value per fused
                    # half, while the serving schema keeps their reduction.
                    shape = (2,)
                staged[parameter_name] = StagingSpec(shape, entry.tensor.dtype)
        if not keys or not staged:
            continue

        def commit(pieces, module=module):
            """Assemble staged expert slices into checkpoint-format tensors."""
            for (parameter_name, expert), slab in pieces.items():
                parameter = module._parameters[parameter_name]
                parameter.data[expert].copy_(slab)

        unit = ReloadUnit(
            name=f"{module_name}.manifest",
            keys=frozenset(keys),
            commit=commit,
            staged=staged,
            deferred=True,
        )
        found.append((module_name, module, ShardCoverageTracker(module, [unit])))
    return found


def _modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """Index every module by its fully qualified name.

    Args:
        model: Model whose module hierarchy is indexed.

    Returns:
        A mapping from qualified module names to module objects.
    """
    return dict(model.named_modules())


def _iter_direct_tensors(
    model: torch.nn.Module,
) -> Iterator[tuple[str, torch.nn.Module, str, str, torch.Tensor]]:
    """Iterate parameters and buffers registered directly on each module.

    Args:
        model: Model whose direct tensor bindings are inspected.

    Yields:
        Tuples containing the module name, module, binding kind, local tensor
        name, and tensor. Bindings whose value is ``None`` are omitted.
    """
    for module_name, module in model.named_modules():
        for name, tensor in module._parameters.items():
            if tensor is not None:
                yield module_name, module, "parameter", name, tensor
        for name, tensor in module._buffers.items():
            if tensor is not None:
                yield module_name, module, "buffer", name, tensor


def _clone_metadata(tensor: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
    """Clone checkpoint metadata and restore references to its owner module.

    Args:
        tensor: Sanitized checkpoint-format meta tensor to clone.
        module: Module that will own the cloned binding.

    Returns:
        A cloned meta tensor with layer references restored.
    """
    clone = to_meta_tensor(tensor)
    return restore_layer_refs(clone, module)


def _original_weight_loader(tensor: torch.Tensor):
    """Unwrap reload-time online loaders to the original weight loader.

    Args:
        tensor: Tensor whose ``weight_loader`` attribute is inspected.

    Returns:
        The original loader, or ``None`` when the tensor has no loader.
    """
    loader = getattr(tensor, "weight_loader", None)
    while loader is not None and getattr(loader, "__name__", "") == (
        "online_process_loader"
    ):
        loader = loader.__wrapped__
    return loader


def _strip_online_loader(tensor: torch.Tensor) -> None:
    """Replace a reload-time loader wrapper with its original loader.

    Args:
        tensor: Tensor whose loader attribute may contain online wrappers.
    """
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
    """Capture serving parameter and buffer objects before a reload starts.

    Args:
        model: Serving model whose live bindings must survive reload.

    Returns:
        Runtime bindings and buffer persistence metadata keyed by owner/name.
    """
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
    skip_modules: frozenset[str] = frozenset(),
) -> None:
    """Remove replaceable parameter and buffer bindings from a model.

    Args:
        model: Model whose direct bindings are cleared.
        preserve_nonpersistent_buffers: Whether runtime-only buffers remain
            attached while checkpoint bindings are restored.
        skip_modules: Qualified modules whose bindings must remain untouched.
    """
    for module_name, module in model.named_modules():
        if module.__class__.__name__ in SKIP_MODULES:
            continue
        if module_name in skip_modules:
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
    model: torch.nn.Module,
    metadata: _ModelMetadata,
    skip_modules: frozenset[str] = frozenset(),
) -> None:
    """Restore the construction-time checkpoint tensor schema on meta device.

    Args:
        model: Model to rebind to its checkpoint-format schema.
        metadata: Construction-time parameter and buffer metadata.
        skip_modules: Qualified modules managed by a separate reload path.
    """
    modules = _modules(model)
    _clear_tensor_bindings(
        model, preserve_nonpersistent_buffers=True, skip_modules=skip_modules
    )
    restored: dict[int, torch.Tensor] = {}
    for (module_name, name), entry in metadata.parameters.items():
        if module_name in skip_modules:
            continue
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
        if module_name in skip_modules:
            continue
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
    model: torch.nn.Module,
    device: torch.device,
    target_names: set[str],
) -> None:
    """Materialize only bindings targeted by sources that started loading."""
    targets: set[int] = set()
    for module_name, _, _, name, tensor in _iter_direct_tensors(model):
        target_name = f"{module_name}.{name}" if module_name else name
        if target_name in target_names and tensor.is_meta:
            targets.add(id(tensor))

    materialized: dict[int, torch.Tensor] = {}
    with device:
        for _, module, _, name, tensor in _iter_direct_tensors(model):
            if tensor.is_meta and id(tensor) in targets:
                value = materialized.get(id(tensor))
                if value is None:
                    value = materialize_meta_tensor(tensor)
                    materialized[id(tensor)] = value
                setattr(module, name, value)


def _bind_runtime_storage(
    model: torch.nn.Module,
    runtime: _RuntimeBindings,
    target_names: set[str],
) -> set[str]:
    """Bind compatible checkpoint targets to their serving storage.

    A direct binding avoids staging allocation when checkpoint and runtime
    shape/dtype already match. All aliases of the selected checkpoint tensor
    are rebound together so an untouched tied alias cannot later materialize
    uninitialized storage. Returned names are the requested targets that chose
    this path; callers materialize every remaining target.

    Direct writes are intentionally not rollbackable without a full backup.
    """
    bound: set[str] = set()
    selected: dict[int, torch.Tensor] = {}
    for module_name, module, kind, name, checkpoint in _iter_direct_tensors(model):
        target_name = f"{module_name}.{name}" if module_name else name
        if target_name not in target_names or not checkpoint.is_meta:
            continue
        key = (module_name, name)
        serving = (
            runtime.parameters.get(key)
            if kind == "parameter"
            else runtime.buffers.get(key)
        )
        if (
            serving is None
            or serving.shape != checkpoint.shape
            or serving.dtype != checkpoint.dtype
        ):
            continue
        selected[id(checkpoint)] = serving
        bound.add(target_name)

    for _, module, kind, name, checkpoint in _iter_direct_tensors(model):
        serving = selected.get(id(checkpoint))
        if serving is None:
            continue
        if kind == "parameter":
            value = torch.nn.Parameter(serving.data, requires_grad=False)
        else:
            value = serving.detach()
        value.__dict__.update(checkpoint.__dict__)
        setattr(module, name, value)
    return bound


def _restore_runtime_bindings(
    model: torch.nn.Module,
    runtime: _RuntimeBindings,
    skip_modules: frozenset[str] = frozenset(),
) -> None:
    """Reattach the original serving parameter and buffer objects.

    Args:
        model: Model whose serving schema is restored.
        runtime: Bindings captured before the reload transaction.
        skip_modules: Qualified modules restored by another reload path.
    """
    modules = _modules(model)
    _clear_tensor_bindings(model, skip_modules=skip_modules)
    for (module_name, name), tensor in runtime.parameters.items():
        if module_name in skip_modules:
            continue
        modules[module_name].register_parameter(name, tensor)
    for (module_name, name), tensor in runtime.buffers.items():
        if module_name in skip_modules:
            continue
        module = modules[module_name]
        persistent = runtime.buffer_persistence[(module_name, name)]
        module.register_buffer(name, tensor, persistent=persistent)


def _restore_module_runtime_bindings(
    model: torch.nn.Module,
    runtime: _RuntimeBindings,
    module_names: frozenset[str],
) -> None:
    """Restore serving bindings for selected module subtrees only.

    Args:
        model: Model currently holding checkpoint-format bindings.
        runtime: Original serving bindings captured before reload.
        module_names: Module names whose subtrees must be discarded.
    """
    modules = _modules(model)

    def belongs(name: str) -> bool:
        """Return whether a binding belongs to a selected module subtree."""
        return any(name == root or name.startswith(f"{root}.") for root in module_names)

    for (module_name, name), tensor in runtime.parameters.items():
        if not belongs(module_name):
            continue
        module = modules[module_name]
        if hasattr(module, name):
            delattr(module, name)
        module.register_parameter(name, tensor)
    for (module_name, name), tensor in runtime.buffers.items():
        if not belongs(module_name):
            continue
        module = modules[module_name]
        if hasattr(module, name):
            delattr(module, name)
        module.register_buffer(
            name,
            tensor,
            persistent=runtime.buffer_persistence[(module_name, name)],
        )


def _validate_copy_back(
    processed: _RuntimeBindings,
    runtime: _RuntimeBindings,
    skip_modules: frozenset[str] = frozenset(),
) -> None:
    """Validate that post-processed tensors fit the serving tensor schema.

    Args:
        processed: Bindings produced by post-load processing.
        runtime: Original serving bindings that receive processed values.
        skip_modules: Qualified modules excluded from modelwise copy-back.

    Raises:
        ValueError: If a processed tensor is missing or has an incompatible
            shape or dtype.
    """
    findings = []
    for kind, old_bindings, new_bindings in (
        ("parameter", runtime.parameters, processed.parameters),
        ("buffer", runtime.buffers, processed.buffers),
    ):
        for key, old in old_bindings.items():
            if key[0] in skip_modules:
                continue
            new = new_bindings.get(key)
            if old is None:
                if new is not None:
                    findings.append(f"materialized runtime {kind} {key}")
                continue
            if new is None or new.is_meta:
                if kind == "parameter" and new is None:
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


def _copy_back(
    processed: _RuntimeBindings,
    runtime: _RuntimeBindings,
    skip_modules: frozenset[str] = frozenset(),
) -> None:
    """Copy post-processed values into stable serving tensor storage.

    Args:
        processed: Bindings containing newly processed tensor values.
        runtime: Original serving bindings to update in place.
        skip_modules: Qualified modules committed by another reload path.
    """
    for old_bindings, new_bindings in (
        (runtime.parameters, processed.parameters),
        (runtime.buffers, processed.buffers),
    ):
        for key, old in old_bindings.items():
            if key[0] in skip_modules:
                continue
            new = new_bindings.get(key)
            if old is not None and new is not None and not new.is_meta:
                old.data.copy_(new)


def _module_binding_keys(
    bindings: _RuntimeBindings, module_name: str
) -> set[tuple[str, str]]:
    """Return runtime binding keys owned by a module and its descendants."""
    return {
        key
        for values in (bindings.parameters, bindings.buffers)
        for key in values
        if key[0] == module_name or key[0].startswith(f"{module_name}.")
    }


def _audit_meta_bindings_before_pwal(
    model: torch.nn.Module,
    pwal_modules: frozenset[str],
    skipped_pwal_modules: frozenset[str],
) -> list[str]:
    """Audit every residual meta binding and reject active PWAL dependencies.

    Args:
        model: Model immediately before post-load processing.
        pwal_modules: Quant modules scheduled for this transaction's PWAL.
        skipped_pwal_modules: Quant modules retaining their serving state.

    Returns:
        Human-readable audit records for residual meta bindings.

    Raises:
        RuntimeError: If a meta binding is reachable from a scheduled quant
            module's PWAL subtree.
    """
    quant_modules = {
        name
        for name, module in model.named_modules()
        if isinstance(getattr(module, "quant_method", None), QuantizeMethodBase)
    }
    findings = []
    for module_name, _, kind, name, tensor in _iter_direct_tensors(model):
        if not tensor.is_meta:
            continue
        quant_owners = {
            owner
            for owner in quant_modules
            if module_name == owner or module_name.startswith(f"{owner}.")
        }
        quant_owner = max(quant_owners, key=len) if quant_owners else None
        active_owners = quant_owners & pwal_modules
        if active_owners:
            state = "active-pwal"
        elif quant_owner in skipped_pwal_modules:
            state = "skipped-pwal"
        else:
            state = "no-quant-pwal"
        qualified_name = f"{module_name}.{name}" if module_name else name
        finding = (
            f"{qualified_name} ({kind}, module={module_name or '<root>'}, "
            f"quant_owner={quant_owner or '<none>'}, state={state}, "
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype})"
        )
        findings.append(finding)
    if findings:
        logger.debug(
            "Streaming reload residual meta audit before PWAL (%d):\n  %s",
            len(findings),
            "\n  ".join(findings[:100]),
        )
    _validate_pwal_dependencies_materialized(model, pwal_modules)
    return findings


def _validate_pwal_dependencies_materialized(
    model: torch.nn.Module,
    pwal_modules: frozenset[str],
) -> None:
    """Reject meta parameters or buffers reachable by scheduled quant PWAL.

    Args:
        model: Model immediately before quant post-load processing.
        pwal_modules: Quant module subtrees about to execute PWAL.

    Raises:
        RuntimeError: If any scheduled PWAL subtree still contains a meta
            parameter or buffer.
    """
    active_findings = []
    for module_name, _, kind, name, tensor in _iter_direct_tensors(model):
        if not tensor.is_meta or not any(
            module_name == owner or module_name.startswith(f"{owner}.")
            for owner in pwal_modules
        ):
            continue
        qualified_name = f"{module_name}.{name}" if module_name else name
        active_findings.append(
            f"{qualified_name} ({kind}, module={module_name or '<root>'}, "
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype})"
        )
    if active_findings:
        raise RuntimeError(
            "Streaming reload left meta dependencies reachable by scheduled "
            "PWAL modules:\n  " + "\n  ".join(active_findings[:100])
        )


def _commit_processed_module(
    model: torch.nn.Module,
    runtime: _RuntimeBindings,
    module_name: str,
) -> None:
    """Validate and commit one module subtree after its PWAL completes.

    Validation is completed for the entire subtree before the first copy, so a
    schema mismatch cannot partially update serving storage. Once copied, the
    checkpoint/PWAL bindings are replaced with the original runtime objects,
    releasing their staging storage when no other references remain.
    """
    processed = _capture_runtime_bindings(model)
    keys = _module_binding_keys(runtime, module_name)
    findings = []
    copies: list[tuple[torch.Tensor, torch.Tensor]] = []
    for kind, old_values, new_values in (
        ("parameter", runtime.parameters, processed.parameters),
        ("buffer", runtime.buffers, processed.buffers),
    ):
        for key in keys:
            if key not in old_values:
                continue
            old = old_values[key]
            new = new_values.get(key)
            if old is None:
                if new is not None:
                    findings.append(f"materialized runtime {kind} {key}")
                continue
            if new is None or old.shape != new.shape or old.dtype != new.dtype:
                findings.append(f"incompatible runtime {kind} {key}")
                continue
            copies.append((old, new))
    if findings:
        raise ValueError(
            "Whole-model reload changed the runtime tensor schema:\n  "
            + "\n  ".join(findings[:20])
        )
    for old, new in copies:
        old.data.copy_(new)

    modules = _modules(model)
    for key in keys:
        binding_module_name, name = key
        module = modules[binding_module_name]
        if key in runtime.parameters:
            if name in module._parameters:
                delattr(module, name)
            module.register_parameter(name, runtime.parameters[key])
        else:
            if name in module._buffers:
                delattr(module, name)
            module.register_buffer(
                name,
                runtime.buffers[key],
                persistent=runtime.buffer_persistence[key],
            )


class _LazyCheckpointBindings:
    """Allocate and finalize checkpoint bindings as manifest shards arrive.

    The initial-load rank manifest maps each checkpoint source to the exact
    target and fragment consumed on this rank. Non-PWAL targets may bind to
    compatible runtime storage. Quantized modules allocate checkpoint-format
    storage on first use. Only methods that explicitly declare incremental
    PWAL safety are processed before the model-wide finish boundary.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        runtime: _RuntimeBindings,
        device: torch.device,
        target_device: torch.device,
        expected: set[RankShard],
        skip_modules: frozenset[str],
    ) -> None:
        """Build source-target and quant-module coverage indexes."""
        self.model = model
        self.runtime = runtime
        self.device = device
        self.target_device = target_device
        self.skip_modules = skip_modules
        self.received: set[RankShard] = set()
        self.committed_modules: set[str] = set()
        self._source_targets: dict[str, set[str]] = {}
        modules = _modules(model)
        quant_modules = {
            name
            for name, module in modules.items()
            if isinstance(getattr(module, "quant_method", None), QuantizeMethodBase)
            and name not in skip_modules
        }
        self._quant_modules = quant_modules
        all_modules = set(modules)
        self._all_modules = all_modules
        shards_by_module: dict[str, set[RankShard]] = {}
        for shard in expected:
            self._source_targets.setdefault(shard.source_name, set()).add(
                shard.target_name
            )
            owner = self._owner(shard.target_name, all_modules)
            if owner is not None:
                shards_by_module.setdefault(owner, set()).add(shard)
        self._module_shards = {
            name: shards_by_module[name] for name in modules if name in shards_by_module
        }
        self._touched_modules: set[str] = set()
        self._module_order = [
            name
            for name in self._module_shards
            if getattr(
                getattr(modules[name], "quant_method", None),
                "supports_incremental_pwal",
                False,
            )
        ]
        self._next_module = 0

    @staticmethod
    def _owner(target_name: str, candidates: set[str]) -> str | None:
        """Find the nearest quantized ancestor that owns a target tensor."""
        matches = [
            name
            for name in candidates
            if target_name == name or target_name.startswith(f"{name}.")
        ]
        return max(matches, key=len) if matches else None

    def before_source(self, source_name: str) -> None:
        """Prepare only the destinations used by the next checkpoint source."""
        targets = self._source_targets.get(source_name, {source_name})
        targets = {
            target
            for target in targets
            if not any(
                target == name or target.startswith(f"{name}.")
                for name in self.skip_modules
            )
        }
        quant_targets = {
            target
            for target in targets
            if any(
                target == name or target.startswith(f"{name}.")
                for name in self._quant_modules
            )
        }
        direct = _bind_runtime_storage(
            self.model, self.runtime, targets - quant_targets
        )
        _materialize_checkpoint_bindings(self.model, self.device, targets - direct)

    def _owner_for_target(self, target_name: str) -> str:
        """Return the deepest registered module owning a target tensor."""
        candidates = [
            name
            for name in self._module_shards
            if target_name == name or target_name.startswith(f"{name}.")
        ]
        return max(candidates, key=len) if candidates else ""

    def note_received(self, received: set[RankShard]) -> None:
        """Record received shards and mark their owning modules as touched."""
        for shard in received:
            owner = self._owner(shard.target_name, self._all_modules)
            if owner is not None and owner not in self._module_shards:
                self._module_shards[owner] = {shard}
        self.received.update(received)
        self._touched_modules.update(
            self._owner_for_target(shard.target_name) for shard in received
        )

    def complete_modules(self) -> frozenset[str]:
        """Return touched modules whose complete manifest shard set arrived."""
        return frozenset(
            name
            for name in self._touched_modules
            if self._module_shards.get(name, set()) <= self.received
        )

    def incomplete_modules(self) -> frozenset[str]:
        """Return touched modules with at least one missing manifest shard."""
        return frozenset(self._touched_modules - self.complete_modules())

    def noncomplete_modules(self) -> frozenset[str]:
        """Return every manifest module excluded from model-level PWAL."""
        return frozenset(set(self._module_shards) - self.complete_modules())

    def pwal_modules(self) -> frozenset[str]:
        """Return quant modules whose checkpoint inputs are complete.

        A quant module is active when it owns a complete manifest target or is
        an ancestor of one. Quant modules with no checkpoint dependency in the
        transaction, such as KV-cache scale holders, retain runtime bindings
        and must not rerun PWAL against construction-time meta tensors.
        """
        complete = self.complete_modules()
        return frozenset(
            quant
            for quant in self._quant_modules
            if any(
                owner == quant or owner.startswith(f"{quant}.") for owner in complete
            )
        )

    def skipped_pwal_modules(self) -> frozenset[str]:
        """Return quant modules that must retain their serving state."""
        return frozenset(self._quant_modules - self.pwal_modules())

    def module_for_target(self, target_name: str) -> str:
        """Return the manifest owner module for a target tensor."""
        return self._owner_for_target(target_name)

    def quant_targets(self, target_names: set[str]) -> set[str]:
        """Return targets owned by a quant module that requires staging."""
        return {
            target
            for target in target_names
            if any(
                target == name or target.startswith(f"{name}.")
                for name in self._quant_modules
            )
        }

    def after_source(self, _source_name: str, received: set[RankShard]) -> None:
        """Run and commit PWAL for newly completed quantized modules.

        Generic quant methods retain lazy staging until model-wide PWAL because
        module-local independence cannot be inferred. Persistent construction-
        time buffers without checkpoint shards are materialized immediately
        before an explicitly supported incremental PWAL.
        """
        self.note_received(received)
        while self._next_module < len(self._module_order):
            module_name = self._module_order[self._next_module]
            expected = self._module_shards[module_name]
            if not expected <= self.received:
                break
            from vllm.model_executor.model_loader.utils import (
                process_quant_method_after_loading,
            )

            module = _modules(self.model)[module_name]
            remaining = {
                f"{name}.{tensor_name}" if name else tensor_name
                for name, _, _, tensor_name, tensor in _iter_direct_tensors(self.model)
                if (name == module_name or name.startswith(f"{module_name}."))
                and tensor.is_meta
            }
            _materialize_checkpoint_bindings(self.model, self.device, remaining)
            _validate_pwal_dependencies_materialized(
                self.model, frozenset({module_name})
            )
            process_quant_method_after_loading(module, self.target_device, force=True)
            _commit_processed_module(self.model, self.runtime, module_name)
            self.committed_modules.add(module_name)
            self._next_module += 1


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
        """Initialize an inactive modelwise reload transaction.

        Args:
            model: Serving model whose weights will be reloaded.
            model_config: Model configuration used by post-load processing.
            target_device: Device on which post-load processing runs.
        """
        self.model = model
        self.model_config = model_config
        self.target_device = target_device
        self._runtime: _RuntimeBindings | None = None
        self._original_torchao = False
        self._loaded_weights: set[str] = set()
        self._loaded_weights_unknown = False
        self._received_shards: set[RankShard] = set()
        self._unit_layers: list[tuple[str, torch.nn.Module, ShardCoverageTracker]] = []
        self._skip_modules: frozenset[str] = frozenset()
        self._lazy_bindings: _LazyCheckpointBindings | None = None

    @property
    def active(self) -> bool:
        """Whether the session currently owns captured runtime bindings."""
        return self._runtime is not None

    @torch.no_grad()
    def start(self) -> None:
        """Start a transaction and restore lazy checkpoint-format bindings.

        Raises:
            RuntimeError: If the session is active, construction metadata is
                missing, or no exact rank-local sharding manifest is available.
        """
        if self.active:
            raise RuntimeError("Model-wise reload session is already active")

        metadata = _MODEL_METADATA.get(self.model)
        if metadata is None:
            raise RuntimeError(
                "Whole-model reload metadata was not recorded during model "
                "initialization"
            )

        manifest = get_rank_sharding_manifest(self.model)
        if manifest.state == "unavailable":
            raise RuntimeError(
                "Whole-model reload has no exact rank-local sharding manifest: "
                f"{manifest.reason}"
            )

        runtime = _capture_runtime_bindings(self.model)
        self._original_torchao = getattr(self.model, "_do_torchao_reload", False)
        self.model._do_torchao_reload = False
        # Layers that can commit a reload unit at a time keep their serving
        # tensors bound: they receive checkpoint shards into small staging
        # slabs and write serving values back in place, so they never need a
        # checkpoint-format copy of the whole layer.
        skip_modules = frozenset()
        trackers = _discover_manifest_expert_trackers(
            self.model, metadata, set(manifest.shards), skip_modules
        )
        try:
            _restore_checkpoint_bindings(self.model, metadata, skip_modules)
            install_sharding_recorders(self.model)
            install_trackers(trackers)
        except BaseException:
            uninstall_trackers(trackers)
            _restore_runtime_bindings(self.model, runtime)
            self.model._do_torchao_reload = self._original_torchao
            raise

        if trackers:
            logger.info(
                "Streaming reload: %d expert layer(s) use manifest-driven "
                "per-expert staging",
                len(trackers),
            )

        self._unit_layers = trackers
        self._skip_modules = skip_modules
        self._runtime = runtime
        self._lazy_bindings = _LazyCheckpointBindings(
            self.model,
            runtime,
            metadata.restore_device,
            self.target_device,
            set(manifest.shards),
            skip_modules,
        )
        self._loaded_weights.clear()
        self._loaded_weights_unknown = False
        self._received_shards.clear()

    @torch.no_grad()
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str] | None:
        """Load sources without exposing stale meta bindings to model loaders.

        ``AutoWeightsLoader`` snapshots a module's direct parameters before it
        consumes every sibling source in that module. Materializing a later
        sibling from an iterator callback would replace the module attribute,
        while the loader still holds the old meta tensor. Run one source per
        model-loader traversal so its destination is materialized before any
        parameter snapshot that can route that source.

        Args:
            weights: Checkpoint source tensors for this transaction chunk.

        Returns:
            The union of loaded target names, or ``None`` if any model-loader
            traversal cannot report its loaded names.
        """
        if not self.active:
            raise RuntimeError("Model-wise reload session is not active")

        lazy_bindings = self._lazy_bindings
        assert lazy_bindings is not None
        loaded_in_call: set[str] = set()
        loaded_unknown = False
        for source in weights:
            source_name = source[0]
            with capture_rank_sharding(self.model) as received:
                loaded = self.model.load_weights(
                    observe_weight_sources(
                        [source],
                        before_yield=lazy_bindings.before_source,
                        after_yield=lambda name: lazy_bindings.after_source(
                            name, received
                        ),
                    )
                )
            if not received and loaded is not None:
                received.update(
                    RankShard(source_name, target_name) for target_name in loaded
                )
            lazy_bindings.note_received(received)
            self._received_shards.update(received)
            if loaded is None:
                loaded_unknown = True
            else:
                loaded_in_call.update(loaded)

        if loaded_unknown:
            self._loaded_weights_unknown = True
            return None
        self._loaded_weights.update(loaded_in_call)
        return loaded_in_call

    def _restore_runtime(self) -> None:
        """Restore serving bindings and clear all transaction-local state."""
        runtime = self._runtime
        if runtime is None:
            return
        uninstall_trackers(self._unit_layers)
        _restore_runtime_bindings(self.model, runtime, self._skip_modules)
        self.model._do_torchao_reload = self._original_torchao
        self._unit_layers = []
        self._skip_modules = frozenset()
        self._lazy_bindings = None
        self._runtime = None

    @torch.no_grad()
    def finish(self) -> set[str] | None:
        """Validate completeness, run PWAL, and commit the transaction.

        Returns:
            Loaded target names, or ``None`` if the model loader did not report
            complete loaded-name information.

        Raises:
            RuntimeError: If no modelwise reload transaction is active.
            ValueError: If processed tensors do not match the serving schema.
        """
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("Model-wise reload session is not active")

        lazy_bindings = self._lazy_bindings
        assert lazy_bindings is not None
        noncomplete_modules = lazy_bindings.noncomplete_modules()
        pwal_modules = lazy_bindings.pwal_modules()
        skipped_pwal_modules = lazy_bindings.skipped_pwal_modules()
        logger.debug(
            "Streaming finish modules: complete=%s pwal=%s skipped_pwal=%s "
            "noncomplete=%s",
            sorted(lazy_bindings.complete_modules()),
            sorted(pwal_modules),
            sorted(skipped_pwal_modules),
            sorted(noncomplete_modules),
        )
        skip_modules = (
            self._skip_modules
            | skipped_pwal_modules
            | noncomplete_modules
            | lazy_bindings.committed_modules
        )
        try:
            from vllm.model_executor.model_loader.utils import (
                process_weights_after_loading,
            )

            # A partial module is intentionally not committed. Restore its
            # serving bindings before post-load processing so a trainer may
            # send only selected layers without corrupting untouched weights.
            _restore_module_runtime_bindings(
                self.model,
                runtime,
                noncomplete_modules | skipped_pwal_modules,
            )

            remaining = {
                f"{module_name}.{name}" if module_name else name
                for module_name, _, _, name, tensor in _iter_direct_tensors(self.model)
                if tensor.is_meta
                and any(
                    module_name == owner or module_name.startswith(f"{owner}.")
                    for owner in pwal_modules
                )
            }
            direct_candidates = remaining - lazy_bindings.quant_targets(remaining)
            direct = _bind_runtime_storage(self.model, runtime, direct_candidates)
            _materialize_checkpoint_bindings(
                self.model, lazy_bindings.device, remaining - direct
            )

            # Complete generic expert trackers now have checkpoint-format
            # destination tensors available for their deferred assembly.
            # Partial units are dropped without failing the whole transaction.
            for _, _, tracker in self._unit_layers:
                tracker.finish(fail_on_partial=False)

            _audit_meta_bindings_before_pwal(
                self.model,
                pwal_modules,
                skipped_pwal_modules,
            )
            process_weights_after_loading(
                self.model,
                self.model_config,
                self.target_device,
                force=True,
                skip_modules=skip_modules,
            )
            processed = _capture_runtime_bindings(self.model)
            _validate_copy_back(processed, runtime, skip_modules)
            _copy_back(processed, runtime, skip_modules)
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
        """Configure a one-shot modelwise checkpoint reloader.

        Args:
            model: Serving model whose weights will be reloaded.
            model_config: Model configuration used by post-load processing.
            target_device: Device on which post-load processing runs.
        """
        self.model = model
        self.model_config = model_config
        self.target_device = target_device

    @torch.no_grad()
    def reload(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str] | None:
        """Execute a complete modelwise reload transaction.

        Args:
            weights: Checkpoint source tensors to load.

        Returns:
            Loaded target names, or ``None`` when the model loader cannot
            report complete loaded-name information.

        Raises:
            BaseException: Re-raises any start, load, or finish failure after
                restoring the original serving bindings.
        """
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
