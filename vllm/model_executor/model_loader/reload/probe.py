# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata-only execution of model weight loaders."""

from collections.abc import Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

import torch
from safetensors import safe_open
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from vllm.model_executor.model_loader.reload.plan import (
    get_current_load_source,
    get_recorded_load_plan,
)

__all__ = [
    "LoadProbeError",
    "LoadProbeFinding",
    "LoadProbeMode",
    "LoadProbeReport",
    "probe_model_load",
    "safetensors_meta_weights",
    "safetensors_meta_weights_from_files",
    "validate_probe_plan_coverage",
]


class LoadProbeError(RuntimeError):
    """Raised when a loader cannot be executed without touching tensor data."""


@dataclass(frozen=True)
class LoadProbeFinding:
    code: str
    operation: str
    detail: str

    def format(self) -> str:
        return f"{self.code}: {self.operation}: {self.detail}"


@dataclass
class LoadProbeReport:
    """Result of one metadata-only ``model.load_weights`` execution."""

    loaded_weights: set[str] = field(default_factory=set)
    intercepted_writes: list[str] = field(default_factory=list)
    write_sources: set[str | None] = field(default_factory=set)
    restored_python_mutations: list[str] = field(default_factory=list)
    findings: list[LoadProbeFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    def raise_for_error(self) -> None:
        if self.findings:
            raise LoadProbeError(
                "Weight-loader probe is incomplete:\n  "
                + "\n  ".join(finding.format() for finding in self.findings)
            )


def _tensor_values(args: tuple[Any, ...], kwargs: dict[str, Any]):
    flat, _ = tree_flatten((args, kwargs))
    return [value for value in flat if isinstance(value, torch.Tensor)]


def _write_arguments(
    func,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[torch.Tensor]:
    writes: list[torch.Tensor] = []
    for index, argument in enumerate(func._schema.arguments):
        if index < len(args):
            value = args[index]
        elif argument.name in kwargs:
            value = kwargs[argument.name]
        else:
            continue
        alias = argument.alias_info
        if alias is not None and alias.is_write and isinstance(value, torch.Tensor):
            writes.append(value)
    return writes


def _has_meta_tensor(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    return any(tensor.is_meta for tensor in _tensor_values(args, kwargs))


_FACTORY_OPS = {
    "aten::arange",
    "aten::empty",
    "aten::empty_like",
    "aten::empty_strided",
    "aten::full",
    "aten::full_like",
    "aten::new_empty",
    "aten::new_empty_strided",
    "aten::new_full",
    "aten::new_ones",
    "aten::new_zeros",
    "aten::ones",
    "aten::ones_like",
    "aten::rand",
    "aten::rand_like",
    "aten::randn",
    "aten::randn_like",
    "aten::zeros",
    "aten::zeros_like",
}

_SAFETENSORS_DTYPES = {
    "BOOL": "bool",
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "U8": "uint8",
    "U16": "uint16",
    "U32": "uint32",
    "U64": "uint64",
}


def safetensors_meta_weights_from_files(
    filenames: Iterable[str],
) -> list[tuple[str, torch.Tensor]]:
    """Read safetensors schemas without materializing tensor data."""
    weights = []
    seen = set()
    for filename in filenames:
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for name in list(handle.keys()):
                if name in seen:
                    raise LoadProbeError(f"duplicate safetensors source key {name!r}")
                seen.add(name)
                tensor_slice = handle.get_slice(name)
                dtype_name = tensor_slice.get_dtype()
                torch_name = _SAFETENSORS_DTYPES.get(dtype_name)
                dtype = getattr(torch, torch_name, None) if torch_name else None
                if not isinstance(dtype, torch.dtype):
                    raise LoadProbeError(
                        "unsupported safetensors dtype "
                        f"{dtype_name!r} for source {name!r}"
                    )
                weights.append(
                    (
                        name,
                        torch.empty(
                            tensor_slice.get_shape(),
                            dtype=dtype,
                            device="meta",
                        ),
                    )
                )
    return weights


def safetensors_meta_weights(
    model_path: str,
) -> list[tuple[str, torch.Tensor]]:
    """Read a local safetensors schema without materializing tensor data."""
    from pathlib import Path

    path = Path(model_path)
    if not path.is_dir():
        return []
    filenames = sorted(str(filename) for filename in path.glob("*.safetensors"))
    if not filenames:
        return []
    return safetensors_meta_weights_from_files(filenames)


class LoadProbeMode(TorchDispatchMode):
    """Suppress tensor mutations while preserving loader control flow.

    Source weights must be meta tensors. Metadata/view operations execute
    normally. Mutable dispatcher operations are recorded and return their
    destination without running a kernel.
    """

    def __init__(self) -> None:
        super().__init__()
        self.report = LoadProbeReport()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        writes = _write_arguments(func, args, kwargs)
        if writes:
            self.report.intercepted_writes.append(str(func))
            self.report.write_sources.add(get_current_load_source())
            if len(func._schema.returns) == 0:
                return None
            if len(func._schema.returns) == 1:
                return writes[0]
            finding = LoadProbeFinding(
                code="PROBE_UNSUPPORTED_MUTATION_RETURN",
                operation=str(func),
                detail=f"mutable operator has {len(func._schema.returns)} returns",
            )
            self.report.findings.append(finding)
            raise LoadProbeError(finding.format())

        if func._schema.name in _FACTORY_OPS or (
            _has_meta_tensor(args, kwargs) and func._schema.name == "aten::_to_copy"
        ):
            kwargs["device"] = torch.device("meta")

        try:
            return func(*args, **kwargs)
        except Exception as exc:
            finding = LoadProbeFinding(
                code="PROBE_UNSUPPORTED_OPERATOR",
                operation=str(func),
                detail=f"{type(exc).__name__}: {exc}",
            )
            self.report.findings.append(finding)
            raise LoadProbeError(finding.format()) from exc


def _copy_attribute(value: Any) -> Any:
    if isinstance(value, dict):
        return value.copy()
    if isinstance(value, list):
        return value.copy()
    if isinstance(value, set):
        return value.copy()
    return value


def _same_attribute(current: Any, saved: Any) -> bool:
    if isinstance(saved, dict):
        return (
            isinstance(current, dict)
            and current.keys() == saved.keys()
            and all(current[key] is value for key, value in saved.items())
        )
    if isinstance(saved, list):
        return (
            isinstance(current, list)
            and len(current) == len(saved)
            and all(left is right for left, right in zip(current, saved))
        )
    if isinstance(saved, set):
        return isinstance(current, set) and current == saved
    return current is saved


class _ModelStateGuard(AbstractContextManager):
    """Restore Python-side model bindings after a probe."""

    def __init__(
        self,
        model: torch.nn.Module,
        report: LoadProbeReport,
        *,
        allow_binding_mutations: bool = False,
    ):
        self.model = model
        self.report = report
        self.allow_binding_mutations = allow_binding_mutations
        self.module_state: dict[torch.nn.Module, dict[str, Any]] = {}
        self.tensor_state: dict[torch.Tensor, dict[str, Any]] = {}

    def __enter__(self):
        for module in self.model.modules():
            self.module_state[module] = {
                key: _copy_attribute(value) for key, value in vars(module).items()
            }
            for tensor in (
                *module.parameters(recurse=False),
                *module.buffers(recurse=False),
            ):
                self.tensor_state[tensor] = {
                    key: _copy_attribute(value) for key, value in vars(tensor).items()
                }
        return self

    def __exit__(self, exc_type, exc, traceback):
        mutated = []
        binding_mutations = []
        for module, state in self.module_state.items():
            current = vars(module)
            changed = {
                key
                for key in set(current) | set(state)
                if key not in current
                or key not in state
                or not _same_attribute(current[key], state[key])
            }
            if changed:
                mutated.append(type(module).__name__)
                if changed & {"_parameters", "_buffers", "_modules"}:
                    binding_mutations.append(
                        f"{type(module).__name__}:{sorted(changed)}"
                    )
            current.clear()
            current.update(state)
        for tensor, state in self.tensor_state.items():
            current = vars(tensor)
            if set(current) != set(state):
                mutated.append(f"{type(tensor).__name__}.__dict__")
            current.clear()
            current.update(state)
        if binding_mutations and not self.allow_binding_mutations:
            self.report.findings.append(
                LoadProbeFinding(
                    code="PROBE_TENSOR_BINDING_MUTATION",
                    operation=f"{type(self.model).__name__}.load_weights",
                    detail=f"restored {binding_mutations[:8]}",
                )
            )
        if mutated:
            self.report.restored_python_mutations.extend(sorted(set(mutated)))
        return False


@torch.no_grad()
def probe_model_load(
    model: torch.nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> LoadProbeReport:
    """Run the model's real weight routing without copying tensor data.

    Args:
        model: Initialized model whose ``load_weights`` method is probed.
        weights: Source-name and metadata-tensor pairs. Every tensor must be
            on the meta device.

    Returns:
        The intercepted writes and the names returned by ``model.load_weights``.

    Raises:
        ValueError: If a source tensor owns real storage.
        LoadProbeError: If an operator cannot be represented metadata-only.
    """
    materialized = [(name, tensor) for name, tensor in weights]
    non_meta = [name for name, tensor in materialized if not tensor.is_meta]
    if non_meta:
        raise ValueError(
            "Weight-loader probe requires meta source tensors; got real "
            f"storage for {non_meta[:8]}"
        )

    mode = LoadProbeMode()
    try:
        with _ModelStateGuard(model, mode.report), mode:
            loaded = model.load_weights(materialized)
        if loaded is not None:
            mode.report.loaded_weights = set(loaded)
    except LoadProbeError:
        raise
    except Exception as exc:
        finding = LoadProbeFinding(
            code="PROBE_LOADER_EXCEPTION",
            operation=f"{type(model).__name__}.load_weights",
            detail=f"{type(exc).__name__}: {exc}",
        )
        mode.report.findings.append(finding)
        raise LoadProbeError(finding.format()) from exc
    return mode.report


def validate_probe_plan_coverage(
    model: torch.nn.Module,
    report: LoadProbeReport,
) -> None:
    """Require every intercepted source write to have a recorded load key."""
    recorded_sources = {
        key[0]
        for layer in model.modules()
        for key in (get_recorded_load_plan(layer) or ())
    }
    missing_keys = {
        source
        for source in report.write_sources
        if source is None or source not in recorded_sources
    }
    if missing_keys:
        raise LoadProbeError(
            "Weight-loader probe observed tensor writes without LoadPlan "
            "coverage for source(s): "
            f"{sorted(map(str, missing_keys))[:20]}"
        )
