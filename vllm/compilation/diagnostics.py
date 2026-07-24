# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

_MISSING = object()
_CACHE_FACTOR_KEYS = frozenset(
    {
        "env_hash",
        "config_hash",
        "compiler_hash",
        "code_hash",
    }
)


def _record_error(errors: list[str], name: str, exc: Exception) -> None:
    errors.append(f"{name}: {type(exc).__name__}: {exc}")


def _safe_getattr(obj: Any, name: str, errors: list[str]) -> Any:
    try:
        return getattr(obj, name)
    except AttributeError:
        return _MISSING
    except Exception as exc:
        _record_error(errors, name, exc)
        return _MISSING


def _json_safe(value: Any) -> Any:
    if value is _MISSING:
        return _MISSING
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): safe_value
            for key, item in value.items()
            if (safe_value := _json_safe(item)) is not _MISSING
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            safe_item
            for item in value
            if (safe_item := _json_safe(item)) is not _MISSING
        ]
    return str(value)


def _set_if_available(target: dict[str, Any], key: str, value: Any) -> None:
    safe_value = _json_safe(value)
    if safe_value is not _MISSING:
        target[key] = safe_value


def _is_compilation_enabled(mode: Any) -> bool | None:
    if mode is _MISSING or mode is None:
        return None
    mode_value = getattr(mode, "value", mode)
    mode_name = getattr(mode, "name", None)
    return not (mode_value == 0 or mode_name == "NONE" or str(mode).upper() == "NONE")


def _collect_torch_runtime(errors: list[str]) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:
        cuda_available = False
        _record_error(errors, "torch.cuda.is_available", exc)
    runtime["cuda_available"] = cuda_available

    if cuda_available:
        try:
            runtime["cuda_device_capability"] = list(torch.cuda.get_device_capability())
        except Exception as exc:
            _record_error(errors, "torch.cuda.get_device_capability", exc)

    return runtime


def get_compile_diagnostics(
    vllm_config: Any,
    *,
    cache_dir: str | None = None,
    local_cache_dir: str | None = None,
    cache_factors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect safe, JSON-serializable torch.compile diagnostics.

    The helper intentionally summarizes already-available config/cache metadata.
    It does not recompute cache keys, inspect environment variables, or dump full
    configs.
    """

    errors: list[str] = []
    diagnostics: dict[str, Any] = {}

    compilation_config = _safe_getattr(vllm_config, "compilation_config", errors)
    compilation: dict[str, Any] = {}
    if compilation_config is not _MISSING:
        mode = _safe_getattr(compilation_config, "mode", errors)
        compilation_enabled = _is_compilation_enabled(mode)
        if compilation_enabled is not None:
            compilation["enabled"] = compilation_enabled
        _set_if_available(compilation, "mode", mode)
        for key in (
            "backend",
            "compile_sizes",
            "compile_ranges_split_points",
            "debug_dump_path",
        ):
            _set_if_available(
                compilation,
                key,
                _safe_getattr(compilation_config, key, errors),
            )

    optimization_level = _safe_getattr(vllm_config, "optimization_level", errors)
    _set_if_available(compilation, "optimization_level", optimization_level)

    if compilation:
        diagnostics["compilation"] = compilation

    cache: dict[str, Any] = {}
    _set_if_available(cache, "cache_dir", cache_dir)
    _set_if_available(cache, "local_cache_dir", local_cache_dir)
    if cache_factors is not None:
        try:
            factors = {
                key: value
                for key, value in cache_factors.items()
                if key in _CACHE_FACTOR_KEYS
            }
            if factors:
                cache["cache_factors"] = _json_safe(factors)
        except Exception as exc:
            _record_error(errors, "cache_factors", exc)
    if cache:
        diagnostics["cache"] = cache

    diagnostics["runtime"] = _collect_torch_runtime(errors)
    if errors:
        diagnostics["errors"] = errors

    return diagnostics


def format_compile_diagnostics(diagnostics: Mapping[str, Any]) -> str:
    """Format compile diagnostics for logs."""

    return json.dumps(_json_safe(diagnostics), indent=2, sort_keys=True)
