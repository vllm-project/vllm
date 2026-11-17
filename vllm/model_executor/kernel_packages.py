# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer sealed kernel packages at vLLM functions marked `kernel_site`."""

import functools
import inspect
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_routes: dict[str, list[tuple[dict[str, Any], Callable[..., Any]]]] = {}


def kernel_site(fn: Callable[..., Any]) -> Callable[..., Any]:
    site = f"{fn.__module__}.{fn.__qualname__}"
    params = inspect.signature(fn).parameters

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if routes := _routes.get(site):
            arguments = dict(zip(params, args)) | kwargs
            for definition, run in routes:
                if not (reason := _misfit(definition, arguments)):
                    result = run(**{n: arguments[n] for n in definition["inputs"]})
                    _count(site, "hit")
                    return result
            _count(site, f"fallback_{reason}")
        return fn(*args, **kwargs)

    return wrapper


def refuse_weight_rewrites(*features: str) -> None:
    packages = sys.modules.get("flashinfer.trace_apply.packages")
    if features and packages and (prepared := packages.setup_stats()):
        raise ValueError(
            f"FlashInfer kernel packages for {', '.join(sorted(prepared))} keep state "
            f"prepared from the weights, which {', '.join(features)} would rewrite."
        )


def activate(vllm_config: "VllmConfig") -> None:
    _routes.clear()
    if not (selection := os.environ.get("FLASHINFER_KERNEL_PACKAGES", "").strip()):
        return
    try:
        from flashinfer.trace_apply import packages, preloaded_entry
    except ImportError:
        logger.warning("FLASHINFER_KERNEL_PACKAGES needs FlashInfer >= 0.7.5; ignored.")
        return

    offload = vllm_config.offload_config
    features = {
        "sleep mode": vllm_config.model_config.enable_sleep_mode,
        "EPLB": vllm_config.parallel_config.enable_eplb,
        "weight transfer": vllm_config.weight_transfer_config is not None,
        "CPU offload": offload.uva.cpu_offload_gb > 0
        or offload.prefetch.offload_group_size > 0,
    }
    refuse_weight_rewrites(*(name for name, on in features.items() if on))

    root = Path(os.environ.get(packages.ROOT_ENV) or packages.DEFAULT_PACKAGE_ROOT)
    routed = set()
    for digest in filter(None, map(str.strip, selection.split(","))):
        try:
            package = packages.verify_package(root, digest)
            run = preloaded_entry(package.operation, str(package.solution.get("name")))
            tags = (package.definition or {}).get("tags") or ()
            site = next((t[7:] for t in tags if t.startswith("fi_api:")), "")
        except Exception:
            continue  # FlashInfer refused it at import, with a warning
        if run is None or package.operation in routed:
            continue
        routed.add(package.operation)
        _routes.setdefault(site, []).append((package.definition, run))
        logger.info_once(
            "FlashInfer kernel package %s: %s.", digest[:12], site or package.operation
        )


def _misfit(definition: dict[str, Any], arguments: dict[str, Any]) -> str:
    inputs = definition["inputs"]
    axes = {
        name: axis.get("value")
        for name, axis in (definition.get("axes") or {}).items()
        if axis.get("type") == "const"
    }
    for name, spec in inputs.items():
        if (value := arguments.get(name)) is None:
            return f"missing_{name}"
        if not (shape := spec.get("shape")):
            continue
        if getattr(value, "ndim", None) != len(shape):
            return f"rank_{name}"
        if spec.get("dtype") not in (None, str(value.dtype).removeprefix("torch.")):
            return f"dtype_{name}"
        for axis, size in zip(shape, value.shape):
            bound = axes.setdefault(axis, int(axis) if str(axis).isdigit() else size)
            if bound != size:
                return f"axis_{axis}"
    extra = [n for n, v in arguments.items() if n not in inputs and v is not None]
    return f"extra_{extra[0]}" if extra else ""


def _count(site: str, status: str) -> None:
    from flashinfer.trace_apply.apply import bump_stat

    bump_stat(site, status)
    logger.info_once("%s: FlashInfer kernel package %s.", site, status)
