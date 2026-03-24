# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CLI-local backend diagnostics and selection helpers.

This module intentionally stays in the CLI layer. It surfaces backend and
platform information without changing vLLM's core execution architecture.
"""

from __future__ import annotations

import platform as py_platform
import re
from dataclasses import asdict, dataclass, field
from importlib.metadata import entry_points
from typing import Any

from vllm.plugins import PLATFORM_PLUGINS_GROUP


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    description: str
    gpu_memory_utilization: float | None = None
    enable_prefix_caching: bool | None = None
    enforce_eager: bool | None = None
    notes: tuple[str, ...] = ()


RUNTIME_PROFILES: dict[str, RuntimeProfile] = {
    "balanced": RuntimeProfile(
        name="balanced",
        description="Default local profile with strong throughput and safe memory headroom.",
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        enforce_eager=None,
        notes=(
            "Keeps prefix caching enabled when the backend supports it.",
            "Leaves graph capture mode unchanged when the backend can decide safely.",
        ),
    ),
    "throughput": RuntimeProfile(
        name="throughput",
        description="Favor higher steady-state throughput when memory headroom allows it.",
        gpu_memory_utilization=0.95,
        enable_prefix_caching=True,
        enforce_eager=False,
        notes=(
            "Pushes cache utilization higher for better batch throughput.",
            "Prefers graph-backed execution when the backend supports it.",
        ),
    ),
    "low-memory": RuntimeProfile(
        name="low-memory",
        description="Reduce memory pressure for local machines with tighter budgets.",
        gpu_memory_utilization=0.82,
        enable_prefix_caching=False,
        enforce_eager=True,
        notes=(
            "Disables prefix caching to reduce KV cache pressure.",
            "Prefers eager mode to avoid graph-capture memory overhead.",
        ),
    ),
}


@dataclass
class BackendCapability:
    name: str
    source: str
    available: bool
    selected: bool = False
    device_name: str | None = None
    device_type: str | None = None
    performance_tier: str = "fallback"
    reason: str | None = None
    supported_dtypes: list[str] = field(default_factory=list)
    supported_quantization: list[str] = field(default_factory=list)
    supported_model_families: list[str] = field(default_factory=list)
    attention_backends: list[str] = field(default_factory=list)
    paged_attention: bool | None = None
    prefix_caching: bool | None = None
    graph_capture: bool | None = None
    interop: list[str] = field(default_factory=list)
    total_memory_bytes: int | None = None
    plugin_name: str | None = None
    plugin_target: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BackendSelection:
    requested_backend: str
    selected_backend: str
    selected_reason: str
    fallback_reason: str | None = None
    rejected_backends: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelPreflight:
    model: str
    dtype: str
    quantization: str | None
    max_model_len: int
    parameter_count: float | None
    estimated_weight_bytes: int | None
    estimated_kv_cache_bytes: int | None
    estimated_runtime_overhead_bytes: int | None
    estimated_total_bytes: int | None
    available_memory_bytes: int | None
    fit: bool | None
    summary: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrtllmDiagnostics:
    environment_supported: bool
    model_supported: bool | None
    eligible: bool
    reasons: list[str]
    flashinfer_available: bool
    flashinfer_trtllm_moe_available: bool
    sink_attention_supported: bool
    ragged_mla_supported: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DoctorReport:
    os: str
    architecture: str
    current_platform: str
    current_device_type: str
    current_device_name: str
    requested_backend: str
    selected_backend: str
    selection_reason: str
    fallback_reason: str | None
    available_plugins: list[dict[str, str]]
    backends: list[BackendCapability]
    model: str | None = None
    profile: str = "balanced"
    preflight: ModelPreflight | None = None
    trtllm: TrtllmDiagnostics | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backends"] = [backend.to_dict() for backend in self.backends]
        if self.preflight is not None:
            payload["preflight"] = self.preflight.to_dict()
        if self.trtllm is not None:
            payload["trtllm"] = self.trtllm.to_dict()
        return payload


def get_runtime_profile(name: str) -> RuntimeProfile:
    try:
        return RUNTIME_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown profile `{name}`. Choose from: {', '.join(RUNTIME_PROFILES)}."
        ) from exc


def _safe_import_torch():
    try:
        import torch
    except Exception:
        return None
    return torch


def _safe_import_psutil():
    try:
        import psutil
    except Exception:
        return None
    return psutil


def _safe_import_current_platform():
    try:
        from vllm.platforms import current_platform
    except Exception:
        return None
    return current_platform


def _discover_platform_plugins() -> list[dict[str, str]]:
    plugins = []
    try:
        discovered = entry_points(group=PLATFORM_PLUGINS_GROUP)
    except Exception:
        return plugins

    for plugin in discovered:
        plugins.append({"name": plugin.name, "value": plugin.value})
    return plugins


def _apple_plugin_record(
    plugins: list[dict[str, str]],
) -> dict[str, str] | None:
    for plugin in plugins:
        haystack = f"{plugin['name']} {plugin['value']}".lower()
        if any(token in haystack for token in ("metal", "mlx", "apple")):
            return plugin
    return None


def _dtype_names(dtypes: list[Any]) -> list[str]:
    names = []
    for dtype in dtypes:
        value = getattr(dtype, "name", None)
        if value is None:
            value = str(dtype).replace("torch.", "")
        names.append(value)
    return names


def _family_hint(model_ref: str) -> str:
    lowered = model_ref.lower()
    for family in (
        "deepseek",
        "llama",
        "qwen",
        "gemma",
        "mistral",
        "mixtral",
        "phi",
        "smollm",
    ):
        if family in lowered:
            return family
    return "generic"


def _parse_parameter_count(model_ref: str) -> float | None:
    lowered = model_ref.lower()
    moe_match = re.search(r"(\d+)x(\d+(?:\.\d+)?)b", lowered)
    if moe_match:
        experts = float(moe_match.group(1))
        per_expert = float(moe_match.group(2))
        return experts * per_expert * 1_000_000_000

    size_match = re.search(r"(\d+(?:\.\d+)?)([bm])", lowered)
    if not size_match:
        return None

    value = float(size_match.group(1))
    unit = size_match.group(2)
    if unit == "b":
        return value * 1_000_000_000
    return value * 1_000_000


def _bytes_per_parameter(dtype: str, quantization: str | None, backend: str) -> float:
    quant = (quantization or "").lower()
    dtype = dtype.lower()

    if quant:
        if any(token in quant for token in ("4", "awq", "gptq")):
            return 0.5
        if any(token in quant for token in ("8", "fp8", "int8")):
            return 1.0

    if dtype in {"float32", "fp32"}:
        return 4.0
    if dtype in {"float16", "fp16", "bfloat16", "bf16", "half"}:
        return 2.0
    if dtype == "auto":
        return 4.0 if backend == "cpu" else 2.0
    return 2.0


def _format_bytes(num_bytes: int | None) -> str | None:
    if num_bytes is None:
        return None
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TiB"


def _get_available_memory_bytes(selected_backend: str, current_platform) -> int | None:
    if current_platform is not None and selected_backend in {
        "cuda",
        "rocm",
        "xpu",
        "apple-metal",
        "cpu",
    }:
        try:
            return int(current_platform.get_device_total_memory())
        except Exception:
            pass

    psutil = _safe_import_psutil()
    if psutil is None:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def collect_backend_capabilities() -> tuple[list[BackendCapability], dict[str, str], Any]:
    system = py_platform.system()
    machine = py_platform.machine().lower()
    torch = _safe_import_torch()
    current_platform = _safe_import_current_platform()
    plugins = _discover_platform_plugins()
    apple_plugin = _apple_plugin_record(plugins)

    current_name = "unavailable"
    current_device_type = "unknown"
    current_device_name = "unknown"
    current_dtypes: list[str] = []
    current_quants: list[str] = []
    current_memory: int | None = None
    current_graph_capture: bool | None = None

    if current_platform is not None:
        current_name = current_platform.__class__.__name__
        current_device_type = getattr(current_platform, "device_type", "unknown")
        current_device_name = getattr(current_platform, "device_name", "unknown")
        with_memory = getattr(current_platform, "get_device_total_memory", None)
        if callable(with_memory):
            try:
                current_memory = int(with_memory())
            except Exception:
                current_memory = None
        current_dtypes = _dtype_names(getattr(current_platform, "supported_dtypes", []))
        current_quants = list(getattr(current_platform, "supported_quantization", []))
        support_static = getattr(current_platform, "support_static_graph_mode", None)
        if callable(support_static):
            try:
                current_graph_capture = bool(support_static())
            except Exception:
                current_graph_capture = None

    cuda_available = bool(current_platform and current_platform.is_cuda())
    rocm_available = bool(current_platform and current_platform.is_rocm())
    xpu_available = bool(current_platform and current_platform.is_xpu())
    cpu_available = True
    apple_plugin_active = bool(
        current_platform is not None
        and current_platform.is_out_of_tree()
        and current_device_type != "cpu"
    )
    apple_available = bool(
        system == "Darwin" and machine in {"arm64", "aarch64"} and apple_plugin_active
    )

    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        cuda_available = cuda_available or True

    backends = [
        BackendCapability(
            name="cuda",
            source="builtin",
            available=cuda_available,
            selected=bool(current_platform and current_platform.is_cuda()),
            device_name=(
                current_device_name if cuda_available and current_platform else None
            ),
            device_type="cuda",
            performance_tier="production",
            reason=(
                "Detected NVIDIA CUDA runtime."
                if cuda_available
                else "CUDA runtime was not detected."
            ),
            supported_dtypes=(
                current_dtypes
                if cuda_available
                else ["bfloat16", "float16", "float32"]
            ),
            supported_quantization=(
                current_quants if cuda_available else ["awq", "gptq", "fp8"]
            ),
            supported_model_families=[
                "generic",
                "deepseek",
                "llama",
                "qwen",
                "mixtral",
                "phi",
            ],
            attention_backends=["flashinfer", "triton", "torch_sdpa"],
            paged_attention=True if cuda_available else None,
            prefix_caching=True,
            graph_capture=current_graph_capture if cuda_available else True,
            interop=["TensorRT-LLM", "FlashInfer"],
            total_memory_bytes=current_memory if cuda_available else None,
        ),
        BackendCapability(
            name="rocm",
            source="builtin",
            available=rocm_available,
            selected=bool(current_platform and current_platform.is_rocm()),
            device_name=(
                current_device_name if rocm_available and current_platform else None
            ),
            device_type="cuda",
            performance_tier="production",
            reason=(
                "Detected ROCm runtime."
                if rocm_available
                else "ROCm runtime was not detected."
            ),
            supported_dtypes=(
                current_dtypes
                if rocm_available
                else ["bfloat16", "float16", "float32"]
            ),
            supported_quantization=current_quants if rocm_available else [],
            supported_model_families=[
                "generic",
                "llama",
                "qwen",
                "mixtral",
                "phi",
            ],
            attention_backends=["triton", "torch_sdpa"],
            paged_attention=True if rocm_available else None,
            prefix_caching=True,
            graph_capture=current_graph_capture if rocm_available else None,
            interop=[],
            total_memory_bytes=current_memory if rocm_available else None,
        ),
        BackendCapability(
            name="xpu",
            source="builtin",
            available=xpu_available,
            selected=bool(current_platform and current_platform.is_xpu()),
            device_name=(
                current_device_name if xpu_available and current_platform else None
            ),
            device_type="xpu",
            performance_tier="production",
            reason=(
                "Detected Intel XPU runtime."
                if xpu_available
                else "XPU runtime was not detected."
            ),
            supported_dtypes=(
                current_dtypes
                if xpu_available
                else ["bfloat16", "float16", "float32"]
            ),
            supported_quantization=current_quants if xpu_available else [],
            supported_model_families=["generic", "llama", "qwen"],
            attention_backends=["torch_sdpa"],
            paged_attention=True if xpu_available else None,
            prefix_caching=True,
            graph_capture=current_graph_capture if xpu_available else None,
            interop=[],
            total_memory_bytes=current_memory if xpu_available else None,
        ),
        BackendCapability(
            name="apple-metal",
            source="plugin",
            available=apple_available,
            selected=bool(
                current_platform
                and current_platform.is_out_of_tree()
                and system == "Darwin"
                and machine in {"arm64", "aarch64"}
            ),
            device_name=(
                current_device_name
                if apple_available and current_platform
                else "Apple Silicon"
            ),
            device_type="metal",
            performance_tier="local",
            reason=(
                f"Active Apple GPU plugin `{apple_plugin['name']}` is selected."
                if apple_available and apple_plugin is not None
                else (
                    (
                        "Detected Apple GPU plugin package(s), but no Apple GPU "
                        "platform is active."
                    )
                    if apple_plugin is not None
                    else (
                        "Apple GPU backend requires an out-of-tree plugin such "
                        "as vllm-metal."
                    )
                )
            ),
            supported_dtypes=(
                current_dtypes
                if apple_available
                else ["float16", "bfloat16", "float32"]
            ),
            supported_quantization=current_quants if apple_available else [],
            supported_model_families=[
                "generic",
                "llama",
                "qwen",
                "phi",
                "gemma",
            ],
            attention_backends=["metal", "mlx"],
            paged_attention=None,
            prefix_caching=True,
            graph_capture=current_graph_capture if apple_available else None,
            interop=[],
            total_memory_bytes=current_memory if apple_available else None,
            plugin_name=apple_plugin["name"] if apple_plugin is not None else None,
            plugin_target=(
                apple_plugin["value"] if apple_plugin is not None else None
            ),
        ),
        BackendCapability(
            name="cpu",
            source="builtin",
            available=cpu_available,
            selected=bool(current_platform and current_platform.is_cpu()),
            device_name=(
                current_device_name
                if current_platform and current_platform.is_cpu()
                else "CPU"
            ),
            device_type="cpu",
            performance_tier="fallback",
            reason="CPU fallback is always available.",
            supported_dtypes=(
                current_dtypes
                if current_platform and current_platform.is_cpu()
                else ["float32", "bfloat16"]
            ),
            supported_quantization=(
                current_quants
                if current_platform and current_platform.is_cpu()
                else []
            ),
            supported_model_families=[
                "generic",
                "llama",
                "qwen",
                "phi",
                "gemma",
            ],
            attention_backends=["torch_sdpa"],
            paged_attention=None,
            prefix_caching=False,
            graph_capture=False,
            interop=[],
            total_memory_bytes=(
                current_memory
                if current_platform and current_platform.is_cpu()
                else None
            ),
        ),
    ]

    environment = {
        "os": system,
        "architecture": machine,
        "current_platform": current_name,
        "current_device_type": current_device_type,
        "current_device_name": current_device_name,
    }
    return backends, environment, current_platform


def select_backend(
    requested_backend: str = "auto",
) -> tuple[BackendSelection, list[BackendCapability], dict[str, str], Any]:
    backends, environment, current_platform = collect_backend_capabilities()
    backend_map = {backend.name: backend for backend in backends}
    rejected: dict[str, str] = {}

    def mark_selected(name: str) -> None:
        for backend in backends:
            backend.selected = backend.name == name

    if requested_backend != "auto":
        chosen = backend_map.get(requested_backend)
        if chosen is None:
            raise ValueError(
                f"Unknown backend `{requested_backend}`. Choose from: auto, "
                + ", ".join(backend_map)
            )
        if chosen.available:
            mark_selected(chosen.name)
            return (
                BackendSelection(
                    requested_backend=requested_backend,
                    selected_backend=chosen.name,
                    selected_reason=f"Selected `{chosen.name}` because the user requested it.",
                ),
                backends,
                environment,
                current_platform,
            )
        rejected[chosen.name] = chosen.reason or "backend unavailable"
        fallback = backend_map["cpu"]
        mark_selected(fallback.name)
        return (
            BackendSelection(
                requested_backend=requested_backend,
                selected_backend=fallback.name,
                selected_reason=(
                    f"Falling back to `{fallback.name}` because the requested "
                    f"backend `{requested_backend}` is unavailable."
                ),
                fallback_reason=chosen.reason or "Requested backend unavailable.",
                rejected_backends=rejected,
            ),
            backends,
            environment,
            current_platform,
        )

    if environment["os"] == "Darwin" and environment["architecture"] in {"arm64", "aarch64"}:
        if backend_map["apple-metal"].available:
            mark_selected("apple-metal")
            return (
                BackendSelection(
                    requested_backend="auto",
                    selected_backend="apple-metal",
                    selected_reason=(
                        "Selected `apple-metal` because an Apple GPU plugin is "
                        "available on Apple Silicon."
                    ),
                ),
                backends,
                environment,
                current_platform,
            )
        rejected["apple-metal"] = backend_map["apple-metal"].reason or "plugin unavailable"

    for candidate, reason in (
        ("cuda", "Selected `cuda` because NVIDIA acceleration is available."),
        ("rocm", "Selected `rocm` because AMD ROCm acceleration is available."),
        ("xpu", "Selected `xpu` because Intel XPU acceleration is available."),
    ):
        if backend_map[candidate].available:
            mark_selected(candidate)
            return (
                BackendSelection(
                    requested_backend="auto",
                    selected_backend=candidate,
                    selected_reason=reason,
                    rejected_backends=rejected,
                ),
                backends,
                environment,
                current_platform,
            )
        rejected[candidate] = backend_map[candidate].reason or "backend unavailable"

    mark_selected("cpu")
    fallback_reason = None
    if rejected:
        fallback_reason = "; ".join(
            f"{name}: {reason}" for name, reason in rejected.items() if reason
        )
    return (
        BackendSelection(
            requested_backend="auto",
            selected_backend="cpu",
            selected_reason=(
                "Selected `cpu` because no accelerator backend was available."
            ),
            fallback_reason=fallback_reason,
            rejected_backends=rejected,
        ),
        backends,
        environment,
        current_platform,
    )


def estimate_model_preflight(
    model: str,
    *,
    selected_backend: str,
    dtype: str = "auto",
    quantization: str | None = None,
    max_model_len: int | None = None,
    current_platform=None,
) -> ModelPreflight:
    max_model_len = max_model_len or 8192
    params = _parse_parameter_count(model)
    family = _family_hint(model)

    if params is None:
        return ModelPreflight(
            model=model,
            dtype=dtype,
            quantization=quantization,
            max_model_len=max_model_len,
            parameter_count=None,
            estimated_weight_bytes=None,
            estimated_kv_cache_bytes=None,
            estimated_runtime_overhead_bytes=None,
            estimated_total_bytes=None,
            available_memory_bytes=_get_available_memory_bytes(
                selected_backend, current_platform
            ),
            fit=None,
            summary=(
                "Could not infer model size from the model name. Use an explicit "
                "Hugging Face model ID that includes the parameter size or run "
                "a real load attempt for authoritative validation."
            ),
            notes=["Preflight uses name-based heuristics when model metadata is unavailable."],
        )

    bytes_per_param = _bytes_per_parameter(dtype, quantization, selected_backend)
    weight_bytes = int(params * bytes_per_param)
    kv_ratio = 0.18 * (max_model_len / 8192.0)
    kv_cache_bytes = int(weight_bytes * kv_ratio)
    runtime_ratio = {
        "cpu": 0.25,
        "apple-metal": 0.2,
        "cuda": 0.12,
        "rocm": 0.14,
        "xpu": 0.14,
    }.get(selected_backend, 0.15)
    runtime_overhead = int(weight_bytes * runtime_ratio)
    total_bytes = weight_bytes + kv_cache_bytes + runtime_overhead
    available = _get_available_memory_bytes(selected_backend, current_platform)
    fit = None if available is None else total_bytes <= available

    summary = (
        f"Estimated {family} working set is {_format_bytes(total_bytes)} "
        f"for backend `{selected_backend}` using dtype `{dtype}`."
    )
    if fit is True:
        summary += f" Detected memory {_format_bytes(available)} appears sufficient."
    elif fit is False:
        summary += f" Detected memory {_format_bytes(available)} appears insufficient."
    else:
        summary += " Available memory could not be determined on this host."

    return ModelPreflight(
        model=model,
        dtype=dtype,
        quantization=quantization,
        max_model_len=max_model_len,
        parameter_count=params,
        estimated_weight_bytes=weight_bytes,
        estimated_kv_cache_bytes=kv_cache_bytes,
        estimated_runtime_overhead_bytes=runtime_overhead,
        estimated_total_bytes=total_bytes,
        available_memory_bytes=available,
        fit=fit,
        summary=summary,
        notes=[
            "Preflight is heuristic and intentionally conservative.",
            "KV cache and runtime overhead depend on sequence length, "
            "batch size, and backend-specific planners.",
        ],
    )


def inspect_trtllm(
    *,
    selected_backend: str,
    model: str | None = None,
) -> TrtllmDiagnostics:
    reasons: list[str] = []

    try:
        from vllm.utils.flashinfer import (
            has_flashinfer,
            has_flashinfer_trtllm_fused_moe,
            supports_trtllm_attention,
        )
    except Exception:
        return TrtllmDiagnostics(
            environment_supported=False,
            model_supported=None,
            eligible=False,
            reasons=[
                "FlashInfer / TensorRT-LLM helpers are unavailable in this "
                "environment."
            ],
            flashinfer_available=False,
            flashinfer_trtllm_moe_available=False,
            sink_attention_supported=False,
            ragged_mla_supported=False,
        )

    flashinfer_available = has_flashinfer()
    flashinfer_trtllm_moe_available = has_flashinfer_trtllm_fused_moe()
    environment_supported = selected_backend == "cuda" and supports_trtllm_attention()
    if not flashinfer_available:
        reasons.append("FlashInfer is not installed or not usable on this system.")
    if selected_backend != "cuda":
        reasons.append(
            "TensorRT-LLM interoperability is only relevant on NVIDIA CUDA backends."
        )
    elif not environment_supported:
        reasons.append(
            "Current CUDA environment does not meet the TensorRT-LLM attention requirements."
        )

    ragged_mla_supported = False
    model_supported: bool | None = None
    if model is not None:
        lowered = model.lower()
        model_supported = any(
            token in lowered for token in ("deepseek", "llama", "qwen", "mixtral")
        )
        if not model_supported:
            reasons.append(
                "Model family is not one of the common families where vLLM "
                "already exposes TRT-LLM-related kernels."
            )
        ragged_mla_supported = environment_supported and "deepseek" in lowered
        if "deepseek" in lowered and not ragged_mla_supported:
            reasons.append(
                "DeepSeek TRT-LLM ragged MLA prefill needs a Blackwell-class "
                "environment with the right FlashInfer support."
            )

    eligible = environment_supported and (model_supported is not False)
    sink_attention_supported = environment_supported
    if eligible and not reasons:
        reasons.append(
            "Environment looks compatible with vLLM's existing TensorRT-LLM interoperability hooks."
        )

    return TrtllmDiagnostics(
        environment_supported=environment_supported,
        model_supported=model_supported,
        eligible=eligible,
        reasons=reasons,
        flashinfer_available=flashinfer_available,
        flashinfer_trtllm_moe_available=flashinfer_trtllm_moe_available,
        sink_attention_supported=sink_attention_supported,
        ragged_mla_supported=ragged_mla_supported,
    )


def build_doctor_report(
    *,
    requested_backend: str = "auto",
    model: str | None = None,
    dtype: str = "auto",
    quantization: str | None = None,
    max_model_len: int | None = None,
    profile: str = "balanced",
) -> DoctorReport:
    selection, backends, environment, current_platform = select_backend(
        requested_backend=requested_backend
    )
    report = DoctorReport(
        os=environment["os"],
        architecture=environment["architecture"],
        current_platform=environment["current_platform"],
        current_device_type=environment["current_device_type"],
        current_device_name=environment["current_device_name"],
        requested_backend=requested_backend,
        selected_backend=selection.selected_backend,
        selection_reason=selection.selected_reason,
        fallback_reason=selection.fallback_reason,
        available_plugins=_discover_platform_plugins(),
        backends=backends,
        model=model,
        profile=profile,
    )

    if model is not None:
        report.preflight = estimate_model_preflight(
            model,
            selected_backend=selection.selected_backend,
            dtype=dtype,
            quantization=quantization,
            max_model_len=max_model_len,
            current_platform=current_platform,
        )
        report.trtllm = inspect_trtllm(
            selected_backend=selection.selected_backend,
            model=model,
        )
    else:
        report.trtllm = inspect_trtllm(selected_backend=selection.selected_backend)
    return report
