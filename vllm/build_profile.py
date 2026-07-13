# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable native build capabilities for reduced vLLM artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_PROFILE_PATH = Path(__file__).with_name("_build_profile.json")
_VALID_PROFILES = {"full", "rwkv"}
_RWKV_LOAD_FORMATS = {"auto", "hf", "pt"}


@dataclass(frozen=True)
class BuildProfileMetadata:
    profile: str
    configured_targets: tuple[str, ...]
    external_projects: tuple[str, ...]
    unrestricted: bool = True
    supported_architectures: tuple[str, ...] = ()
    supported_weight_suffixes: tuple[str, ...] = ()
    supported_device_types: tuple[str, ...] = ()
    supported_runner_types: tuple[str, ...] = ()
    supported_serving_features: tuple[str, ...] = ()
    supported_tensor_parallel_sizes: tuple[int, ...] = ()
    supported_pipeline_parallel_sizes: tuple[int, ...] = ()
    supported_data_parallel_sizes: tuple[int, ...] = ()

    def has_target(self, target: str) -> bool:
        return target in self.configured_targets


def load_build_profile_metadata(path: Path = _PROFILE_PATH) -> BuildProfileMetadata:
    if not path.exists():
        # Source checkouts and legacy/full wheels predate build metadata. Treating
        # those as full preserves existing behavior; reduced builds always write
        # the manifest before their extension is installed.
        return BuildProfileMetadata("full", (), (), unrestricted=True)

    payload = json.loads(path.read_text())
    profile = payload.get("profile")
    if profile not in _VALID_PROFILES:
        raise RuntimeError(
            f"Invalid native build profile metadata {profile!r} in {path}"
        )
    targets = payload.get("configured_targets")
    projects = payload.get("external_projects")
    if not isinstance(targets, list) or not all(isinstance(v, str) for v in targets):
        raise RuntimeError(f"Invalid configured_targets in {path}")
    if not isinstance(projects, list) or not all(isinstance(v, str) for v in projects):
        raise RuntimeError(f"Invalid external_projects in {path}")
    unrestricted = payload.get("unrestricted")
    if not isinstance(unrestricted, bool):
        raise RuntimeError(f"Invalid unrestricted capability in {path}")

    def string_tuple(key: str) -> tuple[str, ...]:
        values = payload.get(key)
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise RuntimeError(f"Invalid {key} in {path}")
        return tuple(values)

    def int_tuple(key: str) -> tuple[int, ...]:
        values = payload.get(key)
        if not isinstance(values, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in values
        ):
            raise RuntimeError(f"Invalid {key} in {path}")
        return tuple(values)

    metadata = BuildProfileMetadata(
        profile=profile,
        configured_targets=tuple(targets),
        external_projects=tuple(projects),
        unrestricted=unrestricted,
        supported_architectures=string_tuple("supported_architectures"),
        supported_weight_suffixes=string_tuple("supported_weight_suffixes"),
        supported_device_types=string_tuple("supported_device_types"),
        supported_runner_types=string_tuple("supported_runner_types"),
        supported_serving_features=string_tuple("supported_serving_features"),
        supported_tensor_parallel_sizes=int_tuple("supported_tensor_parallel_sizes"),
        supported_pipeline_parallel_sizes=int_tuple(
            "supported_pipeline_parallel_sizes"
        ),
        supported_data_parallel_sizes=int_tuple("supported_data_parallel_sizes"),
    )
    if profile == "rwkv" and (
        metadata.unrestricted
        or metadata.configured_targets
        != ("_rapid_sampling", "rwkv7_ops")
        or metadata.external_projects
    ):
        raise RuntimeError(f"Inconsistent RWKV build profile metadata in {path}")
    return metadata


@cache
def get_build_profile_metadata() -> BuildProfileMetadata:
    return load_build_profile_metadata()


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def _is_multimodal(model_config: Any) -> bool:
    value = getattr(model_config, "is_multimodal_model", False)
    return bool(value() if callable(value) else value)


def validate_build_profile_capabilities(
    vllm_config: VllmConfig,
    metadata: BuildProfileMetadata | None = None,
) -> None:
    metadata = metadata or get_build_profile_metadata()
    if metadata.unrestricted:
        return

    model_config = vllm_config.model_config
    if model_config is None:
        return
    parallel_config = vllm_config.parallel_config
    load_config = vllm_config.load_config
    reasons: list[str] = []

    architecture = getattr(model_config, "architecture", None)
    if architecture is None:
        architectures = getattr(model_config, "architectures", ())
        architecture = architectures[0] if architectures else None
    if architecture not in metadata.supported_architectures:
        reasons.append(
            "requires architecture " + ", ".join(metadata.supported_architectures)
        )

    model = str(getattr(model_config, "model", ""))
    if not model.endswith(metadata.supported_weight_suffixes):
        reasons.append(
            "requires an RWKV7 raw "
            + "/".join(metadata.supported_weight_suffixes)
            + " checkpoint"
        )

    if (
        getattr(model_config, "quantization", None) is not None
        or getattr(model_config, "quantization_config", None) is not None
    ):
        reasons.append("does not support quantization")
    if _is_multimodal(model_config):
        reasons.append("does not support multimodal models")
    if getattr(vllm_config, "speculative_config", None) is not None:
        reasons.append("does not support speculative decoding")
    if getattr(vllm_config, "lora_config", None) is not None:
        reasons.append("does not support LoRA")
    if getattr(model_config, "enable_sleep_mode", False) or getattr(
        model_config, "enable_cumem_allocator", False
    ):
        reasons.append("does not support sleep mode or the CuMem allocator")

    offload_config = getattr(vllm_config, "offload_config", None)
    if offload_config is not None and (
        getattr(getattr(offload_config, "uva", None), "cpu_offload_gb", 0) > 0
        or getattr(getattr(offload_config, "prefetch", None), "offload_group_size", 0)
        > 0
    ):
        reasons.append("does not support model offload")
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is not None and getattr(
        kv_transfer_config, "kv_connector", None
    ):
        reasons.append("does not support KV transfer or offload connectors")

    tp = getattr(parallel_config, "tensor_parallel_size", 1)
    pp = getattr(parallel_config, "pipeline_parallel_size", 1)
    dp = getattr(parallel_config, "data_parallel_size", 1)
    pcp = getattr(parallel_config, "prefill_context_parallel_size", 1)
    if tp not in metadata.supported_tensor_parallel_sizes:
        reasons.append(
            "requires TP="
            + "/".join(map(str, metadata.supported_tensor_parallel_sizes))
        )
    if pp not in metadata.supported_pipeline_parallel_sizes:
        reasons.append(
            "requires PP="
            + "/".join(map(str, metadata.supported_pipeline_parallel_sizes))
        )
    if dp not in metadata.supported_data_parallel_sizes:
        reasons.append(
            "requires DP=" + "/".join(map(str, metadata.supported_data_parallel_sizes))
        )
    if pcp != 1:
        reasons.append("requires prefill context parallel size 1")
    executor_backend = getattr(parallel_config, "distributed_executor_backend", None)
    if executor_backend not in (None, "uni"):
        reasons.append("requires the single-process executor")

    load_format = str(_value(getattr(load_config, "load_format", "auto"))).lower()
    if load_format not in _RWKV_LOAD_FORMATS:
        reasons.append(
            "requires load format 'auto', 'hf', or 'pt' for a raw .pth checkpoint"
        )

    device_type = getattr(
        getattr(vllm_config, "device_config", None), "device_type", None
    )
    if device_type is not None and device_type not in metadata.supported_device_types:
        reasons.append("requires a CUDA device")
    runner_type = getattr(model_config, "runner_type", "generate")
    if runner_type not in metadata.supported_runner_types:
        reasons.append("requires the generation runner")

    if reasons:
        details = "; ".join(reasons)
        raise ValueError(
            f"The RWKV build profile cannot run this configuration: {details}. "
            "Install a full build of vLLM for this configuration."
        )
