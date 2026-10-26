# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, fields
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class Step5SparseConfig:
    enabled: bool = True
    proxy_dim: int = 256
    sparse_indexer_rope_dim: int = 32
    sparse_indexer_use_rope: bool = True
    sparse_indexer_num_heads: int = 16
    sparse_indexer_num_k_heads: int = 1
    sparse_indexer_q_norm_type: str = "rmsnorm"
    sparse_indexer_k_norm_type: str = "layernorm"
    sparse_indexer_csa_z_norm_type: str = "none"
    index_tp_size: int = 4
    topk: int = 512
    region_block_size: int = 8
    compression_method: str = "csa_block_compress"
    attention_impl: str = "sparse_gqa"
    sparse_indexer_softmax_variant: str = "softmax"
    sparse_indexer_ssmax_s_granularity: str = "q_head"
    apply_to_layer_types: tuple[str, ...] = ("full_attention",)


_UNSUPPORTED_SPARSE_SECTION_NAMES = ("step5_sparse_config", "step3p5_sparse_config")
_SUPPORTED_SOFTMAX_VARIANTS = {"softmax", "standard", "ssmax"}
_SUPPORTED_INDEXER_Q_HEADS_PER_PROVIDER = (2, 4)
_STEP5_SPARSE_PROXY_DIM = 256
_STEP5_SPARSE_REGION_BLOCK_SIZE = 8
_STEP5_SPARSE_TOPK_MAX = 1024
_STEP5_SPARSE_FIXED_OPTIONS = {
    "sparse_indexer_use_rope": True,
    "sparse_indexer_q_norm_type": "rmsnorm",
    "sparse_indexer_k_norm_type": "layernorm",
    "sparse_indexer_csa_z_norm_type": "none",
    "compression_method": "csa_block_compress",
    "attention_impl": "sparse_gqa",
    "sparse_indexer_ssmax_s_granularity": "q_head",
}


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _get_sparse_config_section(config: Any) -> Any:
    for name in _UNSUPPORTED_SPARSE_SECTION_NAMES:
        if _config_value(config, name) is not None:
            raise NotImplementedError(
                f"Step5 reads the sparse layout from sparse_config; {name} is "
                "unsupported."
            )
    return _config_value(config, "sparse_config")


def checkpoint_has_step5_sparse_config(config: Any) -> bool:
    """Return whether the checkpoint declares a sparse KV layout.

    Section presence determines the shared KV layout, even when sparse
    execution is disabled in the checkpoint.
    """
    return _get_sparse_config_section(config) is not None


def get_step5_sparse_config(config: Any) -> Step5SparseConfig | None:
    defaults = Step5SparseConfig()
    section = _get_sparse_config_section(config)
    if section is None:
        return None
    if _config_value(section, "softmax_variant") is not None:
        raise NotImplementedError(
            "Step5 sparse_config.softmax_variant is unsupported; use "
            "sparse_indexer_softmax_variant."
        )
    enabled = _config_value(section, "enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError(
            f"Step5 sparse_config.enabled must be a bool, got {enabled!r}."
        )
    if enabled is False:
        return None
    softmax_variant = _config_value(section, "sparse_indexer_softmax_variant")
    if softmax_variant is not None and str(softmax_variant).lower() not in (
        _SUPPORTED_SOFTMAX_VARIANTS
    ):
        raise ValueError(
            "Step5 sparse indexer softmax variant is unsupported by the "
            f"current implementation: {softmax_variant!r}. Supported variants are "
            f"{sorted(_SUPPORTED_SOFTMAX_VARIANTS)}."
        )
    if softmax_variant is not None and str(softmax_variant).lower() == "ssmax":
        logger.warning_once(
            "Step5 sparse config carries ssmax metadata, but the DSA "
            "selector uses its fixed weighted-ReLU scoring path; ssmax_s is "
            "loaded for checkpoint completeness but is not consumed."
        )
    values: dict[str, Any] = {}
    for field in fields(Step5SparseConfig):
        default = getattr(defaults, field.name)
        if field.name == "apply_to_layer_types":
            continue
        value = _config_value(section, field.name)
        if field.name == "index_tp_size" and value is None:
            # The checkpoint schema calls this field num_provider_groups.
            value = _config_value(section, "num_provider_groups")
        if value is None or value == "":
            value = default
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(
                    f"Step5 sparse_config.{field.name} must be a bool, got {value!r}."
                )
        elif isinstance(default, int):
            value = int(value)
        elif isinstance(default, str):
            value = str(value)
        values[field.name] = value
    apply_to_layer_types = _config_value(section, "apply_to_layer_types")
    if apply_to_layer_types is None:
        apply_to_layer_types = defaults.apply_to_layer_types
    apply_to_layer_types = tuple(
        str(item).strip().lower() for item in apply_to_layer_types if str(item).strip()
    )
    if apply_to_layer_types != ("full_attention",):
        raise ValueError(
            "Step5 DSA integration requires "
            "apply_to_layer_types=('full_attention',); sliding, empty, or "
            f"mixed layer selections are unsupported, got {apply_to_layer_types!r}."
        )
    values["apply_to_layer_types"] = apply_to_layer_types
    proxy_dim = values["proxy_dim"]
    if proxy_dim != _STEP5_SPARSE_PROXY_DIM:
        raise ValueError(f"Step5 DSA requires proxy_dim=256, got {proxy_dim}.")
    region_block_size = values["region_block_size"]
    if region_block_size != _STEP5_SPARSE_REGION_BLOCK_SIZE:
        raise ValueError(
            f"Step5 DSA requires region_block_size=8, got {region_block_size}."
        )
    topk = values["topk"]
    if not 1 <= topk <= _STEP5_SPARSE_TOPK_MAX:
        raise ValueError(f"Step5 DSA requires topk in [1, 1024], got {topk}.")
    indexer_num_k_heads = values["sparse_indexer_num_k_heads"]
    if indexer_num_k_heads != 1:
        raise ValueError(
            "Step5 sparse indexer currently requires exactly one KV head, got "
            f"sparse_indexer_num_k_heads={indexer_num_k_heads}."
        )
    index_tp_size = values["index_tp_size"]
    if index_tp_size <= 0:
        raise ValueError(
            f"Step5 sparse indexer index_tp_size must be positive, got {index_tp_size}."
        )
    indexer_num_heads = values["sparse_indexer_num_heads"]
    if indexer_num_heads <= 0 or indexer_num_heads % index_tp_size != 0:
        raise ValueError(
            "Step5 sparse_indexer_num_heads must be positive and divisible by "
            "index_tp_size, got "
            f"sparse_indexer_num_heads={indexer_num_heads}, "
            f"index_tp_size={index_tp_size}."
        )
    indexer_q_heads_per_provider = indexer_num_heads // index_tp_size
    if indexer_q_heads_per_provider not in _SUPPORTED_INDEXER_Q_HEADS_PER_PROVIDER:
        raise ValueError(
            "Step5 DSA indexer requires "
            "sparse_indexer_num_heads / index_tp_size in (2, 4), got "
            f"{indexer_q_heads_per_provider} "
            f"({indexer_num_heads} / {index_tp_size})."
        )
    indexer_rope_dim = values["sparse_indexer_rope_dim"]
    if (
        indexer_rope_dim <= 0
        or indexer_rope_dim > proxy_dim
        or indexer_rope_dim % 2 != 0
    ):
        raise ValueError(
            "Step5 sparse indexer RoPE dimension must be positive, even, and "
            "no larger than proxy_dim, got "
            f"sparse_indexer_rope_dim={indexer_rope_dim}, "
            f"proxy_dim={proxy_dim}."
        )
    for name, expected in _STEP5_SPARSE_FIXED_OPTIONS.items():
        actual = values[name]
        normalized = (
            bool(actual) if isinstance(expected, bool) else str(actual).strip().lower()
        )
        if normalized != expected:
            raise ValueError(f"Step5 DSA requires {name}={expected!r}, got {actual!r}.")
        values[name] = expected
    return Step5SparseConfig(**values)
