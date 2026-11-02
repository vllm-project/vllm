# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from dataclasses import fields
from typing import Any

from pydantic import BaseModel

import vllm.envs as envs
from vllm import __version__
from vllm.config import VllmConfig
from vllm.config.utils import (
    SupportsHash,
    get_hash_factors,
    hash_factors,
    normalize_value,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig

logger = init_logger(__name__)


def _environment_factors() -> dict[str, object]:
    ignored = {
        "VLLM_API_KEY",
        "VLLM_BUILD_COMMIT",  # Included explicitly in the deployment identity.
        "VLLM_IMAGE_TAG",
        "VLLM_BUILD_PIPELINE",
        "VLLM_BUILD_URL",
        "VLLM_HOST_IP",
        "VLLM_PORT",
        "VLLM_DP_MASTER_IP",
        "VLLM_DP_MASTER_PORT",
        "VLLM_DP_RANK",
        "VLLM_DP_RANK_LOCAL",
        "VLLM_RAY_BUNDLE_INDICES",
        "VLLM_RAY_DP_PLACEMENT_NODE_IPS",
        "VLLM_RPC_BASE_PATH",
        "VLLM_NIXL_SIDE_CHANNEL_HOST",
        "VLLM_NIXL_SIDE_CHANNEL_PORT",
        "VLLM_EC_SIDE_CHANNEL_HOST",
        "VLLM_EC_SIDE_CHANNEL_PORT",
        "VLLM_MOONCAKE_BOOTSTRAP_PORT",
        "VLLM_LOOPBACK_IP",
        "VLLM_CACHE_ROOT",
        "VLLM_CONFIG_ROOT",
        "VLLM_XLA_CACHE_PATH",
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "VLLM_LOG_STATS_INTERVAL",
        "VLLM_NO_USAGE_STATS",
        "VLLM_DO_NOT_TRACK",
        "VLLM_USAGE_STATS_SERVER",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
    }
    library_prefixes = ("NCCL_", "CUBLAS_", "CUBLASLT_", "FLASHINFER_", "TRITON_")
    library_names = {"NVIDIA_TF32_OVERRIDE", "LD_LIBRARY_PATH"}
    names = set(envs.environment_variables) | set(os.environ) | library_names
    factors: dict[str, object] = {}
    for name in sorted(names):
        if name in ignored or name.startswith("VLLM_LOGGING_"):
            continue
        if not (name.startswith(("VLLM_", *library_prefixes)) or name in library_names):
            continue
        # Use parsed defaults for registered settings. Unknown vLLM/plugin and
        # library flags use their raw value. Invalid getters must fail startup.
        getter = envs.environment_variables.get(name)
        try:
            factors[name] = normalize_value(
                getter() if getter else os.environ.get(name)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid {name} for MooncakeStore's automatic cache prefix"
            ) from exc
    return factors


def _quantization_json_default(value: object) -> Any:
    if isinstance(value, QuantizationConfig):
        # Online config args describe quantization; loading timers and the
        # per-process inventory of quantized layers do not affect KV compatibility.
        state = (
            {"args": value.args}
            if isinstance(value, OnlineQuantizationConfig)
            else vars(value)
        )
        return {"type": normalize_value(type(value)), "state": state}
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return normalize_value(value)


def resolve_cache_prefix(vllm_config: VllmConfig) -> str:
    """Namespace Store KV by deployment identity when cache_prefix is 'auto'."""
    transfer_config = vllm_config.kv_transfer_config
    assert transfer_config is not None, "MooncakeStore requires KVTransferConfig"
    extra_config = transfer_config.kv_connector_extra_config
    prefix = str(extra_config.get("cache_prefix", ""))
    if prefix != "auto":
        return prefix

    image_tag = envs.VLLM_IMAGE_TAG.strip()
    build_commit = envs.VLLM_BUILD_COMMIT.strip()
    if build_commit in ("", "unknown") and image_tag in (
        "",
        "local/vllm-openai:dev",
    ):
        raise ValueError(
            "MooncakeStore cache_prefix='auto' requires VLLM_BUILD_COMMIT "
            "or an immutable VLLM_IMAGE_TAG identifying the deployed code."
        )

    # Use component hashes, not VllmConfig's truncated compilation hash.
    # Their runtime exclusions keep engine IDs, ranks and ports out of the key.
    config_factors: dict[str, object] = {}
    ignored = {
        "instance_id",
        "kv_transfer_config",
        "ec_transfer_config",
        "kv_events_config",
        "observability_config",
        "logging_config",
        "profiler_config",
        "watchdog_config",
        "shutdown_timeout",
        "weight_transfer_config",
    }
    hashed_separately = {
        "model_config",
        "load_config",
        "scheduler_config",
        "compilation_config",
        "quant_config",
    }
    for config_field in fields(vllm_config):
        name = config_field.name
        if name in ignored or name in hashed_separately:
            continue
        value = getattr(vllm_config, name)
        config_factors[name] = (
            value.compute_hash()
            if isinstance(value, SupportsHash)
            else normalize_value(value)
        )

    # Include declared settings by default, excluding capacity and runtime state.
    config_factors["load_config"] = get_hash_factors(
        vllm_config.load_config, {"download_dir", "use_tqdm_on_load"}
    )
    config_factors["scheduler_config"] = get_hash_factors(
        vllm_config.scheduler_config,
        {
            "max_num_batched_tokens",
            "max_num_scheduled_tokens",
            "max_num_seqs",
            "max_num_active_seqs",
            "max_num_queued_reqs",
            "max_num_queued_tokens",
            "max_num_encoder_input_tokens",
            "encoder_cache_size",
        },
    )
    compilation = vllm_config.compilation_config
    config_factors["compilation_config"] = get_hash_factors(
        compilation,
        {
            "debug_dump_path",
            "cache_dir",
            "local_cache_dir",
            "traced_files",
            "compilation_time",
            "encoder_compilation_time",
            "enabled_custom_ops",
            "disabled_custom_ops",
            "static_forward_context",
            "pass_config",
            "dynamic_shapes_config",
            # Capture/compile batch sizes are derived from scheduler capacities.
            "compile_sizes",
            "compile_ranges_endpoints",
            "cudagraph_capture_sizes",
            "max_cudagraph_capture_size",
            "decoder_replay_cudagraph_capture_sizes",
        },
    )
    config_factors["compilation_passes"] = compilation.pass_config.compute_hash()
    config_factors["dynamic_shapes"] = compilation.dynamic_shapes_config.compute_hash()
    quantization = vllm_config.quant_config
    config_factors["quant_config"] = (
        json.dumps(
            quantization,
            default=_quantization_json_default,
            ensure_ascii=False,
            sort_keys=True,
        )
        if quantization is not None
        else None
    )
    model = vllm_config.model_config
    if model is not None:
        config_factors["model_config"] = get_hash_factors(
            model,
            {
                "hf_token",
                "hf_overrides",  # Resolved HF configs are already model fields.
            },
        )
        config_factors["hf_commit_hash"] = getattr(
            model.hf_config, "_commit_hash", None
        )
    parallel = vllm_config.parallel_config
    config_factors["parallel_compute_options"] = normalize_value(
        {
            "disable_custom_all_reduce": parallel.disable_custom_all_reduce,
            "worker_cls": parallel.worker_cls,
            "worker_extension_cls": parallel.worker_extension_cls,
        }
    )
    config_factors["prefix_caching_hash_algo"] = (
        vllm_config.cache_config.prefix_caching_hash_algo
    )
    config_factors["hash_block_size"] = vllm_config.cache_config.hash_block_size
    config_factors["prefix_match_unit"] = vllm_config.cache_config.prefix_match_unit
    config_factors["store_format_options"] = normalize_value(
        {
            name: extra_config.get(name)
            for name in (
                "enable_cross_layers_blocks",
                "store_tp_size",
                "enable_store_tp_lcm",
                "prefill_tp_sizes",
                "store_chunk_size",
            )
        }
    )
    factors = {
        "schema": "vllm-mooncake-store-auto-v1",
        "vllm_version": __version__,
        "image_tag": image_tag,
        "build_commit": build_commit,
        "config": config_factors,
        "env": _environment_factors(),
    }
    prefix = f"auto-{hash_factors(factors)}"
    logger.info("MooncakeStore resolved cache_prefix: %s", prefix)
    return prefix
