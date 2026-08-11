# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibration and loading helpers for Kimi-K3 FP8 prefill scales."""

import json
import os
from pathlib import Path
from typing import Any

import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index

logger = init_logger(__name__)

_QKV_NAMES = ("q", "k", "v")
_CACHE_MODE = "bf16_latent_cache"


def _kimi_layers(vllm_config: VllmConfig) -> list[tuple[int, str, Any]]:
    layers = []
    for (
        runtime_name,
        layer,
    ) in vllm_config.compilation_config.static_forward_context.items():
        calibration_amax = getattr(layer, "_kimi_k3_fp8_calibration_amax", None)
        if calibration_amax is None:
            continue
        layer_idx = extract_layer_index(runtime_name)
        if layer_idx is None:
            continue
        layers.append((layer_idx, runtime_name, layer))
    return sorted(layers)


def _validate_calibration_mode(vllm_config: VllmConfig) -> None:
    compilation_config = vllm_config.compilation_config
    if compilation_config.mode != CompilationMode.NONE:
        raise ValueError("Kimi-K3 FP8 calibration requires compilation mode NONE")
    if compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
        raise ValueError("Kimi-K3 FP8 calibration requires CUDAGraph mode NONE")
    parallel_config = vllm_config.parallel_config
    if parallel_config.data_parallel_size != 1:
        raise ValueError("Kimi-K3 FP8 calibration requires data_parallel_size=1")
    if parallel_config.enable_dbo:
        raise ValueError("Kimi-K3 FP8 calibration does not support DBO")
    if parallel_config.prefill_context_parallel_size != 1:
        raise ValueError("Kimi-K3 FP8 calibration does not support PCP")
    if parallel_config.decode_context_parallel_size != 1:
        raise ValueError("Kimi-K3 FP8 calibration does not support DCP")


def _checkpoint_identity(vllm_config: VllmConfig) -> str:
    model_config = vllm_config.model_config
    if model_config.revision:
        revision = model_config.revision.lower()
        if len(revision) >= 7 and all(char in "0123456789abcdef" for char in revision):
            return revision
        raise ValueError("Kimi-K3 FP8 scale revision must be an immutable commit hash")
    model_path = Path(model_config.model)
    snapshot = model_path.name.lower()
    if (
        model_path.parent.name == "snapshots"
        and len(snapshot) >= 7
        and all(char in "0123456789abcdef" for char in snapshot)
    ):
        return snapshot
    raise ValueError(
        "Kimi-K3 FP8 scales require an immutable model revision or snapshot path"
    )


def _validate_cache_mode(layers: list[tuple[int, str, Any]]) -> None:
    supported = {"auto", "bf16", "bfloat16"}
    incompatible = {
        runtime_name: layer.kv_cache_dtype
        for _, runtime_name, layer in layers
        if getattr(layer, "kv_cache_dtype", "auto") not in supported
    }
    if incompatible:
        raise ValueError(
            f"Kimi-K3 FP8 scales require a BF16 latent cache, got {incompatible}"
        )


def _validate_static_backends(layers: list[tuple[int, str, Any]]) -> None:
    for _, runtime_name, layer in layers:
        backend = getattr(layer, "prefill_backend", None)
        if (
            backend is None
            or backend.get_name() != "ROCM_AITER_FA"
            or not getattr(backend, "_fp8_prefill_enabled", False)
            or getattr(backend, "_fp8_static_quant_func", None) is None
        ):
            raise ValueError(
                f"Kimi-K3 static FP8 is unavailable for expected layer {runtime_name}"
            )


def prepare_kimi_k3_fp8_scales(
    vllm_config: VllmConfig,
    *,
    arm_calibration: bool = False,
) -> None:
    """Arm calibration or load immutable static scales before graph capture."""
    attention_config = vllm_config.attention_config
    save_path = attention_config.rocm_kimi_k3_fp8_prefill_scale_save_path
    load_path = attention_config.rocm_kimi_k3_fp8_prefill_scale_path
    layers = _kimi_layers(vllm_config)
    if save_path is None and load_path is None:
        return

    if save_path is not None:
        _validate_calibration_mode(vllm_config)
        _checkpoint_identity(vllm_config)
        _validate_cache_mode(layers)
        if not arm_calibration:
            return
        for _, _, layer in layers:
            layer._kimi_k3_fp8_calibration_amax.zero_()
            layer._kimi_k3_fp8_calibration_state["armed"] = True
        logger.info(
            "Armed Kimi-K3 FP8 calibration for %d layers; rank shards will be "
            "written under %s",
            len(layers),
            save_path,
        )
        return

    if arm_calibration:
        return

    assert load_path is not None
    checkpoint_id = _checkpoint_identity(vllm_config)
    _validate_cache_mode(layers)
    _validate_static_backends(layers)
    from safetensors import safe_open

    tp_rank = get_tp_group().rank_in_group
    tp_size = len(get_tp_group().ranks)
    model_config = vllm_config.model_config
    with safe_open(load_path, framework="pt", device="cpu") as artifact:
        metadata = artifact.metadata()
        if metadata.get("schema") != "1":
            raise ValueError("Unsupported Kimi-K3 FP8 scale artifact schema")
        if metadata.get("model") != model_config.model:
            raise ValueError("Kimi-K3 FP8 scale artifact model mismatch")
        if metadata.get("checkpoint_id") != checkpoint_id:
            raise ValueError("Kimi-K3 FP8 scale artifact checkpoint mismatch")
        if int(metadata.get("tp_size", "0")) != tp_size:
            raise ValueError("Kimi-K3 FP8 scale artifact TP size mismatch")
        if int(metadata.get("pp_size", "0")) != len(get_pp_group().ranks):
            raise ValueError("Kimi-K3 FP8 scale artifact PP size mismatch")
        if metadata.get("fp8_dtype") != "float8_e4m3fnuz":
            raise ValueError("Kimi-K3 FP8 scale artifact dtype mismatch")
        if metadata.get("cache_mode") != _CACHE_MODE:
            raise ValueError("Kimi-K3 FP8 scale artifact cache mode mismatch")
        if int(metadata.get("qk_head_dim", "0")) != 192:
            raise ValueError("Kimi-K3 FP8 scale artifact QK dimension mismatch")
        if int(metadata.get("v_head_dim", "0")) != 128:
            raise ValueError("Kimi-K3 FP8 scale artifact V dimension mismatch")
        artifact_keys = set(artifact.keys())
        artifact_layer_ids = {
            int(key.split(".")[1])
            for key in artifact_keys
            if key.startswith("layers.") and key.endswith(".q_descale")
        }
        if len(artifact_layer_ids) != int(metadata.get("num_layers", "0")):
            raise ValueError("Kimi-K3 FP8 artifact layer metadata mismatch")

        for layer_idx, _, layer in layers:
            local_heads = layer._kimi_k3_fp8_calibration_amax.shape[1]
            start = tp_rank * local_heads
            stop = start + local_heads
            descales = []
            for tensor_name in _QKV_NAMES:
                key = f"layers.{layer_idx}.{tensor_name}_descale"
                if key not in artifact_keys:
                    raise ValueError(f"Missing Kimi-K3 FP8 scale tensor {key}")
                global_descale = artifact.get_tensor(key)
                if global_descale.ndim != 1 or global_descale.numel() != (
                    local_heads * tp_size
                ):
                    raise ValueError(f"Invalid shape for Kimi-K3 FP8 scale {key}")
                descales.append(global_descale[start:stop])
            stacked = torch.stack(descales).to(
                device=layer._kimi_k3_fp8_calibration_amax.device,
                dtype=torch.float32,
            )
            if not torch.isfinite(stacked).all() or not (stacked > 0).all():
                raise ValueError(f"Invalid values for Kimi-K3 FP8 layer {layer_idx}")
            layer._kimi_k3_fp8_static_descale.resize_as_(stacked).copy_(stacked)
    logger.info("Loaded immutable Kimi-K3 FP8 scales for %d layers", len(layers))


def save_kimi_k3_fp8_calibration(vllm_config: VllmConfig) -> None:
    """Write one atomic rank-local JSON shard during worker shutdown."""
    attention_config = vllm_config.attention_config
    save_path = attention_config.rocm_kimi_k3_fp8_prefill_scale_save_path
    if save_path is None:
        return
    layers = _kimi_layers(vllm_config)

    if layers:
        torch.cuda.synchronize()
    tp_rank = get_tp_group().rank_in_group
    pp_rank = get_pp_group().rank_in_group
    payload: dict[str, Any] = {
        "schema": 1,
        "model": vllm_config.model_config.model,
        "revision": vllm_config.model_config.revision or "",
        "checkpoint_id": _checkpoint_identity(vllm_config),
        "calibration_id": attention_config.rocm_kimi_k3_fp8_prefill_calibration_id,
        "tp_size": len(get_tp_group().ranks),
        "tp_rank": tp_rank,
        "pp_size": len(get_pp_group().ranks),
        "pp_rank": pp_rank,
        "fp8_dtype": "float8_e4m3fnuz",
        "cache_mode": _CACHE_MODE,
        "local_heads": layers[0][2]._kimi_k3_fp8_calibration_amax.shape[1]
        if layers
        else 12,
        "qk_head_dim": 192,
        "v_head_dim": 128,
        "margin": attention_config.rocm_kimi_k3_fp8_prefill_scale_margin,
        "layers": {},
    }
    for layer_idx, runtime_name, layer in layers:
        if not layer._kimi_k3_fp8_calibration_state["armed"]:
            continue
        maxima = layer._kimi_k3_fp8_calibration_amax.detach().cpu()
        if not torch.isfinite(maxima).all() or not (maxima > 0).all():
            raise RuntimeError(
                f"Incomplete Kimi-K3 FP8 calibration for layer {layer_idx}"
            )
        payload["layers"][str(layer_idx)] = {
            "runtime_name": runtime_name,
            **{
                f"{name}_amax": maxima[index].tolist()
                for index, name in enumerate(_QKV_NAMES)
            },
        }

    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"kimi-k3-fp8-scales-pp{pp_rank:02d}-tp{tp_rank:02d}.json"
    )
    temporary_path = output_path.with_suffix(f".tmp-{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    logger.info("Saved Kimi-K3 FP8 calibration shard to %s", output_path)
