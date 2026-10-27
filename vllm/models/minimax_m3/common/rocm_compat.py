# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm MiniMax-M3.1 serving constraints and the AMD compatibility recipe.

CDNA4 has no native NVFP4 KV/indexer kernels. The supported AMD path loads the
official MiniMax-M3.1 NVFP4 checkpoint, requantizes routed experts to MXFP4,
and keeps the existing FP8 MSA/indexer caches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from vllm.platforms import current_platform

RocmIndexerKVDType = Literal["bf16", "fp8"]

ROCM_DEFAULT_INDEXER_KV_DTYPE: RocmIndexerKVDType = "fp8"

MINIMAX_DSPARK_ARCHITECTURES = (
    "DSparkMiniMaxDraftModel",
    "MiniMaxM3DSparkModel",
)

MINIMAX_M31_ROCM_RECIPE = (
    "--kv-cache-dtype fp8 "
    '--attention-config \'{"indexer_kv_dtype": "fp8"}\' '
    "--quantization-config.moe mxfp4"
)


def default_minimax_indexer_kv_dtype() -> RocmIndexerKVDType:
    """Indexer cache dtype when ``attention_config.indexer_kv_dtype`` is auto.

    MI350/MI355 (gfx950) MSA indexer is fp8 MFMA, so those GPUs default to
    fp8. Other platforms keep bf16 unless the user overrides it.
    """
    if not current_platform.is_rocm():
        return "bf16"
    try:
        from vllm.platforms.rocm import on_gfx950
    except ImportError:
        return "bf16"
    return ROCM_DEFAULT_INDEXER_KV_DTYPE if on_gfx950() else "bf16"


def is_minimax_draft_weight(name: str) -> bool:
    """True for MiniMax MTP or nested DSpark tensors the target model skips."""
    return "mtp." in name or "dspark." in name


def minimax_m31_rocm_nvfp4_unsupported(kind: str) -> str:
    """User-facing error for NVFP4 caches that ROCm MiniMax cannot serve."""
    return (
        f"MiniMax-M3.1 {kind} is not supported on ROCm. "
        f"Serve the official NVFP4 checkpoint with {MINIMAX_M31_ROCM_RECIPE}. "
        "That requantizes routed experts NVFP4→MXFP4 at load and keeps the "
        "supported FP8 MSA/indexer path."
    )


def is_minimax_m3_hf_config(hf_config: Any) -> bool:
    """True when ``hf_config`` is a MiniMax-M3 / M3.1 text or VL config."""
    model_type = str(getattr(hf_config, "model_type", "") or "")
    if model_type in (
        "minimax_m3",
        "minimax_m3_text",
        "minimax_m3_vl",
        "minimax_m3_mtp",
    ):
        return True
    text = getattr(hf_config, "text_config", None)
    return str(getattr(text, "model_type", "") or "") in (
        "minimax_m3",
        "minimax_m3_text",
    )


def minimax_m3_sparse_attention_is_full(hf_config: Any) -> bool:
    """True when every layer uses MSA (MiniMax-M3.1), not mixed dense+MSA (M3)."""
    text = hf_config
    getter = getattr(hf_config, "get_text_config", None)
    if callable(getter):
        text = getter()
    elif getattr(hf_config, "text_config", None) is not None:
        text = hf_config.text_config
    cfg = getattr(text, "sparse_attention_config", None) or {}
    freq = cfg.get("sparse_attention_freq")
    return bool(freq) and all(int(flag) != 0 for flag in freq)


def nested_minimax_dspark_model(target_model: str, hf_config: Any) -> str | None:
    """Return MiniMax-M3.1's nested DSpark draft path when one exists.

    Official MiniMax-M3.1 checkpoints ship the DSpark drafter under
    ``{model}/dspark/`` rather than as a sibling Hugging Face repo. Prefer a
    local ``dspark/config.json``; fall back to the Hub subpath when every
    layer is MSA (the M3.1 layout). MiniMax-M3 keeps MTP modules in the
    target weights, so it must not take this path.

    Keep this policy in sync with
    ``vllm.config.speculative._minimax_m3_nested_dspark_model``.
    """
    if not target_model:
        return None
    local = Path(target_model) / "dspark"
    if (local / "config.json").is_file():
        return str(local)
    if not is_minimax_m3_hf_config(hf_config):
        return None
    if minimax_m3_sparse_attention_is_full(hf_config):
        return f"{str(target_model).rstrip('/')}/dspark"
    return None


def preserves_minimax_dspark_architecture(architectures: Any) -> bool:
    """True when HF config rewrite must keep the nested MiniMax DSpark arch."""
    if not architectures:
        return False
    return str(architectures[0]) in MINIMAX_DSPARK_ARCHITECTURES
