# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration for vLLM multimodal tests."""

import os
import warnings

import torch

from vllm.platforms import current_platform


def pytest_configure(config):
    """Early ROCm configuration that must happen before test collection."""
    if not current_platform.is_rocm():
        return

    # Disable skinny GEMM on ROCm to avoid non-deterministic results
    # from atomic reductions in wvSplitKrc kernel.
    # See: https://github.com/vllm-project/vllm/pull/33493#issuecomment-3906083975
    os.environ["VLLM_ROCM_USE_SKINNY_GEMM"] = "0"
    warnings.warn(
        "ROCm: Set VLLM_ROCM_USE_SKINNY_GEMM=0 to avoid non-deterministic "
        "results from skinny GEMM atomic reductions",
        UserWarning,
        stacklevel=1,
    )


def pytest_collection_modifyitems(config, items):
    """Configure ROCm-specific settings based on collected tests."""
    if not current_platform.is_rocm():
        return

    skip_patterns = ["test_granite_speech.py"]
    if any(pattern in str(arg) for arg in config.args for pattern in skip_patterns):
        return

    # Disable Flash/MemEfficient SDP on ROCm to avoid HF Transformers
    # accuracy issues: https://github.com/vllm-project/vllm/issues/30167
    # TODO: Remove once ROCm SDP accuracy issues are resolved on HuggingFace
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    warnings.warn(
        "ROCm: Disabled flash_sdp and mem_efficient_sdp, enabled math_sdp "
        "to avoid HuggingFace Transformers accuracy issues",
        UserWarning,
        stacklevel=1,
    )


def _patch_encoder_layers(encoder):
    """Set _attn_implementation='sdpa' on all encoder self_attn layers."""
    for layer in encoder.layers:
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            for cfg_attr in ("vision_config", "config"):
                cfg = getattr(attn, cfg_attr, None)
                if cfg is not None and hasattr(cfg, "_attn_implementation"):
                    cfg._attn_implementation = "sdpa"
                    break


def patch_hf_vision_attn_for_rocm(model):
    """Force SDPA for HF vision encoders on ROCm.

    HF's flash_attention_2 has accuracy issues on ROCm that bypass
    torch.backends.cuda settings. This forces SDPA which then uses
    math_sdp via the pytest_collection_modifyitems settings.

    Supports both Isaac-style models (vision_embedding) and
    SigLIP-based models like Nemotron VL (vision_model).
    """
    if not current_platform.is_rocm():
        return

    inner = getattr(model, "model", model)

    # Isaac-style: inner.vision_embedding[0].encoder
    if hasattr(inner, "vision_embedding") and inner.vision_embedding:
        vit = inner.vision_embedding[0]
        if hasattr(vit, "encoder"):
            _patch_encoder_layers(vit.encoder)

    # SigLIP-based (e.g. Nemotron VL): inner.vision_model.vision_model.encoder
    # or inner.vision_model.encoder
    if hasattr(inner, "vision_model"):
        vm = inner.vision_model
        # SiglipVisionModel wraps SiglipVisionTransformer as .vision_model
        vm_inner = getattr(vm, "vision_model", vm)
        if hasattr(vm_inner, "encoder"):
            _patch_encoder_layers(vm_inner.encoder)
