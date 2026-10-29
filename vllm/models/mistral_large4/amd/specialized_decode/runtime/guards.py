# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loaded-model, device and KV-cache refusals (each list holds every reason)."""

from __future__ import annotations

import torch

from vllm.platforms import current_platform

from .spec import DecodeConfig

LATENT = 576  # MLA latent row (bf16 or E4M3 values): kv_lora 512 + rope 64
INDEX_ROW = 132  # DSA indexer key row: 128 E4M3 bytes + one fp32 scale


def model_refusals(model, config: DecodeConfig) -> list[str]:
    """After load: the layer mix, each kernel layer's weights, the device."""
    from vllm.models.mistral_large4.amd.specialized_decode.mono.common.runtime.consensus import (  # noqa: E501
        MonoUnsupported,
    )

    from .weights import check_moe_layer, moe_kinds

    reasons: list[str] = []
    layers = list(model.layers)
    try:
        moe_kinds(layers)
    except MonoUnsupported as err:
        reasons.append(str(err))
    for layer in layers[1:]:
        try:
            check_moe_layer(layer)
        except (MonoUnsupported, AttributeError) as err:
            reasons.append(f"layer {layer.self_attn.spec.layer_idx}: {err}")
    # aux hidden-state taps (DSpark) are cloned off the residual stream between
    # launches by MonoRunner.forward
    if not current_platform.is_rocm():
        reasons.append("requires ROCm")
        return reasons
    try:
        import flydsl  # noqa: F401
        from aiter.ops.flydsl.kernels import buffer_ops  # noqa: F401

        from vllm.platforms.rocm import get_cdna_version
    except ImportError as err:
        reasons.append(f"requires FlyDSL and AITER FlyDSL helpers ({err})")
        return reasons
    if get_cdna_version() != 4:
        reasons.append("requires CDNA4 / gfx950")
    index = torch.accelerator.current_device_index()
    cus = current_platform.get_device_properties(index).multi_processor_count
    if cus != config.blocks:
        reasons.append(f"requires {config.blocks} compute units, found {cus}")
    return reasons


def cache_refusals(model, max_model_len: int) -> list[str]:
    """At bind: every kernel layer's latent and indexer caches."""
    reasons: list[str] = []
    page = None
    for layer in list(model.layers)[1:]:
        attn = layer.self_attn
        i = attn.spec.layer_idx
        kv = attn.attn.kv_cache
        # bf16, or fp8 (E4M3) which vLLM stores as uint8 and views as fp8 per call
        if (
            kv.dim() != 3
            or kv.dtype not in (torch.bfloat16, torch.uint8, torch.float8_e4m3fn)
            or kv.shape[-1] != LATENT
        ):
            reasons.append(f"layer {i}: KV cache {kv.dtype} {tuple(kv.shape)}")
            continue
        if not kv.is_contiguous():
            reasons.append(f"layer {i}: KV cache pages not contiguous {kv.stride()}")
        page = kv.shape[1] if page is None else page
        if kv.shape[1] != page or page % 16:
            reasons.append(f"layer {i}: KV cache page {kv.shape[1]}")
        icache = attn.indexer.k_cache.kv_cache if attn.indexer is not None else None
        if icache is not None and (
            icache.dim() != 3
            or icache.dtype != torch.uint8
            or tuple(icache.shape[1:]) != (page, INDEX_ROW)
            or not icache.is_contiguous()
        ):
            reasons.append(
                f"layer {i}: index cache {icache.dtype} {tuple(icache.shape)}"
            )
    return reasons
