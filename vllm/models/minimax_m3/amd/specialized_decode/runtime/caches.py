# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named cache ABI validation and identity tracking."""

from __future__ import annotations

from dataclasses import dataclass

import torch

TensorIdentity = tuple[
    int,
    int,
    torch.dtype,
    tuple[int, ...],
    int,
    torch.device,
    str,
]


def tensor_identity(tensor: torch.Tensor, layout: str) -> TensorIdentity:
    return (
        tensor.data_ptr(),
        tensor.numel(),
        tensor.dtype,
        tensor.stride(),
        tensor.storage_offset(),
        tensor.device,
        layout,
    )


@dataclass(frozen=True)
class CacheBinding:
    main: torch.Tensor
    key_pages: torch.Tensor
    value_pages: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    index_key: torch.Tensor
    sparse_table: torch.Tensor
    sparse_context: torch.Tensor

    def identity(self) -> dict[str, TensorIdentity]:
        layouts = {
            "main": "vllm-fp8-block128",
            "key_pages": "aiter-page16-k",
            "value_pages": "aiter-page16-v",
            "k_scale": "scalar",
            "v_scale": "scalar",
            "index_key": "minimax-index-k",
            "sparse_table": "aiter-sparse-table",
            "sparse_context": "aiter-sparse-context",
        }
        return {
            name: tensor_identity(value, layouts[name])
            for name, value in vars(self).items()
        }


def cache_binding(layer) -> tuple[CacheBinding | None, list[str]]:
    reasons: list[str] = []
    layer_id = layer.layer_id
    attn = layer.self_attn
    try:
        key_pages, value_pages = attn.get_aiter_sparse_pa_kv_cache()
        binding = CacheBinding(
            main=attn.kv_cache,
            key_pages=key_pages,
            value_pages=value_pages,
            k_scale=attn._k_scale,
            v_scale=attn._v_scale,
            index_key=attn.indexer.index_cache.kv_cache,
            sparse_table=attn.sparse_bt_buffer,
            sparse_context=attn.sparse_ctx_buffer,
        )
    except (AttributeError, RuntimeError, ValueError) as err:
        return None, [f"layer {layer_id} cache binding unavailable ({err})"]

    main = binding.main
    if (
        main.numel() == 0
        or main.device.type != "cuda"
        or main.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or main.ndim < 3
        or main.shape[2] != 128
    ):
        reasons.append(f"layer {layer_id} has an invalid main FP8 KV cache")
    if (
        key_pages.device != main.device
        or value_pages.device != main.device
        or key_pages.stride(-1) != 1
        or value_pages.stride(-1) != 1
    ):
        reasons.append(f"layer {layer_id} has invalid AITER page-16 K/V views")
    for name, scale in (("K", binding.k_scale), ("V", binding.v_scale)):
        if (
            not isinstance(scale, torch.Tensor)
            or scale.device != main.device
            or scale.dtype != torch.float32
            or scale.numel() != 1
        ):
            reasons.append(f"layer {layer_id} has an invalid {name} cache scale")
    if (
        binding.index_key.device != main.device
        or binding.index_key.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or binding.index_key.ndim < 2
    ):
        reasons.append(f"layer {layer_id} has an invalid index-key cache")
    for name, scratch in (
        ("sparse table", binding.sparse_table),
        ("sparse context", binding.sparse_context),
    ):
        if (
            not isinstance(scratch, torch.Tensor)
            or scratch.device != main.device
            or scratch.dtype != torch.int32
            or not scratch.is_contiguous()
        ):
            reasons.append(f"layer {layer_id} has invalid {name} scratch")
    return binding, reasons


def checked_cache_bindings(layers) -> dict[int, CacheBinding]:
    bindings: dict[int, CacheBinding] = {}
    reasons: list[str] = []
    for layer in layers:
        binding, layer_reasons = cache_binding(layer)
        reasons.extend(layer_reasons)
        if binding is not None:
            bindings[layer.layer_id] = binding
    if reasons:
        raise ValueError(
            "MiniMax-M3 specialized decode caches unsupported: " + "; ".join(reasons)
        )
    return bindings
