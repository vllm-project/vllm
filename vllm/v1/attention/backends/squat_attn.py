# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SQuat (Subspace-orthogonal KV Cache Quantization) attention backend.

Extends TurboQuant 4bit-nc quantization with a learned correction that
keeps quantization errors orthogonal to the query subspace. Uses the same
packed cache layout and decode kernels as turboquant_4bit_nc.

Reference: "SQuat: Subspace-orthogonal KV Cache Quantization"
(Wang et al., COLM 2025; preprint arXiv:2503.24358)

Usage:
    SQUAT_ROTATION_PATH=rotations/model.pt \\
    vllm serve model --kv-cache-dtype squat
"""

import functools
import math
import os
from typing import Any, ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.v1.attention.backends.turboquant_attn import (
    TurboQuantAttentionBackend,
    TurboQuantAttentionImpl,
    TurboQuantMetadataBuilder,
)

_SQUAT_TQ_PRESET = "turboquant_4bit_nc"


@functools.cache
def _load_rotation_file(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


@functools.cache
def _build_hadamard(d: int, device_str: str) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str))


def _get_layer_index(layer: torch.nn.Module) -> int | None:
    return getattr(layer, "_squat_layer_idx", None)


class SQuatAttentionBackend(TurboQuantAttentionBackend):
    """Attention backend using SQuat KV-cache compression."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["squat"]

    @staticmethod
    def get_name() -> str:
        return "SQUAT"

    @staticmethod
    def get_impl_cls() -> type["SQuatAttentionImpl"]:
        return SQuatAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TurboQuantMetadataBuilder]:
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "squat",
    ) -> tuple[int, ...]:
        return TurboQuantAttentionBackend.get_kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str=_SQUAT_TQ_PRESET,
        )

    @classmethod
    def supports_kv_cache_dtype(
        cls,
        kv_cache_dtype: CacheDType | None,
    ) -> bool:
        return kv_cache_dtype == "squat"


class SQuatAttentionImpl(TurboQuantAttentionImpl):
    """SQuat attention with precomputed subspace correction matrices."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "squat",
        logits_soft_cap: float | None = None,
        attn_type: str = "decoder",
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=_SQUAT_TQ_PRESET,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            **kwargs,
        )
        self._rotation_path = os.environ.get("SQUAT_ROTATION_PATH")
        self._rotations_loaded: dict | None = None

    def _get_rotations(self) -> dict | None:
        if self._rotations_loaded is not None:
            return self._rotations_loaded
        if self._rotation_path and os.path.exists(self._rotation_path):
            self._rotations_loaded = _load_rotation_file(self._rotation_path)
            return self._rotations_loaded
        return None

    def _ensure_on_device(self, layer: Any, device: torch.device) -> None:
        if hasattr(layer, "_tq_cached"):
            return

        D = self.head_size
        H = _build_hadamard(D, str(device))
        layer._tq_PiT = H
        layer._tq_Pi = H
        layer._squat_M_update = None

        rotations = self._get_rotations()
        if rotations is not None:
            layer_idx = _get_layer_index(layer)
            if layer_idx is not None:
                layer_key = f"layer_{layer_idx}"
                if layer_key in rotations and "M_update" in rotations[layer_key]:
                    M = rotations[layer_key]["M_update"]
                    total_kv_heads = M.shape[0]
                    local_kv_heads = self.num_kv_heads
                    if total_kv_heads > local_kv_heads:
                        tp_rank = device.index if device.index is not None else 0
                        start_h = tp_rank * local_kv_heads
                        M = M[start_h : start_h + local_kv_heads]
                    layer._squat_M_update = M.to(device=device, dtype=torch.float32)

        layer._tq_centroids = get_centroids(
            D,
            self.tq_config.centroid_bits,
        ).to(device=device, dtype=torch.float32)
        layer._tq_Pi_half = layer._tq_Pi.to(torch.float16)
        c_sorted, _ = layer._tq_centroids.sort()
        layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
        layer._tq_centroids_sorted = c_sorted
        layer._tq_cached = True

    def _store_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer: Any,
    ) -> None:
        from vllm.v1.attention.ops.triton_squat_store import (
            squat_triton_store_mse,
        )

        correction_scale = float(os.environ.get("SQUAT_CORRECTION_SCALE", "1.0"))

        squat_triton_store_mse(
            key,
            value,
            kv_cache,
            slot_mapping,
            layer._tq_PiT,
            layer._tq_midpoints,
            layer._tq_centroids_sorted,
            layer._squat_M_update,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            correction_scale=correction_scale,
        )
