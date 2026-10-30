# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DSpark context-KV write for all draft layers in one CUDA kernel.

Replaces the permute copy, per-layer K-norm launches + stack, positions.repeat,
rotary_embedding and per-layer reshape_and_cache_flash calls in
``DFlashQwen3Model.precompute_and_store_context_kv``. K/V are read straight
from the fused projection output ``all_kv`` ``[num_ctx, L, 2, nkv, 128]`` and
written into each layer's paged cache (FLASH_ATTN, FLASHINFER_TS)
``[num_blocks, nkv, block_size, 2 * 128]`` (K at ``[..., :128]``, V at
``[..., 128:]``), matching ``reshape_and_cache_flash``: fp8 e4m3 of
``x / scale`` for kv_cache_dtype fp8, raw bf16 otherwise; slots < 0 are padding
and are not written.

K gets the Gemma RMSNorm of its layer (``x * rsqrt(mean(x^2) + eps) * (1 + w)``
in fp32, materialized to bf16 like the unfused flashinfer path) followed by
partial NeoX RoPE with the token's position (see
``fused_minimax_m3_dspark_context_kv``). Each layer's cache is a separate
allocation, addressed as layer 0's cache plus a per-layer element offset, and
each fp8 scale as layer 0's k scale plus a float offset; the offset tables are
built once and cached (keyed by the data_ptrs) so the call stays CUDA-graph
safe.
"""

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

_SUPPORTED_KV_DTYPES = ("auto", "bfloat16", "fp8", "fp8_e4m3")
_HEAD_DIM = 128


def rotary_dims_supported(rotary_dim: int, head_dim: int) -> bool:
    """The kernel rotates the leading 16, 32 or 64 dims of a 128-dim head."""
    return head_dim == _HEAD_DIM and rotary_dim in (16, 32, 64)


class DSparkContextKVWriter:
    """Writes precomputed context K/V into every draft layer's KV cache.

    Built from the draft's ``Attention`` layers; the cache- and scale-offset
    tables are cached, keyed by their data_ptrs, so calls captured in a CUDA
    graph keep addressing the same memory.
    """

    def __init__(self, attn_layers: list) -> None:
        layers = list(attn_layers)
        error = self._validate(layers)
        if error is not None:
            raise ValueError(f"unsupported DSpark context-KV layout: {error}")
        self._layers = layers
        cache = layers[0].kv_cache
        self._num_layers = len(layers)
        self._num_kv_heads = cache.shape[1]
        self._kv_cache_dtype = "fp8" if cache.dtype == torch.uint8 else "auto"
        self._table_key: tuple | None = None
        self._cache_offsets: torch.Tensor | None = None
        self._scale_offsets: torch.Tensor | None = None

    @staticmethod
    def _validate(layers: list) -> str | None:
        if not layers:
            return "no layers"
        ref = None
        for a in layers:
            kv_dtype = getattr(a, "kv_cache_dtype", None)
            if kv_dtype not in _SUPPORTED_KV_DTYPES:
                return f"kv_cache_dtype {kv_dtype!r}"
            cache = getattr(a, "kv_cache", None)
            if not isinstance(cache, torch.Tensor) or cache.numel() == 0:
                return "KV cache not bound"
            if cache.dim() != 4 or cache.shape[3] != 2 * _HEAD_DIM:
                return f"unexpected cache shape {tuple(cache.shape)}"
            block_size = cache.shape[2]
            if block_size & (block_size - 1):
                return f"block size {block_size} is not a power of two"
            if not cache.is_contiguous():
                return "cache is not contiguous"
            if kv_dtype in ("fp8", "fp8_e4m3"):
                if cache.dtype != torch.uint8:
                    return f"fp8 cache has dtype {cache.dtype}"
            elif cache.dtype != torch.bfloat16:
                return f"cache has dtype {cache.dtype}"
            for name in ("_k_scale", "_v_scale"):
                scale = getattr(a, name, None)
                if (
                    not isinstance(scale, torch.Tensor)
                    or scale.dtype != torch.float32
                    or not scale.is_cuda
                ):
                    return f"{name} is not a CUDA fp32 tensor"
            key = (cache.shape, cache.stride(), cache.dtype, kv_dtype)
            if ref is None:
                ref = key
            elif key != ref:
                return "layers do not share cache shape/stride/dtype"
            # Caches are addressed from layer 0's as 16-element aligned offsets.
            delta = cache.data_ptr() - layers[0].kv_cache.data_ptr()
            if delta % (16 * cache.element_size()):
                return "layer caches are not 16-element aligned to each other"
        return None

    @staticmethod
    def supported(attn_layers: list) -> bool:
        # The fp8 store is OCP E4M3; ROCm fnuz caches keep the base path.
        if not current_platform.is_cuda() or not hasattr(
            torch.ops._C, "fused_minimax_m3_dspark_context_kv"
        ):
            return False
        return DSparkContextKVWriter._validate(list(attn_layers)) is None

    def _ensure_tables(self) -> None:
        layers = self._layers
        key = tuple(
            (a.kv_cache.data_ptr(), a._k_scale.data_ptr(), a._v_scale.data_ptr())
            for a in layers
        )
        if key == self._table_key:
            return
        cache0, scale0 = layers[0].kv_cache, layers[0]._k_scale

        def table(offsets: list[int]) -> torch.Tensor:
            return torch.tensor(offsets, dtype=torch.int64, device=cache0.device)

        self._cache_offsets = table(
            [
                (a.kv_cache.data_ptr() - cache0.data_ptr()) // cache0.element_size()
                for a in layers
            ]
        )
        # [2, L]: k scales, then v scales, as float offsets from layer 0's k.
        self._scale_offsets = table(
            [
                (getattr(a, name).data_ptr() - scale0.data_ptr()) // 4
                for name in ("_k_scale", "_v_scale")
                for a in layers
            ]
        )
        self._table_key = key

    def __call__(
        self,
        all_kv: torch.Tensor,
        k_norm_weight: torch.Tensor,
        eps: float,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        rotary_dim: int,
        slot_mapping: torch.Tensor,
    ) -> None:
        num_ctx, L, _, nkv, _ = all_kv.shape
        assert self._num_layers == L and self._num_kv_heads == nkv
        if num_ctx == 0:
            return
        self._ensure_tables()
        ops.fused_minimax_m3_dspark_context_kv(
            all_kv,
            k_norm_weight,
            cos_sin_cache,
            positions,
            slot_mapping,
            self._layers[0].kv_cache,
            self._cache_offsets,
            self._layers[0]._k_scale,
            self._scale_offsets,
            rotary_dim,
            eps,
            self._kv_cache_dtype,
        )
