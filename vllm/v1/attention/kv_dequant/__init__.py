# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV dequantization dispatch scaffold for attention backends."""

from types import ModuleType

from vllm.v1.kv_cache_interface import KVQuantMode

from . import flashinfer_tile, triton_tile

_BACKEND_TILE_MODULE: dict[str, ModuleType] = {
    "TritonAttentionBackend": triton_tile,
    "TritonAttentionImpl": triton_tile,
    "FlashInferBackend": flashinfer_tile,
    "FlashInferImpl": flashinfer_tile,
}


def assert_backend_supports_kv_quant_mode(
    backend_name: str,
    quant_mode: KVQuantMode,
) -> None:
    """Raise when a backend has not declared support for the kv quant mode."""
    if quant_mode in (KVQuantMode.NONE, KVQuantMode.FP8_PER_TENSOR):
        return

    module = _BACKEND_TILE_MODULE.get(backend_name)
    if module is None or quant_mode not in module.SUPPORTED_MODES:
        raise RuntimeError(
            f"kv-cache quantization mode '{quant_mode.name.lower()}' is not yet "
            f"supported by '{backend_name}'."
        )
