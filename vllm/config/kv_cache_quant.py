# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KV-cache fake-quantization config.

Picklable dataclass that flows through ``VllmConfig`` to every worker. The
heavy SmoothKV calib tensors are NOT stored here -- only the file path is.
Each worker lazy-loads the calib (with caching) inside
``attach_kv_quant_to_layer``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVCacheQuantConfig:
    """Configuration for KV-cache fake-quantization.

    Pass to ``LLM(...)`` via ``kv_cache_quant_config=KVCacheQuantConfig(...)``.

    Attributes:
        method: One of bf16 / fp16 / fp8 / pertoken / smoothkv / smoothkv_fused.
            "bf16" / "fp16" are no-op baselines (no quant).
        group_size: Per-group size for the quant kernels (default 128).
        bits: Bit width for int{2,4} pertoken / smoothkv (default 4).
        calib_path: Required for smoothkv / smoothkv_fused. Points to a `.pt`
            file with keys "s_K" and "s_V" of shape
            ``(num_layers, num_kv_heads, head_dim)``.
        dtype: dtype the calib scales are cast to before being held on CPU.
            "bfloat16" or "float16".
    """

    method: str = "bf16"
    group_size: int = 128
    bits: int = 4
    calib_path: str | None = None
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        valid = {"bf16", "fp16", "fp8", "pertoken", "smoothkv", "smoothkv_fused"}
        if self.method not in valid:
            raise ValueError(
                f"Unknown method {self.method!r}; expected one of {sorted(valid)}"
            )
        if self.method in ("smoothkv", "smoothkv_fused") and not self.calib_path:
            raise ValueError(f"method={self.method!r} requires calib_path")
        if self.method == "kivi2":  # forward-compat: rejected by validation above
            self.bits = 2

    def is_active(self) -> bool:
        """Returns True if this config requires per-step quantization work."""
        return self.method not in ("bf16", "fp16")
