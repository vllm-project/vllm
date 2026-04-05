# SPDX-License-Identifier: Apache-2.0
"""TurboQuant: Near-optimal KV-cache quantization for vLLM.

PolarQuant compression: random rotation + per-coordinate Lloyd-Max
scalar quantization for keys, uniform quantization for values.

Reference: "TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate" (ICLR 2026), Zandieh et al.
"""

from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig

__all__ = ["TurboQuantConfig"]
