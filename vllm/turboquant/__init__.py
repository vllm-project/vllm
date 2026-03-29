# SPDX-License-Identifier: Apache-2.0
"""TurboQuant: Near-optimal KV-cache quantization for vLLM.

Two-stage compression:
  Stage 1 (MSE): Random rotation + per-coordinate Lloyd-Max quantization
  Stage 2 (QJL): 1-bit sign quantization on residuals for unbiased inner products

Reference: "TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate" (ICLR 2026), Zandieh et al.
"""

from vllm.turboquant.config import TurboQuantConfig
from vllm.turboquant.quantizer import TurboQuantizer

__all__ = ["TurboQuantConfig", "TurboQuantizer"]
