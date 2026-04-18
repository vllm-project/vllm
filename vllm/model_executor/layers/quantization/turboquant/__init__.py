# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant: KV-cache quantization for vLLM.

Hadamard rotation + per-coordinate Lloyd-Max scalar quantization for
keys, uniform quantization for values.

The technique implemented here consists of the scalar case of the HIGGS
quantization method (Malinovskii et al., "Pushing the Limits of Large
Language Model Quantization via the Linearity Theorem", NAACL 2025;
preprint arXiv:2411.17525): rotation + optimized grid + optional
re-normalization, applied to KV cache compression. A first application
of this approach to KV-cache compression is in "Cache Me If You Must:
Adaptive Key-Value Quantization for Large Language Models" (Shutova
et al., ICML 2025; preprint arXiv:2501.19392). Both these references
pre-date the TurboQuant paper (Zandieh et al., ICLR 2026).
"""

from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig

__all__ = ["TurboQuantConfig"]
