# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Re-export of upstream ``QuantFP8``.

Compilation passes (e.g. ``MatcherQuantFP8`` in
``vllm/compilation/passes/fusion/``) match against the upstream class
identity to rewrite torch.compile subgraphs. A vendored copy would be a
different class and silently fail to match.
"""

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8

__all__ = ["QuantFP8"]
