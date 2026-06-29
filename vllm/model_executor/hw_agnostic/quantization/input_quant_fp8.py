# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Re-export of ``QuantFP8``.

Compilation passes (e.g. ``MatcherQuantFP8`` in
``vllm/compilation/passes/fusion/``) match against this class identity to
rewrite torch.compile subgraphs; a separate class here would silently
fail to match.
"""

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8

__all__ = ["QuantFP8"]
