# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Re-export of ``QuantFP8``. Compilation-pass matchers key on this class
identity; subclassing here would silently fail to match."""

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8

__all__ = ["QuantFP8"]
