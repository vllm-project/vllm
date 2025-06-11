# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def logfmt8_quantize(weight: torch.Tensor, n_bits: int = 8):
    # Take absolute values
    abs_weight = weight.abs()
    # Compute log of abs values
    log_weight = torch.log(abs_weight + 1e-12)
    # Find min and max in log space
    min_val = log_weight.min()
    max_val = log_weight.max()
    # Compute step size
    step = (max_val - min_val) / (2**n_bits - 2)
    # Quantize
    q = ((log_weight - min_val) / step).round().clamp(0, 2**n_bits - 2)
    # Encode sign bit separately
    sign = torch.sign(weight)
    # Zero is a special case
    q[weight == 0] = 0
    # Return quantized tensor and quantization parameters
    return q.to(torch.int8), min_val.unsqueeze(0), step.unsqueeze(0), sign


def logfmt8_dequantize(qweight: torch.Tensor,
                       min_val: torch.Tensor,
                       step: torch.Tensor,
                       sign: torch.Tensor,
                       n_bits: int = 8):
    # Dequantize from log space
    log_weight = min_val + qweight.float() * step
    # Exponentiate and restore sign
    weight = sign * torch.exp(log_weight)
    # Zero is a special case
    weight[qweight == 0] = 0.0
    return weight
