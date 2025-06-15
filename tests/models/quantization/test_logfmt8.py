# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.layers.quantization.logfmt8 import (
    LogFMT8Config, LogFMT8LinearMethod)
from vllm.model_executor.layers.quantization.utils.logfmt8_utils import (
    logfmt8_dequantize, logfmt8_quantize)


def test_logfmt8_quant_dequant_roundtrip():
    torch.manual_seed(0)
    weight = torch.randn(8, 8)
    q, min_val, step, sign = logfmt8_quantize(weight, n_bits=8)
    deq = logfmt8_dequantize(q, min_val, step, sign, n_bits=8)
    # Should be close for nonzero values
    mask = weight != 0
    assert torch.allclose(weight[mask], deq[mask], atol=1e-1)


def test_logfmt8_linear_layer():
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    weight = torch.randn(8, 8)
    bias = torch.randn(8)

    class DummyLayer(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(weight.clone())
            self.logfmt8_min = torch.zeros(1)
            self.logfmt8_step = torch.zeros(1)
            self.logfmt8_sign = torch.zeros(1)

    layer = DummyLayer()
    quant_config = LogFMT8Config()
    method = LogFMT8LinearMethod(quant_config)
    method.process_weights_after_loading(layer)
    out = method.apply(layer, x, bias)
    # Compare to FP32 linear
    ref = torch.nn.functional.linear(x, weight, bias)
    assert out.shape == ref.shape
    # Should be close for most values
    assert torch.allclose(out, ref, atol=1e-1)
