# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run `pytest tests/model_executor/test_deep_gemm_warmup.py`."""

from unittest.mock import Mock

import torch

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.warmup.deep_gemm_warmup import (
    _block_fp8_linear_kernel,
    _extract_data_from_linear_base_module,
)


def test_fp8_linear_method_warmup_discovery():
    """Warmup now reads the block size off the layer rather than
    ``quant_method.quant_config``, which ModelOptLinearMethod does not have.
    Fp8LinearMethod.create_weights sets ``layer.weight_block_size`` from the
    same value, so the two must stay in agreement -- otherwise every block-FP8
    model warms up against the wrong shape, silently.
    """
    kernel = Mock()
    method = Mock(spec=Fp8LinearMethod)
    method.__class__ = Fp8LinearMethod
    method.block_quant = True
    method.use_marlin = False
    method.fp8_linear = kernel
    method.quant_config = Mock(weight_block_size=[128, 128])

    layer = Mock(spec=LinearBase)
    layer.__class__ = LinearBase
    layer.quant_method = method
    layer.weight = torch.empty((256, 512), dtype=torch.float8_e4m3fn)
    layer.weight_scale_inv = torch.empty((2, 4), dtype=torch.float32)
    layer.weight_block_size = [128, 128]

    assert _block_fp8_linear_kernel(method) is kernel
    _, _, block_size = _extract_data_from_linear_base_module(layer)
    assert block_size == method.quant_config.weight_block_size
