# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
    CompressedTensorsKVCacheMethod,
)
from vllm.platforms import current_platform

_SCHEME = {
    "type": "float",
    "num_bits": 8,
    "strategy": "tensor",
    "symmetric": True,
}


@pytest.mark.parametrize(
    ("is_fnuz", "has_scheme", "has_q_scale", "kv_expected", "q_expected"),
    [
        pytest.param(True, True, True, 0.6, 0.6, id="fnuz_with_q"),
        pytest.param(True, True, False, 0.6, 1.0, id="fnuz_no_q"),
        pytest.param(False, True, True, 0.3, 0.3, id="not_fnuz"),
        pytest.param(True, False, True, 0.3, 0.3, id="fnuz_no_scheme"),
    ],
)
def test_compressed_tensors_kv_scales_fnuz_doubling(
    monkeypatch, is_fnuz, has_scheme, has_q_scale, kv_expected, q_expected
):
    """Loaded KV scales double on fnuz; a missing q_scale stays 1.0."""
    monkeypatch.setattr(current_platform, "is_fp8_fnuz", lambda: is_fnuz)
    method = CompressedTensorsKVCacheMethod(
        CompressedTensorsConfig(
            target_scheme_map={},
            ignore=[],
            quant_format="float-quantized",
            kv_cache_scheme=_SCHEME if has_scheme else None,
        )
    )

    layer = torch.nn.Module()
    set_default_quant_scales(layer, register_buffer=True)
    layer.kv_cache_dtype = "fp8"
    layer.num_kv_heads = 1
    method.create_weights(layer)
    for name in ("k_scale", "v_scale"):
        getattr(layer, name).data.fill_(0.3)
    if has_q_scale:
        layer.q_scale.data.fill_(0.3)

    method.process_weights_after_loading(layer)
    assert layer._k_scale.item() == pytest.approx(kv_expected)
    assert layer._k_scale_float == pytest.approx(kv_expected)
    assert layer._v_scale_float == pytest.approx(kv_expected)
    assert layer._q_scale_float == pytest.approx(q_expected)
    assert layer._k_scale_cpu.item() == pytest.approx(kv_expected)

    # A second call must not compound the factor.
    method.process_weights_after_loading(layer)
    assert layer._k_scale_float == pytest.approx(kv_expected)
    assert layer._q_scale_float == pytest.approx(q_expected)
