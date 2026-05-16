# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention


@pytest.mark.skip_global_cleanup
def test_attention_calc_kv_scales_updates_tensor_and_float_scales():
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.0),
        _k_scale=torch.tensor(0.0),
        _v_scale=torch.tensor(0.0),
        q_range=torch.tensor(2.0),
        k_range=torch.tensor(4.0),
        v_range=torch.tensor(8.0),
        calculate_kv_scales=True,
    )
    query = torch.tensor([-2.0, 4.0, -6.0])
    key = torch.tensor([-3.0, 1.0, 5.0])
    value = torch.tensor([16.0, -8.0, 4.0])

    Attention.calc_kv_scales(layer, query, key, value)

    assert layer._q_scale_float == pytest.approx(3.0)
    assert layer._k_scale_float == pytest.approx(1.25)
    assert layer._v_scale_float == pytest.approx(2.0)
    assert layer.calculate_kv_scales is False
    torch.testing.assert_close(layer._q_scale, torch.tensor(3.0))
    torch.testing.assert_close(layer._k_scale, torch.tensor(1.25))
    torch.testing.assert_close(layer._v_scale, torch.tensor(2.0))


@pytest.mark.skip_global_cleanup
def test_mla_calc_kv_scales_uses_shared_kv_abs_max():
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.0),
        _k_scale=torch.tensor(0.0),
        _v_scale=torch.tensor(0.0),
        q_range=torch.tensor(2.0),
        k_range=torch.tensor(5.0),
        v_range=torch.tensor(10.0),
        calculate_kv_scales=True,
    )
    q = torch.tensor([-1.0, 3.0, -5.0])
    kv_c_normed = torch.tensor([-2.0, 4.0, -8.0])
    k_pe = torch.tensor([0.5, -0.5, 1.0])

    MLAAttention.calc_kv_scales(layer, q, kv_c_normed, k_pe)

    assert layer._q_scale_float == pytest.approx(2.5)
    assert layer._k_scale_float == pytest.approx(1.6)
    assert layer._v_scale_float == pytest.approx(0.8)
    assert layer.calculate_kv_scales is False
    torch.testing.assert_close(layer._q_scale, torch.tensor(2.5))
    torch.testing.assert_close(layer._k_scale, torch.tensor(1.6))
    torch.testing.assert_close(layer._v_scale, torch.tensor(0.8))