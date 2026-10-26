# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.models.muse_spark.amd import model as muse_model


@pytest.mark.parametrize("enabled", [False, True])
def test_tuned_linear_routes_to_tgemm_only_when_enabled(monkeypatch, enabled) -> None:
    calls = []

    def mm(x, w, bias):
        calls.append(x.shape)
        return F.linear(x, w, bias) + 1

    tuned_gemm = SimpleNamespace(tgemm=SimpleNamespace(mm=mm))
    monkeypatch.setitem(sys.modules, "aiter.tuned_gemm", tuned_gemm)
    monkeypatch.setattr(muse_model, "_use_aiter_tuned_gemm", lambda: enabled)
    layer = SimpleNamespace(weight=torch.randn(8, 4))
    x = torch.randn(3, 4)

    out = muse_model.MuseSparkTunedLinearMethod().apply(layer, x)
    torch.testing.assert_close(out, F.linear(x, layer.weight) + int(enabled))
    assert len(calls) == int(enabled)
    torch.testing.assert_close(
        muse_model.MuseSparkTorchLinearMethod().apply(layer, x),
        F.linear(x, layer.weight),
    )
