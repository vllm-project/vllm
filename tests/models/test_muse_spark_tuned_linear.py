# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.models.muse_spark.amd import model as muse_model

# Production projection shapes as (in_features, out_features): the pooler
# compresses hidden_size 8192 to moe_hidden_size 4096, the unpooler expands back.
POOLER = (8192, 4096)
UNPOOLER = (4096, 8192)


@pytest.mark.parametrize("m", [1, 64], ids=lambda m: f"M{m}")
@pytest.mark.parametrize("enabled", [False, True], ids=["gate_closed", "gate_open"])
@pytest.mark.parametrize(
    "projection, method_cls, shape",
    [
        ("pooler", muse_model.MuseSparkTorchLinearMethod, POOLER),
        ("unpooler", muse_model.MuseSparkTunedLinearMethod, UNPOOLER),
    ],
    ids=["pooler", "unpooler"],
)
def test_only_the_unpooler_routes_to_tgemm(
    monkeypatch, projection, method_cls, shape, enabled, m
) -> None:
    """Fails if the pooler reaches AITER tuned GEMM, or the unpooler ignores it."""
    calls = []

    def mm(x, w, bias):
        calls.append((x.shape, w.shape))
        return F.linear(x, w, bias) + 1

    monkeypatch.setitem(
        sys.modules, "aiter.tuned_gemm", SimpleNamespace(tgemm=SimpleNamespace(mm=mm))
    )
    monkeypatch.setattr(muse_model, "_use_aiter_tuned_gemm", lambda: enabled)
    in_features, out_features = shape
    layer = SimpleNamespace(
        weight=torch.randn(out_features, in_features, dtype=torch.bfloat16)
    )
    x = torch.randn(m, in_features, dtype=torch.bfloat16)

    out = method_cls().apply(layer, x)

    routed = enabled and projection == "unpooler"
    assert calls == ([(x.shape, layer.weight.shape)] if routed else [])
    torch.testing.assert_close(out, F.linear(x, layer.weight) + int(routed))


@pytest.mark.parametrize("m", [1, 8, 48, 256], ids=lambda m: f"M{m}")
def test_unpooler_tgemm_matches_f_linear_at_production_shape(m) -> None:
    """Fails if a tuned kernel for the unpooler shape loses BF16 accuracy."""
    if not muse_model._use_aiter_tuned_gemm():
        pytest.skip("AITER tuned GEMM is not enabled on this platform")

    in_features, out_features = UNPOOLER
    torch.manual_seed(0)
    layer = SimpleNamespace(
        weight=(
            torch.randn(out_features, in_features, device="cuda") * in_features**-0.5
        ).to(torch.bfloat16)
    )
    x = torch.randn(m, in_features, device="cuda", dtype=torch.bfloat16)

    out = muse_model.MuseSparkTunedLinearMethod().apply(layer, x)

    ref = F.linear(x.float(), layer.weight.float())
    assert out.shape == (m, out_features)
    # Error against an FP32 reference, normalized by the largest output. The
    # worst tuned row measured is 0.0047 (M=48), F.linear's is 0.0025.
    assert (out.float() - ref).abs().max() / ref.abs().max() < 1e-2
