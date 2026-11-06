# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.muse_spark.amd import model as muse_model
from vllm.platforms import current_platform

# Production shapes as (in_features, out_features): the pooler projects
# hidden_size 8192 to moe_hidden_size 4096, the unpooler projects it back.
POOLER = (8192, 4096)
UNPOOLER = (4096, 8192)


@pytest.mark.parametrize(
    "is_rocm, cdna, expected",
    [(False, 3, False), (True, 2, False), (True, 3, True), (True, 4, False)],
    ids=["not_rocm", "cdna2", "cdna3", "cdna4"],
)
def test_pooler_default_gemm_gate(monkeypatch, is_rocm, cdna, expected) -> None:
    """Fails if the pooler leaves F.linear anywhere but CDNA3."""
    monkeypatch.setattr(muse_model.current_platform, "is_rocm", lambda: is_rocm)
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms.rocm",
        SimpleNamespace(get_cdna_version=lambda: cdna),
    )
    assert muse_model._pooler_uses_default_gemm() is expected


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires a ROCm GPU")
@pytest.mark.parametrize("m", [1, 4, 8, 48, 256], ids=lambda m: f"M{m}")
@pytest.mark.parametrize("shape", [POOLER, UNPOOLER], ids=["pooler", "unpooler"])
def test_default_gemm_matches_f_linear_at_production_shape(
    monkeypatch, shape, m
) -> None:
    """Fails if the default GEMM dispatch loses BF16 accuracy where it is used."""
    if shape == POOLER and not muse_model._pooler_uses_default_gemm():
        pytest.skip("the pooler stays on F.linear here")
    from vllm._aiter_ops import rocm_aiter_ops

    in_features, out_features = shape
    torch.manual_seed(0)
    layer = SimpleNamespace(
        weight=(
            torch.randn(out_features, in_features, device="cuda") * in_features**-0.5
        ).to(torch.bfloat16)
    )
    x = torch.randn(m, in_features, device="cuda", dtype=torch.bfloat16)

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()
    try:
        out = UnquantizedLinearMethod().apply(layer, x)
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()

    ref = F.linear(x.float(), layer.weight.float())
    assert out.shape == (m, out_features)
    # Error against an FP32 reference, normalized by the largest output. The
    # worst measured is 0.0054 (unpooler, gfx950, M=192), F.linear's is 0.0035.
    assert (out.float() - ref).abs().max() / ref.abs().max() < 1e-2
    # AITER's lossy gfx950 pooler kernels put 3.5-9.3% of elements outside this.
    torch.testing.assert_close(out.float(), ref, rtol=1.6e-2, atol=1e-5)
