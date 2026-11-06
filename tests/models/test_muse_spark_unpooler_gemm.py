# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.platforms import current_platform

# Production unpooler shape as (in_features, out_features): moe_hidden_size 4096
# expands back to hidden_size 8192.
UNPOOLER = (4096, 8192)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires a ROCm GPU")
@pytest.mark.parametrize("m", [1, 4, 8, 48, 256], ids=lambda m: f"M{m}")
def test_unpooler_default_gemm_matches_f_linear_at_production_shape(
    monkeypatch, m
) -> None:
    """Fails if the default GEMM dispatch loses BF16 accuracy on the unpooler."""
    from vllm._aiter_ops import rocm_aiter_ops

    in_features, out_features = UNPOOLER
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
    # worst measured is 0.0054 (gfx950, M=192), F.linear's is 0.0035.
    assert (out.float() - ref).abs().max() / ref.abs().max() < 1e-2
