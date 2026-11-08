# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    LinearBase,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.models.deepseek_v32.amd import mtp as deepseek_v32_amd_mtp
from vllm.platforms import current_platform

HIDDEN_SIZE = 64


class _StubModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


def _build_layer(monkeypatch) -> nn.Module:
    """Construct the AMD MTP layer with the heavy submodules stubbed out."""
    monkeypatch.setattr(deepseek_v32_amd_mtp, "SharedHead", _StubModule)
    monkeypatch.setattr(deepseek_v32_amd_mtp, "DeepseekV32DecoderLayer", _StubModule)
    monkeypatch.setattr(current_platform, "device_type", "cpu", raising=False)

    hf_config = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        rms_norm_eps=1e-6,
        index_topk=4,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_config)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        quant_config=None,
    )
    return deepseek_v32_amd_mtp.DeepseekV32MultiTokenPredictorLayer(
        vllm_config, prefix="model.layers.0"
    )


def test_eh_proj_reaches_the_unquantized_gemm_dispatch(
    default_vllm_config, dist_init, monkeypatch
) -> None:
    """Fails if eh_proj goes back to nn.Linear and stops reaching the dispatch."""
    eh_proj = _build_layer(monkeypatch).eh_proj

    assert isinstance(eh_proj, ReplicatedLinear)
    assert isinstance(eh_proj, LinearBase)
    # Only UnquantizedLinearMethod.apply calls dispatch_unquantized_gemm.
    assert isinstance(eh_proj.quant_method, UnquantizedLinearMethod)


def test_eh_proj_keeps_the_nn_linear_weight_contract(
    default_vllm_config, dist_init, monkeypatch
) -> None:
    """Fails if the checkpoint key, shape, dtype or bias of eh_proj.weight move."""
    eh_proj = _build_layer(monkeypatch).eh_proj

    params = dict(eh_proj.named_parameters())
    assert "weight" in params
    assert params["weight"].shape == (HIDDEN_SIZE, HIDDEN_SIZE * 2)
    assert params["weight"].dtype == torch.get_default_dtype()
    assert eh_proj.bias is None


def test_eh_proj_forward_returns_a_bare_tensor(
    default_vllm_config, dist_init, monkeypatch
) -> None:
    """Fails if return_bias=False is dropped, which would hand the call site a tuple."""
    eh_proj = _build_layer(monkeypatch).eh_proj
    torch.nn.init.normal_(eh_proj.weight, std=0.02)

    out = eh_proj(torch.randn(2, HIDDEN_SIZE * 2))

    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, HIDDEN_SIZE)


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="exercises the ROCm GEMM dispatch"
)
def test_eh_proj_forward_calls_the_skinny_kernel(
    default_vllm_config, dist_init, monkeypatch
) -> None:
    """Fails if eh_proj reaches a fallback GEMM instead of wvSplitK at decode M."""
    from vllm import _custom_ops as ops

    eh_proj = _build_layer(monkeypatch).eh_proj
    eh_proj.to(device="cuda", dtype=torch.bfloat16)

    called = []
    original = ops.wvSplitK
    monkeypatch.setattr(
        ops,
        "wvSplitK",
        lambda *args, **kwargs: (called.append(1), original(*args, **kwargs))[1],
    )

    eh_proj(torch.randn(2, HIDDEN_SIZE * 2, dtype=torch.bfloat16, device="cuda"))

    assert called, "eh_proj did not reach wvSplitK"


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="exercises the ROCm GEMM dispatch"
)
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 5, 8, 128])
def test_rocm_dispatch_matches_the_reference_at_eh_proj_shape(num_tokens) -> None:
    """Fails if a skinny kernel is numerically wrong at an eh_proj decode shape."""
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    torch.manual_seed(0)
    out_features, in_features = 1536, 3072
    x = torch.randn(num_tokens, in_features, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16, device="cuda")
    weight /= in_features**0.5

    out = rocm_unquantized_gemm_impl(x, weight, None)
    reference = torch.nn.functional.linear(x.float(), weight.float())

    assert out.shape == (num_tokens, out_features)
    torch.testing.assert_close(
        out.float(), reference, rtol=2e-2, atol=2e-2, equal_nan=False
    )
