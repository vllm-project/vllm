# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools

import pytest
import torch
import torch.nn.functional as F

import vllm.utils.flashinfer as fi_utils
from vllm.model_executor.layers import utils as layer_utils
from vllm.platforms import current_platform
from vllm.utils.flashinfer import is_flashinfer_cutedsl_bf16_gemm_supported


@pytest.fixture(scope="module", autouse=True)
def require_flashinfer_bf16_cutedsl() -> None:
    if not torch.accelerator.is_available():
        pytest.skip("CUDA is required")
    if not is_flashinfer_cutedsl_bf16_gemm_supported():
        pytest.skip("FlashInfer BF16 cute-dsl backend is unavailable")


@pytest.mark.parametrize("n,k", [(1024, 1024), (2048, 3072)])
@pytest.mark.parametrize(
    "m,use_bias,pdl",
    [
        (1, False, False),
        (8, True, False),
        (16, False, True),
        (32, True, True),
        (64, True, True),
        (512, False, False),
    ],
)
def test_flashinfer_bf16_cutedsl_correctness(
    m: int,
    n: int,
    k: int,
    use_bias: bool,
    pdl: bool,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
    bias = (
        torch.randn(n, device="cuda", dtype=torch.bfloat16) * 0.1 if use_bias else None
    )

    actual = layer_utils.cuda_flashinfer_bf16_gemm_impl(
        x, weight, bias, pdl, "flashinfer_cutedsl"
    )
    expected = F.linear(x, weight, bias)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 1024, 1024),
        (64, 2048, 3072),
        (512, 1024, 1000),
        # N not a multiple of 8 (MoE shared_expert_gate, in_proj_ba at TP8) goes
        # through FlashInfer auto.
        (64, 1, 2048),
        (64, 12, 2048),
        (4, 7, 2048),
        # K not a multiple of 8: FlashInfer keeps TGV and TinyGEMM out of auto
        # (flashinfer-ai/flashinfer#9610).
        (64, 12, 2052),
        (64, 128, 1001),
    ],
)
@pytest.mark.parametrize("tune", [False, True])
def test_flashinfer_bf16_auto_correctness(
    m: int, n: int, k: int, tune: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--linear-backend=flashinfer dispatches to mm_bf16(backend="auto") with
    an explicit pdl, with and without autotuning across FlashInfer's BF16 backends."""
    from flashinfer import autotune

    # No vLLM config here, so pretend a tuning file for the model is loaded and
    # has an entry for this GEMM (its M bucket is m).
    monkeypatch.setattr(
        layer_utils, "load_flashinfer_bf16_tuning_for_current_model", lambda: True
    )
    gemm = layer_utils.dispatch_unquantized_gemm("flashinfer")
    monkeypatch.setattr(fi_utils, "_bf16_tuning_points", frozenset({(m, n, k)}))
    monkeypatch.setattr(fi_utils, "_bf16_gemm_tuning_buckets", (m,))
    # A plain function here means the F.linear fallback.
    assert isinstance(gemm, functools.partial)
    assert gemm.keywords == {
        "vllm_backend": "flashinfer",
        "pdl": current_platform.is_arch_support_pdl(),
    }

    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1

    with autotune(tune):
        actual = gemm(None, x, weight)
    expected = F.linear(x, weight)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize("m", [1, 11, 64])
def test_flashinfer_bf16_gemm_without_a_tuned_entry_stays_on_torch(
    m: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GEMM the loaded tuning file has no entry for (here its M bucket) is
    computed by torch, not by FlashInfer's default tactic."""
    monkeypatch.setattr(
        layer_utils, "load_flashinfer_bf16_tuning_for_current_model", lambda: True
    )
    gemm = layer_utils.dispatch_unquantized_gemm("flashinfer")
    assert isinstance(gemm, functools.partial)
    # Entries for another M bucket only, and capture sizes 11, 16 and 64.
    monkeypatch.setattr(fi_utils, "_bf16_tuning_points", frozenset({(16, 1024, 1024)}))
    monkeypatch.setattr(fi_utils, "_bf16_gemm_tuning_buckets", (11, 16, 64))

    torch.manual_seed(0)
    x = torch.randn(m, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16) * 0.1

    torch.testing.assert_close(
        gemm(None, x, weight), F.linear(x, weight), rtol=0, atol=0
    )
