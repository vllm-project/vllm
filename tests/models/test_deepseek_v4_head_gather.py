# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather-first head (VLLM_MOE_HEAD_GATHER_FIRST) numerical contract.

The optimization gathers the logits rows of the [T, hc, H] head input
before hc_head + final RMSNorm instead of running both on all T rows and
gathering afterwards. Both ops are row-wise:

- hc_head (fused tilelang kernel) is bitwise identical per row regardless
  of how many rows are in the batch.
- The vllm_c RMSNorm kernel may pick a different reduction schedule for
  different row counts, so the full chain can differ by at most 1 bf16 ULP
  on rare elements. The test pins that envelope so any regression beyond
  reduction-order noise fails loudly.
"""

import pytest
import torch

from vllm.model_executor.kernels.mhc.tilelang import hc_head_fused_kernel_tilelang
from vllm.platforms import current_platform

HIDDEN_SIZE = 4096
HC_MULT = 4
RMS_EPS = 1e-6
HC_EPS = 1e-6
DTYPE = torch.bfloat16

# Observed on H20 (sm90): a few elements per ~1e5 differ by 1 ULP when the
# rms_norm row count changes. Anything larger indicates a real bug.
MAX_MISMATCH_RATE = 1e-4


def _mhc_tilelang_available() -> bool:
    """CUDA and the TileLang kernel dependency must both be present.

    The op is registered at import time even without TileLang, so checking
    ``torch.ops.vllm.hc_head_fused_kernel_tilelang`` alone is not enough;
    the lazy kernel import would fail on such builds at runtime.
    """
    if not current_platform.is_cuda():
        return False
    try:
        import tilelang  # noqa: F401
    except ImportError:
        return False
    return True


def _make_inputs(num_tokens: int, device: torch.device):
    torch.manual_seed(20260828)
    hs = torch.randn(num_tokens, HC_MULT, HIDDEN_SIZE, device=device).to(DTYPE)
    fn = torch.randn(HC_MULT, HC_MULT * HIDDEN_SIZE, device=device) * 0.02
    hc_scale = torch.ones(1, device=device)
    hc_base = torch.zeros(HC_MULT, device=device)
    norm_weight = torch.empty(HIDDEN_SIZE, device=device, dtype=DTYPE).uniform_(
        0.5, 1.5
    )
    return hs, fn, hc_scale, hc_base, norm_weight


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    torch.ops._C.rms_norm(out, x, weight, RMS_EPS)
    return out


def _hc_head(hs, fn, hc_scale, hc_base):
    return hc_head_fused_kernel_tilelang(hs, fn, hc_scale, hc_base, RMS_EPS, HC_EPS)


@pytest.mark.skipif(
    not _mhc_tilelang_available(), reason="requires CUDA + tilelang"
)
@pytest.mark.parametrize("num_tokens", [512, 2048])
@pytest.mark.parametrize("num_logits", [1, 64])
def test_hc_head_gather_first_bitwise(num_tokens: int, num_logits: int) -> None:
    """hc_head rows are bitwise identical whether gathered before or after."""
    device = torch.device("cuda:0")
    hs, fn, hc_scale, hc_base, _ = _make_inputs(num_tokens, device)
    idx = torch.linspace(0, num_tokens - 1, num_logits, device=device).long()

    full_then_gather = _hc_head(hs, fn, hc_scale, hc_base)[idx]
    gather_first = _hc_head(hs[idx], fn, hc_scale, hc_base)
    assert torch.equal(full_then_gather, gather_first)


@pytest.mark.skipif(
    not _mhc_tilelang_available(), reason="requires CUDA + tilelang"
)
def test_head_norm_chain_all_rows_bitwise() -> None:
    """L == T (decode-shaped) gather-first is bitwise identical end to end."""
    device = torch.device("cuda:0")
    num_tokens = 256
    hs, fn, hc_scale, hc_base, weight = _make_inputs(num_tokens, device)
    idx = torch.arange(num_tokens, device=device)

    full_then_gather = _rms_norm(_hc_head(hs, fn, hc_scale, hc_base), weight)[idx]
    gather_first = _rms_norm(_hc_head(hs[idx], fn, hc_scale, hc_base), weight)
    assert torch.equal(full_then_gather, gather_first)


@pytest.mark.skipif(
    not _mhc_tilelang_available(), reason="requires CUDA + tilelang"
)
@pytest.mark.parametrize("num_tokens", [2048, 8192])
def test_head_norm_chain_within_one_ulp(num_tokens: int) -> None:
    """Full chain (hc_head + vllm_c rms_norm) matches within 1 bf16 ULP."""
    device = torch.device("cuda:0")
    num_logits = 64
    hs, fn, hc_scale, hc_base, weight = _make_inputs(num_tokens, device)
    idx = torch.linspace(0, num_tokens - 1, num_logits, device=device).long()

    full_then_gather = _rms_norm(_hc_head(hs, fn, hc_scale, hc_base), weight)[idx]
    gather_first = _rms_norm(_hc_head(hs[idx], fn, hc_scale, hc_base), weight)

    mismatch = full_then_gather != gather_first
    rate = mismatch.float().mean().item()
    assert rate <= MAX_MISMATCH_RATE, f"mismatch rate {rate:.2e} too high"
    if mismatch.any():
        a = full_then_gather[mismatch].float()
        b = gather_first[mismatch].float()
        eps = torch.finfo(DTYPE).eps
        assert ((a - b).abs() <= eps * torch.maximum(a.abs(), b.abs())).all()
