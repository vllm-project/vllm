# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the gfx942 (MI300X) fp8 MQA-logits indexer kernel.

Exercises the native-fp8-MFMA path used by the DeepSeek-V4 sparse-attention
indexer (``fp8_mqa_logits_gfx942``), specifically the features it adds over the
software fp8->fp16 expansion path:

* the OCP ``e4m3fn`` -> ``e4m3fnuz`` operand bitcast (native ``v_mfma_*_f8``),
* the host-side ``0x80`` scrub (``-0.0`` in OCP e4m3, ``NaN`` in e4m3fnuz), and
* the fused out-of-window ``-inf`` epilogue (``OOW_FILL``).

Only gfx942 takes this path (see ``rocm_aiter_mla_sparse``), so every test is
skipped elsewhere.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="gfx942 fp8 MQA-logits kernel is ROCm-only",
)


def _on_gfx942() -> bool:
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import _ON_GFX942

        return bool(_ON_GFX942)
    except Exception:
        return False


requires_gfx942 = pytest.mark.skipif(
    not _on_gfx942(),
    reason="native fp8 MFMA MQA-logits path is only implemented for AMD gfx942",
)

FP8 = torch.float8_e4m3fn


def _make_inputs(M, N, H, D, *, device, full_window=True, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    # keep magnitudes modest so the fp8 cast stays in-range
    q = (torch.randn(M, H, D, generator=g, device=device, dtype=torch.bfloat16) * 0.25).to(FP8)
    k = (torch.randn(N, D, generator=g, device=device, dtype=torch.bfloat16) * 0.25).to(FP8)
    scale = torch.rand(N, 1, generator=g, device=device, dtype=torch.float32) * 0.5 + 0.5
    weights = torch.rand(M, H, generator=g, device=device, dtype=torch.float32)
    if full_window:
        ks = torch.zeros(M, dtype=torch.int32, device=device)
        ke = torch.full((M,), N, dtype=torch.int32, device=device)
    else:
        # staggered windows strictly inside [0, N)
        base = torch.arange(M, device=device, dtype=torch.int32)
        ks = (base % max(1, N // 4)).to(torch.int32)
        ke = torch.clamp(ks + (N // 2), max=N).to(torch.int32)
    return q, k, scale, weights, ks, ke


def _run(q, k, scale, weights, ks, ke):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import fp8_mqa_logits_torch
    from vllm.v1.attention.ops.triton_fp8_mqa_logits import fp8_mqa_logits_gfx942

    got = fp8_mqa_logits_gfx942(q, k, scale, weights, ks, ke)
    ref = fp8_mqa_logits_torch(q, (k, scale), weights, ks, ke)
    return got, ref


def _topk_overlap(got, ref, kk):
    top_got = got.topk(kk, dim=1).indices
    top_ref = ref.topk(kk, dim=1).indices
    fr = [
        len(set(a.tolist()) & set(b.tolist())) / kk
        for a, b in zip(top_got, top_ref)
    ]
    return torch.tensor(fr)


@requires_gfx942
@pytest.mark.parametrize("M,N,H,D", [(128, 512, 64, 128), (64, 256, 32, 128)])
def test_matches_reference_full_window(M, N, H, D):
    device = torch.device("cuda")
    q, k, scale, weights, ks, ke = _make_inputs(M, N, H, D, device=device)
    got, ref = _run(q, k, scale, weights, ks, ke)

    assert got.shape == (M, N)
    assert torch.isfinite(got).all(), "full-window logits must be finite"

    # Scale-robust per-row similarity: both arms share the same fp8 inputs and
    # differ only in matmul precision (native fp8 MFMA vs bf16 upcast), so the
    # rows should be nearly collinear.
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=1)
    assert cos.mean() >= 0.99, f"mean row cosine too low: {cos.mean():.4f}"
    assert cos.min() >= 0.95, f"worst row cosine too low: {cos.min():.4f}"

    # The indexer only consumes the top-k, so top-k agreement is the
    # functionally meaningful correctness signal.
    overlap = _topk_overlap(got, ref, kk=64)
    assert overlap.mean() >= 0.85, f"top-64 overlap too low: {overlap.mean():.3f}"


@requires_gfx942
def test_out_of_window_is_neg_inf():
    device = torch.device("cuda")
    M, N, H, D = 96, 384, 32, 128
    q, k, scale, weights, ks, ke = _make_inputs(
        M, N, H, D, device=device, full_window=False
    )
    got, ref = _run(q, k, scale, weights, ks, ke)

    col = torch.arange(N, device=device)[None, :]
    in_window = (col >= ks[:, None]) & (col < ke[:, None])

    # OOW_FILL: positions outside [ks, ke) are pre-filled with -inf by the
    # kernel itself so the top-k consumer needs no separate masking pass.
    assert torch.isneginf(got[~in_window]).all(), (
        "out-of-window entries must be -inf"
    )
    assert torch.isfinite(got[in_window]).all(), (
        "in-window entries must be finite"
    )


@requires_gfx942
def test_neg_zero_fnuz_scrub_produces_no_nan():
    """Byte ``0x80`` is ``-0.0`` in OCP e4m3fn but ``NaN`` in e4m3fnuz. The
    kernel scrubs it host-side (``0x80`` -> ``0x81``) before the fnuz bitcast so
    a single ``-0.0`` byte cannot poison an entire output column with NaN."""
    device = torch.device("cuda")
    M, N, H, D = 64, 256, 32, 128
    q, k, scale, weights, ks, ke = _make_inputs(M, N, H, D, device=device)

    # Plant 0x80 (-0.0) bytes into a band of in-window K columns.
    k = k.contiguous()
    k.view(torch.uint8)[10:20, :] = 0x80

    from vllm.v1.attention.ops.triton_fp8_mqa_logits import fp8_mqa_logits_gfx942

    got = fp8_mqa_logits_gfx942(q, k, scale, weights, ks, ke)
    assert torch.isfinite(got).all(), (
        "0x80 (-0.0) K bytes must not yield NaN/Inf after the fnuz scrub"
    )
