# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end: drive the production sparse-MLA op with an MXFP4 packed cache.

The equivalence that matters: attention over a packed MXFP4 cache must equal
attention over the *same values* materialized as bf16. The unpack produces
exactly the reference-dequantized bf16 tile, so the two runs should agree
bit-for-bit -- any difference is a wiring bug, not a quantization effect.

Also asserts the bf16 path is untouched, because a live benchmark serves from
this tree.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from vllm.v1.attention.ops import mxfp4_mla as mx


def _on_rocm_gpu() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_rocm() and torch.cuda.is_available()


pytestmark = pytest.mark.skipif(not _on_rocm_gpu(), reason="mxfp4_mla is ROCm-only")

SQ, H, LATENT, SKV = 4, 16, 512, 128
ROW = 272
GROUP = 32
PER_QUERY = 64


def _attend(q, kv, indices, indptr):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill

    out = torch.empty(SQ, H, LATENT, dtype=torch.bfloat16, device="cuda")
    rocm_sparse_attn_prefill(
        q=q,
        kv=kv.unsqueeze(1),
        indices=None,
        topk_length=None,
        scale=LATENT**-0.5,
        head_dim=LATENT,
        nope_head_dim=LATENT,
        rope_head_dim=0,
        attn_sink=None,
        output=out,
        ragged_indices=indices,
        ragged_indptr=indptr,
        kv_cache_dtype="mxfp4_mla" if kv.dtype == torch.uint8 else "auto",
    )
    return out


def _attend_split(q, kv, indices, indptr, num_splits: int):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        rocm_sparse_attn_decode_bf16,
    )

    out = torch.empty(SQ, H, LATENT, dtype=torch.bfloat16, device="cuda")
    rocm_sparse_attn_decode_bf16(
        q=q,
        kv=kv.unsqueeze(1),
        scale=LATENT**-0.5,
        head_dim=LATENT,
        nope_head_dim=LATENT,
        rope_head_dim=0,
        attn_sink=None,
        output=out,
        ragged_indices=indices,
        ragged_indptr=indptr,
        num_splits=num_splits,
        kv_cache_dtype="mxfp4_mla" if kv.dtype == torch.uint8 else "auto",
    )
    return out


def _fixture(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(SQ, H, LATENT, generator=g, dtype=torch.bfloat16).cuda()
    latent = torch.randn(SKV, LATENT, generator=g, dtype=torch.float32)
    latent[:, ::97] *= 8.0
    indices = torch.randint(
        0, SKV, (SQ * PER_QUERY,), generator=g, dtype=torch.int32
    ).cuda()
    indptr = torch.arange(SQ + 1, dtype=torch.int32).cuda() * PER_QUERY
    return q, latent, indices, indptr


def _packed_cache(latent: torch.Tensor) -> torch.Tensor:
    from vllm.v1.attention.ops.mxfp4_mla_store import store_mxfp4_mla

    cache = torch.zeros(SKV, ROW, dtype=torch.uint8, device="cuda")
    store_mxfp4_mla(
        latent.to(torch.bfloat16).cuda(),
        torch.arange(SKV).cuda().to(torch.int64),
        cache,
    )
    return cache


def _assert_packed_matches_dequantized(seed: int, num_splits: int = 1):
    """``num_splits > 1`` runs the split-K decode kernel instead."""
    q, latent, indices, indptr = _fixture(seed)

    def attend(kv):
        if num_splits == 1:
            return _attend(q, kv, indices, indptr).cpu()
        return _attend_split(q, kv, indices, indptr, num_splits).cpu()

    got = attend(_packed_cache(latent))
    # The same values, materialized as a plain bf16 cache.
    want = attend(mx.quantize_dequantize(latent.to(torch.bfloat16), GROUP).cuda())

    assert torch.equal(got, want), (
        "attention over the packed cache disagrees with attention over the "
        "same values in bf16 -- this is a wiring bug, not quantization"
    )


@pytest.mark.parametrize("num_splits", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 1])
def test_packed_cache_matches_dequantized_bf16_cache(seed: int, num_splits: int):
    _assert_packed_matches_dequantized(seed, num_splits)


def test_software_unpack_matches_dequantized_bf16_cache():
    """Force the software unpack used off gfx950 through both attention kernels.

    Triton reads _HW_UNPACK at compile time and refuses to relaunch a kernel
    after it changes, so it is set in a fresh process before the first launch.
    """
    script = (
        "from vllm.triton_utils import tl\n"
        "from vllm.v1.attention.ops import mxfp4_mla_read\n"
        "mxfp4_mla_read._HW_UNPACK = tl.constexpr(False)\n"
        "import test_mxfp4_mla_e2e\n"
        "test_mxfp4_mla_e2e._assert_packed_matches_dequantized(seed=0)\n"
        "test_mxfp4_mla_e2e._assert_packed_matches_dequantized(0, num_splits=4)\n"
    )
    path = os.pathsep.join(
        [os.path.dirname(__file__), os.environ.get("PYTHONPATH", "")]
    )
    subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": path},
        check=True,
    )


def test_packed_cache_tracks_unquantized_attention():
    """Sanity bound only -- this is not the accuracy instrument.

    The fixture is deliberately adversarial: 8x outliers planted every 97th
    channel, and a random Q that spreads attention evenly instead of
    concentrating it the way a trained query does. A few percent relative MSE
    here is expected: on GLM-5.3-Flash, GSM8K showed no accuracy loss for FP4
    at group 32 (96.66/96.74% against 96.89% for bf16).

    So the threshold is set where genuine divergence would show -- a wrong
    scale exponent, a permuted latent axis, a dropped sign -- all of which move
    this by an order of magnitude, not a few percent. Model-level evaluation
    remains the authoritative accuracy check.
    """
    q, latent, indices, indptr = _fixture(seed=2)

    got = _attend(q, _packed_cache(latent), indices, indptr).float().cpu()
    exact = _attend(q, latent.to(torch.bfloat16).cuda(), indices, indptr).float().cpu()

    rel = ((got - exact).pow(2).sum() / exact.pow(2).sum()).item()
    assert rel < 0.15, f"relative MSE vs bf16 attention is {rel:.4f}, too large"

    # A structural bug reads as near-total decorrelation, so also require the
    # outputs to be strongly correlated rather than merely small in difference.
    corr = torch.corrcoef(torch.stack([got.flatten(), exact.flatten()]))[0, 1].item()
    assert corr > 0.97, f"correlation with bf16 attention is only {corr:.4f}"


def test_bf16_path_is_untouched():
    """The dtype branch must not perturb the unquantized path at all."""
    q, latent, indices, indptr = _fixture(seed=3)
    kv = latent.to(torch.bfloat16).cuda()
    a = _attend(q, kv, indices, indptr)
    b = _attend(q, kv, indices, indptr)
    assert torch.equal(a, b), "bf16 path is not deterministic"


def test_cache_is_3_76x_smaller():
    """The actual point: bytes moved per token."""
    bf16_row = LATENT * 2
    assert ROW == 272
    ratio = bf16_row / ROW
    assert 3.7 < ratio < 3.8, ratio


def test_dense_indices_entry_point_also_reaches_the_mxfp4_path():
    """The other entry point into the op must handle MXFP4 too.

    ``rocm_sparse_attn_prefill`` branches: with ``ragged_indices`` it calls the
    ragged kernel directly, and otherwise it goes through
    ``_rocm_sparse_attn_prefill_triton`` with a dense ``indices`` tensor. Only
    the ragged kernel reads a packed cache, so the dense branch is correct only
    because it converts its indices and delegates to that same launcher.

    Nothing enforces that delegation. If the dense branch ever grew its own
    kernel launch, a packed cache would be read as bf16 and return plausible
    garbage rather than failing, so the equivalence is asserted here from the
    outside instead of by reading the source.

    The shipping backend always passes ragged arguments, so this covers an entry
    point that is currently unreached rather than a live one.
    """
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill

    q, latent, indices, _ = _fixture(seed=0)
    dense = indices.reshape(SQ, PER_QUERY)

    def attend_dense(kv):
        out = torch.empty(SQ, H, LATENT, dtype=torch.bfloat16, device="cuda")
        rocm_sparse_attn_prefill(
            q=q,
            kv=kv.unsqueeze(1),
            indices=dense,
            topk_length=None,
            scale=LATENT**-0.5,
            head_dim=LATENT,
            nope_head_dim=LATENT,
            rope_head_dim=0,
            attn_sink=None,
            output=out,
            ragged_indices=None,
            ragged_indptr=None,
            kv_cache_dtype="mxfp4_mla" if kv.dtype == torch.uint8 else "auto",
        )
        return out

    got = attend_dense(_packed_cache(latent)).cpu()
    dequantized = mx.quantize_dequantize(latent.to(torch.bfloat16), GROUP).cuda()
    want = attend_dense(dequantized).cpu()

    assert torch.equal(got, want), (
        "the dense-indices entry point disagrees with bf16 over the same "
        "values, so it is not reaching the MXFP4-aware launcher"
    )
