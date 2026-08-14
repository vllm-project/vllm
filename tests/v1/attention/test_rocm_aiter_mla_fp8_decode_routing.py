# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-kernel routing for the ROCm AITER MLA backend.

Gluon exposes a single fp8-KV regime, bh16bn128. It is a bf16-query kernel that
upcasts the cache in registers with a hardcoded scale of 1.0, and it asserts
batch_size == 1, so it cannot serve a decode batch. Every fp8 shape therefore
has to land on the asm kernels, which ship real fp8 variants for gqa=16. These
tests pin that down at the predicates, since the failure it prevents is either a
batch assertion or -- worse -- a silently wrong result.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm AITER MLA tests", allow_module_level=True)

from vllm.v1.attention.backends.mla import rocm_aiter_mla  # noqa: E402
from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper  # noqa: E402

FP8_DTYPES = ["fp8", "fp8_e4m3", "fp8_e5m2"]
UNQUANTIZED_DTYPES = ["auto", "bfloat16"]


@pytest.fixture
def gluon_available(monkeypatch):
    """Pin the arch gate and the mode knob so only the dtype rules are in play.

    Gluon ships a gfx950 build only, and VLLM_ROCM_AITER_MLA_ASM_PADDING can
    force the asm path on any arch. Without pinning both, the expectations that
    bf16 *keeps* Gluon would pass on a gfx950 host and fail on gfx942 for
    reasons that have nothing to do with what these tests cover.
    """
    monkeypatch.setattr(rocm_aiter_mla, "_gluon_mla_decode_supported", lambda: True)
    monkeypatch.setattr(rocm_aiter_mla, "_aiter_mla_small_head_mode", lambda: "auto")


@pytest.mark.parametrize("kv_cache_dtype", FP8_DTYPES)
@pytest.mark.parametrize("num_heads", [1, 2, 4, 5, 6, 8, 12, 16, 32, 128])
@pytest.mark.parametrize("max_qo_len", [1, 2, 4, 5, 8, 15])
def test_fp8_never_routes_to_gluon(kv_cache_dtype, num_heads, max_qo_len):
    """No fp8 shape reaches either Gluon entry point.

    The head count is deliberately swept across divisors of 16 as well as
    non-divisors: the divisor case (e.g. 8 heads at TP8) is the one that stays
    on Gluon for bf16 and so is the one an fp8 guard has to override.
    """
    assert not AiterMLAHelper.use_gluon_decode(num_heads, max_qo_len, kv_cache_dtype)
    assert not AiterMLAHelper.use_gluon_verify(num_heads, max_qo_len, kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", FP8_DTYPES)
@pytest.mark.parametrize("num_heads", [1, 2, 4, 8, 12])
@pytest.mark.parametrize("mode", ["auto", "gluon", "asm"])
def test_fp8_never_routes_to_gluon_under_any_mode(
    monkeypatch, kv_cache_dtype, num_heads, mode
):
    """VLLM_ROCM_AITER_MLA_ASM_PADDING cannot force an fp8 cache onto Gluon.

    The dtype guard deliberately precedes the mode knob: honouring an explicit
    "gluon" request under fp8 would hand Gluon the batch it asserts against, so
    it is overridden rather than obeyed. Pinned here because the override is a
    correctness decision, not a preference.
    """
    monkeypatch.setattr(rocm_aiter_mla, "_gluon_mla_decode_supported", lambda: True)
    monkeypatch.setattr(rocm_aiter_mla, "_aiter_mla_small_head_mode", lambda: mode)
    assert not AiterMLAHelper.use_gluon_decode(num_heads, 1, kv_cache_dtype)
    assert not AiterMLAHelper.use_gluon_verify(num_heads, 8, kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", FP8_DTYPES + UNQUANTIZED_DTYPES)
@pytest.mark.parametrize("num_heads", [16, 17, 32, 64, 128])
@pytest.mark.parametrize("max_qo_len", [1, 4, 8])
def test_large_head_counts_never_use_gluon(kv_cache_dtype, num_heads, max_qo_len):
    """>= 16 heads has always been asm-only; the dtype guard must not change it."""
    assert not AiterMLAHelper.use_gluon_decode(num_heads, max_qo_len, kv_cache_dtype)
    assert not AiterMLAHelper.use_gluon_verify(num_heads, max_qo_len, kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", UNQUANTIZED_DTYPES)
@pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
def test_unquantized_divisor_heads_keep_gluon_decode(
    gluon_available, kv_cache_dtype, num_heads
):
    """Divisor head counts keep the Gluon single-token decode for bf16.

    Gluon does not scale with KV length the way the asm persistent decode does,
    so this is the faster path where it is usable.
    """
    assert AiterMLAHelper.use_gluon_decode(num_heads, 1, kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", UNQUANTIZED_DTYPES)
@pytest.mark.parametrize("num_heads", [3, 5, 6, 7, 9, 12, 15])
def test_unquantized_non_divisor_heads_use_asm_decode(kv_cache_dtype, num_heads):
    """Non-divisor head counts are padded to 16 and take the asm decode."""
    assert not AiterMLAHelper.use_gluon_decode(num_heads, 1, kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", UNQUANTIZED_DTYPES)
@pytest.mark.parametrize("num_heads", [5, 8, 12])
@pytest.mark.parametrize("max_qo_len", [2, 4, 8, 15])
def test_unquantized_small_head_verify_keeps_gluon(
    gluon_available, kv_cache_dtype, num_heads, max_qo_len
):
    """bf16 has no gqa<16, qseqlen>1 asm kernel, so verify still flattens onto Gluon.

    Unlike the decode predicate, this one does not care whether the head count
    divides 16 -- the flatten reshapes to qseqlen=1 either way.
    """
    assert AiterMLAHelper.use_gluon_verify(num_heads, max_qo_len, kv_cache_dtype)
    # The verify flatten is a separate entry point from the single-token decode.
    assert not AiterMLAHelper.use_gluon_decode(num_heads, max_qo_len, kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", FP8_DTYPES + UNQUANTIZED_DTYPES)
@pytest.mark.parametrize("num_heads", [5, 8, 12, 16, 32])
def test_decode_and_verify_are_disjoint(kv_cache_dtype, num_heads):
    """The two Gluon entry points partition on qlen and never both fire."""
    for max_qo_len in (1, 2, 8):
        assert not (
            AiterMLAHelper.use_gluon_decode(num_heads, max_qo_len, kv_cache_dtype)
            and AiterMLAHelper.use_gluon_verify(num_heads, max_qo_len, kv_cache_dtype)
        )


@pytest.mark.parametrize("num_heads", [1, 2, 3, 5, 6, 7, 8, 9, 12, 15])
def test_padded_query_is_contiguous(num_heads):
    """asm_mla.cu:805 requires Q.is_contiguous().

    Padding a non-divisor head count to 16 tiles the query and slices it back
    down, which yields a non-contiguous view whenever more than one tile is
    needed (12 heads -> repeat to 24 -> slice to 16). The asm kernel rejects
    that outright, so the padding has to materialize the result.
    """
    q = torch.randn(4, num_heads, 576)
    padded = AiterMLAHelper.get_mla_padded_q(num_heads, q)

    assert padded.shape[1] == max(16, num_heads)
    assert padded.is_contiguous()


@pytest.mark.parametrize("num_heads", [1, 2, 3, 5, 6, 7, 8, 9, 12, 15, 16, 32])
def test_pad_unpad_round_trip_preserves_head_order(num_heads):
    """Unpadding recovers each original head from the padded output, in order."""
    q = torch.arange(num_heads, dtype=torch.float32).view(1, num_heads, 1)
    padded = AiterMLAHelper.get_mla_padded_q(num_heads, q)
    unpadded = AiterMLAHelper.get_mla_unpadded_o(num_heads, padded)

    assert unpadded.shape == q.shape
    torch.testing.assert_close(unpadded, q)
